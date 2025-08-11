#!/usr/bin/env python

import matplotlib.pyplot as pyplot
import matplotlib
import pandas as pd
import collections
import re
from typing import Sequence, Union
import itertools
import logging
import math
import numpy as np
import sys
import json
from matplotlib.ticker import ScalarFormatter, FixedLocator
from scipy.stats import zscore

"""Hard-coded parameters for CNVkit. These should not change between runs."""
# Filter thresholds used in constructing the reference (log2 scale)
MIN_REF_COVERAGE = -5.0
MAX_REF_SPREAD = 1.0
NULL_LOG2_COVERAGE = -20.0

# Thresholds used in GC-content masking of bad bins at 'fix' step
GC_MIN_FRACTION = 0.3
GC_MAX_FRACTION = 0.7

# Fragment size for paired-end reads
INSERT_SIZE = 250

# Target/bin names that are not meaningful gene names
# (In some UCSF panels, "CGH" probes denote selected intergenic regions)
IGNORE_GENE_NAMES = ("-", ".", "CGH")
ANTITARGET_NAME = "Antitarget"
ANTITARGET_ALIASES = (ANTITARGET_NAME, "Background")

# PAR1/2 start/end definitions
PSEUDO_AUTSOMAL_REGIONS = {
    "grch37": {"PAR1X": [60000, 2699520], "PAR2X": [154931043, 155260560], "PAR1Y": [10000, 2649520], "PAR2Y": [59034049, 59363566] },
    "grch38": {"PAR1X": [10000, 2781479], "PAR2X": [155701382, 156030895], "PAR1Y": [10000, 2781479], "PAR2Y": [56887902, 57217415] },
}
SUPPORTED_GENOMES_FOR_PAR_HANDLING = PSEUDO_AUTSOMAL_REGIONS.keys()

"""Handle text genomic ranges as named tuples.

A range specification should look like ``chromosome:start-end``, e.g.
``chr1:1234-5678``, with 1-indexed integer coordinates. We also allow
``chr1:1234-`` or ``chr1:-5678``, where missing start becomes 0 and missing end
becomes None.
"""

Region = collections.namedtuple("Region", "chromosome start end")
NamedRegion = collections.namedtuple("NamedRegion", "chromosome start end gene")

re_label = re.compile(r"(\w[\w.]*)?:(\d+)?-(\d+)?\s*(\S+)?")


def from_label(text: str, keep_gene: bool = True) -> Union[Region, NamedRegion]:
    """Parse a chromosomal range specification.

    Parameters
    ----------
    text : string
        Range specification, which should look like ``chr1:1234-5678`` or
        ``chr1:1234-`` or ``chr1:-5678``, where missing start becomes 0 and
        missing end becomes None.
    keep_gene : bool
        If True, include gene names as a 4th field where available; otherwise return a
        3-field Region of chromosomal coordinates without gene labels.
    """
    match = re_label.match(text)
    if not match:
        raise ValueError(
            f"Invalid range spec: {text} (should be like: chr1:2333000-2444000)"
        )

    chrom, start, end, gene = match.groups()
    start = int(start) - 1 if start else None
    end = int(end) if end else None
    if keep_gene:
        gene = gene or ""
        return NamedRegion(chrom, start, end, gene)
    return Region(chrom, start, end)


def to_label(row: Region) -> str:
    """Convert a Region tuple to a region label."""
    return f"{row.chromosome}:{row.start + 1}-{row.end}"


def unpack_range(a_range: Union[str, Sequence]) -> Region:
    """Extract chromosome, start, end from a string or tuple.

    Examples::

        "chr1" -> ("chr1", None, None)
        "chr1:100-123" -> ("chr1", 99, 123)
        ("chr1", 100, 123) -> ("chr1", 100, 123)
    """
    if not a_range:
        return Region(None, None, None)
    if isinstance(a_range, str):
        if ":" in a_range and "-" in a_range:
            return from_label(a_range, keep_gene=False)  # type: ignore
        return Region(a_range, None, None)
    if isinstance(a_range, (list, tuple)):
        if len(a_range) == 3:
            return Region(*a_range)
        if len(a_range) == 4:
            return Region(*a_range[:3])
    raise ValueError(f"Not a range: {a_range!r}")

"""Plotting utilities."""
MB = 1e-6  # To rescale from bases to megabases

def choose_segment_color(segment, highlight_color, default_bright=True):
    """Choose a display color based on a segment's CNA status.

    Uses the fields added by the 'call' command. If these aren't present, use
    `highlight_color` for everything.

    For sex chromosomes, some single-copy deletions or gains might not be
    highlighted, since sample sex isn't used to infer the neutral ploidies.
    """
    neutral_color = TREND_COLOR
    if "cn" not in segment._fields:
        # No 'call' info
        return highlight_color if default_bright else neutral_color

    # Detect copy number alteration
    expected_ploidies = {"chrY": (0, 1), "Y": (0, 1), "chrX": (1, 2), "X": (1, 2)}
    if segment.cn not in expected_ploidies.get(segment.chromosome, [2]):
        return highlight_color

    # Detect regions of allelic imbalance / LOH
    if (
        segment.chromosome not in expected_ploidies
        and "cn1" in segment._fields
        and "cn2" in segment._fields
        and (segment.cn1 != segment.cn2)
    ):
        return highlight_color

    return neutral_color
    
def plot_chromosome_dividers(axis, chrom_sizes, pad=None, along="x"):
    """Given chromosome sizes, plot divider lines and labels.

    Draws black lines between each chromosome, with padding. Labels each chromosome range with the chromosome name,
    centered in the region, under a tick. Sets the axis limits to the covered range.

    By default, the dividers are vertical and the labels are on the X axis of the plot. If the `along` parameter is 'y',
    this is transposed to horizontal dividers and the labels on the Y axis.

    Returns
    -------
    OrderedDict
        A table of the position offsets of each chromosome along the specified axis.
    """
    assert isinstance(chrom_sizes, collections.OrderedDict)
    if pad is None:
        pad = 0.0035 * sum(chrom_sizes.values())
    dividers = []
    centers = []
    starts = collections.OrderedDict()
    curr_offset = pad
    for label, size in list(chrom_sizes.items()):
        starts[label] = curr_offset
        centers.append(curr_offset + 0.5 * size)
        dividers.append(curr_offset + size + pad)
        curr_offset += size + 2 * pad

    if along not in ("x", "y"):
        raise ValueError(
            "Direction for plotting chromosome dividers and labels along must be either x or y."
        )

    if along == "x":
        axis.set_xlim(0, curr_offset)
        for xposn in dividers[:-1]:
            axis.axvline(x=xposn, color="k", linestyle="-")  # Make vertical lines dotted
        # Use chromosome names as x-axis labels (instead of base positions)
        chrom_names = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,'X','Y']
        axis.xaxis.set_ticks_position('top')    # Move the ticks to the top
        axis.xaxis.set_label_position('top')    # Move the labels to the top
    
        axis.set_xticks([])
        axis.tick_params(axis="x", which="both", direction='out')

        # Place all chromosome labels at the same y level (e.g., y=12)
        label_y = 12
        for center, label in zip(centers, chrom_names):
            axis.text(center, label_y, label, ha='center', va='bottom', fontsize=14)

    else:
        axis.set_ylim(0, max(centers) + 2)
        for yposn in dividers[:-1]:
            axis.axhline(y=yposn, color="k", linestyle="-")  # Make horizontal lines dotted
        # Use chromosome names as y-axis labels (instead of base positions)
        axis.set_yticks(centers)
        axis.set_yticklabels(list(chrom_sizes.keys()))
        axis.tick_params(labelsize=100)
        axis.tick_params(axis="y", length=0)
        axis.get_xaxis().tick_bottom()

    return starts

# ________________________________________
# Internal supporting functions

def chromosome_sizes(probes, to_mb=False):
    """Create an ordered mapping of chromosome names to sizes."""
    chrom_sizes = collections.OrderedDict()
    for chrom, rows in probes.by_chromosome():
        chrom_sizes[chrom] = rows["end"].max()
        if to_mb:
            chrom_sizes[chrom] *= MB
    return chrom_sizes

def translate_region_to_bins(region, bins):
    """Map genomic coordinates to bin indices.

    Return a tuple of (chrom, start, end), just like unpack_range.
    """
    if region is None:
        return Region(None, None, None)
    chrom, start, end = unpack_range(region)
    if start is None and end is None:
        return Region(chrom, start, end)
    if start is None:
        start = 0
    if end is None:
        end = float("inf")
    # NB: only bin start positions matter here
    c_bin_starts = bins.data.loc[bins.data.chromosome == chrom, "start"].values
    r_start, r_end = np.searchsorted(c_bin_starts, [start, end])
    return Region(chrom, r_start, r_end)


def translate_segments_to_bins(segments, bins):
    if "probes" in segments and segments["probes"].sum() == len(bins):
        # Segments and .cnr bins already match
        return update_binwise_positions_simple(segments)

    logging.warning(
        "Segments %s 'probes' sum does not match the number of bins in %s",
        segments.sample_id,
        bins.sample_id,
    )
    # Must re-align segments to .cnr bins
    _x, segments, _v = update_binwise_positions(bins, segments)
    return segments


def update_binwise_positions_simple(cnarr):
    start_chunks = []
    end_chunks = []
    is_segment = "probes" in cnarr
    if is_segment:
        cnarr = cnarr[cnarr["probes"] > 0]
    for _chrom, c_arr in cnarr.by_chromosome():
        if is_segment:
            # Segments -- each row can cover many bins
            ends = c_arr["probes"].values.cumsum()
            starts = np.r_[0, ends[:-1]]
        else:
            # Bins -- enumerate rows
            n_bins = len(c_arr)
            starts = np.arange(n_bins)
            ends = np.arange(1, n_bins + 1)
        start_chunks.append(starts)
        end_chunks.append(ends)
    return cnarr.as_dataframe(
        cnarr.data.assign(
            start=np.concatenate(start_chunks), end=np.concatenate(end_chunks)
        )
    )


def update_binwise_positions(cnarr, segments=None, variants=None):
    """Convert start/end positions from genomic to bin-wise coordinates.

    Instead of chromosomal basepairs, the positions indicate enumerated bins.

    Revise the start and end values for all GenomicArray instances at once,
    where the `cnarr` bins are mapped to corresponding `segments`, and
    `variants` are grouped into `cnarr` bins as well -- if multiple `variants`
    rows fall within a single bin, equally-spaced fractional positions are used.

    Returns copies of the 3 input objects with revised `start` and `end` arrays.
    """
    cnarr = cnarr.copy()
    if segments:
        segments = segments.copy()
        seg_chroms = set(segments.chromosome.unique())
    if variants:
        variants = variants.copy()
        var_chroms = set(variants.chromosome.unique())

    # ENH: look into pandas groupby innards to get group indices
    for chrom in cnarr.chromosome.unique():
        # Enumerate bins, starting from 0
        # NB: plotted points will be at +0.5 offsets
        c_idx = cnarr.chromosome == chrom
        c_bins = cnarr[c_idx]  # .copy()
        if segments and chrom in seg_chroms:
            # Match segment boundaries to enumerated bins
            c_seg_idx = (segments.chromosome == chrom).values
            seg_starts = np.searchsorted(
                c_bins.start.values, segments.start.values[c_seg_idx]
            )
            seg_ends = np.r_[seg_starts[1:], len(c_bins)]
            segments.data.loc[c_seg_idx, "start"] = seg_starts
            segments.data.loc[c_seg_idx, "end"] = seg_ends

        if variants and chrom in var_chroms:
            # Match variant positions to enumerated bins, and
            # add fractional increments to multiple variants within 1 bin
            c_varr_idx = (variants.chromosome == chrom).values
            c_varr_df = variants.data[c_varr_idx]
            # Get binwise start indices of the variants
            v_starts = np.searchsorted(c_bins.start.values, c_varr_df.start.values)
            # Overwrite runs of repeats with fractional increments,
            #   adding the cumulative fraction to each repeat
            for idx, size in list(get_repeat_slices(v_starts)):
                v_starts[idx] += np.arange(size) / size
            variant_sizes = c_varr_df.end - c_varr_df.start
            variants.data.loc[c_varr_idx, "start"] = v_starts
            variants.data.loc[c_varr_idx, "end"] = v_starts + variant_sizes

        c_starts = np.arange(len(c_bins))  # c_idx.sum())
        c_ends = np.arange(1, len(c_bins) + 1)
        cnarr.data.loc[c_idx, "start"] = c_starts
        cnarr.data.loc[c_idx, "end"] = c_ends

    return cnarr, segments, variants


def get_repeat_slices(values):
    """Find the location and size of each repeat in `values`."""
    # ENH: look into pandas groupby innards
    offset = 0
    for idx, (_val, rpt) in enumerate(itertools.groupby(values)):
        size = len(list(rpt))
        if size > 1:
            i = idx + offset
            slc = slice(i, i + size)
            yield slc, size
            offset += size - 1


# ________________________________________
# Utilies used by other modules


def cvg2rgb(cvg, desaturate):
    """Choose a shade of red or blue representing log2-coverage value."""
    cutoff = 1.33  # Values above this magnitude are shown with max intensity
    x = min(abs(cvg) / cutoff, 1.0)
    if desaturate:
        # Adjust intensity sigmoidally -- reduce near 0, boost near 1
        # Exponent <1 shifts the fixed point leftward (from x=0.5)
        x = ((1.0 - math.cos(x * math.pi)) / 2.0) ** 0.8
        # Slight desaturation of colors at lower coverage
        s = x**1.2
    else:
        s = x
    if cvg < 0:
        rgb = (1 - s, 1 - s, 1 - 0.25 * x)  # Blueish
    else:
        rgb = (1 - 0.25 * x, 1 - s, 1 - s)  # Reddish
    return cvg


# XXX should this be a CopyNumArray method?
# or: use by_genes internally
# or: have by_genes use this internally
def gene_coords_by_name(probes, names):
    """Find the chromosomal position of each named gene in probes.

    Returns
    -------
    dict
        Of: {chromosome: [(start, end, gene name), ...]}
    """
    names = list(filter(None, set(names)))
    if not names:
        return {}

    # Create an index of gene names
    gene_index = collections.defaultdict(set)
    for i, gene in enumerate(probes["gene"]):
        for gene_name in gene.split(","):
            if gene_name in names:
                gene_index[gene_name].add(i)
    # Retrieve coordinates by name
    all_coords = collections.defaultdict(lambda: collections.defaultdict(set))
    for name in names:
        gene_probes = probes.data.take(sorted(gene_index.get(name, [])))
        if not len(gene_probes):
            raise ValueError(f"No targeted gene named {name!r} found")
        # Find the genomic range of this gene's probes
        start = gene_probes["start"].min()
        end = gene_probes["end"].max()
        chrom = core.check_unique(gene_probes["chromosome"], name)
        # Deduce the unique set of gene names for this region
        uniq_names = set()
        for oname in set(gene_probes["gene"]):
            uniq_names.update(oname.split(","))
        all_coords[chrom][start, end].update(uniq_names)
    # Consolidate each region's gene names into a string
    uniq_coords = {}
    for chrom, hits in all_coords.items():
        uniq_coords[chrom] = [
            (start, end, ",".join(sorted(gene_names)))
            for (start, end), gene_names in hits.items()
        ]
    return uniq_coords


def gene_coords_by_range(probes, chrom, start, end, ignore=IGNORE_GENE_NAMES):
    """Find the chromosomal position of all genes in a range.

    Returns
    -------
    dict
        Of: {chromosome: [(start, end, gene), ...]}
    """
    ignore += ANTITARGET_ALIASES
    # Tabulate the genes in the selected region
    genes = collections.OrderedDict()
    for row in probes.in_range(chrom, start, end):
        name = str(row.gene)
        if name in genes:
            genes[name][1] = row.end
        elif name not in ignore:
            genes[name] = [row.start, row.end]
    # Reorganize the data structure
    return {
        chrom: [(gstart, gend, name) for name, (gstart, gend) in list(genes.items())]
    }

def read_correction_regions_json(json_file):
    """Read the correction regions JSON file containing arm coordinates and copy numbers."""
    correction_regions = {}
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            for arm, info in data.items():
                # Store as tuple of (chromosome, start, end, copy_number)
                correction_regions[arm] = (
                    info['chromosome'],
                    info['start'],
                    info['end'],
                    info['copy_number']
                )
        print(f"Loaded {len(correction_regions)} correction regions from {json_file}")
    except FileNotFoundError:
        print(f"Warning: Correction regions JSON file not found: {json_file}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return {}
    except Exception as e:
        print(f"Error reading correction regions JSON: {e}")
        return {}
    return correction_regions

def apply_copy_number_corrections_from_json(cnr, correction_regions):
    """Apply copy number corrections from JSON to specific chromosome arms."""
    if not correction_regions:
        return cnr
    
    # Make a copy to avoid modifying the original
    cnr_corrected = cnr.copy()
    
    for arm, (chrom, start, end, copy_number) in correction_regions.items():
        # Ensure chromosome naming consistency
        if not chrom.startswith('chr') and cnr_corrected.data['chromosome'].iloc[0].startswith('chr'):
            chrom = 'chr' + chrom
        elif chrom.startswith('chr') and not cnr_corrected.data['chromosome'].iloc[0].startswith('chr'):
            chrom = chrom[3:]
        
        # Find rows in this chromosome arm region
        mask = (cnr_corrected.data['chromosome'] == chrom) & \
               (cnr_corrected.data['start'] >= start) & \
               (cnr_corrected.data['end'] <= end)
        
        if mask.any():
            # Calculate the log2 ratio for the specified copy number
            # Assuming diploid reference (copy number 2), log2(cn/2)
            corrected_log2 = np.log2(copy_number / 2.0)
            
            # Add log2Seg column if it doesn't exist
            if 'log2Seg' not in cnr_corrected.data.columns:
                cnr_corrected.data['log2Seg'] = cnr_corrected.data['log2'].copy()
            
            # Apply the correction to log2Seg
            cnr_corrected.data.loc[mask, 'log2Seg'] = corrected_log2
            
            print(f"Applied correction: {arm} ({chrom}:{start}-{end}) set to copy number {copy_number} (log2={corrected_log2:.3f})")
        else:
            print(f"Warning: No data found for arm {arm} ({chrom}:{start}-{end})")
    
    return cnr_corrected

# Plotting constants
POINT_COLOR = "black"
DEPTH_THRESHOLD = 0
SEG_COLOR = 'black'

def do_scatter(
    cnarr,
    segments=None,
    variants=None,
    show_range=None,
    show_gene=None,
    do_trend=False,
    by_bin=False,
    window_width=1e6,
    y_min=None,
    y_max=None,
    fig_size=(18,12),
    antitarget_marker=None,
    segment_color='black',
    title=None,
    highlight=None
):
    """Plot probe log2 coverages and segmentation calls together."""
    if by_bin:
        # Fixed line: Access 'end' column through the __getitem__ method
        bp_per_bin = sum(c["end"].iat[-1] for _, c in cnarr.by_chromosome()) / len(cnarr)
        window_width /= bp_per_bin
        show_range_bins = translate_region_to_bins(show_range, cnarr)
        cnarr, segments, variants = update_binwise_positions(
            cnarr, segments, variants
        )
        global MB
        orig_mb = MB
        MB = 1

    if not show_gene and not show_range:
        fig = genome_scatter(
            cnarr, segments, variants, do_trend, y_min, y_max, title, segment_color, highlight
        )
    else:
        if by_bin:
            show_range = show_range_bins
        fig = chromosome_scatter(
            cnarr,
            segments,
            variants,
            show_range,
            show_gene,
            antitarget_marker,
            do_trend,
            by_bin,
            window_width,
            y_min,
            y_max,
            title,
            'black',
            highlight
        )

    if by_bin:
        # Reset to avoid permanently altering the value of cnvlib.scatter.MB
        MB = orig_mb
    fig.set_dpi(300)
    if fig_size:
        width, height = fig_size
        fig.set_size_inches(w=width, h=height)
    return fig

# === Genome-level scatter plots ===
def genome_scatter(
    cnarr,
    segments=None,
    variants=None,
    do_trend=False,
    y_min=None,
    y_max=None,
    title=None,
    segment_color=SEG_COLOR,
    highlight=None
):
    """Plot all chromosomes, concatenated on one plot."""
    if (cnarr or segments) and variants:
        # Lay out top 3/5 for the CN scatter, bottom 2/5 for SNP plot
        axgrid = pyplot.GridSpec(5, 1, hspace=0.85)
        axis = pyplot.subplot(axgrid[:3])
        axis2 = pyplot.subplot(axgrid[3:], sharex=axis)
        # Place chromosome labels between the CNR and SNP plots
        axis2.tick_params(labelbottom=False)
        chrom_sizes = chromosome_sizes(cnarr or segments)
        axis2 = cnv_on_genome(
            axis2, variants, chrom_sizes, segments, do_trend, segment_color, highlight
        )
    else:
        _fig, axis = pyplot.subplots()
    if title is None:
        title = (cnarr or segments or variants).sample_id
    if cnarr or segments:
        # axis.set_title(title)
        axis.set_title(title, loc='center', fontsize=26, pad=50)
        axis = cnv_on_genome(
            axis, cnarr, segments, do_trend, y_min, y_max, segment_color, highlight
        )
    else:
        axis.set_title(f"Variant allele frequencies: {title}", loc='left', fontsize=16, pad=20)
        chrom_sizes = collections.OrderedDict(
            (chrom, subarr["end"].max()) for chrom, subarr in variants.by_chromosome()
        )
        axis = cnv_on_genome(
            axis, variants, chrom_sizes, segments, do_trend, segment_color
        )
    return axis.get_figure()

def highlight_positions(subprobes_filtered, highlight_ranges, highlight_color="blue"):
    """
    Highlights specified positions on the chromosome based on proportional mapping
    between genomic coordinates and bin indices.
    
    Parameters:
    - subprobes_filtered: The filtered probes CopyNumArray with index-based positions.
    - highlight_ranges: A dictionary with genomic coordinates to highlight.
    - highlight_color: The color used to highlight the specified range.
    """
    # Create a new array for colors, initially set to POINT_COLOR
    colors = np.full(len(subprobes_filtered), POINT_COLOR)

    if highlight_ranges is None:
        subprobes_filtered.data['color'] = 'black'
        return subprobes_filtered

    # Reference chromosome sizes (approximate, in bp)
    chrom_sizes = {
        "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
        "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 146364022,
        "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
        "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
        "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
        "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415
    }

    # Current chromosome being processed
    current_chrom = subprobes_filtered.data['chromosome'].iloc[0] if len(subprobes_filtered) > 0 else None
    
    if current_chrom not in highlight_ranges:
        subprobes_filtered.data['color'] = POINT_COLOR
        return subprobes_filtered
    
    # Get the total number of bins for this chromosome
    total_bins = len(subprobes_filtered)
    
    # Process highlight ranges for this chromosome
    for start_pos, end_pos in highlight_ranges[current_chrom]:
        # Get the chromosome size
        chrom_size = chrom_sizes.get(current_chrom, None)
        if not chrom_size:
            continue
        
        # Convert genomic positions to bin indices using proportions
        # Add a small buffer to ensure all points are included
        start_proportion = max(0, (start_pos / chrom_size) - 0.005)
        end_proportion = min(1.0, (end_pos / chrom_size) + 0.005)
        
        start_bin = int(start_proportion * total_bins)
        end_bin = int(end_proportion * total_bins)
        
        # Create a mask for all bins in this range
        bin_indices = np.arange(len(subprobes_filtered))
        is_in_range = (bin_indices >= start_bin) & (bin_indices <= end_bin)
        
        # Apply highlight color to all bins in range
        colors[is_in_range] = highlight_color
    
    subprobes_filtered.data['color'] = colors
    return subprobes_filtered

def cnv_on_genome(
    axis,
    probes,
    segments,
    do_trend=False,
    y_min=None,
    y_max=None,
    segment_color=SEG_COLOR,
    highlight=None
):
    """Plot bin ratios and/or segments for all chromosomes on one plot."""
    # Configure axes etc.
    axis.axhline(color="k")
    axis.set_ylabel("Copy Number", fontsize=25, labelpad=8)
    axis.set_xlabel("", fontsize=30, labelpad=57) 

    if not (y_min and y_max):
        if segments:
            # Auto-scale y-axis according to segment mean-coverage values
            # (Avoid spuriously low log2 values in HLA and chrY)
            low_chroms = segments.chromosome.isin(("6", "chr6", "Y", "chrY"))
            seg_auto_vals = segments[~low_chroms]["log2"].dropna()
            if not y_min:
                y_min = (
                    np.nanmin([seg_auto_vals.min() - 0.2, -1.5])
                    if len(seg_auto_vals)
                    else -2.5
                )
            if not y_max:
                y_max = (
                    np.nanmax([seg_auto_vals.max() + 0.2, 1.5])
                    if len(seg_auto_vals)
                    else 2.5
                )
        else:
            if not y_min:
                y_min = -2.5
            if not y_max:
                y_max = 2.5
    # axis.set_ylim(y_min, y_max)
    axis.set_ylim(0, y_max)
    axis.tick_params(axis='y', labelsize=30)
    axis.set_yticks(range(0, y_max + 1))
    
    # Group probes by chromosome (to calculate plotting coordinates)
    if probes:
        chrom_sizes = chromosome_sizes(probes)
        chrom_probes = dict(probes.by_chromosome())
        # Precalculate smoothing window size so all chromosomes have similar
        # degree of smoothness
        # NB: Target panel has ~1k bins/chrom. -> 250-bin window
        #     Exome: ~10k bins/chrom. -> 2500-bin window
        window_size = int(round(0.15 * len(probes) / probes.chromosome.nunique()))
    else:
        chrom_sizes = chromosome_sizes(segments)
    # Same for segment calls
    chrom_segs = dict(segments.by_chromosome()) if segments else {}
    
    copy_nums=np.arange(7)
    ratio_thresholds=np.log2((copy_nums+.5) / 3)
    # print(ratio_thresholds)
    # Plot points & segments
    x_starts = plot_chromosome_dividers(axis, chrom_sizes)
    for chrom, x_offset in x_starts.items():
        if probes and chrom in chrom_probes:
            subprobes = chrom_probes[chrom]
            subprobes_filtered = subprobes[
                (subprobes['log2'] >= ratio_thresholds[0] - 0.5) & 
                (subprobes['log2'] <= ratio_thresholds[-1] + 0.5)
            ]
            log2_values = subprobes_filtered['log2']
            original_count = len(subprobes_filtered)
            
            # Work with the underlying data DataFrame
            subprobes_filtered.data['z_scores'] = zscore(log2_values)
            subprobes_filtered = subprobes_filtered[(subprobes_filtered.data['z_scores'] >= -3) & (subprobes_filtered.data['z_scores'] <= 3)]
            
            # Highlight specified positions
            if highlight is not None:
                subprobes_filtered = highlight_positions(subprobes_filtered, highlight_ranges=highlight)
            else:
                subprobes_filtered = highlight_positions(subprobes_filtered, highlight_ranges=None)
            
            x = 0.5 * (subprobes_filtered["start"] + subprobes_filtered["end"]) + x_offset
            alpha_values = np.ones(len(subprobes_filtered))
            highlight_color = "blue"
            not_highlighted = subprobes_filtered.data['color'] != highlight_color
            positive_mask = not_highlighted & (subprobes_filtered.data['z_scores'] >= 2) & (subprobes_filtered.data['z_scores'] <= 3)
            alpha_values[positive_mask] = 1 - (subprobes_filtered.data['z_scores'][positive_mask] - 2) / (3 - 2)
            negative_mask = not_highlighted & (subprobes_filtered.data['z_scores'] <= -2) & (subprobes_filtered.data['z_scores'] >= -3)
            alpha_values[negative_mask] = 1 - (abs(subprobes_filtered.data['z_scores'][negative_mask]) - 2) / (3 - 2)
            
            # Plot the main data points (black dots)
            axis.scatter(x, 2 * 2**subprobes_filtered["log2"], marker=".", color=subprobes_filtered.data['color'], alpha=alpha_values)
            
            # Plot the corrected segmentation line (red dots) - use log2Seg if available, otherwise log2
            if "log2Seg" in subprobes_filtered.data.columns:
                seg_values = subprobes_filtered.data["log2Seg"]
            else:
                seg_values = subprobes_filtered["log2"]
            axis.scatter(x, 2 * 2**seg_values, marker=".", color='red', alpha=0.2, s=20)
            
            axis.set_yscale('log', base=10)
            ticks = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 10]
            axis.set_ylim(0.05, 11)
            axis.yaxis.set_major_locator(FixedLocator(ticks))
            axis.set_yticks(ticks)
            axis.set_yticklabels(
                [str(int(t)) if t == int(t) else str(t) for t in ticks], fontsize=18
            )
            for y in ticks:
                axis.axhline(y=y, color='black', linestyle='-', linewidth=0.7)
    return axis

# GenomicArray and CopyNumArray classes with proper attribute access
class GenomicArray:
    def __init__(self, data_table, meta_dict=None):
        self.data = data_table if data_table is not None else pd.DataFrame()
        self.meta = meta_dict if meta_dict is not None else {}
    
    def copy(self):
        return GenomicArray(self.data.copy(), self.meta.copy())
    
    def by_chromosome(self):
        for chrom, group in self.data.groupby('chromosome', sort=False):
            yield chrom, GenomicArray(group, self.meta)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        else:
            return GenomicArray(self.data[key], self.meta)
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying DataFrame."""
        if name in self.data.columns:
            return self.data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def in_range(self, chrom, start, end):
        """Return rows within the specified genomic range."""
        mask = (self.data['chromosome'] == chrom)
        if start is not None:
            mask &= (self.data['end'] >= start)
        if end is not None:
            mask &= (self.data['start'] <= end)
        return self.data[mask].itertuples(index=False)
    
    def as_dataframe(self, df):
        """Return a new instance with updated dataframe."""
        return self.__class__(df, self.meta)
    
    def get(self, key, default=None):
        """Get a column value, with a default if it doesn't exist."""
        if key in self.data.columns:
            return self.data[key]
        return default


class CopyNumArray(GenomicArray):
    @property
    def sample_id(self):
        return self.meta.get('sample_id', '')


def read_cna(infile, sample_id=None, meta=None):
    """Read a CNVkit file (.cnn, .cnr, .cns) to create a CopyNumArray object."""
    try:
        data = pd.read_csv(infile, sep='\t', dtype={'chromosome': str})
        if meta is None:
            meta = {}
        if sample_id:
            meta['sample_id'] = sample_id
        return CopyNumArray(data, meta)
    except Exception as e:
        print(f"Error reading file {infile}: {e}")
        return CopyNumArray(pd.DataFrame(), {})

# Main execution
if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) not in (5, 6, 7):
        sys.exit("Usage: Plotting.py <intersect_cnr> <sorted_cns> <cleanname> <output> <correction_regions.json> [highlight_json]")

    # Get file paths and sample name from command-line arguments
    intersect_cnr = sys.argv[1]
    sorted_cns = sys.argv[2]
    cleanname = sys.argv[3]
    output = sys.argv[4]
    correction_regions_json = sys.argv[5]
    
    highlight_ranges = None
    
    if len(sys.argv) == 7:
        try:
            with open(sys.argv[6], 'r') as f:
                highlight_ranges = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read highlight file: {e}")

    # Read the correction regions JSON file
    correction_regions = read_correction_regions_json(correction_regions_json)

    # Load the .cnr file
    cnr = read_cna(intersect_cnr, sample_id=cleanname)
    cns = read_cna(sorted_cns, sample_id=cleanname)

    # Apply copy number corrections if any
    if correction_regions:
        cnr = apply_copy_number_corrections_from_json(cnr, correction_regions)
        print(f"Applied {len(correction_regions)} copy number corrections")

    # Generate the plot
    fig = do_scatter(cnr, segments=cns, title=cleanname, y_min=0, y_max=8, by_bin=True, highlight=highlight_ranges)

    # Save the plot to a file
    fig.savefig(output, bbox_inches="tight")
    print(f"Plot saved to: {output}")
