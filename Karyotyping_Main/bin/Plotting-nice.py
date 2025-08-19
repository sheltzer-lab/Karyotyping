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
import os

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

    
def plot_chromosome_dividers(axis, chrom_sizes, pad=None, along="x", y_max=None):
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
            axis.axvline(x=xposn, color="k")
        # Use chromosome names as x-axis labels (instead of base positions)
        chrom_names = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,'X','Y']

        # First, clear the original x-ticks and labels
        axis.xaxis.set_ticks_position('top')    # Move the ticks to the top
        axis.xaxis.set_label_position('top')    # Move the labels to the top
    
        axis.set_xticks([])
        # Place ticks on the top axis
        axis.tick_params(axis="x", which="both", direction='out')
        for center, label in zip(centers, chrom_names):
            axis.text(center, y_max, label, ha='center', va='bottom', fontsize=14)
            
        # Alternate the positions of the labels
        # for i, (center, label) in enumerate(zip(centers, chrom_names)):
        #     if i % 2 == 0:
        #         axis.text(center, 1.066*y_max, label, ha='center', va='top', fontsize=25)
        #     else:
        #         axis.text(center, 1.033*y_max, label, ha='center', va='top', fontsize=25)

        # Ensure y ticks are on the left side of the y-axis
        axis.get_yaxis().tick_left()

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

# ________________________________________
# Utilies used by other modules


def read_tab(infile):
    """Read tab-separated data with column names in the first row.

    The format is BED-like, but with a header row included and with
    arbitrary extra columns.
    """
    dframe = pd.read_csv(infile, sep="\t", dtype={"chromosome": "str"})
    if "log2" in dframe.columns:
        # Every bin needs a log2 value; the others can be NaN
        d2 = dframe.dropna(subset=["log2"])
        if len(d2) < len(dframe):
            logging.warning(
                "Dropped %d rows with missing log2 values", len(dframe) - len(d2)
            )
            dframe = d2.copy()
    return dframe


from typing import Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Union
from collections import OrderedDict

Numeric = Union[int, float, np.number]

class GenomicArray:
    """An array of genomic intervals. Base class for genomic data structures.

    Can represent most BED-like tabular formats with arbitrary additional
    columns.
    """

    _required_columns = ("chromosome", "start", "end")
    _required_dtypes = (str, int, int)

    def __init__(
        self,
        data_table: Optional[Union[Sequence, pd.DataFrame]],
        meta_dict: Optional[Mapping] = None,
    ):
        # Validation
        if (
            data_table is None
            or (isinstance(data_table, (list, tuple)) and not len(data_table))
            or (isinstance(data_table, pd.DataFrame) and not len(data_table.columns))
        ):
            data_table = self._make_blank()
        else:
            if not isinstance(data_table, pd.DataFrame):
                # Rarely if ever needed -- prefer from_rows, from_columns, etc.
                data_table = pd.DataFrame(data_table)
            if not all(c in data_table.columns for c in self._required_columns):
                raise ValueError(
                    "data table must have at least columns "
                    + f"{self._required_columns!r}; got {tuple(data_table.columns)!r}"
                )
            # Ensure columns are the right type
            # (in case they've been automatically converted to the wrong type,
            # e.g. chromosome names as integers; genome coordinates as floats)
            if len(data_table):

                def ok_dtype(col, dtype):
                    return isinstance(data_table[col].iat[0], dtype)

            else:

                def ok_dtype(col, dtype):
                    return data_table[col].dtype == np.dtype(dtype)

            recast_cols = {
                col: dtype
                for col, dtype in zip(self._required_columns, self._required_dtypes)
                if not ok_dtype(col, dtype)
            }
            if recast_cols:
                data_table = data_table.astype(recast_cols)

        self.data = data_table
        self.meta = dict(meta_dict) if meta_dict is not None and len(meta_dict) else {}

    @classmethod
    def _make_blank(cls) -> pd.DataFrame:
        """Create an empty dataframe with the columns required by this class."""
        spec = list(zip(cls._required_columns, cls._required_dtypes))
        try:
            arr = np.zeros(0, dtype=spec)
            return pd.DataFrame(arr)
        except TypeError as exc:
            raise TypeError(r"{exc}: {spec}") from exc

    @classmethod
    def from_columns(
        cls, columns: Mapping[str, Iterable], meta_dict: Optional[Mapping] = None
    ):
        """Create a new instance from column arrays, given as a dict."""
        table = pd.DataFrame.from_dict(columns)
        ary = cls(table, meta_dict)
        ary.sort_columns()
        return ary

    @classmethod
    def from_rows(
        cls,
        rows: Iterable,
        columns: Optional[Sequence[str]] = None,
        meta_dict: Optional[Mapping] = None,
    ):
        """Create a new instance from a list of rows, as tuples or arrays."""
        if columns is None:
            columns = cls._required_columns
        if isinstance(rows, pd.DataFrame):
            table = rows[columns].reset_index(drop=True)
        else:
            table = pd.DataFrame.from_records(rows, columns=columns)
        return cls(table, meta_dict)

    def as_columns(self, **columns):
        """Wrap the named columns in this instance's metadata."""
        return self.__class__.from_columns(columns, self.meta)
        # return self.__class__(self.data.loc[:, columns], self.meta.copy())

    def as_dataframe(self, dframe: pd.DataFrame, reset_index: bool = False):
        """Wrap the given pandas DataFrame in this instance's metadata."""
        if reset_index:
            dframe = dframe.reset_index(drop=True)
        return self.__class__(dframe, self.meta.copy())

    def as_series(self, arraylike: Iterable) -> pd.Series:
        """Coerce `arraylike` to a Series with this instance's index."""
        return pd.Series(arraylike, index=self.data.index)

    def as_rows(self, rows: Iterable):
        """Wrap the given rows in this instance's metadata."""
        try:
            out = self.from_rows(rows, columns=self.data.columns, meta_dict=self.meta)
        except AssertionError as exc:
            columns = self.data.columns.tolist()
            firstrow = next(iter(rows))
            raise RuntimeError(
                f"Passed {len(columns)} columns {columns!r}, but "
                f"{len(firstrow)} elements in first row: {firstrow}"
            ) from exc
        return out

    # Container behaviour

    def __bool__(self) -> bool:
        return bool(len(self.data))

    def __eq__(self, other) -> bool:
        return isinstance(other, self.__class__) and self.data.equals(other.data)

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key) -> bool:
        return key in self.data.columns

    def __getitem__(self, index) -> Union[pd.Series, pd.DataFrame]:
        """Access a portion of the data.

        Cases:

        - single integer: a row, as pd.Series
        - string row name: a column, as pd.Series
        - a boolean array: masked rows, as_dataframe
        - tuple of integers: selected rows, as_dataframe
        """
        if isinstance(index, int):
            # A single row
            return self.data.iloc[index]
            # return self.as_dataframe(self.data.iloc[index:index+1])
        if isinstance(index, str):
            # A column, by name
            return self.data[index]
        if (
            isinstance(index, tuple)
            and len(index) == 2
            and index[1] in self.data.columns
        ):
            # Row index, column index -> cell value
            return self.data.loc[index]
        if isinstance(index, slice):
            # return self.as_dataframe(self.data.take(index))
            return self.as_dataframe(self.data[index])
        # Iterable -- selected row indices or boolean array, probably
        try:
            if isinstance(index, type(None)) or len(index) == 0:
                empty = pd.DataFrame(columns=self.data.columns)
                return self.as_dataframe(empty)
        except TypeError as exc:
            raise TypeError(
                f"object of type {type(index)!r} "
                f"cannot be used as an index into a {self.__class__.__name__}"
            ) from exc
        return self.as_dataframe(self.data[index])
        # return self.as_dataframe(self.data.take(index))

    def __setitem__(self, index, value):
        """Assign to a portion of the data."""
        if isinstance(index, int):
            self.data.iloc[index] = value
        elif isinstance(index, str):
            self.data[index] = value
        elif (
            isinstance(index, tuple)
            and len(index) == 2
            and index[1] in self.data.columns
        ):
            self.data.loc[index] = value
        else:
            assert isinstance(index, slice) or len(index) > 0
            self.data[index] = value

    def __delitem__(self, index):
        return NotImplemented

    def __iter__(self):
        return self.data.itertuples(index=False)

    __next__ = next

    @property
    def chromosome(self) -> pd.Series:
        """Get column 'chromosome'."""
        return self.data["chromosome"]

    @property
    def start(self) -> pd.Series:
        """Get column 'start'."""
        return self.data["start"]

    @property
    def end(self) -> pd.Series:
        """Get column 'end'."""
        return self.data["end"]

    @property
    def sample_id(self) -> pd.Series:
        """Get metadata field 'sample_id'."""
        return self.meta.get("sample_id")

    # Traversal

    def autosomes(self, also=None):
        """Select chromosomes w/ integer names, ignoring any 'chr' prefixes."""
        is_auto = self.chromosome.str.match(r"(chr)?\d+$", na=False)
        if not is_auto.any():
            # The autosomes, if any, are not named with plain integers
            return self
        if also is not None:
            if isinstance(also, pd.Series):
                is_auto |= also
            else:
                # The assumption is that `also` is a single chromosome name or an iterable thereof.
                if isinstance(also, str):
                    also = [also]
                for a_chrom in also:
                    is_auto |= self.chromosome == a_chrom
        return self[is_auto]

    def by_arm(self, min_gap_size: Union[int, float] = 1e5, min_arm_bins: int = 50):
        """Iterate over bins grouped by chromosome arm (inferred)."""
        # ENH:
        # - Accept GArray of actual centromere regions as input
        #   -> find largest gap (any size) within cmere region, split there
        # - Cache centromere locations once found
        self.data.chromosome = self.data.chromosome.astype(str)
        for chrom, subtable in self.data.groupby("chromosome", sort=False):
            margin = max(min_arm_bins, int(round(0.1 * len(subtable))))
            if len(subtable) > 2 * margin + 1:
                # Found a candidate centromere
                gaps = (
                    subtable.start.values[margin + 1 : -margin]
                    - subtable.end.values[margin : -margin - 1]
                )
                cmere_idx = gaps.argmax() + margin + 1
                cmere_size = gaps[cmere_idx - margin - 1]
            else:
                cmere_idx = 0
                cmere_size = 0
            if cmere_idx and cmere_size >= min_gap_size:
                logging.debug(
                    "%s centromere at %d of %d bins (size %s)",
                    chrom,
                    cmere_idx,
                    len(subtable),
                    cmere_size,
                )
                p_arm = subtable.index[:cmere_idx]
                yield chrom, self.as_dataframe(subtable.loc[p_arm, :])
                q_arm = subtable.index[cmere_idx:]
                yield chrom, self.as_dataframe(subtable.loc[q_arm, :])
            else:
                # No centromere found -- emit the whole chromosome
                if cmere_idx:
                    logging.debug(
                        "%s: Ignoring centromere at %d of %d bins (size %s)",
                        chrom,
                        cmere_idx,
                        len(subtable),
                        cmere_size,
                    )
                else:
                    logging.debug("%s: Skipping centromere search, too small", chrom)
                yield chrom, self.as_dataframe(subtable)

    def by_chromosome(self) -> Iterator:
        """Iterate over bins grouped by chromosome name."""
        for chrom, subtable in self.data.groupby("chromosome", sort=False):
            yield chrom, self.as_dataframe(subtable)

    def by_ranges(
        self, other, mode: str = "outer", keep_empty: bool = True
    ) -> Iterator:
        """Group rows by another GenomicArray's bin coordinate ranges.

        For example, this can be used to group SNVs by CNV segments.

        Bins in this array that fall outside the other array's bins are skipped.

        Parameters
        ----------
        other : GenomicArray
            Another GA instance.
        mode : string
            Determines what to do with bins that overlap a boundary of the
            selection. Possible values are:

            - ``inner``: Drop the bins on the selection boundary, don't emit them.
            - ``outer``: Keep/emit those bins as they are.
            - ``trim``: Emit those bins but alter their boundaries to match the
              selection; the bin start or end position is replaced with the
              selection boundary position.
        keep_empty : bool
            Whether to also yield `other` bins with no overlapping bins in
            `self`, or to skip them when iterating.

        Yields
        ------
        tuple
            (other bin, GenomicArray of overlapping rows in self)
        """
        for bin_row, subrange in by_ranges(self.data, other.data, mode, keep_empty):
            if len(subrange):
                yield bin_row, self.as_dataframe(subrange)
            elif keep_empty:
                yield bin_row, self.as_rows(subrange)

    def coords(self, also: Union[str, Iterable[str]] = ()):
        """Iterate over plain coordinates of each bin: chromosome, start, end.

        Parameters
        ----------
        also : str, or iterable of strings
            Also include these columns from `self`, in addition to chromosome,
            start, and end.

        Example, yielding rows in BED format:

        >>> probes.coords(also=["gene", "strand"])
        """
        cols = list(GenomicArray._required_columns)
        if also:
            if isinstance(also, str):
                cols.append(also)
            else:
                cols.extend(also)
        coordframe = self.data.loc[:, cols]
        return coordframe.itertuples(index=False)

    def labels(self) -> pd.Series:
        """Get chromosomal coordinates as genomic range labels."""
        return self.data.apply(to_label, axis=1)

    def in_range(
        self,
        chrom: Optional[str] = None,
        start: Optional[Numeric] = None,
        end: Optional[Numeric] = None,
        mode: str = "outer",
    ):
        """Get the GenomicArray portion within the given genomic range.

        Parameters
        ----------
        chrom : str or None
            Chromosome name to select. Use None if `self` has only one
            chromosome.
        start : int or None
            Start coordinate of range to select, in 0-based coordinates.
            If None, start from 0.
        end : int or None
            End coordinate of range to select. If None, select to the end of the
            chromosome.
        mode : str
            As in `by_ranges`: ``outer`` includes bins straddling the range
            boundaries, ``trim`` additionally alters the straddling bins'
            endpoints to match the range boundaries, and ``inner`` excludes
            those bins.

        Returns
        -------
        GenomicArray
            The subset of `self` enclosed by the specified range.
        """
        starts = [int(start)] if start is not None else None
        ends = [int(end)] if end is not None else None
        results = iter_ranges(self.data, chrom, starts, ends, mode)
        return self.as_dataframe(next(results))

    def in_ranges(
        self,
        chrom: Optional[str] = None,
        starts: Optional[Sequence[Numeric]] = None,
        ends: Optional[Sequence[Numeric]] = None,
        mode: str = "outer",
    ):
        """Get the GenomicArray portion within the specified ranges.

        Similar to `in_ranges`, but concatenating the selections of all the
        regions specified by the `starts` and `ends` arrays.

        Parameters
        ----------
        chrom : str or None
            Chromosome name to select. Use None if `self` has only one
            chromosome.
        starts : int array, or None
            Start coordinates of ranges to select, in 0-based coordinates.
            If None, start from 0.
        ends : int array, or None
            End coordinates of ranges to select. If None, select to the end of the
            chromosome. If `starts` and `ends` are both specified, they must be
            arrays of equal length.
        mode : str
            As in `by_ranges`: ``outer`` includes bins straddling the range
            boundaries, ``trim`` additionally alters the straddling bins'
            endpoints to match the range boundaries, and ``inner`` excludes
            those bins.

        Returns
        -------
        GenomicArray
            Concatenation of all the subsets of `self` enclosed by the specified
            ranges.
        """
        table = pd.concat(iter_ranges(self.data, chrom, starts, ends, mode), sort=False)
        return self.as_dataframe(table)

    def into_ranges(
        self, other, column: str, default, summary_func: Optional[Callable] = None
    ):
        """Re-bin values from `column` into the corresponding ranges in `other`.

        Match overlapping/intersecting rows from `other` to each row in `self`.
        Then, within each range in `other`, extract the value(s) from `column`
        in `self`, using the function `summary_func` to produce a single value
        if multiple bins in `self` map to a single range in `other`.

        For example, group SNVs (self) by CNV segments (other) and calculate the
        median (summary_func) of each SNV group's allele frequencies.

        Parameters
        ----------
        other : GenomicArray
            Ranges into which the overlapping values of `self` will be
            summarized.
        column : string
            Column name in `self` to extract values from.
        default
            Value to assign to indices in `other` that do not overlap any bins in
            `self`. Type should be the same as or compatible with the output
            field specified by `column`, or the output of `summary_func`.
        summary_func : callable, dict of string-to-callable, or None
            Specify how to reduce 1 or more `other` rows into a single value for
            the corresponding row in `self`.

                - If callable, apply to the `column` field each group of rows in
                  `other` column.
                - If a single-element dict of column name to callable, apply to that
                  field in `other` instead of `column`.
                - If None, use an appropriate summarizing function for the datatype
                  of the `column` column in `other` (e.g. median of numbers,
                  concatenation of strings).
                - If some other value, assign that value to `self` wherever there is
                  an overlap.

        Returns
        -------
        pd.Series
            The extracted and summarized values from `self` corresponding to
            other's genomic ranges, the same length as `other`.
        """
        if column not in self:
            logging.warning("No '%s' column available for summary calculation", column)
            return pd.Series(np.repeat(default, len(other)))
        return into_ranges(self.data, other.data, column, default, summary_func)

    def iter_ranges_of(
        self, other, column: str, mode: str = "outer", keep_empty: bool = True
    ):
        """Group rows by another GenomicArray's bin coordinate ranges.

        For example, this can be used to group SNVs by CNV segments.

        Bins in this array that fall outside the other array's bins are skipped.

        Parameters
        ----------
        other : GenomicArray
            Another GA instance.
        column : string
            Column name in `self` to extract values from.
        mode : string
            Determines what to do with bins that overlap a boundary of the
            selection. Possible values are:

            - ``inner``: Drop the bins on the selection boundary, don't emit them.
            - ``outer``: Keep/emit those bins as they are.
            - ``trim``: Emit those bins but alter their boundaries to match the
              selection; the bin start or end position is replaced with the
              selection boundary position.
        keep_empty : bool
            Whether to also yield `other` bins with no overlapping bins in
            `self`, or to skip them when iterating.

        Yields
        ------
        tuple
            (other bin, GenomicArray of overlapping rows in self)
        """
        if column not in self.data.columns:
            raise ValueError(f"No column named {column!r} in this object")
        ser = self.data[column]
        for slc in iter_slices(self.data, other.data, mode, keep_empty):
            yield ser[slc]

    # Modification

    def add(self, other):
        """Combine this array's data with another GenomicArray (in-place).

        Any optional columns must match between both arrays.
        """
        if not isinstance(other, self.__class__):
            raise ValueError(
                f"Argument (type {type(other)}) is not a {self.__class__} instance"
            )
        if len(other.data):
            self.data = pd.concat([self.data, other.data], ignore_index=True)
            self.sort()

    def concat(self, others):
        """Concatenate several GenomicArrays, keeping this array's metadata.

        This array's data table is not implicitly included in the result.
        """
        table = pd.concat([otr.data for otr in others], ignore_index=True)
        result = self.as_dataframe(table)
        result.sort()
        return result

    def copy(self):
        """Create an independent copy of this object."""
        return self.as_dataframe(self.data.copy())

    def add_columns(self, **columns):
        """Add the given columns to a copy of this GenomicArray.

        Parameters
        ----------
        **columns : array
            Keyword arguments where the key is the new column's name and the
            value is an array of the same length as `self` which will be the new
            column's values.

        Returns
        -------
        GenomicArray or subclass
            A new instance of `self` with the given columns included in the
            underlying dataframe.
        """
        return self.as_dataframe(self.data.assign(**columns))

    def keep_columns(self, colnames):
        """Extract a subset of columns, reusing this instance's metadata."""
        colnames = self.data.columns.intersection(colnames)
        return self.__class__(self.data.loc[:, colnames], self.meta.copy())

    def drop_extra_columns(self):
        """Remove any optional columns from this GenomicArray.

        Returns
        -------
        GenomicArray or subclass
            A new copy with only the minimal set of columns required by the
            class (e.g. chromosome, start, end for GenomicArray; may be more for
            subclasses).
        """
        table = self.data.loc[:, self._required_columns]
        return self.as_dataframe(table)

    def filter(self, func=None, **kwargs):
        """Take a subset of rows where the given condition is true.

        Parameters
        ----------
        func : callable
            A boolean function which will be applied to each row to keep rows
            where the result is True.
        **kwargs : string
            Keyword arguments like ``chromosome="chr7"`` or
            ``gene="Antitarget"``, which will keep rows where the keyed field
            equals the specified value.

        Return
        ------
        GenomicArray
            Subset of `self` where the specified condition is True.
        """
        table = self.data
        if func is not None:
            table = table[table.apply(func, axis=1)]
        for key, val in list(kwargs.items()):
            assert key in self
            table = table[table[key] == val]
        return self.as_dataframe(table)

    def shuffle(self):
        """Randomize the order of bins in this array (in-place)."""
        order = np.arange(len(self.data))
        np.random.seed(0xA5EED)
        np.random.shuffle(order)
        self.data = self.data.iloc[order]
        return order

    def sort(self):
        """Sort this array's bins in-place, with smart chromosome ordering."""
        sort_key = self.data.chromosome.apply(sorter_chrom)
        self.data = (
            self.data.assign(_sort_key_=sort_key)
            .sort_values(by=["_sort_key_", "start", "end"], kind="mergesort")
            .drop("_sort_key_", axis=1)
            .reset_index(drop=True)
        )

    def sort_columns(self):
        """Sort this array's columns in-place, per class definition."""
        extra_cols = []
        for col in self.data.columns:
            if col not in self._required_columns:
                extra_cols.append(col)
        sorted_colnames = list(self._required_columns) + sorted(extra_cols)
        assert len(sorted_colnames) == len(self.data.columns)
        self.data = self.data.reindex(columns=sorted_colnames)

    # Genome arithmetic

    def merge(
        self,
        bp: int = 0,
        stranded: bool = False,
        combine: Optional[Dict[str, Callable]] = None,
    ):
        """Merge adjacent or overlapping regions into single rows.

        Similar to 'bedtools merge'.
        """
        return self.as_dataframe(merge(self.data, bp, stranded, combine))

    




def read_cna(infile, sample_id=None, meta=None):
    """Read a CNVkit file (.cnn, .cnr, .cns) to create a CopyNumArray object."""
    return read(infile, into=CopyNumArray, sample_id=sample_id)


def fbase(fname):
    """Strip directory and all extensions from a filename."""
    base = os.path.basename(fname)
    # Gzip extension usually follows another extension
    if base.endswith(".gz"):
        base = base[:-3]
    # Cases to drop more than just the last dot
    known_multipart_exts = (
        ".antitargetcoverage.cnn",
        ".targetcoverage.cnn",
        ".antitargetcoverage.csv",
        ".targetcoverage.csv",
        # Pipeline suffixes
        ".recal.bam",
        ".deduplicated.realign.bam",
    )
    for ext in known_multipart_exts:
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    else:
        base = base.rsplit(".", 1)[0]
    return base



def get_filename(infile):
    if isinstance(infile, str):
        return infile
    if hasattr(infile, 'name') and infile not in (sys.stdout, sys.stderr):
        # File(-like) handle
        return infile.name
        
def read(infile, into=None, sample_id=None):
    """Read tab-delimited CN data into a GenomicArray subclass.
       Always called with (infile, sample_id)."""
    target_cls = into or GenomicArray

    # Build metadata from inputs
    fname = get_filename(infile)
    meta = {"sample_id": sample_id}
    if fname:
        meta["filename"] = fname

    # Read the table (use helper that drops rows with missing 'log2')
    try:
        dframe = read_tab(infile)
    except pd.errors.EmptyDataError:
        logging.info("Blank tab file?: %s", infile)
        dframe = target_cls._make_blank()  # correct empty schema

    # Instantiate + normalize
    result = target_cls(dframe, meta)
    result.sort_columns()
    result.sort()
    return result

from typing import Iterable, Tuple
from itertools import takewhile

def sorter_chrom(label: str) -> Tuple[int, str]:
    """Create a sorting key from chromosome label.

    Sort by integers first, then letters or strings. The prefix "chr"
    (case-insensitive), if present, is stripped automatically for sorting.

    E.g. chr1 < chr2 < chr10 < chrX < chrY < chrM
    """
    # Strip "chr" prefix
    chrom = label[3:] if label.lower().startswith("chr") else label
    if chrom in ("X", "Y"):
        key = (1000, chrom)
    else:
        # Separate numeric and special chromosomes
        nums = "".join(takewhile(str.isdigit, chrom))
        chars = chrom[len(nums):]
        nums = int(nums) if nums else 0
        if not chars:
            key = (nums, "")
        elif len(chars) == 1:
            key = (2000 + nums, chars)
        else:
            key = (3000 + nums, chars)
    return key

def biweight_location(a, initial=None, c=6.0, epsilon=1e-3, max_iter=5):
    """Compute the biweight location for an array.

    The biweight is a robust statistic for estimating the central location of a
    distribution.
    """

    def biloc_iter(a, initial):
        # Weight the observations by distance from initial estimate
        d = a - initial
        mad = np.median(np.abs(d))
        w = d / max(c * mad, epsilon)
        w = (1 - w**2) ** 2
        # Omit the outlier points
        mask = w < 1
        weightsum = w[mask].sum()
        if weightsum == 0:
            # Insufficient variation to improve the initial estimate
            return initial
        return initial + (d[mask] * w[mask]).sum() / weightsum

    if initial is None:
        initial = np.median(a)
    for _i in range(max_iter):
        result = biloc_iter(a, initial)
        if abs(result - initial) <= epsilon:
            break
        initial = result
    return result
    
def modal_location(a):
    """Return the modal value of an array's values.

    The "mode" is the location of peak density among the values, estimated using
    a Gaussian kernel density estimator.

    Parameters
    ----------
    a : np.array
        A 1-D array of floating-point values, e.g. bin log2 ratio values.
    """
    sarr = np.sort(a)
    kde = stats.gaussian_kde(sarr)
    y = kde.evaluate(sarr)
    peak = sarr[y.argmax()]
    return peak
   
def weighted_median(a, weights):
    """Weighted median of a 1-D numeric array."""
    order = a.argsort()
    a = a[order]
    weights = weights[order]
    midpoint = 0.5 * weights.sum()
    if (weights > midpoint).any():
        # Any point with the majority of total weight must be the median
        return a[weights.argmax()]
    cumulative_weight = weights.cumsum()
    midpoint_idx = cumulative_weight.searchsorted(midpoint)
    if (
        midpoint_idx > 0
        and cumulative_weight[midpoint_idx - 1] - midpoint < sys.float_info.epsilon
    ):
        # Midpoint of 2 array values
        return a[midpoint_idx - 1 : midpoint_idx + 1].mean()
    return a[midpoint_idx]
    
class CopyNumArray(GenomicArray):
    """An array of genomic intervals, treated like aCGH probes.

    Required columns: chromosome, start, end, gene, log2

    Optional columns: gc, rmask, spread, weight, probes
    """

    _required_columns = ("chromosome", "start", "end", "gene", "log2")  # type: ignore
    _required_dtypes = (str, int, int, str, float)  # type: ignore
    # ENH: make gene optional
    # Extra columns, in order:
    # "depth", "gc", "rmask", "spread", "weight", "probes"

    def __init__(self, data_table, meta_dict=None):
        GenomicArray.__init__(self, data_table, meta_dict)

    @property
    def log2(self):
        return self.data["log2"]

    @log2.setter
    def log2(self, value):
        self.data["log2"] = value

    @property
    def chr_x_label(self):
        """The name of the X chromosome.

        This is either "X" or "chrX".
        """
        key = "chr_x"
        if key in self.meta:
            return self.meta[key]
        if len(self):
            chr_x_label = "chrX" if self.chromosome.iat[0].startswith("chr") else "X"
            self.meta[key] = chr_x_label
            return chr_x_label
        return ""

    @property
    def chr_y_label(self):
        """The name of the Y chromosome."""
        if "chr_y" in self.meta:
            return self.meta["chr_y"]
        if len(self):
            chr_y = "chrY" if self.chr_x_label.startswith("chr") else "Y"
            self.meta["chr_y"] = chr_y
            return chr_y
        return ""


    





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

    for col in ("start", "end"):
        cnr_corrected.data[col] = pd.to_numeric(cnr_corrected.data[col], errors="raise").astype(int)
    
    for arm, vals in correction_regions.items():
        chrom = vals["chromosome"]
        start = int(vals["start"])
        end = int(vals["end"])
        copy_number = float(vals["copy_number"])
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

from scipy.stats import zscore

POINT_COLOR = "black"
DEPTH_THRESHOLD = 0
SEG_COLOR = 'black'

def do_scatter(
    cnarr,
    segments=None,
    variants=None,
    y_min=None,
    y_max=None,
    do_trend=False,
    fig_size=(18,12),
    segment_color='black',
    title=None,
    by_bin=True,
    show_range=None,
    window_width=1e6,
    highlight=None,
    base=2,
    y_log=False
):
    if by_bin:
        bp_per_bin = sum(c.end.iat[-1] for _, c in cnarr.by_chromosome()) / len(cnarr)
        window_width /= bp_per_bin
        show_range_bins = translate_region_to_bins(show_range, cnarr)
        cnarr, segments, variants = update_binwise_positions(
            cnarr, segments, variants
        )
        global MB
        orig_mb = MB
        MB = 1

    fig = genome_scatter(
        cnarr, segments, variants, do_trend, y_min, y_max, title, segment_color, highlight,base,y_log
    )
    
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
    highlight=None,
    base=None,
    y_log=False
):
    """Plot all chromosomes, concatenated on one plot."""
    if (cnarr or segments) and variants:
        # Lay out top 3/5 for the CN scatter, bottom 2/5 for SNP plot
        axgrid = pyplot.GridSpec(5, 1, hspace=0.85)
        axis = pyplot.subplot(axgrid[:3])
        axis2 = pyplot.subplot(axgrid[3:], sharex=axis)
        # Place chromosome labels between the CNR and SNP plots
        axis2.tick_params(labelbottom=False)
        chrom_sizes = plots.chromosome_sizes(cnarr or segments)
        axis2 = cnv_on_genome(
            axis2, variants, chrom_sizes, segments, do_trend, segment_color, highlight
        )
    else:
        _fig, axis = pyplot.subplots()
    if title is None:
        title = (cnarr or segments or variants).sample_id
    if cnarr or segments:
        axis.set_title(title, loc='left', fontsize=30, pad=10)
        axis = cnv_on_genome(
            axis, cnarr, segments, do_trend, y_min, y_max, segment_color, highlight, base, y_log
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
        subprobes_filtered['color'] = 'black'
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
    current_chrom = subprobes_filtered['chromosome'].iloc[0] if len(subprobes_filtered) > 0 else None
    
    if current_chrom not in highlight_ranges:
        subprobes_filtered['color'] = POINT_COLOR
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
    
    subprobes_filtered['color'] = colors
    return subprobes_filtered

from matplotlib.ticker import FixedLocator, NullFormatter

def cnv_on_genome(
    axis,
    probes,
    segments,
    do_trend: bool = False,
    y_min: int | float | None = None,
    y_max: int | float | None = None,
    segment_color: str = SEG_COLOR,
    highlight=None,
    base: int | float | None = None,
    y_log: bool = False,
):
    """Plot bin ratios and/or segments for all chromosomes on one plot."""

    # --- Axis style ---
    axis.axhline(color="k")
    axis.set_ylabel("Copy Number", fontsize=30, labelpad=5)
    axis.set_xlabel("Chromosome", fontsize=30, labelpad=25)
    for spine in axis.spines.values():
        spine.set_linewidth(2)

    # --- Y-axis handling (linear vs log) ---
    if y_log:
        # Default log window if not specified
        y_min = 0.5
        y_max = 5
        axis.set_yscale("log", base=10)
        axis.set_ylim(y_min, y_max)
        ticks = [0.5, 1, 2, 3, 4, 5]
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.yaxis.set_major_locator(FixedLocator(ticks))
        axis.set_yticks(ticks)
        axis.set_yticklabels([str(int(t)) if t == int(t) else str(t) for t in ticks], fontsize=20)
    else:
        # Default linear window if not specified
        if y_min is None: y_min = 0
        if y_max is None: y_max = 6
        axis.set_ylim(y_min, y_max)
        axis.tick_params(axis="y", labelsize=20)
        axis.set_yticks(range(int(y_min), int(y_max) + 1))

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
    x_starts = plot_chromosome_dividers(axis, chrom_sizes, y_max=y_max)
    for chrom, x_offset in x_starts.items():
        if probes and chrom in chrom_probes:
            subprobes = chrom_probes[chrom]
            subprobes_filtered = subprobes[
                (subprobes['log2'] >= ratio_thresholds[0] - 0.5) & 
                (subprobes['log2'] <= ratio_thresholds[-1] + 0.5)
            ]
            log2_values = subprobes_filtered['log2']
            original_count = len(subprobes_filtered)
            subprobes_filtered['z_scores'] = zscore(log2_values)
            subprobes_filtered = subprobes_filtered[(subprobes_filtered['z_scores'] >= -3) & (subprobes_filtered['z_scores'] <= 3)]
            # Highlight specified positions
            if highlight is not None:
                subprobes_filtered = highlight_positions(subprobes_filtered, highlight_ranges=highlight)
    
            else:
                subprobes_filtered = highlight_positions(subprobes_filtered, highlight_ranges=None)
            x = 0.5 * (subprobes_filtered["start"] + subprobes_filtered["end"]) + x_offset
            alpha_values = np.ones(len(subprobes_filtered))
            highlight_color = "blue"
            not_highlighted = subprobes_filtered['color'] != highlight_color
            positive_mask = not_highlighted & (subprobes_filtered['z_scores'] >= 2) & (subprobes_filtered['z_scores'] <= 3)
            alpha_values[positive_mask] = 1 - (subprobes_filtered['z_scores'][positive_mask] - 2) / (3 - 2)
            negative_mask = not_highlighted & (subprobes_filtered['z_scores'] <= -2) & (subprobes_filtered['z_scores'] >= -3)
            alpha_values[negative_mask] = 1 - (abs(subprobes_filtered['z_scores'][negative_mask]) - 2) / (3 - 2)
            axis.scatter(x, base * 2**subprobes_filtered["log2"], marker=".", color=subprobes_filtered['color'], alpha=alpha_values)
            axis.scatter(x, base * 2**subprobes_filtered['log2Seg'], marker=".", color='red', alpha=0.2, s=20)
            # axis.scatter(x, subprobes_filtered['cn'], marker=".", color='yellow', alpha=0.2, s=20)
    
    return axis


# Ensure the correct number of arguments are passed
if len(sys.argv) != 9:
    sys.exit(
        "Usage: Plotting.py <intersect_cnr> <sorted_cns> <cleanname> <output> <base> <y_log> <highlight_json> <correction_json>"
    )

intersect_cnr = sys.argv[1]
sorted_cns   = sys.argv[2]
cleanname    = sys.argv[3]
output       = sys.argv[4]

# Mandatory args
try:
    base = int(sys.argv[5])
except ValueError:
    sys.exit(f"Error: base must be an integer, got '{sys.argv[5]}'")

y_log_arg = sys.argv[6].strip().lower()
if y_log_arg == "true":
    y_log = True
elif y_log_arg == "false":
    y_log = False
else:
    sys.exit(f"Error: y_log must be 'True' or 'False', got '{sys.argv[6]}'")

highlight_path   = sys.argv[7]
correction_path  = sys.argv[8]

def load_json_or_none(path: str, label: str):
    """
    Return parsed JSON, or None if the file is missing, empty, '{}'/'[]',
    or the path is a sentinel like '-', 'None', 'null'.
    """
    if path in ("-", "None", "none", "NULL", "null", ""):
        return None
    if not os.path.exists(path):
        # Treat missing as None to keep CLI stable; comment next line to make it strict
        return None
    try:
        if os.path.getsize(path) == 0:
            return None
        with open(path, "r") as f:
            data = json.load(f)
        # Coerce empty objects/arrays to None
        if not data:
            return None
        return data
    except json.JSONDecodeError as e:
        sys.exit(f"Error: {label} JSON is not valid: {e}")

highlight_ranges   = load_json_or_none(highlight_path,  "highlight")
correction_regions = load_json_or_none(correction_path, "correction")


# Load the .cnr file
cnr = read_cna(intersect_cnr, sample_id=cleanname)
cns = read_cna(sorted_cns, sample_id=cleanname)


# Apply copy number corrections if any
if correction_regions:
    cnr = apply_copy_number_corrections_from_json(cnr, correction_regions)
    print(f"Applied {len(correction_regions)} copy number corrections")

# Generate the plot
fig = do_scatter(cnr, segments=cns, title=cleanname, y_min=0, y_max=5, by_bin=True, highlight=highlight_ranges, base=base, y_log=y_log)

# Save the plot to a file
fig.savefig(output, bbox_inches="tight")
