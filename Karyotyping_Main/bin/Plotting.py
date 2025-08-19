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

        # Alternate the positions of the labels
        for i, (center, label) in enumerate(zip(centers, chrom_names)):
            if i % 2 == 0:
                axis.text(center, 1.066*y_max, label, ha='center', va='top', fontsize=25)
            else:
                axis.text(center, 1.033*y_max, label, ha='center', va='top', fontsize=25)

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

    def cut(self, other, combine=None):
        """Split this array's regions at the boundaries in `other`."""
        # TODO
        return NotImplemented

    def flatten(
        self,
        combine: Optional[Dict[str, Callable]] = None,
        split_columns: Optional[Iterable[str]] = None,
    ):
        """Split this array's regions where they overlap."""
        return self.as_dataframe(
            flatten(self.data, combine=combine, split_columns=split_columns)
        )

    def intersection(self, other, mode: str = "outer"):
        """Select the bins in `self` that overlap the regions in `other`.

        The extra fields of `self`, but not `other`, are retained in the output.
        """
        # TODO options for which extra fields to keep
        #   by default, keep just the fields in 'table'
        if mode == "trim":
            # Slower
            chunks = [
                chunk.data
                for _, chunk in self.by_ranges(other, mode=mode, keep_empty=False)
            ]
            return self.as_dataframe(pd.concat(chunks))
        # Faster
        slices = iter_slices(self.data, other.data, mode, False)
        indices = np.concatenate(list(slices))
        return self.as_dataframe(self.data.loc[indices])

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

    def resize_ranges(self, bp: int, chrom_sizes: Optional[Mapping[str, Numeric]] = None):
        """Resize each genomic bin by a fixed number of bases at each end.

        Bin 'start' values have a minimum of 0, and `chrom_sizes` can
        specify each chromosome's maximum 'end' value.

        Similar to 'bedtools slop'.

        Parameters
        ----------
        bp : int
            Number of bases in each direction to expand or shrink each bin.
            Applies to 'start' and 'end' values symmetrically, and may be
            positive (expand) or negative (shrink).
        chrom_sizes : dict of string-to-int
            Chromosome name to length in base pairs. If given, all chromosomes
            in `self` must be included.
        """
        table = self.data
        limits = {"lower": 0}
        if chrom_sizes:
            limits["upper"] = self.chromosome.map(chrom_sizes)
        table = table.assign(
            start=(table["start"] - bp).clip(**limits),
            end=(table["end"] + bp).clip(**limits),
        )
        if bp < 0:
            # Drop any bins that now have zero or negative size
            ok_size = table["end"] - table["start"] > 0
            logging.debug("Dropping %d bins with size <= 0", (~ok_size).sum())
            table = table[ok_size]
        # Don't modify the original
        return self.as_dataframe(table.copy())

    def squash(self, combine=None):
        """Combine some groups of rows, by some criteria, into single rows."""
        # TODO
        return NotImplemented

    def subdivide(self, avg_size: int, min_size: int = 0, verbose: bool = False):
        """Split this array's regions into roughly equal-sized sub-regions."""
        return self.as_dataframe(subdivide(self.data, avg_size, min_size, verbose))

    def subtract(self, other):
        """Remove the overlapping regions in `other` from this array."""
        return self.as_dataframe(subtract(self.data, other.data))

    def total_range_size(self) -> int:
        """Total number of bases covered by all (merged) regions."""
        if not len(self):
            return 0
        regions = merge(self.data, bp=1)
        return regions.end.sum() - regions.start.sum()

    def _get_gene_map(self) -> OrderedDict:
        """Map unique gene names to their indices in this array.

        Returns
        -------
        OrderedDict
            An (ordered) dictionary of unique gene names and the data indices of
            their segments in the order of occurrence (genomic order).
        """
        if "gene" not in self.data:
            return OrderedDict()

        genes: OrderedDict = OrderedDict()
        for idx, genestr in self.data["gene"].items():
            if pd.isnull(genestr):
                continue
            for gene in genestr.split(","):
                if gene not in genes:
                    genes[gene] = []
                genes[gene].append(idx)
        return genes

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

    def chr_x_filter(self, diploid_parx_genome=None):
        """ All regions on X, potentially without PAR1/2. """
        x = self.chromosome == self.chr_x_label
        if diploid_parx_genome is not None:
            # Exclude PAR since they are expected to be diploid (i.e. autosomal).
            x &= ~self.parx_filter(genome_build=diploid_parx_genome)
        return x

    def parx_filter(self, genome_build):
        """ All PAR1/2 regions on X. """
        genome_build = genome_build.lower()
        assert genome_build in SUPPORTED_GENOMES_FOR_PAR_HANDLING
        f = self.chromosome == self.chr_x_label
        par1_start, par1_end = PSEUDO_AUTSOMAL_REGIONS[genome_build]["PAR1X"]
        par2_start, par2_end = PSEUDO_AUTSOMAL_REGIONS[genome_build]["PAR2X"]
        f &= ((self.start >= par1_start) & (self.end <= par1_end)) | ((self.start >= par2_start) & (self.end <= par2_end))
        return f

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

    def pary_filter(self, genome_build):
        """ All PAR1/2 regions on Y. """
        genome_build = genome_build.lower()
        assert genome_build in SUPPORTED_GENOMES_FOR_PAR_HANDLING
        f = self.chromosome == self.chr_y_label
        par1_start, par1_end = PSEUDO_AUTSOMAL_REGIONS[genome_build]["PAR1Y"]
        par2_start, par2_end = PSEUDO_AUTSOMAL_REGIONS[genome_build]["PAR2Y"]
        f &= ((self.start >= par1_start) & (self.end <= par1_end)) | ((self.start >= par2_start) & (self.end <= par2_end))
        return f

    def chr_y_filter(self, diploid_parx_genome=None):
        """ All regions on Y, potentially without PAR1/2. """
        y = self.chromosome == self.chr_y_label
        if diploid_parx_genome is not None:
            # Exclude PAR on Y since they cannot be covered (everything is mapped to X).
            y &= ~self.pary_filter(genome_build=diploid_parx_genome)
        return y

    def autosomes(self, diploid_parx_genome=None, also=None):
        """Overrides GenomeArray.autosomes()."""
        if diploid_parx_genome is not None:
            if also is None:
                also = self.parx_filter(diploid_parx_genome)
            elif isinstance(also, pd.Series):
                also |= self.parx_filter(diploid_parx_genome)
            else:
                raise NotImplementedError("Cannot combine pd.Series with non-Series.")
        return super().autosomes(also=also)

    # More meta to store:
    #   is_sample_male = bool
    #   is_haploid_x_reference = bool
    #   * invalidate 'chr_x' if .chromosome/['chromosome'] is set

    # Traversal

    def by_gene(self, ignore=IGNORE_GENE_NAMES):
        """Iterate over probes grouped by gene name.

        Group each series of intergenic bins as an "Antitarget" gene; any
        "Antitarget" bins within a gene are grouped with that gene.

        Bins' gene names are split on commas to accommodate overlapping genes
        and bins that cover multiple genes.

        Parameters
        ----------
        ignore : list or tuple of str
            Gene names to treat as "Antitarget" bins instead of real genes,
            grouping these bins with the surrounding gene or intergenic region.
            These bins will still retain their name in the output.

        Yields
        ------
        tuple
            Pairs of: (gene name, CNA of rows with same name)
        """
        ignore += ANTITARGET_ALIASES
        for _chrom, subgary in self.by_chromosome():
            prev_idx = 0
            for gene, gene_idx in subgary._get_gene_map().items():
                if gene not in ignore:
                    if not len(gene_idx):
                        logging.warning(
                            "Specified gene name somehow missing: %s", gene
                        )
                        continue
                    start_idx = gene_idx[0]
                    end_idx = gene_idx[-1] + 1
                    if prev_idx < start_idx:
                        # Include intergenic regions
                        yield ANTITARGET_NAME, subgary.as_dataframe(
                            subgary.data.loc[prev_idx:start_idx]
                        )
                    yield gene, subgary.as_dataframe(
                        subgary.data.loc[start_idx:end_idx]
                    )
                    prev_idx = end_idx
            if prev_idx < len(subgary) - 1:
                # Include the telomere
                yield ANTITARGET_NAME, subgary.as_dataframe(
                    subgary.data.loc[prev_idx:]
                )

    # Manipulation

    def center_all(
        self, estimator=pd.Series.median, by_chrom=True, skip_low=False, verbose=False, diploid_parx_genome=None
    ):
        """Re-center log2 values to the autosomes' average (in-place).

        Parameters
        ----------
        estimator : str or callable
            Function to estimate central tendency. If a string, must be one of
            'mean', 'median', 'mode', 'biweight' (for biweight location). Median
            by default.
        skip_low : bool
            Whether to drop very-low-coverage bins (via `drop_low_coverage`)
            before estimating the center value.
        by_chrom : bool
            If True, first apply `estimator` to each chromosome separately, then
            apply `estimator` to the per-chromosome values, to reduce the impact
            of uneven targeting or extreme aneuploidy. Otherwise, apply
            `estimator` to all log2 values directly.
        diploid_parx_genome : String
             Whether to include the PAR1/2 on chr X from the given genome (build)
             as part of the autosomes
        """
        est_funcs = {
            "mean": pd.Series.mean,
            "median": pd.Series.median,
            "mode": modal_location,
            "biweight": biweight_location,
        }
        if isinstance(estimator, str):
            if estimator in est_funcs:
                estimator = est_funcs[estimator]
            else:
                raise ValueError(
                    "Estimator must be a function or one of: "
                    + ", ".join(map(repr, est_funcs))
                )
        cnarr = (
            self.drop_low_coverage(verbose=verbose) if skip_low else self
        ).autosomes(diploid_parx_genome=diploid_parx_genome)
        if cnarr:
            if by_chrom:
                values = pd.Series(
                    [
                        estimator(subarr["log2"])
                        for _c, subarr in cnarr.by_chromosome()
                        if len(subarr)
                    ]
                )
            else:
                values = cnarr["log2"]
            shift = -estimator(values)
            if verbose:
                logging.info("Shifting log2 values by %f", shift)
            self.data["log2"] += shift

    def drop_low_coverage(self, verbose=False):
        """Drop bins with extremely low log2 coverage or copy ratio values.

        These are generally bins that had no reads mapped due to sample-specific
        issues. A very small log2 ratio or coverage value may have been
        substituted to avoid domain or divide-by-zero errors.
        """
        min_cvg = params.NULL_LOG2_COVERAGE - MIN_REF_COVERAGE
        drop_idx = self.data["log2"] < min_cvg
        if "depth" in self:
            drop_idx |= self.data["depth"] == 0
        if verbose and drop_idx.any():
            logging.info("Dropped %d low-coverage bins", drop_idx.sum())
        return self[~drop_idx]

    def squash_genes(
        self,
        summary_func=biweight_location,
        squash_antitarget=False,
        ignore=IGNORE_GENE_NAMES,
    ):
        """Combine consecutive bins with the same targeted gene name.

        Parameters
        ----------
        summary_func : callable
            Function to summarize an array of log2 values to produce a
            new log2 value for a "squashed" (i.e. reduced) region. By default
            this is the biweight location, but you might want median, mean, max,
            min or something else in some cases.
        squash_antitarget : bool
            If True, also reduce consecutive "Antitarget" bins into a single
            bin. Otherwise, keep "Antitarget" and ignored bins as they are in
            the output.
        ignore : list or tuple of str
            Bin names to be treated as "Antitarget" instead of as unique genes.

        Return
        ------
        CopyNumArray
            Another, usually smaller, copy of `self` with each gene's bins
            reduced to a single bin with appropriate values.
        """

        def squash_rows(name, rows):
            """Combine multiple rows (for the same gene) into one row."""
            if len(rows) == 1:
                return tuple(rows.iloc[0])
            chrom = core.check_unique(rows.chromosome, "chromosome")
            start = rows.start.iat[0]
            end = rows.end.iat[-1]
            cvg = summary_func(rows.log2)
            outrow = [chrom, start, end, name, cvg]
            # Handle extra fields
            # ENH - no coverage stat; do weighted average as appropriate
            for xfield in ("depth", "gc", "rmask", "spread", "weight"):
                if xfield in self:
                    outrow.append(summary_func(rows[xfield]))
            if "probes" in self:
                outrow.append(sum(rows["probes"]))
            return tuple(outrow)

        outrows = []
        for name, subarr in self.by_gene(ignore):
            if not len(subarr):
                continue
            if name in params.ANTITARGET_ALIASES and not squash_antitarget:
                outrows.extend(subarr.data.itertuples(index=False))
            else:
                outrows.append(squash_rows(name, subarr.data))
        return self.as_rows(outrows)

    # Chromosomal sex

    def shift_xx(self, is_haploid_x_reference=False, is_xx=None, diploid_parx_genome=None):
        """Adjust chrX log2 ratios to match the ploidy of the reference sex.

        I.e. add 1 to chrX log2 ratios for a male sample vs. female reference,
        or subtract 1 for a female sample vs. male reference, so that chrX log2
        values are comparable across samples with different chromosomal sex.
        """
        outprobes = self.copy()
        if is_xx is None:
            is_xx = self.guess_xx(is_haploid_x_reference=is_haploid_x_reference, diploid_parx_genome=diploid_parx_genome)
        if is_xx and is_haploid_x_reference:
            # Female: divide X coverages by 2 (in log2: subtract 1)
            outprobes[outprobes.chromosome == self.chr_x_label, "log2"] -= 1.0
            # Male: no change
        elif not is_xx and not is_haploid_x_reference:
            # Male: multiply X coverages by 2 (in log2: add 1)
            outprobes[outprobes.chromosome == self.chr_x_label, "log2"] += 1.0
            # Female: no change
        return outprobes

    def guess_xx(self, is_haploid_x_reference=False, diploid_parx_genome=None, verbose=True):
        """Detect chromosomal sex; return True if a sample is probably female.

        Uses `compare_sex_chromosomes` to calculate coverage ratios of the X and
        Y chromosomes versus autosomes.

        Parameters
        ----------
        is_haploid_x_reference : bool
            Was this sample normalized to a male reference copy number profile?
        verbose : bool
            If True, print (i.e. log to console) the ratios of the log2
            coverages of the X and Y chromosomes versus autosomes, the
            "maleness" ratio of male vs. female expectations for each sex
            chromosome, and the inferred chromosomal sex.

        Returns
        -------
        bool
            True if the coverage ratios indicate the sample is female.
        """
        is_xy, stats = self.compare_sex_chromosomes(is_haploid_x_reference, diploid_parx_genome)
        if is_xy is None:
            return None
        if verbose:
            logging.info(
                "Relative log2 coverage of %s=%.3g, %s=%.3g "
                "(maleness=%.3g x %.3g = %.3g) --> assuming %s",
                self.chr_x_label,
                stats["chrx_ratio"],
                self.chr_y_label,
                stats["chry_ratio"],
                stats["chrx_male_lr"],
                stats["chry_male_lr"],
                stats["chrx_male_lr"] * stats["chry_male_lr"],
                "male" if is_xy else "female",
            )
        return ~is_xy

    def compare_sex_chromosomes(self, is_haploid_x_reference=False, diploid_parx_genome=None, skip_low=False):
        """Compare coverage ratios of sex chromosomes versus autosomes.

        Perform 4 Mood's median tests of the log2 coverages on chromosomes X and
        Y, separately shifting for assumed male and female chromosomal sex.
        Compare the chi-squared values obtained to infer whether the male or
        female assumption fits the data better.

        Parameters
        ----------
        is_haploid_x_reference : bool
            Whether a male reference copy number profile was used to normalize
            the data. If so, a male sample should have log2 values of 0 on X and
            Y, and female +1 on X, deep negative (below -3) on Y. Otherwise, a
            male sample should have log2 values of -1 on X and 0 on
            Y, and female 0 on X, deep negative (below -3) on Y.
        skip_low : bool
            If True, drop very-low-coverage bins (via `drop_low_coverage`)
            before comparing log2 coverage ratios. Included for completeness,
            but shouldn't affect the result much since the M-W test is
            nonparametric and p-values are not used here.

        Returns
        -------
        bool
            True if the sample appears male.
        dict
            Calculated values used for the inference: relative log2 ratios of
            chromosomes X and Y versus the autosomes; the Mann-Whitney U values
            from each test; and ratios of U values for male vs. female
            assumption on chromosomes X and Y.
        """
        if not len(self):
            return None, {}

        chrx = self[self.chr_x_filter(diploid_parx_genome)]
        if not len(chrx):
            logging.warning(
                "No %s found in sample; is the input truncated?", self.chr_x_label
            )
            return None, {}

        auto = self.autosomes(diploid_parx_genome=diploid_parx_genome)
        if skip_low:
            chrx = chrx.drop_low_coverage()
            auto = auto.drop_low_coverage()
        auto_l = auto["log2"].values
        use_weight = "weight" in self
        auto_w = auto["weight"].values if use_weight else None

        def compare_to_auto(vals, weights):
            # Mood's median test stat is chisq -- near 0 for similar median
            try:
                stat, _p, _med, cont = median_test(
                    auto_l, vals, ties="ignore", lambda_="log-likelihood"
                )
            except ValueError:
                # "All values are below the grand median (0.0)"
                stat = None
            else:
                if stat == 0 and 0 in cont:
                    stat = None
            # In case Mood's test failed for either sex
            if use_weight:
                med_diff = abs(
                    weighted_median(auto_l, auto_w)
                    - weighted_median(vals, weights)
                )
            else:
                med_diff = abs(np.median(auto_l) - np.median(vals))
            return (stat, med_diff)

        def compare_chrom(vals, weights, female_shift, male_shift):
            """Calculate "maleness" ratio of test statistics.

            The ratio is of the female vs. male chi-square test statistics from
            the median test. If the median test fails for either sex, (due to
            flat/trivial input), use the ratio of the absolute difference in
            medians.
            """
            female_stat, f_diff = compare_to_auto(vals + female_shift, weights)
            male_stat, m_diff = compare_to_auto(vals + male_shift, weights)
            # Statistic is smaller for similar-median sets
            if female_stat is not None and male_stat is not None:
                return female_stat / max(male_stat, 0.01)
            # Difference in medians is also smaller for similar-median sets
            return f_diff / max(m_diff, 0.01)

        female_x_shift, male_x_shift = (-1, 0) if is_haploid_x_reference else (0, +1)
        chrx_male_lr = compare_chrom(
            chrx["log2"].values,
            (chrx["weight"].values if use_weight else None),
            female_x_shift,
            male_x_shift,
        )
        combined_score = chrx_male_lr
        # Similar for chrY if it's present
        chry = self[self.chr_y_filter(diploid_parx_genome)]
        if len(chry):
            if skip_low:
                chry = chry.drop_low_coverage()
            chry_male_lr = compare_chrom(
                chry["log2"].values,
                (chry["weight"].values if use_weight else None),
                +3,
                0,
            )
            if np.isfinite(chry_male_lr):
                combined_score *= chry_male_lr
        else:
            # If chrY is missing, don't sabotage the inference
            chry_male_lr = np.nan
        # Relative log2 values, for convenient reporting
        auto_mean = segment_mean(auto, skip_low=skip_low)
        chrx_mean = segment_mean(chrx, skip_low=skip_low)
        chry_mean = segment_mean(chry, skip_low=skip_low)
        return (
            combined_score > 1.0,
            dict(
                chrx_ratio=chrx_mean - auto_mean,
                chry_ratio=chry_mean - auto_mean,
                combined_score=combined_score,
                # For debugging, mainly
                chrx_male_lr=chrx_male_lr,
                chry_male_lr=chry_male_lr,
            ),
        )

    def expect_flat_log2(self, is_haploid_x_reference=None, diploid_parx_genome=None):
        """Get the uninformed expected copy ratios of each bin.

        Create an array of log2 coverages like a "flat" reference.

        This is a neutral copy ratio at each autosome (log2 = 0.0) and sex
        chromosomes based on whether the reference is male (XX or XY).
        """
        if is_haploid_x_reference is None:
            is_haploid_x_reference = not self.guess_xx(diploid_parx_genome=diploid_parx_genome, verbose=False)
        cvg = np.zeros(len(self), dtype=np.float_)
        if is_haploid_x_reference:
            # Single-copy X, Y
            idx = self.chr_x_filter(diploid_parx_genome).values | (self.chr_y_filter(diploid_parx_genome)).values
        else:
            # Y will be all noise, so replace with 1 "flat" copy, including PAR1/2.
            idx = (self.chr_y_filter()).values
        cvg[idx] = -1.0
        return cvg

    # Reporting

    def residuals(self, segments=None):
        """Difference in log2 value of each bin from its segment mean.

        Parameters
        ----------
        segments : GenomicArray, CopyNumArray, or None
            Determines the "mean" value to which `self` log2 values are relative:

            - If CopyNumArray, use the log2 values as the segment means to
              subtract.
            - If GenomicArray with no log2 values, group `self` by these ranges
              and subtract each group's median log2 value.
            - If None, subtract each chromosome's median.

        Returns
        -------
        array
            Residual log2 values from `self` relative to `segments`; same length
            as `self`.
        """
        if not segments:
            resids = [
                subcna.log2 - subcna.log2.median()
                for _chrom, subcna in self.by_chromosome()
            ]
        elif "log2" in segments:
            resids = [
                bins_lr - seg_lr
                for seg_lr, bins_lr in zip(
                    segments["log2"],
                    self.iter_ranges_of(
                        segments, "log2", mode="inner", keep_empty=True
                    ),
                )
                if len(bins_lr)
            ]
        else:
            resids = [
                lr - lr.median()
                for lr in self.iter_ranges_of(segments, "log2", keep_empty=False)
            ]
        return pd.concat(resids) if resids else pd.Series([])

    def smooth_log2(self, bandwidth=None, by_arm=True):
        """Smooth log2 values with a sliding window.

        Account for chromosome and (optionally) centromere boundaries. Use bin
        weights if present.

        Returns
        -------
        array
            Smoothed log2 values from `self`, the same length as `self`.
        """
        if bandwidth is None:
            bandwidth = smoothing.guess_window_size(
                self.log2, weights=(self["weight"] if "weight" in self else None)
            )

        if by_arm:
            parts = self.by_arm()
        else:
            parts = self.by_chromosome()
        if "weight" in self:
            out = [
                smoothing.savgol(
                    subcna["log2"].values, bandwidth, weights=subcna["weight"].values
                )
                for _chrom, subcna in parts
            ]
        else:
            out = [
                smoothing.savgol(subcna["log2"].values, bandwidth)
                for _chrom, subcna in parts
            ]
        return np.concatenate(out)

    def _guess_average_depth(self, segments=None, window=100, diploid_parx_genome=None):
        """Estimate the effective average read depth from variance.

        Assume read depths are Poisson distributed, converting log2 values to
        absolute counts. Then the mean depth equals the variance , and the average
        read depth is the estimated mean divided by the estimated variance.
        Use robust estimators (Tukey's biweight location and midvariance) to
        compensate for outliers and overdispersion.

        With `segments`, take the residuals of this array's log2 values from
        those of the segments to remove the confounding effect of real CNVs.

        If `window` is an integer, calculate and subtract a smoothed trendline
        to remove the effect of CNVs without segmentation (skipped if `segments`
        are given).

        See: http://www.evanmiller.org/how-to-read-an-unlabeled-sales-chart.html
        """
        # Try to drop allosomes
        cnarr = self.autosomes(diploid_parx_genome=diploid_parx_genome)
        if not len(cnarr):
            cnarr = self
        # Remove variations due to real/likely CNVs
        y_log2 = cnarr.residuals(segments)
        if segments is None and window:
            y_log2 -= smoothing.savgol(y_log2, window)
        # Guess Poisson parameter from absolute-scale values
        y = np.exp2(y_log2)
        # ENH: use weight argument to these stats
        loc = biweight_location(y)
        spread = biweight_midvariance(y, loc)
        if spread > 0:
            return loc / spread**2
        return loc

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
    base=2
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
        cnarr, segments, variants, do_trend, y_min, y_max, title, segment_color, highlight,base
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
    base=None
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
            axis, cnarr, segments, do_trend, y_min, y_max, segment_color, highlight, base
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

def cnv_on_genome(
    axis,
    probes,
    segments,
    do_trend=False,
    y_min=None,
    y_max=None,
    segment_color=SEG_COLOR,
    highlight=None,
    base=None
):
    """Plot bin ratios and/or segments for all chromosomes on one plot."""
    # Configure axes etc.
    axis.axhline(color="k")
    axis.set_ylabel("Copy Number", fontsize=30, labelpad=25)
    axis.set_xlabel("Chromosome", fontsize=30, labelpad=57) 
    axis.set_ylim(y_min, y_max)
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
if len(sys.argv) < 5 or len(sys.argv) > 7:
    sys.exit("Usage: Plotting.py <intersect_cnr> <sorted_cns> <cleanname> <output> [highlight_json] [base]")

intersect_cnr = sys.argv[1]
sorted_cns   = sys.argv[2]
cleanname    = sys.argv[3]
output       = sys.argv[4]

highlight_ranges = None
base = 2  # default

# Optional arguments
if len(sys.argv) >= 6:
    arg5 = sys.argv[5]
    if arg5.endswith(".json"):  # highlight file
        with open(arg5, 'r') as f:
            highlight_ranges = json.load(f)
        if len(sys.argv) == 7:
            base = int(sys.argv[6])
    else:  # base directly
        base = int(arg5)

# Load the .cnr file
cnr = read_cna(intersect_cnr, sample_id=cleanname)
cns = read_cna(sorted_cns, sample_id=cleanname)

# Generate the plot
fig = do_scatter(cnr, segments=cns, title=cleanname, y_min=0, y_max=5, by_bin=True, highlight=highlight_ranges, base=base)

# Save the plot to a file
fig.savefig(output, bbox_inches="tight")
