"""Plotting functions."""
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_FONTSIZE = 20
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def filterdf(
    df: pd.DataFrame,
    src_block: Optional[int] = None,
    dst_block: Optional[int] = None,
    src_type: Optional[str] = None,
    dst_type: Optional[str] = None,
    term_type: Optional[str] = None,
) -> pd.DataFrame:
    """Filters the rows of a composition df by a set of conditions."""
    if src_block is not None:
        subdf = df[df.src_block == src_block]

    if dst_block is not None:
        subdf = df[df.dst_block == dst_block]

    if src_type is not None:
        subdf = df[df.src_type == src_type]

    if dst_type is not None:
        subdf = df[df.dst_type == dst_type]

    if term_type is not None:
        subdf = df[df.term_type == term_type]

    return subdf


def plot_all_head_input_dists(
    df: pd.DataFrame,
    src_type: Optional[str] = None,
    src_block: Optional[int] = None,
    term_type: Optional[str] = None,
    logy: Optional[bool] = True,
) -> plt.Figure:
    """Plot the input term_value distribution for every head within a dataframe."""
    xlim = (0, 1)
    bins = np.linspace(0, 1, 21)

    df = filterdf(df, src_type=src_type, src_block=src_block, term_type=term_type)

    blocks = sorted(pd.unique(df.dst_block).tolist())
    blocks = list(filter(lambda x: x != 0, blocks))  # removing block 0

    heads = sorted(pd.unique(df.dst_index).tolist())

    num_blocks = len(blocks)
    num_heads = len(heads)

    for block in blocks:
        for head in heads:
            plt.subplot(num_blocks, num_heads, num_heads * (block - 1) + head + 1)

            plt.title(f"head: {head} block: {block}")
            subdf = df[
                (df.dst_type == "att_head")
                & (df.dst_index == head)
                & (df.dst_block == block)
            ]

            plt.hist(subdf.term_value, color="k", bins=bins)
            plt.xlim(*xlim)

            if logy:
                plt.yscale("log")

    plt.tight_layout()


def logspace_bins(
    data: np.ndarray, numbins: int = 30, eps: float = 1e-10
) -> np.ndarray:
    datamin = np.log(data.min())
    datamax = np.log(data.max())

    return np.exp(np.linspace(datamin - eps, datamax + eps, numbins))


def percentiles_by_type(
    df: pd.DataFrame,
    type_colname: str,
    types: Optional[list[str]] = None,
    nbins: int = 20,
    ax: Optional[plt.Axes] = None,
    fontsize: int = DEFAULT_FONTSIZE,
    colors: list = COLORS,
    legend: bool = True,
) -> plt.Axes:
    """Bar plot with a type breakdown for percentile groups.

    Args:
        df: Dataframe containing composition/singular values.
        type_colname: The column name that describes how to split values.
        types: An ordered list for the types to plot and how to display them.
        nbins: The number of percentile bins to use. Percentile bin width = 100/nbins.
        ax: The matplotlib axis to use when plotting.
        fontsize: The axis fontsize.
        colors: The colors to use when plotting each group.
            Default to a list with len=10.
        legend: Whether or not to plot the legend describing the types.
    Returns:
        An axis filled with the desired plot.
    """
    ax = plt.gca() if ax is None else ax

    plotdf = df.copy()

    plotdf["bin"] = compute_percentile_bins(df, nbins=nbins)

    grouped = (
        plotdf.groupby(["bin", type_colname]).count() / (plotdf.shape[0] / nbins)
    ).reset_index()

    y_ = np.zeros((nbins,), dtype=np.float32)
    types = sorted(pd.unique(grouped[type_colname])) if types is None else types
    for (t, c) in zip(types, colors):
        subdf = grouped[grouped[type_colname] == t].set_index("bin")
        ax.bar(
            x=subdf.index,
            height=subdf["term_value"],
            bottom=y_[subdf.index - 1],
            width=1,
            edgecolor="k",
            label=t,
            color=c,
        )
        y_[subdf.index - 1] += subdf["term_value"]

    perc_interval = 100 / nbins
    ax.set_xticks(
        np.arange(nbins) + 1,
        labels=[
            f"{i*perc_interval:.0f}p-{(i+1)*perc_interval:.0f}p" for i in range(nbins)
        ],
        rotation=45,
    )
    ax.set_xlabel("Value percentile", fontsize=fontsize)
    ax.set_ylabel("Percentage", fontsize=fontsize)
    if legend:
        ax.legend()

    ax.set_xlim(0.5, grouped.bin.max() + 0.5)
    ax.set_ylim(0, 1)

    return ax


def compute_percentile_bins(
    df: pd.DataFrame,
    nbins: int = 20,
    colname: str = "term_value",
    eps: float = 1e-10,
) -> np.ndarray:
    """Bins the values within colname by percentiles.

    Args:
        df: Dataframe containing composition/singular values.
        nbins: The number of percentile bins to use. Percentile bin width = 100/nbins.
        colname: The column name that describes how to split values.
        eps: A small nudge to the percentile values to clean the binning.
    Returns:
        An np array with length=length(df[colname]) where each value is assigned
        to a percentile bin.
    """
    perc_interval = 100 / nbins
    percs = np.percentile(df[colname], np.arange(nbins + 1) * perc_interval)

    # pushing the boundaries beyond the extrema
    percs[0] -= eps
    percs[-1] += eps

    return np.digitize(df[colname], bins=percs)


def plot_blockdists(
    df: pd.DataFrame,
    color="k",
    ax: plt.Axes = None,
    fontsize: int = DEFAULT_FONTSIZE,
    **kwargs,
) -> plt.Axes:
    """Plots the distribution of block distance between edges in a DataFrame.

    The block distance is equal to the difference between the destination block
    index and the source block index.

    Args:
        df: Dataframe containing composition/singular values.
        colors: The bar color
        ax: The matplotlib axis to use when plotting.
        fontsize: The axis fontsize.
    Returns:
        An axis filled with the desired plot.
    """
    ax = plt.gca() if ax is None else ax

    diffs = df.dst_block - df.src_block

    ax.hist(diffs, bins=np.arange(29), color=color),
    ax.set_xlabel("Block distance", fontsize=fontsize)
    ax.set_ylabel("Count", fontsize=fontsize)
    ax.set_xlim(0, 28)

    return ax


def plot_grp_blockdist_hist(
    df: pd.DataFrame,
    grp: int = 20,
    grp_colname: str = "overall_5p_grp",
    color="k",
    ax: plt.Axes = None,
    fontsize: int = DEFAULT_FONTSIZE,
) -> plt.Axes:
    """Plots the distribution of block distance between edges in a DataFrame.

    The block distance is equal to the difference between the destination block
    index and the source block index.

    Args:
        df: Dataframe containing composition/singular values.
        grp: The group value to plot
        grp_colname: The group column name that indices which rows to plot
        colors: The bar color
        ax: The matplotlib axis to use when plotting.
        fontsize: The axis fontsize.
    Returns:
        An axis filled with the desired plot.
    """
    groupdf = df[df[grp_colname] == grp]

    return plot_blockdists(groupdf, color=color, ax=ax, fontsize=fontsize)


def plot_ipc_percentiles(
    percs: np.ndarray,
    color="k",
    ax: plt.Axes = None,
    fontsize: int = DEFAULT_FONTSIZE,
    **kwargs,
) -> plt.Axes:
    """Plots the input path complexities over percentile threshold values.

    See graph.ipc_percentiles for more information on the inputs to this function.

    Args:
        percs: The IPC values for each percentile threshold value (101 x 5) np.ndarray.
        colors: The base color to use for the lines and shaded regions.
        ax: The matplotlib axis to use when plotting.
        fontsize: The axis fontsize.
    Returns:
        An axis filled with the desired plot.
    """
    ax = plt.gca() if ax is None else ax

    assert percs.shape[1] == 5, "not sure what the columns refer to"

    ax.plot(percs[:, 2], color)  # median
    ax.plot(percs[:, 0], color, ls="--")  # min
    ax.plot(percs[:, 4], color, ls="--")  # max

    # IQR
    ax.fill_between(
        np.arange(percs.shape[0]),
        percs[:, 1],
        percs[:, 3],
        alpha=0.4,
        color=color,
    )

    ax.set_xlabel("Percentile threshold", fontsize=fontsize)
    ax.set_ylabel("IPC", fontsize=fontsize)

    return ax
