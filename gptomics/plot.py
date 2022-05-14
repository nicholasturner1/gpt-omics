"""Plotting functions."""
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_FONTSIZE = 20
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def percentiles_by_type(
    df: pd.DataFrame,
    type_colname: str,
    types: Optional[list[str]] = None,
    nbins: Optional[int] = 20,
    ax: Optional[plt.Axes] = None,
    fontsize: Optional[int] = DEFAULT_FONTSIZE,
    colors: Optional[list] = COLORS,
    legend: Optional[bool] = True,
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
        plotdf.groupby(["bin", type_colname]).count()
        / (plotdf.shape[0] / nbins)
    ).reset_index()

    y_ = np.zeros((nbins,), dtype=np.float32)
    types = sorted(pd.unique(grouped[type_colname])) if types is None else types
    for (t, c) in zip(types, colors):
        subdf = grouped[grouped[type_colname] == t].set_index("bin")
        ax.bar(
            x=subdf.index,
            height=subdf["term_value"],
            bottom=y_[subdf.index - 1],
            label=t,
            color=c,
        )
        y_[subdf.index - 1] += subdf["term_value"]

    perc_interval = 100 / nbins
    ax.set_xticks(
        np.arange(nbins) + 1,
        labels=[f"{i*perc_interval:.0f}p-{(i+1)*perc_interval:.0f}p" for i in range(nbins)],
        rotation=45,
    )
    ax.set_xlabel("Value percentile", fontsize=fontsize)
    ax.set_ylabel("Percentage", fontsize=fontsize)
    if legend:
        ax.legend()

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


def plot_layerdists(
    df: pd.DataFrame,
    color="k",
    ax: plt.Axes = None,
    fontsize: int = DEFAULT_FONTSIZE,
    **kwargs
) -> plt.Axes:
    """Plots the distribution of layer distance between edges in a DataFrame.

    The layer distance is equal to the difference between the destination layer
    index and the source layer index.

    Args:
        df: Dataframe containing composition/singular values.
        colors: The bar color
        ax: The matplotlib axis to use when plotting.
        fontsize: The axis fontsize.
    Returns:
        An axis filled with the desired plot.
    """
    ax = plt.gca() if ax is None else ax

    diffs = df.dst_layer - df.src_layer

    ax.hist(diffs, bins=np.arange(29), color=color),
    ax.set_xlabel("Layer distance", fontsize=fontsize)
    ax.set_ylabel("Count", fontsize=fontsize)
    ax.set_xlim(0, 28)

    return ax


def plot_grp_layerdist_hist(
    df: pd.DataFrame,
    grp: int = 20,
    grp_colname: str = "overall_5p_grp",
    color='k',
    ax: plt.Axes = None,
    fontsize: int = DEFAULT_FONTSIZE,
) -> plt.Axes:
    """Plots the distribution of layer distance between edges in a DataFrame.

    The layer distance is equal to the difference between the destination layer
    index and the source layer index.

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

    return plot_layerdists(groupdf, color=color, ax=ax, fontsize=fontsize)
