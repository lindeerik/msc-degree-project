"""
Map visualizations with Geopandas dataframes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.patches import Patch
import contextily as ctx


def plotFloatMap(
    gdf,
    dependentCol,
    useLogScale=None,
    title=None,
    baseMapSource=ctx.providers.CartoDB.Positron,
    zoom=12,
    figSize=(12, 8),
    show=False,
    markerSize=5,
    alpha=1,
):
    vmin = gdf[dependentCol].min()
    vmax = gdf[dependentCol].max()

    if useLogScale:
        norm = LogNorm(vmin=vmin + 0.1, vmax=vmax)
    else:
        norm = None

    _, ax = plt.subplots(figsize=figSize)
    gdf.plot(
        column=dependentCol,
        cmap="inferno",
        legend=True,
        markersize=markerSize,
        alpha=alpha,
        norm=norm,
        ax=ax,
    )
    ax.axis("off")
    if title:
        ax.set_title(title)
    ctx.add_basemap(ax, source=baseMapSource, zoom=zoom)
    if show:
        plt.show()


def plotCategoricalMap(
    gdf,
    dependentCol,
    title=None,
    baseMapSource=ctx.providers.CartoDB.Positron,
    zoom=12,
    figSize=(12, 8),
    show=False,
    markerSize=5,
    alpha=1,
    customLabels=None,
):

    categories = np.sort(gdf[dependentCol].unique())
    cmap = plt.get_cmap("viridis", len(categories))

    _, ax = plt.subplots(figsize=figSize)

    norm = BoundaryNorm(boundaries=range(len(categories) + 1), ncolors=len(categories))
    gdf.plot(
        column=dependentCol,
        cmap=cmap,
        legend=True,
        markersize=markerSize,
        alpha=alpha,
        ax=ax,
        categorical=True,
        norm=norm,
    )

    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14)

    if customLabels:
        legendPatches = [
            Patch(color=cmap(i), label=customLabels.get(cat, cat))
            for i, cat in enumerate(categories)
        ]
        ax.legend(handles=legendPatches, loc="best", bbox_to_anchor=(1, 1))

    ctx.add_basemap(ax, source=baseMapSource, zoom=zoom)
    if show:
        plt.show()
