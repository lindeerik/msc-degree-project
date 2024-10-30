"""
Map visualization for uplink throughput between Stockholm-Södertälje
"""

import pandas as pd
import geopandas as gpd

from visualization.maps import plotFloatMap, plotCategoricalMap
from visualization.save_figures import saveFiguresWithDateTime


def main():
    stockholmSodertaljeVisualization20241004()
    stockholmSodertaljeVisualization20241028()
    stockholmSodertaljeVisualization20241029()
    saveFiguresWithDateTime("figures/")


def stockholmSodertaljeVisualization20241004():
    fileName = "2024.10.04_11.19.11.csv"
    dataDir = "data/intermediate/sthlm-sodertalje/"
    date = "2024-10-04"
    dependentCol = "UL_bitrate"
    threshholdCol = "UL_threshhold"
    threshholds = [15, 30]
    stockholmSodertaljeVisualization(
        fileName, dataDir, date, dependentCol, threshholdCol, threshholds
    )


def stockholmSodertaljeVisualization20241028():
    fileName = "2024.10.28_17.20.20.csv"
    dataDir = "data/intermediate/sthlm-sodertalje/"
    date = "2024-10-28"
    dependentCol = "UL_bitrate"
    threshholdCol = "UL_threshhold"
    threshholds = [15, 30]
    stockholmSodertaljeVisualization(
        fileName, dataDir, date, dependentCol, threshholdCol, threshholds
    )


def stockholmSodertaljeVisualization20241029():
    fileName = "2024.10.29_07.18.51.csv"
    dataDir = "data/intermediate/sthlm-sodertalje/"
    date = "2024-10-29"
    dependentCol = "UL_bitrate"
    threshholdCol = "UL_threshhold"
    threshholds = [15, 30]
    stockholmSodertaljeVisualization(
        fileName, dataDir, date, dependentCol, threshholdCol, threshholds
    )


def stockholmSodertaljeVisualization(
    fileName, dataDir, date, dependentCol, threshholdCol, threshholds
):
    df = pd.read_csv(dataDir + fileName)
    # Mbps from Kbps
    df[dependentCol] = df[dependentCol] / 1024

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
    )
    gdf = gdf.to_crs(epsg=3857)

    plotFloatMap(
        gdf,
        dependentCol,
        title=f"Uplink Bitrate (Mbps) Sthlm - Södertalje Drive Test {date}",
        useLogScale=True,
        alpha=0.1,
    )

    plotCategoricalMap(
        gdf,
        "NetworkMode",
        title=f"Uplink Bitrate (Mbps) Sthlm - Södertalje Drive Test {date}",
        alpha=0.1,
    )

    for threshhold in threshholds:
        gdf[threshholdCol] = gdf[dependentCol] >= threshhold
        customLabels = {False: f"<{threshhold} mbps", True: f">={threshhold} mbps"}
        plotCategoricalMap(
            gdf,
            threshholdCol,
            title=f"Uplink Bitrate (Mbps) Sthlm - Södertalje Drive Test {date}",
            alpha=0.1,
            customLabels=customLabels,
        )


main()
