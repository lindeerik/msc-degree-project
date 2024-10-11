"""
Map visualization for uplink throughput
"""
import pandas as pd
import geopandas as gpd

from data.data_loader import loadDataCsv
from data.data_processing import transformTimestamp
from visualization.visualize import plotFloatMap, plotCategoricalMap


def stockholmSodertaljeVisualization():
    dataDir = "data/intermediate/sthlm-sodertalje/"
    df = loadDataCsv(dataDir, "")
    dependentCol = "UL_bitrate"
    threshholdCol = "UL_threshhold"
    # Mbps from Kbps
    df[dependentCol] = df[dependentCol] / 1024
    df = transformTimestamp(
        df, "Timestamp", timeOfDayCol="Time_of_day", timeOfYearCol="Time_of_year"
    )

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
    )
    gdf = gdf.to_crs(epsg=3857)

    plotFloatMap(
        gdf,
        dependentCol,
        title="Uplink Bitrate (Mbps) Sthlm - Södertalje Drive Test 2024-10-04",
        useLogScale=True,
        alpha=0.1,
    )

    gdf[threshholdCol] = gdf[dependentCol] >= 15
    customLabels = {False: "<15 mbps", True: ">=15 mbps"}
    plotCategoricalMap(
        gdf,
        threshholdCol,
        title="Uplink Bitrate (Mbps) Sthlm - Södertalje Drive Test 2024-10-04",
        alpha=0.1,
        customLabels=customLabels,
    )
    plotCategoricalMap(
        gdf,
        "NetworkMode",
        title="Uplink Bitrate (Mbps) Sthlm - Södertalje Drive Test 2024-10-04",
        alpha=0.1,
    )


def ooklaE4Visualization():
    path = "data/raw/ookla/SDK_data_e4_polygon.csv"
    df = pd.read_csv(path)
    selectedCols = ["longitude", "latitude", "ul_throughput"]
    df = df[selectedCols].dropna()
    dependentCol = "ul_throughput"
    threshholdCol = "ul_threshhold"

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )
    gdf = gdf.to_crs(epsg=3857)

    plotFloatMap(
        gdf,
        dependentCol,
        title="Uplink Throughput (Mbps) E4 Ookla Data",
        useLogScale=True,
        alpha=0.1,
    )

    gdf[threshholdCol] = gdf[dependentCol] >= 15
    customLabels = {False: "<15 mbps", True: ">=15 mbps"}
    plotCategoricalMap(
        gdf,
        threshholdCol,
        title="Uplink Throughput (Mbps) E4 Ookla Data",
        alpha=0.1,
        customLabels=customLabels,
    )


def ooklaI45Visualization():
    path = "data/raw/ookla/SDK_data_i45_polygon.csv"
    df = pd.read_csv(path)
    selectedCols = ["longitude", "latitude", "ul_throughput"]
    df = df[selectedCols].dropna()
    dependentCol = "ul_throughput"
    threshholdCol = "ul_threshhold"

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )
    gdf = gdf.to_crs(epsg=3857)

    plotFloatMap(
        gdf,
        dependentCol,
        title="Uplink Throughput (Mbps) I45 Ookla Data",
        zoom=11,
        useLogScale=True,
        alpha=0.1,
    )

    gdf[threshholdCol] = gdf[dependentCol] >= 15
    customLabels = {False: "<15 mbps", True: ">=15 mbps"}
    plotCategoricalMap(
        gdf,
        threshholdCol,
        title="Uplink Throughput (Mbps) I45 Ookla Data",
        zoom=11,
        alpha=0.1,
        customLabels=customLabels,
    )


def main():
    stockholmSodertaljeVisualization()
    # ooklaE4Visualization()
    # ooklaI45Visualization()


main()
