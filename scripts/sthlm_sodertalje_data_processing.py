"""
Script for cleaning raw data recorded between Stockholm and Södertälje
"""

import pandas as pd


def main():
    cleanData241004()
    cleanData241028()
    cleanData241029()


def cleanData241004(standardizeCoordinates=True):
    fileName = "2024.10.04_11.19.11.csv"
    dirRaw = "data/raw/sthlm-sodertalje/"
    dirClean = "data/intermediate/sthlm-sodertalje/"
    # drop rows due to stops in iperf test
    indicesToDropTestStops = (
        list(range(0, 10))
        + list(range(2160, 2176))
        + list(range(3939, 3949))
        + list(range(6508, 6553))
        + list(range(8063, 8084))
        + list(range(8689, 8734))
    )
    # drop rows due to tunnels
    indicesToDropTunnels = list(range(269, 363)) + list(range(530, 711))
    indicesToDrop = indicesToDropTestStops + indicesToDropTunnels
    cleanData(fileName, dirRaw, dirClean, indicesToDrop, standardizeCoordinates, 200)


def cleanData241028(standardizeCoordinates=True):
    fileName = "2024.10.28_17.20.20.csv"
    dirRaw = "data/raw/sthlm-sodertalje/"
    dirClean = "data/intermediate/sthlm-sodertalje/"
    # drop rows due to stops in iperf test
    indicesToDrop = (
        list(range(0, 7)) + list(range(4631, 5090)) + list(range(8942, 8959))
    )
    cleanData(fileName, dirRaw, dirClean, indicesToDrop, standardizeCoordinates, 200)


def cleanData241029(standardizeCoordinates=True):
    fileName = "2024.10.29_07.18.51.csv"
    dirRaw = "data/raw/sthlm-sodertalje/"
    dirClean = "data/intermediate/sthlm-sodertalje/"
    # drop rows due to stops in iperf test
    indicesToDrop = (
        list(range(0, 7))
        + list(range(3933, 4720))
        + list(range(6105, 6445))
        + list(range(8849, 8881))
    )
    cleanData(fileName, dirRaw, dirClean, indicesToDrop, standardizeCoordinates, 200)


def cleanData(
    fileName,
    dirRaw,
    dirClean,
    indicesToDrop,
    standardizeCoordinates=True,
    threshhold=None,
):
    dfRaw = pd.read_csv(dirRaw + fileName, na_values=["", "-"])
    df = dfRaw.dropna(axis=1, how="all")
    dfClean = df.drop(df.index[indicesToDrop])
    if standardizeCoordinates:
        lonLimit = 18.019982
        latLimit = 59.300837
        dfClean = dfClean[
            (dfClean["Longitude"] < lonLimit) & (dfClean["Latitude"] < latLimit)
        ]
    if threshhold:
        dfClean = dfClean[dfClean["UL_bitrate"] < threshhold * 1024]
    dfClean.to_csv(dirClean + fileName)


main()
