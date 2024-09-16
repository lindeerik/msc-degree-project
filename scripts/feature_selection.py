"""
Script for investigating feature selection
"""

from sklearn.ensemble import RandomForestRegressor

from data.data_loader import loadDataParquet
from models.model import Model
from features.feature_selection import getBestFeatures


def main():
    dirParquet = "data/intermediate/"
    df = loadDataParquet(dirParquet)

    dependentCol = "UL_bitrate"

    selectedFloatCols = [
        "Longitude",
        "Latitude",
        "Speed",
        "RSRP",
        "RSRQ",
        "SNR",
        "NRxRSRP",
        "NRxRSRQ",
        "PINGAVG",
    ]
    selectedCatCols = ["CellID"]

    model = Model(RandomForestRegressor(), "Random Forest")

    bestFloatCols = getBestFeatures(
        df, selectedFloatCols, selectedCatCols, dependentCol, model
    )
    print(bestFloatCols)


main()
