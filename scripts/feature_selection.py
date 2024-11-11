"""
Script for conducting feature selection with LASSO
"""

import pandas as pd

from data.data_loader import loadDataCsv
from data.data_saver import saveExperimentData
from features.lasso import lassoFeatureSelection


def main():
    dirCsv = "data/intermediate/sthlm-sodertalje/"
    df = loadDataCsv(dirCsv, "")

    dependentCol = "UL_bitrate"
    selectedFloatCols = [
        "Longitude",
        "Latitude",
        "Speed",
        "SNR",
        "CQI",
        "Level",
        "Qual",
    ]
    selectedCatCols = [
        "CellID",
        "Node",
        "NetworkMode",
        "State",
        "BAND",
        "BANDWIDTH",
        "LAC",
        "PSC",
        "CA",
    ]
    selectedVariables, lasso = lassoFeatureSelection(
        df, selectedFloatCols, selectedCatCols, dependentCol
    )
    data = pd.DataFrame({"Selected Variables": selectedVariables})
    saveExperimentData(
        data,
        "experiments/feature-selection/",
        "lasso",
        selectedFloatCols,
        selectedCatCols,
        [lasso],
        "",
    )


main()
