"""
Script for investigating feature selection
"""

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from data.data_loader import loadDataCsv
from data.data_processing import transformTimestamp
from data.data_saver import saveExperimentData
from models.model import Model
from features.feature_selection import BackwardFeatureSelector, ForwardFeatureSelector


def main():
    dirCsv = "data/intermediate/sthlm-sodertalje/"
    df = loadDataCsv(dirCsv, "")
    df = transformTimestamp(df, "Timestamp", timeOfDayCol="Time_of_day")

    dependentCol = "UL_bitrate"
    selectedFloatCols = [
        "Longitude",
        "Latitude",
        "Speed",
        "SNR",
        "CQI",
        "Time_of_day",
    ]
    selectedCatCols = [
        "CellID",
        "NetworkMode",
        "Operatorname",
        "CELLHEX",
        "NODEHEX",
        "LACHEX",
        "RAWCELLID",
        "State",
    ]
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    model = Model(rf, "Random Forest")

    backwardSelector = BackwardFeatureSelector(model)
    bestFloatCols, bestCatCols = backwardSelector.getBestFeatures(
        df, selectedFloatCols, selectedCatCols, dependentCol
    )
    backwardDf = pd.DataFrame({"Best Columns": bestFloatCols + bestCatCols})
    saveExperimentData(
        backwardDf,
        "experiments/feature-selection/",
        "backward:feature_selection",
        selectedFloatCols,
        selectedCatCols,
        [model],
        "",
    )

    forwardSelector = ForwardFeatureSelector(model)
    bestFloatCols, bestCatCols = forwardSelector.getBestFeatures(
        df, selectedFloatCols, selectedCatCols, dependentCol
    )
    forwardDf = pd.DataFrame({"Best Columns": bestFloatCols + bestCatCols})
    saveExperimentData(
        forwardDf,
        "experiments/feature-selection/",
        "forward_feature_selection",
        selectedFloatCols,
        selectedCatCols,
        [model],
        "",
    )


main()
