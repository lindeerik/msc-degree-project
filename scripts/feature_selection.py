"""
Script for investigating feature selection
"""

from sklearn.ensemble import RandomForestRegressor


from data.data_loader import loadDataCsv
from data.data_processing import transformTimestamp
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
    print(f"Best float columns: {bestFloatCols}")
    print(f"Best categorical columns: {bestCatCols}")

    forwardSelector = ForwardFeatureSelector(model)
    bestFloatCols, bestCatCols = forwardSelector.getBestFeatures(
        df, selectedFloatCols, selectedCatCols, dependentCol
    )
    print(f"Best float columns: {bestFloatCols}")
    print(f"Best categorical columns: {bestCatCols}")


main()
