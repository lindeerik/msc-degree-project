import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from sklearn.model_selection import train_test_split
from data.data_loader import *
from models.model import Model
from features.feature_selection import getBestFeatures

def main():
    dirParquet = "data/intermediate/"
    df = loadDataParquet(dirParquet)

    dependentCol = "UL_bitrate"

    selectedFloatCols = ["Longitude", "Latitude", "Speed", "RSRP","RSRQ","SNR", "NRxRSRP", "NRxRSRQ", "PINGAVG"]
    selectedCatCols = ["CellID"]

    model = Model(RandomForestRegressor(), "Random Forest")

    bestFloatCols = getBestFeatures(df, selectedFloatCols, selectedCatCols, dependentCol, model)
    print(bestFloatCols)

main()
