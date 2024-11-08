import pandas as pd

# Sci-kit
from sklearn.ensemble import RandomForestRegressor

# Torch
from torch import nn
from torch import optim
from skorch import NeuralNetRegressor

# xg-boost
import xgboost as xgb

from data.data_loader import loadCsv
from data.data_processing import processData, getDataProcessor
from models.model import Model
from models.neuralnetwork.architecture import ThroughputPredictor


def main():
    dirStd = "data/intermediate/sthlm-sodertalje/"

    df241004 = loadCsv(dirStd + "2024.10.04_11.19.11.csv", ["", "-"])
    df241028 = loadCsv(dirStd + "2024.10.28_17.20.20.csv", ["", "-"])
    df241029 = loadCsv(dirStd + "2024.10.29_07.18.51.csv", ["", "-"])

    ### DATA PREPARATION ###
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
        "NetworkMode",
        "BAND",
        "BANDWIDTH",
    ]

    # Iterate by training models on all but one drive test (which beomces test data)
    dfs = [df241029, df241004, df241028]
    for i, dfTest in enumerate(dfs):
        dfTrain = pd.concat(
            [df for j, df in enumerate(dfs) if j != i], ignore_index=True, join="outer"
        )

        processor = getDataProcessor(
            selectedFloatCols, selectedCatCols, applyScaler=True, binaryEncoding=True
        )
        xTrain, yTrain = processData(
            dfTrain, selectedFloatCols, selectedCatCols, dependentCol, processor
        )
        xTest, yTest = processData(
            dfTest,
            selectedFloatCols,
            selectedCatCols,
            dependentCol,
            processor,
            fitProcessor=False,
        )

        rf = RandomForestRegressor()
        paramGridRf = {
            "n_estimators": [300],
            "max_depth": [20],
            "min_samples_split": [5],
            "min_samples_leaf": [5],
            "max_features": ["sqrt"],
        }

        xGradBoost = xgb.XGBRegressor()
        paramGridXgb = {
            "n_estimators": [200],
            "learning_rate": [0.05],
            "max_depth": [20],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "gamma": [0.1],
            "reg_alpha": [0.01],
            "reg_lambda": [1.5],
        }

        net = NeuralNetRegressor(
            ThroughputPredictor,
            module__input_size=xTrain.shape[1],
            optimizer=optim.Adam,
            criterion=nn.MSELoss,
            verbose=0,
            train_split=None,
        )
        paramGridNet = {
            "lr": [0.01],
            "max_epochs": [100],
            "optimizer__weight_decay": [0.001],
            "batch_size": [16],
        }
        models = [
            Model(rf, "RF", paramGridRf),
            Model(xGradBoost, "XGB", paramGridXgb),
            Model(net, "NN", paramGridNet),
        ]
        for model in models:
            print(f"Model: {model.getName()}")
            model.fit(xTrain, yTrain)
            print(f"Training R2: {model.getR2(xTrain, yTrain):.4f}")
            print(f"Training R2: {model.getR2(xTest, yTest):.4f}\n")


main()
