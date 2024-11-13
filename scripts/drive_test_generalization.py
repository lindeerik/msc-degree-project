import pandas as pd

# Sci-kit
from sklearn.ensemble import RandomForestRegressor

# Torch
from torch import nn
from torch import optim
from skorch import NeuralNetRegressor

# xg-boost
import xgboost as xgb

from data.data_loader import loadCsv, loadHyperparams
from data.data_processing import processData, getDataProcessor
from data.data_saver import saveExperimentData
from models.model import Model
from models.neuralnetwork.architecture import ThroughputPredictor


def main():
    dataDir = "data/intermediate/sthlm-sodertalje/"
    hyperparamsDir = "config/hyperparameters/"


    df241004 = loadCsv(dataDir + "2024.10.04_11.19.11.csv", ["", "-"])
    df241028 = loadCsv(dataDir + "2024.10.28_17.20.20.csv", ["", "-"])
    df241029 = loadCsv(dataDir + "2024.10.29_07.18.51.csv", ["", "-"])

    ### DATA PREPARATION ###
    dependentCol = "UL_bitrate"
    selectedFloatCols = [
        "Longitude",
        "Latitude",
        "Speed",
        "SNR",
        "Level",
        "Qual",
    ]
    selectedCatCols = [
        "CellID",
        "Node",
        "NetworkMode",
        "BAND",
        "BANDWIDTH",
        "LAC",
        "PSC",
    ]
    data = []
    cols = ["Model", "Index of Drive Test for Testing Data", "Train R2", "Test R2"]
    # Iterate by training models on all but one drive test (which beomces test data)
    dfs = [df241004, df241028, df241029]
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
        paramGridRf = loadHyperparams(hyperparamsDir + "rf.json")

        xGradBoost = xgb.XGBRegressor()
        paramGridXgb = loadHyperparams(hyperparamsDir + "xgb.json")

        net = NeuralNetRegressor(
            ThroughputPredictor,
            module__input_size=xTrain.shape[1],
            optimizer=optim.Adam,
            criterion=nn.MSELoss,
            verbose=0,
            train_split=None,
        )
        paramGridNet = loadHyperparams(hyperparamsDir + "nn.json")

        models = [
            Model(rf, "RF", paramGridRf),
            Model(xGradBoost, "XGB", paramGridXgb),
            Model(net, "NN", paramGridNet),
        ]
        for model in models:
            model.fit(xTrain, yTrain)
            trainR2 = model.getR2(xTrain, yTrain)
            testR2 = model.getR2(xTest, yTest)
            data.append([model.getName(), i, trainR2, testR2])

    df = pd.DataFrame(data, columns=cols)
    saveExperimentData(
        df,
        "experiments/generalization/",
        "drive_test_generalization",
        selectedFloatCols,
        selectedCatCols,
        models,
        "Drive tests are in order 2024-10-04, 2024-10-28, 2024-10-29",
    )


main()
