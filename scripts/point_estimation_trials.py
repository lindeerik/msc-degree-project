"""
Generating results from running trials of point-estimation models
"""

import pandas as pd

# Sci-kit
from sklearn.ensemble import RandomForestRegressor

# Torch
from torch import optim
from torch import nn
from skorch import NeuralNetRegressor

# xg-boost
import xgboost as xgb

from data.data_loader import loadDataCsv, loadHyperparams
from data.data_processing import (
    processData,
    getDataProcessor,
    trainTestSplit,
)
from data.data_saver import saveExperimentData
from models.model import Model
from models.neuralnetwork.architecture import ThroughputPredictor


def main():
    saveDir = "experiments/point-estimation/"
    modelCol = "Model"
    trainRatioCol = "Train ratio"
    samplesCol = "Number of Samples"
    samplesEvalCol = "Evaluation Samples"
    r2Col = "Coefficient of Determination (R2)"
    rmseCol = "Root Mean Squared Error"
    maeCol = "Mean Absolute Error"

    cols = [
        modelCol,
        trainRatioCol,
        samplesCol,
        samplesEvalCol,
        r2Col,
        rmseCol,
        maeCol,
    ]
    trainRatios = [0.7, 0.8, 0.9]
    numTrials = 20
    runTrialsAndSaveData(cols, trainRatios, numTrials, saveDir)

# pylint: disable-msg=too-many-locals, too-many-statements
def runTrialsAndSaveData(cols, trainRatios, numTrials, saveDir, verbose=True):
    dataDir = "data/intermediate/sthlm-sodertalje/"
    hyperparamsDir = "config/hyperparameters/"
    df = loadDataCsv(dataDir, "")

    ### DATA PREPARATION ###
    dependentCol = "UL_bitrate"
    # Mbps from Kbps
    df[dependentCol] = df[dependentCol] / 1024

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

    processor = getDataProcessor(selectedFloatCols, selectedCatCols, applyScaler=True)
    dataX, dataY = processData(
        df, selectedFloatCols, selectedCatCols, dependentCol, processor
    )

    data = []
    ### SELECT MODELS ###
    rf = RandomForestRegressor()
    paramGridRf = loadHyperparams(hyperparamsDir + "rf.json")

    xGradBoost = xgb.XGBRegressor()
    paramGridXgb = loadHyperparams(hyperparamsDir + "xgb.json")

    net = NeuralNetRegressor(
        ThroughputPredictor,
        module__input_size=dataX.shape[1],
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

    ### TRIALS ###
    for j, trainRatio in enumerate(trainRatios):
        for _ in range(numTrials):
            xTrain, xTest, yTrain, yTest = trainTestSplit(dataX, dataY, trainRatio)
            samples = xTrain.shape[0]
            testSamples = xTest.shape[0]

            for model in models:
                model.fit(xTrain, yTrain)

            for model in models:
                name = model.getName()
                addMetricsToList(
                    data,
                    model,
                    name,
                    trainRatio,
                    samples,
                    testSamples,
                    xTest,
                    yTest,
                )
        if verbose:
            completedShare = (j + 1) / len(trainRatios)
            print(f"Completed {completedShare*100:.2f}% of trials")

    df = pd.DataFrame(data, columns=cols)
    saveExperimentData(
        df,
        saveDir,
        "point_estimation_trials",
        selectedFloatCols,
        selectedCatCols,
        models,
        "",
    )


def addMetricsToList(data, model, name, trainRatio, samples, testSamples, xTest, yTest):
    r2 = model.getR2(xTest, yTest)
    rmse = model.getRmse(xTest, yTest)
    mae = model.getMae(xTest, yTest)
    data.append([name, trainRatio, samples, testSamples, r2, rmse, mae])


main()
