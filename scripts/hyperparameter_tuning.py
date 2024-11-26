"""
Tuning hyperparameters of point estimation and uncertainty intervals (divdied accordingly)
"""

import time
import random
import pandas as pd


# Sci-kit
from sklearn.ensemble import RandomForestRegressor

# Torch
import torch
from torch import optim
from torch import nn
from skorch import NeuralNetRegressor

# xg-boost
import xgboost as xgb

# Random Forest Quantile
from sklearn_quantile import RandomForestQuantileRegressor

from data.data_loader import loadDataCsv, loadHyperparams
from data.data_processing import (
    processData,
    getDataProcessor,
)
from data.data_saver import saveExperimentData, saveModelBestParamsToJson
from models.model import Model
from models.neuralnetwork.architecture import ThroughputPredictor
from models.conformalprediction.conformalizing_scalar import (
    ConformalizingScalarPredictor,
)
from models.conformalprediction.quantile_regression import (
    ConformalizedQuantileRegressor,
    QuantileRegressorNeuralNet,
    QuantileRegressorRandomForest,
)
from models.conformalprediction.pinball import (
    PinballLoss,
    pinballLossScorer,
    doublePinballLossScorer,
)


# pylint: disable-msg=too-many-locals, too-many-statements
def main():
    torch.manual_seed(0)
    random.seed(0)
    saveDir = "experiments/hyperparameter-tuning/"
    hyperparamsDir = "config/hyperparameters/"

    dirCsv = "data/intermediate/sthlm-sodertalje/train/"
    df = loadDataCsv(dirCsv, "")

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

    pointEstimationModelsTuning(
        dataX, dataY, selectedFloatCols, selectedCatCols, saveDir, hyperparamsDir
    )
    predictionIntervalModelsTuning(
        dataX, dataY, selectedFloatCols, selectedCatCols, saveDir, hyperparamsDir
    )


def pointEstimationModelsTuning(
    dataX, dataY, floatCols, catCols, saveDir, hyperparamDir=None
):
    rf = RandomForestRegressor(random_state=42)
    paramGridRf = getRandomForestFullParamGrid()

    xGradBoost = xgb.XGBRegressor(random_state=42)
    paramGridXgb = getXgboostFullParamGrid()

    net = NeuralNetRegressor(
        ThroughputPredictor,
        module__input_size=dataX.shape[1],
        optimizer=optim.Adam,
        criterion=nn.MSELoss,
        verbose=0,
        train_split=None,
    )
    paramGridNet = getNeuralNetFullParamGrid()

    models = [
        Model(rf, "RF", paramGridRf),
        Model(xGradBoost, "XGB", paramGridXgb),
        Model(net, "NN", paramGridNet),
    ]

    ### TUNING ###
    for model in models:
        print(f"Tuning {model.getName()}")
        timeStart = time.time()
        model.gridSearchFit(dataX, dataY, 4)
        timeend = time.time()
        print(f"Finished tuning {model.getName()} in {timeend-timeStart} seconds")

    ### SAVING DATA ###
    cols = ["Model", "Parameter", "Value"]
    data = []
    for model in models:
        if hyperparamDir:
            saveModelBestParamsToJson(model, hyperparamDir)

        bestParams = model.getBestParams()
        modelName = model.getName()
        for key, value in bestParams.items():
            data.append([modelName, key, value])

    df = pd.DataFrame(data, columns=cols)
    saveExperimentData(
        df,
        saveDir,
        "hyperparameter_tuning_point_estimation",
        floatCols,
        catCols,
        models,
        "",
    )


def predictionIntervalModelsTuning(
    dataX, dataY, floatCols, catCols, saveDir, hyperparamDir=None
):
    alpha = 0.1

    ### QUANTILE REGRESSORS ###
    lowerNet = NeuralNetRegressor(
        ThroughputPredictor,
        module__input_size=dataX.shape[1],
        optimizer=optim.Adam,
        criterion=PinballLoss(alpha / 2),
        verbose=0,
        train_split=None,
    )
    upperNet = NeuralNetRegressor(
        ThroughputPredictor,
        module__input_size=dataX.shape[1],
        optimizer=optim.Adam,
        criterion=PinballLoss(1 - alpha / 2),
        verbose=0,
        train_split=None,
    )
    paramGridNetLower = getNeuralNetFullParamGrid()
    paramGridNetUpper = getNeuralNetFullParamGrid()

    lowerScorer = pinballLossScorer(alpha / 2)
    upperScorer = pinballLossScorer(1 - alpha / 2)
    lowerModel = Model(lowerNet, "NN_lower", paramGridNetLower, lowerScorer)
    upperModel = Model(upperNet, "UB_lower", paramGridNetUpper, upperScorer)

    quantileNeuralNetRegressor = QuantileRegressorNeuralNet(
        [lowerModel, upperModel], alpha, "QNN"
    )
    conformalQuantileNeuralNetRegressor = ConformalizedQuantileRegressor(
        quantileNeuralNetRegressor, name="CQNN"
    )

    rfq = RandomForestQuantileRegressor(q=[alpha / 2, 1 - alpha / 2], random_state=42)
    paramGridRfq = getRandomForestQuantileParamGrid()
    doublePinballScorer = doublePinballLossScorer(alpha / 2, 1 - alpha / 2)
    rqfModel = Model(rfq, "QRF", paramGridRfq, doublePinballScorer)

    quantileForestRegressor = QuantileRegressorRandomForest([rqfModel], alpha, "QRF")
    conformalQuantileForestRegressor = ConformalizedQuantileRegressor(
        quantileForestRegressor, name="CQRF"
    )

    ### CONFORMALIZING SCALAR PREDICTORS ###
    rfBase = RandomForestRegressor(random_state=42)
    paramGridRfBase = loadHyperparams(hyperparamDir + "rf.json")

    rfError = RandomForestRegressor(random_state=42)
    paramGridRfError = getRandomForestFullParamGrid()

    rfBaseModel = Model(rfBase, "RF_base", paramGridRfBase)
    rfErrorModel = Model(rfError, "RF_error", paramGridRfError)

    conformalizingScalarRf = ConformalizingScalarPredictor(
        rfBaseModel, rfErrorModel, alpha, name="L-RF"
    )

    xgbBase = xgb.XGBRegressor(random_state=42)
    paramGridXgbBase = loadHyperparams(hyperparamDir + "xgb.json")

    xgbError = xgb.XGBRegressor(random_state=42)
    paramGridXgbError = getXgboostFullParamGrid()

    xgbBaseModel = Model(xgbBase, "XGB_base", paramGridXgbBase)
    xgbErrorModel = Model(xgbError, "XGB_error", paramGridXgbError)

    conformalizingScalarXgb = ConformalizingScalarPredictor(
        xgbBaseModel, xgbErrorModel, alpha, name="L-XGB"
    )

    conformalModels = [
        conformalQuantileNeuralNetRegressor,
        conformalQuantileForestRegressor,
        conformalizingScalarRf,
        conformalizingScalarXgb,
    ]

    baseModels = [
        lowerModel,
        upperModel,
        rqfModel,
        rfBaseModel,
        rfErrorModel,
        xgbBaseModel,
        xgbErrorModel,
    ]

    ### TUNING ###
    for conformalModel in conformalModels:
        # fit with training data as reserved data, conformal scores do not impact tuning
        print(f"Tuning {conformalModel.getName()}")
        timeStart = time.time()
        conformalModel.fit(dataX, dataY, dataX, dataY, 4)
        timeend = time.time()
        print(
            f"Finished tuning {conformalModel.getName()} in {timeend-timeStart} seconds"
        )

    ### SAVING DATA ###
    cols = ["Model", "Parameter", "Value"]
    data = []
    for model in baseModels:
        if hyperparamDir:
            saveModelBestParamsToJson(model, hyperparamDir)

        bestParams = model.getBestParams()
        if bestParams:
            modelName = model.getName()
            for key, value in bestParams.items():
                data.append([modelName, key, value])
            df = pd.DataFrame(data, columns=cols)
            saveExperimentData(
                df,
                saveDir,
                "hyperparameter_tuning_prediction_interval",
                floatCols,
                catCols,
                conformalModels,
                "",
            )


def getRandomForestFullParamGrid():
    return {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [0.3, "sqrt", "log2"],
    }


def getXgboostFullParamGrid():
    return {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [None, 10, 20],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.3],
        "reg_alpha": [0.1],
        "reg_lambda": [0.1],
    }


def getNeuralNetFullParamGrid():
    return {
        "module__dropout_rate": [0.2, 0.3],
        "lr": [0.005, 0.01, 0.02],
        "optimizer__weight_decay": [0, 0.001],
        "optimizer__betas": [(0.9, 0.999), (0.95, 0.999)],
        "max_epochs": [100, 200, 300],
        "batch_size": [64, 128],
    }


def getRandomForestQuantileParamGrid():
    return {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": [0.3, "sqrt", "log2"],
    }


main()
