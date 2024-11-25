"""
Accessing tuned models for point estimation, quantile regression and conformal prediction
"""

# Sci-kit
from sklearn.ensemble import RandomForestRegressor

# Torch
from torch import nn
from torch import optim
from skorch import NeuralNetRegressor

# xg-boost
import xgboost as xgb

# Random Forest Quantile
from sklearn_quantile import RandomForestQuantileRegressor

from data.data_loader import loadHyperparams
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


def getPointEstimationModels(hyperparamsDir, inputSize):
    models = [
        getRandomForestModel(hyperparamsDir),
        getXGBoostModel(hyperparamsDir),
        getNeuralNetworkModel(hyperparamsDir, inputSize),
    ]
    return models


def getQuantileRegressionModels(alpha, hyperparamsDir, inputSize):
    quantileModels = [
        getQuantileRegressionForest(alpha, hyperparamsDir),
        getQuantileNeuralNetwork(alpha, hyperparamsDir, inputSize),
    ]
    return quantileModels


def getConformalModels(alpha, hyperparamsDir, inputSize):
    conformalQuantileRegressionForest = getConformalQuantileRegressionForest(
        alpha, hyperparamsDir
    )
    conformalQuantileRegressionNeuralNet = getConformalQuantileNeuralNetwork(
        alpha, hyperparamsDir, inputSize
    )
    rfConformalizingScalar = getConformalizingScalarRandomForest(alpha, hyperparamsDir)
    xgbConformalizingScalar = getConformalizingScalarXGBoost(alpha, hyperparamsDir)

    conformalPredictors = [
        conformalQuantileRegressionForest,
        conformalQuantileRegressionNeuralNet,
        rfConformalizingScalar,
        xgbConformalizingScalar,
    ]
    return conformalPredictors


def getRandomForestModel(hyperparamsDir):
    rf = RandomForestRegressor()
    paramGridRf = loadHyperparams(hyperparamsDir + "rf.json")
    rfModel = Model(rf, "RF", paramGridRf)
    return rfModel


def getXGBoostModel(hyperparamsDir):
    xGradBoost = xgb.XGBRegressor()
    paramGridXgb = loadHyperparams(hyperparamsDir + "xgb.json")
    xgbModel = Model(xGradBoost, "XGB", paramGridXgb)
    return xgbModel


def getNeuralNetworkModel(hyperparamsDir, inputSize):
    net = NeuralNetRegressor(
        ThroughputPredictor,
        module__input_size=inputSize,
        optimizer=optim.Adam,
        criterion=nn.MSELoss,
        verbose=0,
        train_split=None,
    )
    paramGridNet = loadHyperparams(hyperparamsDir + "nn.json")
    netModel = Model(net, "NN", paramGridNet)
    return netModel


def getRandomForestErrorModel(hyperparamsDir):
    rfError = RandomForestRegressor()
    paramGridRfError = loadHyperparams(hyperparamsDir + "rf_error.json")
    rfErrorModel = Model(rfError, "RF_error", paramGridRfError)
    return rfErrorModel


def getXGBoostErrorModel(hyperparamsDir):
    xgbError = xgb.XGBRegressor()
    paramGridXgbError = loadHyperparams(hyperparamsDir + "xgb_error.json")
    xgbErrorModel = Model(xgbError, "XGB_error", paramGridXgbError)
    return xgbErrorModel


def getQuantileRegressionForest(alpha, hyperparamsDir):
    qrf = RandomForestQuantileRegressor(q=[alpha / 2, 1 - alpha / 2])
    paramGridQrf = loadHyperparams(hyperparamsDir + "qrf.json")
    doublePinballScorer = doublePinballLossScorer(alpha / 2, 1 - alpha / 2)
    qrfModel = Model(qrf, "QRF", paramGridQrf, doublePinballScorer)

    quantileRegressionForest = QuantileRegressorRandomForest([qrfModel], alpha, "QRF")
    return quantileRegressionForest


def getQuantileNeuralNetwork(alpha, hyperparamsDir, inputSize):
    lowerNet = NeuralNetRegressor(
        ThroughputPredictor,
        module__input_size=inputSize,
        optimizer=optim.Adam,
        criterion=PinballLoss(alpha / 2),
        verbose=0,
        train_split=None,
    )
    upperNet = NeuralNetRegressor(
        ThroughputPredictor,
        module__input_size=inputSize,
        optimizer=optim.Adam,
        criterion=PinballLoss(1 - alpha / 2),
        verbose=0,
        train_split=None,
    )
    paramGridNetLower = loadHyperparams(hyperparamsDir + "nn_lower.json")
    paramGridNetUpper = loadHyperparams(hyperparamsDir + "nn_upper.json")

    lowerScorer = pinballLossScorer(alpha / 2)
    upperScorer = pinballLossScorer(1 - alpha / 2)
    lowerModel = Model(lowerNet, "NN_lower", paramGridNetLower, lowerScorer)
    upperModel = Model(upperNet, "NN_upper", paramGridNetUpper, upperScorer)

    quantileNeuralNetRegressor = QuantileRegressorNeuralNet(
        [lowerModel, upperModel], alpha, "QNN"
    )

    return quantileNeuralNetRegressor


def getConformalQuantileRegressionForest(alpha, hyperparamsDir):
    quantileForestRegressor = getQuantileRegressionForest(alpha, hyperparamsDir)

    conformalQuantileForestRegressor = ConformalizedQuantileRegressor(
        quantileForestRegressor, name="CQRF", minVal=0
    )
    return conformalQuantileForestRegressor


def getConformalQuantileNeuralNetwork(alpha, hyperparamsDir, inputSize):
    quantileNeuralNetRegressor = getQuantileNeuralNetwork(
        alpha, hyperparamsDir, inputSize
    )
    conformalQuantileNeuralNetRegressor = ConformalizedQuantileRegressor(
        quantileNeuralNetRegressor, name="CQNN", minVal=0
    )
    return conformalQuantileNeuralNetRegressor


def getConformalizingScalarRandomForest(alpha, hyperparamsDir):
    rfBaseModel = getRandomForestModel(hyperparamsDir)
    rfErrorModel = getRandomForestErrorModel(hyperparamsDir)

    rfConformalizingScalar = ConformalizingScalarPredictor(
        rfBaseModel, rfErrorModel, alpha, name="CSRF", minVal=0
    )
    return rfConformalizingScalar


def getConformalizingScalarXGBoost(alpha, hyperparamsDir):
    xgbBaseModel = getXGBoostModel(hyperparamsDir)
    xgbErrorModel = getXGBoostErrorModel(hyperparamsDir)

    xgbConformalizingScalar = ConformalizingScalarPredictor(
        xgbBaseModel, xgbErrorModel, alpha, name="CSXGB", minVal=0
    )
    return xgbConformalizingScalar
