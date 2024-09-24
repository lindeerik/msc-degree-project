"""
Data loading for tabular data
"""

# Sci-kit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Torch
from torch import optim
from skorch import NeuralNetRegressor

# Random Forest Quantile
from sklearn_quantile import RandomForestQuantileRegressor

from data.data_loader import loadDataParquet
from data.data_processing import processData
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


# pylint: disable-msg=too-many-locals
def main():
    dirParquet = "data/intermediate/"
    df = loadDataParquet(dirParquet)

    ### DATA PREPARATION ###
    dependentCol = "UL_bitrate"

    selectedFloatCols = ["Longitude", "Latitude", "Speed", "RSRP", "RSRQ", "SNR"]
    selectedCatCols = ["CellID"]

    dataX, dataY = processData(
        df, selectedFloatCols, selectedCatCols, dependentCol, True
    )

    ### DIVIDE INTO TRAINING, VALIDATION AND TEST ###
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    xTrain, xTest, yTrain, yTest = train_test_split(
        dataX, dataY, test_size=1 - train_ratio
    )
    xVal, xTest, yVal, yTest = train_test_split(
        xTest, yTest, test_size=test_ratio / (test_ratio + validation_ratio)
    )

    ### NEURAL NET QUANTILE REGRESSOR ###

    alpha = 0.1
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
    paramGridNetLower = {
        "lr": [0.01],
        "max_epochs": [100],
        "optimizer__weight_decay": [0.01],
        "batch_size": [128],
    }
    paramGridNetUpper = {
        "lr": [0.01],
        "max_epochs": [100],
        "optimizer__weight_decay": [0.01],
        "batch_size": [128],
    }

    lowerScorer = pinballLossScorer(alpha / 2)
    upperScorer = pinballLossScorer(1 - alpha / 2)
    lowerModel = Model(
        lowerNet, "Lower Bound Neural Network", paramGridNetLower, lowerScorer
    )
    upperModel = Model(
        upperNet, "Upper Bound Neural Network", paramGridNetUpper, upperScorer
    )

    quantileNeuralNetRegressor = QuantileRegressorNeuralNet(
        [lowerModel, upperModel], alpha, "Neural Network Quantile"
    )
    conformalQuantileNeuralNetRegressor = ConformalizedQuantileRegressor(
        quantileNeuralNetRegressor
    )

    ### RANDOM FOREST QUANTILE REGRESSOR ###

    rfq = RandomForestQuantileRegressor(q=[alpha / 2, 1 - alpha / 2])
    paramGridRfq = {
        "n_estimators": [100],
        "criterion": ["squared_error"],
        "max_depth": [10],
        "min_samples_split": [10],
        "min_samples_leaf": [10],
        "min_weight_fraction_leaf": [0.1],
        "max_features": ["log2"],
    }
    doublePinballScorer = doublePinballLossScorer(alpha / 2, 1 - alpha / 2)
    rqfModel = Model(rfq, "Random Forest Quantile", paramGridRfq, doublePinballScorer)

    quantileForestRegressor = QuantileRegressorRandomForest(
        [rqfModel], alpha, "Random Forest Quantile"
    )
    conformalQuantileForestRegressor = ConformalizedQuantileRegressor(
        quantileForestRegressor
    )

    ### RANDOM FOREST CONFORMALIZING SCALAR PREDICTOR ###
    rfBase = RandomForestRegressor(random_state=42)
    paramGridRfBase = {
        "n_estimators": [300],
        "max_depth": [20],
        "min_samples_split": [5],
        "min_samples_leaf": [2],
        "max_features": ["sqrt"],
    }

    rfError = RandomForestRegressor(random_state=42)
    paramGridRfError = {
        "n_estimators": [300],
        "max_depth": [20],
        "min_samples_split": [5],
        "min_samples_leaf": [2],
        "max_features": ["sqrt"],
    }
    baseModel = Model(rfBase, "Random Forest", paramGridRfBase)
    errorModel = Model(rfError, "Random Forest Error", paramGridRfError)

    conformalizingScalar = ConformalizingScalarPredictor(baseModel, errorModel, alpha)

    ### TRAINING ###
    conformalPredictors = [
        conformalQuantileForestRegressor,
        conformalQuantileNeuralNetRegressor,
        conformalizingScalar,
    ]
    for conformalModel in conformalPredictors:
        conformalModel.fit(xTrain, yTrain, xVal, yVal, 2)

    ### EVALUATION ###
    for conformalModel in conformalPredictors:
        if "Quantile" in conformalModel.getName():
            print(
                f"{conformalModel.getQuantileRegressor().getName()} coverage: {conformalModel.getQuantileRegressor().getCoverageRatio(xTest, yTest)}"
            )
        print(
            f"{conformalModel.getName()} coverage: {conformalModel.getCoverageRatio(xTest, yTest)}"
        )
        print(
            f"Average {conformalModel.getName()} width: {conformalModel.getAverageIntervalWidth(xTest)}"
        )


main()
