"""
Pointestimation Script for training and generating point estimations
"""

# Sci-kit
from sklearn.ensemble import RandomForestRegressor

# Torch
from torch import nn
from torch import optim
from skorch import NeuralNetRegressor

# xg-boost
import xgboost as xgb

from visualization.visualize import plotModelsErrors
from data.data_loader import loadDataParquet
from data.data_processing import processData, trainValTestSplit
from models.training import trainModels
from models.model import Model
from models.neuralnetwork.architecture import ThroughputPredictor


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
    trainRatio = 0.75
    validatioRatio = 0.15

    xTrain, xVal, xTest, yTrain, yVal, yTest = trainValTestSplit(
        dataX, dataY, trainRatio, validatioRatio
    )

    ### SELECT MODELS ###
    rf = RandomForestRegressor(random_state=42)
    paramGridRf = {
        "n_estimators": [300],  # Number of trees in the forest
        "max_depth": [20],  # Maximum depth of the tree
        "min_samples_split": [
            5
        ],  # Minimum number of samples required to split an internal node
        "min_samples_leaf": [
            1,
            2,
            4,
        ],  # Minimum number of samples required to be at a leaf node
        "max_features": [
            "sqrt"
        ],  # Number of features to consider when looking for the best split
    }

    xGradBoost = xgb.XGBRegressor(random_state=42)
    paramGridXgb = {
        "n_estimators": [200],
        "learning_rate": [0.05],
        "max_depth": [5],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "gamma": [0.1],
        "reg_alpha": [0.01],
        "reg_lambda": [1.5],
    }

    net = NeuralNetRegressor(
        ThroughputPredictor,
        module__input_size=dataX.shape[1],  # Pass the input size to the module
        optimizer=optim.Adam,  # Optimizer
        criterion=nn.MSELoss,  # Loss function
        verbose=0,  # Silence verbose output
        train_split=None,  # Disable internal train/val split, we'll use external CV
    )
    paramGridNet = {
        "lr": [0.01],
        "max_epochs": [100],
        "optimizer__weight_decay": [0.01],
        "batch_size": [128],
    }

    models = [
        Model(rf, "Random Forest", paramGridRf),
        Model(xGradBoost, "XGBoost", paramGridXgb),
        Model(net, "Neural Network", paramGridNet),
    ]

    ### TRAINING AND EVALUATION ###
    errors = trainModels(models, xTrain, yTrain, xVal, yVal, xTest, yTest)

    ### CHECK NORMALITY OF ERRORS ###
    plotModelsErrors(errors, models)


main()
