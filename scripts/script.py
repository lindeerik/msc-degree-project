import matplotlib.pyplot as plt

#Sci-kit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#Torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor

from models.neuralnetwork.architecture import ThroughputPredictor


#xg-boost
import xgboost as xgb


from visualization.visualize import plotErrors
from data.data_loader import *
from models.training import trainModels

def main():
    dirParquet = "data/intermediate/"
    df = loadDataParquet(dirParquet)

    ### DATA PREPARATION ###
    dependentCol = "UL_bitrate"

    selectedFloatCols = ["Longitude", "Latitude", "Speed", "RSRP","RSRQ","SNR"]
    selectedCatCols = ["CellID"]

    dataX, dataY = processData(df, selectedFloatCols,selectedCatCols, dependentCol)


    ### DIVIDE INTO TRAINING, VALIDATION AND TEST ###
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    xTrain, xTest, yTrain, yTest = train_test_split(dataX, dataY, test_size=1 - train_ratio)
    xVal, xTest, yVal, yTest = train_test_split(xTest, yTest, test_size=test_ratio/(test_ratio + validation_ratio))


    ### SELECT MODELS ###
    rf = RandomForestRegressor(random_state=42)
    paramGridRf = {
        'n_estimators': [300],        # Number of trees in the forest
        'max_depth': [20],        # Maximum depth of the tree
        'min_samples_split': [5],        # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],          # Minimum number of samples required to be at a leaf node
        'max_features': ['sqrt']        # Number of features to consider when looking for the best split
    }

    xGradBoost = xgb.XGBRegressor(random_state=42)
    paramGridXgb = {
        'n_estimators': [200],
        'learning_rate': [0.05],
        'max_depth': [5],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'gamma': [0.1],
        'reg_alpha': [0.01],
        'reg_lambda': [1.5]
    }

    net = NeuralNetRegressor(
        ThroughputPredictor,
        module__input_size=dataX.shape[1],  # Pass the input size to the module
        optimizer=optim.Adam,               # Optimizer
        criterion=nn.MSELoss,               # Loss function
        verbose=0,                          # Silence verbose output
        train_split=None                    # Disable internal train/val split, we'll use external CV
    )
    paramGridNet = {
        'lr': [0.01],
        'max_epochs': [100],
        'optimizer__weight_decay': [0.01],
        'batch_size': [128]
    }

    models = {net: paramGridNet, rf: paramGridRf, xGradBoost:paramGridXgb}

    ### TRAINING AND EVALUATION ###
    errors = trainModels(models, xTrain, yTrain, xVal, yVal, xTest, yTest)

    ### CHECK NORMALITY OF ERRORS ###
    for err in errors:
        plotErrors(errors, "Model")
    plt.show()

main()