#Sci-kit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#Torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor

#xg-boost
import xgboost as xgb

from visualization.visualize import *
from data.data_loader import *
from models.model import Model
from models.neuralnetwork.architecture import *
from models.quantileregression.conformalprediction import *
from models.quantileregression.pinball import *

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


    alpha = 0.1
    lowerNet = NeuralNetRegressor(
        ThroughputPredictor,
        module__input_size=dataX.shape[1],  
        optimizer=optim.Adam,               
        criterion=PinballLoss(alpha/2),
        verbose=0,                          
        train_split=None                    
    )
    upperNet = NeuralNetRegressor(
        ThroughputPredictor,
        module__input_size=dataX.shape[1],
        optimizer=optim.Adam,
        criterion=PinballLoss(1-alpha/2),
        verbose=0,
        train_split=None
    )
    paramGridNetLower = {
        'lr': [0.01],
        'max_epochs': [100],
        'optimizer__weight_decay': [0.01],
        'batch_size': [128]
    }
    paramGridNetUpper = {
        'lr': [0.01],
        'max_epochs': [100],
        'optimizer__weight_decay': [0.01],
        'batch_size': [128]
    }
    lowerScorer = pinballLossScorer(alpha/2)
    upperScorer = pinballLossScorer(1-alpha/2)
    lowerModel = Model(lowerNet, "Lower Bound Neural Network", paramGridNetLower, lowerScorer)
    upperModel = Model(upperNet, "Upper Bound Neural Network", paramGridNetUpper, upperScorer)

    quantileRegressor = QuantileRegressor(lowerModel, upperModel)

    conformalScoreFunc = lambda X,Y: pinballConformalScoreFunc(lowerModel, upperModel, X, Y)
    conformalPredictor = ConformalizedQuantileRegressor(quantileRegressor, conformalScoreFunc, alpha)
    conformalPredictor.fit(xTrain, yTrain, xVal, yVal, 3)

    print("Quantile regression coverage:", quantileRegressor.getCoverageRatio(xTest, yTest))
    print("Conformal prediction coverage:", conformalPredictor.getCoverageRatio(xTest, yTest))
              
main()