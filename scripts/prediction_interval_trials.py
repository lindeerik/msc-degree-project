"""
Generating results from running trials of conformal prediction models
"""

import pandas as pd

# Sci-kit
from sklearn.ensemble import RandomForestRegressor

# Torch
from torch import optim
from skorch import NeuralNetRegressor

# Random Forest Quantile
from sklearn_quantile import RandomForestQuantileRegressor

from data.data_loader import loadDataCsv
from data.data_processing import (
    processData,
    getDataProcessor,
    trainValTestSplit,
)
from data.data_saver import saveExperimentData
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


def main():
    saveDir = "experiments/uncertainty-intervals/"
    modelCol = "Model"
    alphaCol = "Alpha"
    trainRatioCol = "Train ratio"
    samplesCol = "Number of Samples"
    samplesEvalCol = "Evaluation Samples"
    coverageCol = "Empirical coverage"
    widthCol = "Empirical width"

    # Generate new measurements
    cols = [
        modelCol,
        alphaCol,
        trainRatioCol,
        samplesCol,
        samplesEvalCol,
        coverageCol,
        widthCol,
    ]
    testRatio = 0.1
    trainRatios = [0.7, 0.8, 0.9]
    alphas = [0.1, 0.2]
    numTrials = 20
    runTrialsAndSaveData(cols, testRatio, trainRatios, alphas, numTrials, saveDir)

# pylint: disable-msg=too-many-locals, too-many-statements
def runTrialsAndSaveData(
    cols, testRatio, trainRatios, alphas, numTrials, saveDir, verbose=True
):
    dirCsv = "data/intermediate/sthlm-sodertalje/"
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

    data = []
    for i, alpha in enumerate(alphas):
        ### NEURAL NET QUANTILE REGRESSOR ###
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
            [lowerModel, upperModel], alpha, "QNN"
        )
        conformalQuantileNeuralNetRegressor = ConformalizedQuantileRegressor(
            quantileNeuralNetRegressor, name="CQNN"
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
        rqfModel = Model(rfq, "QRF", paramGridRfq, doublePinballScorer)

        quantileForestRegressor = QuantileRegressorRandomForest(
            [rqfModel], alpha, "QRF"
        )
        conformalQuantileForestRegressor = ConformalizedQuantileRegressor(
            quantileForestRegressor, name="CQRF"
        )

        ### RANDOM FOREST CONFORMALIZING SCALAR PREDICTOR ###
        rfBase = RandomForestRegressor()
        paramGridRfBase = {
            "n_estimators": [300],
            "max_depth": [20],
            "min_samples_split": [5],
            "min_samples_leaf": [2],
            "max_features": ["sqrt"],
        }

        rfError = RandomForestRegressor()
        paramGridRfError = {
            "n_estimators": [300],
            "max_depth": [20],
            "min_samples_split": [5],
            "min_samples_leaf": [2],
            "max_features": ["sqrt"],
        }
        baseModel = Model(rfBase, "RF", paramGridRfBase)
        errorModel = Model(rfError, "RF Error", paramGridRfError)

        conformalizingScalar = ConformalizingScalarPredictor(
            baseModel, errorModel, alpha, name="CSRF"
        )

        conformalPredictors = [
            conformalQuantileForestRegressor,
            conformalQuantileNeuralNetRegressor,
            conformalizingScalar,
        ]
        for j, trainRatio in enumerate(trainRatios):
            trainRatioOfAllData = trainRatio * (1 - testRatio)
            validatioRatio = 1 - trainRatioOfAllData - testRatio
            for _ in range(numTrials):
                ### DIVIDE INTO TRAINING, VALIDATION AND TEST ###
                xTrain, xRes, xTest, yTrain, yRes, yTest = trainValTestSplit(
                    dataX, dataY, trainRatioOfAllData, validatioRatio
                )
                samples = xTrain.shape[0] + xRes.shape[0]
                testSamples = xTest.shape[0]

                ### TRAINING ###
                for conformalModel in conformalPredictors:
                    conformalModel.fit(xTrain, yTrain, xRes, yRes, 2)

                ### EVALUATION ###
                for conformalModel in conformalPredictors:
                    name = conformalModel.getName()
                    addMetricsToList(
                        data,
                        conformalModel,
                        name,
                        alpha,
                        trainRatio,
                        samples,
                        testSamples,
                        xTest,
                        yTest,
                    )
                    try:
                        qModel = conformalModel.getQuantileRegressor()
                        qName = qModel.getName()
                        addMetricsToList(
                            data,
                            qModel,
                            qName,
                            alpha,
                            trainRatio,
                            samples,
                            testSamples,
                            xTest,
                            yTest,
                        )
                    except AttributeError:
                        pass
            if verbose:
                completedShare = (i * len(trainRatios) + j + 1) / (
                    len(alphas) * len(trainRatios)
                )
                print(f"Completed {completedShare*100:.2f}% of trials")

    df = pd.DataFrame(data, columns=cols)
    saveExperimentData(
        df,
        saveDir,
        "uncertainty_intervals_trials",
        selectedFloatCols,
        selectedCatCols,
        conformalPredictors,
        "",
    )


def addMetricsToList(
    data, model, name, alpha, trainRatio, samples, testSamples, xTest, yTest
):
    coverage = model.getCoverageRatio(xTest, yTest)
    width = model.getAverageIntervalWidth(xTest)
    data.append([name, alpha, trainRatio, samples, testSamples, coverage, width])


main()
