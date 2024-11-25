"""
Generating results from running trials of conformal prediction models
"""

import numpy as np
import pandas as pd

from data.data_loader import loadDataCsv, loadCsv
from data.data_processing import (
    processData,
    getDataProcessor,
    trainTestSplit,
)
from data.data_saver import saveExperimentData
from models.tuned_models import getQuantileRegressionModels, getConformalModels


def main():
    saveDir = "experiments/uncertainty-intervals/"
    modelCol = "Model"
    alphaCol = "Alpha"
    samplesCol = "Training samples"
    reserveredRatioCol = "Reserved ratio"
    coverageCol = "Empirical coverage"
    widthCol = "Empirical width"

    cols = [
        modelCol,
        alphaCol,
        samplesCol,
        reserveredRatioCol,
        coverageCol,
        widthCol,
    ]
    evaluateReservedRatio(saveDir + "reserved-ratio/", cols)
    evaluateTargetCoverage(saveDir + "target-coverage/", cols)
    evaluateSampleSizes(saveDir + "samples/", cols)
    evaluateBootstrap(saveDir + "bootstrap/", cols)
    evaluateFailedGeneralization(saveDir + "generalization/", cols)
    evaluateBounds(saveDir + "bounds/")


def evaluateReservedRatio(saveDir, cols):
    experimentName = "evaluate_reserved_ratio"
    trainDataDir = "data/intermediate/sthlm-sodertalje/train/"
    testDataDir = "data/intermediate/sthlm-sodertalje/test/"
    dfTrain = loadDataCsv(trainDataDir, ["", "-"])
    dfTest = loadDataCsv(testDataDir, ["", "-"])

    reservedRatios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    alphas = [0.1, 0.2]
    runTrialsAndSaveData(
        experimentName,
        dfTrain,
        dfTest,
        cols,
        reservedRatios,
        alphas,
        saveDir,
        isQuantile=False,
    )


def evaluateTargetCoverage(saveDir, cols):
    experimentName = "evaluate_target_coverage"
    trainDataDir = "data/intermediate/sthlm-sodertalje/train/"
    testDataDir = "data/intermediate/sthlm-sodertalje/test/"
    dfTrain = loadDataCsv(trainDataDir, ["", "-"])
    dfTest = loadDataCsv(testDataDir, ["", "-"])

    reservedRatios = [0.15]
    alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    runTrialsAndSaveData(
        experimentName, dfTrain, dfTest, cols, reservedRatios, alphas, saveDir
    )


def evaluateSampleSizes(saveDir, cols):
    experimentName = "evaluate_sample_sizes"
    trainDataDir = "data/intermediate/sthlm-sodertalje/train/"
    testDataDir = "data/intermediate/sthlm-sodertalje/test/"
    dfTrain = loadDataCsv(trainDataDir, ["", "-"])
    dfTest = loadDataCsv(testDataDir, ["", "-"])

    reservedRatios = [0.15]
    alphas = [0.1]
    samples = [1000, 2000, 5000, 8000, 12000, 15000, 17368]
    runTrialsAndSaveData(
        experimentName,
        dfTrain,
        dfTest,
        cols,
        reservedRatios,
        alphas,
        saveDir,
        sampleSizes=samples,
    )


def evaluateBootstrap(saveDir, cols):
    experimentName = "evaluate_bootstrap_variations"
    trainDataDir = "data/intermediate/sthlm-sodertalje/train/"
    testDataDir = "data/intermediate/sthlm-sodertalje/test/"
    dfTrain = loadDataCsv(trainDataDir, ["", "-"])
    dfTest = loadDataCsv(testDataDir, ["", "-"])

    reservedRatios = [0.15]
    alphas = [0.1, 0.2]
    numTrials = 30
    runTrialsAndSaveData(
        experimentName,
        dfTrain,
        dfTest,
        cols,
        reservedRatios,
        alphas,
        saveDir,
        numTrials=numTrials,
    )


def evaluateFailedGeneralization(saveDir, cols):
    experimentName = "evaluate_generalization"
    dataDir = "data/intermediate/sthlm-sodertalje/"

    df241004 = loadCsv(dataDir + "2024.10.04_11.19.11.csv", ["", "-"])
    df241028 = loadCsv(dataDir + "2024.10.28_17.20.20.csv", ["", "-"])
    df241029 = loadCsv(dataDir + "2024.10.29_07.18.51.csv", ["", "-"])
    dfs = [df241029, df241004, df241028]

    reservedRatios = [0.15]
    alphas = [0.1]
    for i, dfTest in enumerate(dfs):
        dfTrain = pd.concat(
            [df for j, df in enumerate(dfs) if j != i], ignore_index=True, join="outer"
        )
        runTrialsAndSaveData(
            f"{experimentName}_drivetest_{i+1}",
            dfTrain.copy(),
            dfTest.copy(),
            cols,
            reservedRatios,
            alphas,
            saveDir,
            note="Drive tests indices are in order 2024-10-04, 2024-10-28, 2024-10-29",
        )


# pylint: disable-msg=too-many-locals, too-many-statements, too-many-arguments, dangerous-default-value, too-many-nested-blocks
def runTrialsAndSaveData(
    experimentName,
    dfTrain,
    dfTest,
    cols,
    reservedRatios,
    alphas,
    saveDir,
    numTrials=1,
    sampleSizes=[None],
    verbose=True,
    isQuantile=True,
    note=None,
):
    hyperparamsDir = "config/hyperparameters/"

    ### DATA PREPARATION ###
    dependentCol = "UL_bitrate"
    # Mbps from Kbps
    dfTrain[dependentCol] = dfTrain[dependentCol] / 1024
    dfTest[dependentCol] = dfTest[dependentCol] / 1024

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

    data = []
    bootstrap = numTrials > 1
    for i, alpha in enumerate(alphas):
        for j, reservedRatio in enumerate(reservedRatios):
            for k in range(numTrials):
                for sampleSize in sampleSizes:
                    xTrain, xRes, yTrain, yRes = getTrainReservedData(
                        dataX,
                        dataY,
                        reservedRatio=reservedRatio,
                        sampleSize=sampleSize,
                        bootstrap=bootstrap,
                    )

                    ### TRAINING ###
                    conformalPredictors = getConformalModels(
                        alpha, hyperparamsDir, dataX.shape[1]
                    )
                    for model in conformalPredictors:
                        model.fit(xTrain, yTrain, xRes, yRes)
                    if isQuantile:
                        quantileModels = getQuantileRegressionModels(
                            alpha, hyperparamsDir, dataX.shape[1]
                        )
                        for qModel in quantileModels:
                            qModel.fit(
                                np.vstack((xTrain, xRes)), pd.concat([yTrain, yRes])
                            )
                        models = conformalPredictors + quantileModels
                    else:
                        models = conformalPredictors

                    ### EVALUATION ###
                    for model in models:
                        covRatio = model.getCoverageRatio(xTest, yTest)
                        data.append(
                            [
                                model.getName(),
                                alpha,
                                sampleSize,
                                reservedRatio,
                                covRatio,
                                model.getAverageIntervalWidth(xTest),
                            ]
                        )
                if verbose:
                    totalCompleted = (i * len(reservedRatios) + j) * numTrials + k + 1
                    totalTrials = len(alphas) * len(reservedRatios) * numTrials
                    print(f"Completed {totalCompleted/totalTrials*100:.2f}% of trials")

    dfResults = pd.DataFrame(data, columns=cols)
    saveExperimentData(
        dfResults,
        saveDir,
        experimentName,
        selectedFloatCols,
        selectedCatCols,
        conformalPredictors,
        note,
    )


def getTrainReservedData(x, y, reservedRatio=0.1, sampleSize=None, bootstrap=False):
    trainRatio = 1 - reservedRatio
    if sampleSize:
        indices = np.random.permutation(x.shape[0])
        x = x[indices[:sampleSize], :]
        y = y.iloc[indices[:sampleSize]]
    xTrain, xRes, yTrain, yRes = trainTestSplit(x, y, trainRatio)
    if bootstrap:
        indicesTrain = np.random.choice(
            np.arange(xTrain.shape[0]), size=xTrain.shape[0], replace=True
        )
        xTrain = xTrain[indicesTrain]
        yTrain = yTrain.iloc[indicesTrain]
        indicesRes = np.random.choice(
            np.arange(xRes.shape[0]), size=xRes.shape[0], replace=True
        )
        xRes = xRes[indicesRes]
        yRes = yRes.iloc[indicesRes]
    return xTrain, xRes, yTrain, yRes


# pylint: disable-msg=too-many-locals
def evaluateBounds(saveDir):
    experimentName = "evaluate_bounds"
    hyperparamsDir = "config/hyperparameters/"
    trainDataDir = "data/intermediate/sthlm-sodertalje/train/"
    testDataDir = "data/intermediate/sthlm-sodertalje/test/"
    dfTrain = loadDataCsv(trainDataDir, ["", "-"])
    dfTest = loadDataCsv(testDataDir, ["", "-"])

    cols = [
        "Model",
        "Alpha",
        "Lower Bound Average",
        "Upper Bound Average",
        "Share of points less than Lower Bound",
        "Share of points greater than Upper Bound",
    ]

    ### DATA PREPARATION ###
    dependentCol = "UL_bitrate"
    # Mbps from Kbps
    dfTrain[dependentCol] = dfTrain[dependentCol] / 1024
    dfTest[dependentCol] = dfTest[dependentCol] / 1024

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
    xTrain, xRes, yTrain, yRes = trainTestSplit(dataX, dataY, trainSize=0.85)

    data = []
    alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    for alpha in alphas:
        conformalPredictors = getConformalModels(alpha, hyperparamsDir, dataX.shape[1])
        for model in conformalPredictors:
            model.fit(xTrain, yTrain, xRes, yRes)

        quantileModels = getQuantileRegressionModels(
            alpha, hyperparamsDir, dataX.shape[1]
        )
        for qModel in quantileModels:
            qModel.fit(np.vstack((xTrain, xRes)), pd.concat([yTrain, yRes]))

        models = conformalPredictors + quantileModels

        ### EVALUATION ###
        for model in models:
            yPred = model.predict(xTest)
            lowerBounds = yPred[0]
            upperBounds = yPred[1]
            lowerBoundsAvg = np.average(lowerBounds)
            upperBoundsAvg = np.average(upperBounds)
            lowerBoundsRatio = np.average(lowerBounds > yTest)
            upperBoundsRatio = np.average(upperBounds > yTest)
            data.append(
                [
                    model.getName(),
                    alpha,
                    lowerBoundsAvg,
                    upperBoundsAvg,
                    lowerBoundsRatio,
                    upperBoundsRatio,
                ]
            )

    dfResults = pd.DataFrame(data, columns=cols)
    saveExperimentData(
        dfResults,
        saveDir,
        experimentName,
        selectedFloatCols,
        selectedCatCols,
        conformalPredictors,
    )


main()
