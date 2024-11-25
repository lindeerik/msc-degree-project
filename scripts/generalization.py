"""
Generalization results for point-estimation and prediction intervals
"""

import numpy as np
import pandas as pd

from data.data_loader import loadCsv
from data.data_processing import processData, getDataProcessor
from data.data_saver import saveExperimentData
from models.tuned_models import (
    getPointEstimationModels,
    getQuantileRegressionModels,
    getConformalModels,
)


def main():
    dataDir = "data/intermediate/sthlm-sodertalje/"
    hyperparamsDir = "config/hyperparameters/"

    df241004 = loadCsv(dataDir + "2024.10.04_11.19.11.csv", ["", "-"])
    df241028 = loadCsv(dataDir + "2024.10.28_17.20.20.csv", ["", "-"])
    df241029 = loadCsv(dataDir + "2024.10.29_07.18.51.csv", ["", "-"])

    ### DATA PREPARATION ###
    dependentCol = "UL_bitrate"
    floatCols = [
        "Longitude",
        "Latitude",
        "Speed",
        "SNR",
        "Level",
        "Qual",
    ]
    catCols = [
        "CellID",
        "Node",
        "NetworkMode",
        "BAND",
        "BANDWIDTH",
        "LAC",
        "PSC",
    ]
    df241004[dependentCol] = df241004[dependentCol] / 1024
    df241028[dependentCol] = df241028[dependentCol] / 1024
    df241029[dependentCol] = df241029[dependentCol] / 1024
    dfs = [df241004, df241028, df241029]

    pointEstimationGeneralization(dfs, floatCols, catCols, dependentCol, hyperparamsDir)
    predictionIntervalGeneralization(
        dfs, floatCols, catCols, dependentCol, hyperparamsDir
    )


def pointEstimationGeneralization(
    dfs, floatCols, catCols, dependentCol, hyperparamsDir
):
    data = []
    cols = [
        "Model",
        "Index of Drive Test for Testing Data",
        "Train R2",
        "Test R2",
        "Train RMSE",
        "Test RMSE",
        "Train MAE",
        "Test MAE",
    ]
    # Iterate by training models on all but one drive test (which beomces test data)
    for i, dfTest in enumerate(dfs):
        dfTrain = pd.concat(
            [dfResults for j, dfResults in enumerate(dfs) if j != i],
            ignore_index=True,
            join="outer",
        )

        processor = getDataProcessor(
            floatCols, catCols, applyScaler=True, binaryEncoding=True
        )
        xTrain, yTrain = processData(
            dfTrain, floatCols, catCols, dependentCol, processor
        )
        xTest, yTest = processData(
            dfTest,
            floatCols,
            catCols,
            dependentCol,
            processor,
            fitProcessor=False,
        )

        models = getPointEstimationModels(hyperparamsDir, xTrain.shape[1])
        for model in models:
            model.fit(xTrain, yTrain)
            trainR2 = model.getR2(xTrain, yTrain)
            testR2 = model.getR2(xTest, yTest)
            trainRmse = model.getRmse(xTrain, yTrain)
            testRmse = model.getRmse(xTest, yTest)
            trainMae = model.getMae(xTrain, yTrain)
            testMae = model.getMae(xTest, yTest)
            data.append(
                [
                    model.getName(),
                    i,
                    trainR2,
                    testR2,
                    trainRmse,
                    testRmse,
                    trainMae,
                    testMae,
                ]
            )

    dfResults = pd.DataFrame(data, columns=cols)
    saveExperimentData(
        dfResults,
        "experiments/point-estimation/generalization/",
        "point_estimation_generalization",
        floatCols,
        catCols,
        models,
        "Drive tests are in order 2024-10-04, 2024-10-28, 2024-10-29",
    )


def predictionIntervalGeneralization(
    dfs, floatCols, catCols, dependentCol, hyperparamsDir
):
    cols = [
        "Index of Training Data",
        "Index of Reserved Data",
        "Index of Testing Data",
        "Model",
        "Empirical coverage",
        "Empirical width",
    ]

    fullData = []
    models = []
    for i, dfTest in enumerate(dfs):
        for j, dfRes in enumerate(dfs):
            if i != j:
                dfTrain = pd.concat(
                    [dfResults for k, dfResults in enumerate(dfs) if k not in (i, j)],
                    ignore_index=True,
                    join="outer",
                )
                testIndices = [3 - i - j, i, j]
                data, models = getPredictionIntervalPerformances(
                    dfTrain,
                    dfRes,
                    dfTest,
                    floatCols,
                    catCols,
                    dependentCol,
                    hyperparamsDir,
                    testIndices,
                )
                fullData.extend(data)

    dfResults = pd.DataFrame(fullData, columns=cols)
    saveExperimentData(
        dfResults,
        "experiments/uncertainty-intervals/generalization/",
        "uncertainty_interval_generalization",
        floatCols,
        catCols,
        models,
        "Drive tests are in order 2024-10-04, 2024-10-28, 2024-10-29",
    )


def getPredictionIntervalPerformances(
    dfTrain,
    dfRes,
    dfTest,
    floatCols,
    catCols,
    dependentCol,
    hyperparamsDir,
    testIndices,
):
    alpha = 0.1

    ### DATA PREPARATION ###
    processor = getDataProcessor(floatCols, catCols, applyScaler=True)
    xTrain, yTrain = processData(dfTrain, floatCols, catCols, dependentCol, processor)
    xRes, yRes = processData(
        dfRes, floatCols, catCols, dependentCol, processor, fitProcessor=False
    )
    xTest, yTest = processData(
        dfTest, floatCols, catCols, dependentCol, processor, fitProcessor=False
    )

    conformalPredictors = getConformalModels(alpha, hyperparamsDir, xTrain.shape[1])

    ## TRAINING ###
    for model in conformalPredictors:
        model.fit(xTrain, yTrain, xRes, yRes)

    quantileModels = getQuantileRegressionModels(alpha, hyperparamsDir, xTrain.shape[1])
    for qModel in quantileModels:
        qModel.fit(np.vstack((xTrain, xRes)), pd.concat([yTrain, yRes]))

    models = quantileModels + conformalPredictors

    ### EVALUATION ###
    data = []
    for model in models:
        data.append(
            testIndices
            + [
                model.getName(),
                model.getCoverageRatio(xTest, yTest),
                model.getAverageIntervalWidth(xTest),
            ]
        )

    return data, models


main()
