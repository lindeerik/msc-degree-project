"""
Generating results from running trials of point-estimation models
"""

import numpy as np
import pandas as pd

from data.data_loader import loadDataCsv
from data.data_processing import processData, getDataProcessor
from data.data_saver import saveExperimentData
from models.tuned_models import getPointEstimationModels


def main():
    saveDir = "experiments/point-estimation/samples/"
    modelCol = "Model"
    trainRatioCol = "Train ratio"
    samplesCol = "Number of Samples"
    trainR2Col = "Training R2"
    testR2Col = "Test R2"
    trainRmseCol = "Training RMSE"
    testRmseCol = "Test RMSE"
    trainMaeCol = "Training MAE"
    testMaeCol = "Test MAE"

    cols = [
        modelCol,
        trainRatioCol,
        samplesCol,
        trainR2Col,
        testR2Col,
        trainRmseCol,
        testRmseCol,
        trainMaeCol,
        testMaeCol,
    ]
    trainRatios = np.divide([1000, 2000, 5000, 8000, 12000, 15000, 17368], 17368)
    numTrials = 1
    runTrialsAndSaveData(cols, trainRatios, numTrials, saveDir)


# pylint: disable-msg=too-many-locals, too-many-statements
def runTrialsAndSaveData(cols, trainRatios, numTrials, saveDir, verbose=True):
    trainData = "data/intermediate/sthlm-sodertalje/train/"
    testData = "data/intermediate/sthlm-sodertalje/test/"
    hyperparamsDir = "config/hyperparameters/"
    dfTrain = loadDataCsv(trainData, ["", "-"])
    dfTest = loadDataCsv(testData, ["", "-"])

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
    xTrainFull, yTrainFull = processData(
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
    ### TRIALS ###
    for j, trainRatio in enumerate(trainRatios):
        samples = int(trainRatio * xTrainFull.shape[0])
        for _ in range(numTrials):
            indices = np.random.permutation(xTrainFull.shape[0])
            xTrain = xTrainFull[indices[:samples], :]
            yTrain = yTrainFull.iloc[indices[:samples]]

            models = getPointEstimationModels(hyperparamsDir, xTrain.shape[1])

            for model in models:
                model.fit(xTrain, yTrain)

            for model in models:
                name = model.getName()
                trainR2 = model.getR2(xTrain, yTrain)
                testR2 = model.getR2(xTest, yTest)
                trainRmse = model.getRmse(xTrain, yTrain)
                testRmse = model.getRmse(xTest, yTest)
                trainMae = model.getMae(xTrain, yTrain)
                testMae = model.getMae(xTest, yTest)
                data.append(
                    [
                        name,
                        trainRatio,
                        samples,
                        trainR2,
                        testR2,
                        trainRmse,
                        testRmse,
                        trainMae,
                        testMae,
                    ]
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


main()
