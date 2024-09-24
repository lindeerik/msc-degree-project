"""
Functions for drop-out feature selection
"""

import numpy as np
from sklearn.model_selection import train_test_split
from data.data_processing import processData


def getBestFeatures(
    df, floatCols, catCols, dependentCol, model, trainSize=0.7, isBinaryEncoding=True
):
    selectedFloatCols = floatCols.copy()
    prevScore = -1.0
    scoreDiff = 1.0
    while scoreDiff > 0 and len(selectedFloatCols) > 1:
        worseFloatCol, score = dropWorseFeature(
            df,
            selectedFloatCols,
            catCols,
            dependentCol,
            model,
            trainSize,
            isBinaryEncoding,
        )
        scoreDiff = score - prevScore
        prevScore = score
        if scoreDiff > 0:
            selectedFloatCols.remove(worseFloatCol)
    return selectedFloatCols


def dropWorseFeature(
    df, floatCols, catCols, dependentCol, model, trainSize=0.7, isBinaryEncoding=True
):
    dataX, dataY = processData(df, floatCols, catCols, dependentCol, isBinaryEncoding)
    xTrain, xTest, yTrain, yTest = train_test_split(
        dataX, dataY, test_size=1 - trainSize
    )

    scores = []
    for floatCol in floatCols:
        model.fit(xTrain.drop(floatCol, axis=1), yTrain)
        score = -model.getMse(xTest.drop(floatCol, axis=1), yTest)
        scores.append(score)
    # worst feature has best performance in its absence
    idxWorst = np.argmax(scores)
    return floatCols[idxWorst], scores[idxWorst]
