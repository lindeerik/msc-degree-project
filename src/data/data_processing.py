"""
Data processing for tabular data
"""

from datetime import date
import numpy as np
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def processData(df, selectedFloatCols, selectedCatCols, dependentCol, processor=None):
    selectedCols = selectedFloatCols + selectedCatCols + [dependentCol]
    df = df[selectedCols].dropna()

    processor = (
        getDataProcessor(selectedFloatCols, selectedCatCols)
        if processor is None
        else processor
    )

    dataX = processor.fit_transform(df[selectedFloatCols + selectedCatCols]).astype(
        np.float32
    )
    dataY = df[dependentCol].astype(np.float32)

    return dataX, dataY


def getDataProcessor(
    selectedFloatCols,
    selectedCatCols,
    binaryEncoding=True,
    applyScaler=False,
    applyPCA=False,
):
    steps = []
    if applyScaler:
        steps.append(("scaler", StandardScaler()))
    if applyPCA:
        steps.append(("pca", PCA()))
    numTransformer = Pipeline(steps=steps) if len(steps) > 0 else "passthrough"

    catTransformer = (
        ce.BinaryEncoder(cols=selectedCatCols, drop_invariant=True)
        if binaryEncoding
        else ce.OneHotEncoder(cols=selectedCatCols, drop_invariant=True)
    )
    processor = ColumnTransformer(
        transformers=[
            ("num", numTransformer, selectedFloatCols),
            ("cat", catTransformer, selectedCatCols),
        ]
    )
    return processor


def trainTestSplit(dataX, dataY, trainSize=0.8, randomState=None):
    xTrain, xTest, yTrain, yTest = train_test_split(
        dataX, dataY, test_size=1 - trainSize, random_state=randomState
    )
    return xTrain, xTest, yTrain, yTest


def trainValTestSplit(dataX, dataY, trainSize=0.7, valSize=0.15, randomState=None):
    testSize = 1 - trainSize - valSize
    xTrain, xTest, yTrain, yTest = train_test_split(
        dataX, dataY, test_size=1 - trainSize, random_state=randomState
    )
    xVal, xTest, yVal, yTest = train_test_split(
        xTest,
        yTest,
        test_size=testSize / (testSize + valSize),
        random_state=randomState,
    )
    return xTrain, xVal, xTest, yTrain, yVal, yTest


def transformTimestamp(df, timestampCol, timeOfDayCol=None, timeOfYearCol=None):
    if (not timeOfDayCol) and (not timeOfYearCol):
        return df
    if timeOfDayCol:
        df[timeOfDayCol] = df[timestampCol].apply(readTimeOfDayFromTimestamp)
    if timeOfYearCol:
        df[timeOfYearCol] = df[timestampCol].apply(readTimeOfYearFromTimestamp)
    df.drop(columns=[timestampCol], inplace=True)
    return df


def readTimeOfDayFromTimestamp(timestamp):
    timeOfDayString = timestamp.split("_")[1]
    hours, minutes, seconds = map(int, timeOfDayString.split("."))
    timeOfDay = (hours * 3600 + minutes * 60 + seconds) / 86400.0
    return timeOfDay


def readTimeOfYearFromTimestamp(timestamp):
    timeOfYearString = timestamp.split("_")[0]
    year, month, day = map(int, timeOfYearString.split("."))
    currentDate = date(year, month, day)
    startOfYear = date(year, 1, 1)
    endOfYear = date(year, 12, 31)
    totalDays = (endOfYear - startOfYear).days + 1
    timeOfYear = ((currentDate - startOfYear).days + 1) / totalDays
    return timeOfYear
