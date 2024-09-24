"""
Data processing for tabular data
"""

from datetime import date
import numpy as np
import pandas as pd
import category_encoders as ce


def processData(
    df, selectedFloatCols, selectedCatCols, dependentCol, binaryEncoding=True
):
    selectedCols = selectedFloatCols + selectedCatCols
    selectedCols.append(dependentCol)
    df = df[selectedCols].dropna()

    encoder = (
        ce.BinaryEncoder(cols=selectedCatCols, drop_invariant=True)
        if binaryEncoding
        else ce.OneHotEncoder(cols=selectedCatCols, drop_invariant=True)
    )
    encoded = encoder.fit_transform(df[selectedCatCols])
    catData = encoded.reset_index(drop=True)

    # z-score normalization
    floatDataUnnormalized = df.drop(
        selectedCatCols + [dependentCol], axis=1
    ).reset_index(drop=True)
    floatData = (
        floatDataUnnormalized - floatDataUnnormalized.mean()
    ) / floatDataUnnormalized.std()
    dataX = pd.concat([floatData, catData], axis=1).astype(np.float32)
    dataY = df[dependentCol].astype(np.float32)

    return dataX, dataY


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
