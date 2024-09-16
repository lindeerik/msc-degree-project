"""
Data loading for tabular data
"""

import glob
import os

import pandas as pd
import numpy as np
import category_encoders as ce


def loadDataCsv(filePath, emptyData):

    fileNames = glob.glob(os.path.join(filePath, "*.csv"))
    dataFrames = []

    for fileName in fileNames:
        df = pd.read_csv(fileName, index_col=None, header=0, na_values=emptyData)
        dataFrames.append(df)

    return pd.concat(dataFrames, axis=0, ignore_index=True)


def loadDataParquet(filePath):
    fileNames = glob.glob(os.path.join(filePath, "*.parquet"))
    dataFrames = []

    for fileName in fileNames:
        df = pd.read_parquet(fileName)
        dataFrames.append(df)

    return pd.concat(dataFrames, axis=0, ignore_index=True)


def saveDataParquet(dirCsv, dirParquet, emptyData=""):
    fileNames = glob.glob(os.path.join(dirCsv, "*.csv"))

    for fileName in fileNames:
        df = pd.read_csv(fileName, index_col=None, header=0, na_values=emptyData)
        fileNameParquet = os.path.basename(fileName.replace("csv", "parquet"))
        df.to_parquet(dirParquet + fileNameParquet, engine="pyarrow")


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
