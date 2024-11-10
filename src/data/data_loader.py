"""
Data loading for tabular data
"""

import glob
import os
import pickle

import pandas as pd


def loadDataCsv(filePath, emptyData):
    fileNames = glob.glob(os.path.join(filePath, "*.csv"))
    dataFrames = []

    for fileName in fileNames:
        df = loadCsv(fileName, emptyData)
        dataFrames.append(df)

    return pd.concat(dataFrames, axis=0, ignore_index=True)


def loadCsv(filePath, emptyData):
    df = pd.read_csv(filePath, index_col=None, header=0, na_values=emptyData)
    return df


def loadDataParquet(filePath):
    fileNames = glob.glob(os.path.join(filePath, "*.parquet"))
    dataFrames = []

    for fileName in fileNames:
        df = pd.read_parquet(fileName)
        dataFrames.append(df)

    return pd.concat(dataFrames, axis=0, ignore_index=True)


def loadModel(modelPath):
    with open(modelPath, "rb") as f:
        return pickle.load(f)
