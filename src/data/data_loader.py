"""
Data loading for tabular data
"""

import glob
import os
import json
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


def loadHyperparams(filePath):
    try:
        with open(filePath, "r", encoding="utf-8") as f:
            hyperparams = json.load(f)
        return hyperparams
    except FileNotFoundError:
        print(
            f"Warning: Hyperparameter file not found at '{filePath}'. Using default values."
        )
        return {}
