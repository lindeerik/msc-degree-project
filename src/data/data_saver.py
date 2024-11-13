"""
Data saving for tabular data
"""

import os
import json
import glob
from datetime import datetime
import pandas as pd


def saveExperimentData(
    df, path, experimentName, floatCols, catCols, models, notes=None
):
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    dirName = os.path.join(path, timestamp)
    os.makedirs(dirName, exist_ok=True)

    saveDataToCsv(df, os.path.join(dirName, "data.csv"))
    saveMetadataToJson(
        experimentName,
        timestamp,
        floatCols,
        catCols,
        models,
        notes,
        os.path.join(dirName, "metadata.json"),
    )
    saveModelsToPkl(models, dirName)


def saveDataToCsv(df, filePath):
    df.to_csv(filePath, index=False)


def saveMetadataToJson(
    experimentName, timestamp, floatCols, catCols, models, notes, filePath
):
    modelMetadata = {}
    for model in models:
        modelMetadata[model.getName()] = model.getMetadata()
    metadata = {
        "experiment_name": f"{experimentName}_{timestamp}",
        "float_columns": floatCols,
        "categorical_columns": catCols,
        "models": modelMetadata,
        "notes": notes,
    }
    with open(filePath, "w", encoding="utf-8") as jsonFile:
        json.dump(metadata, jsonFile, indent=4)


def saveModelsToPkl(models, dirName):
    subDirName = os.path.join(dirName, "models")
    os.makedirs(subDirName, exist_ok=True)
    for model in models:
        path = os.path.join(
            subDirName, f"{model.getName()}.pkl".lower().replace(" ", "_")
        )
        model.saveModel(path)


def saveModelBestParamsToJson(model, dirName):
    bestParams = model.getBestParams()
    if bestParams:
        fileName = model.getName().lower().replace(" ", "_")
        file = os.path.join(dirName, f"{fileName}.json")
        bestParams = {key: [value] for key, value in bestParams.items()}
        with open(file, "w", encoding="utf-8") as jsonFile:
            json.dump(bestParams, jsonFile, indent=4)


def saveCsvWithDateTime(data, path):
    if not os.path.exists(path):
        os.makedirs(path)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    filename = os.path.join(path, timestamp + ".csv")
    data.to_csv(filename, index=False)


def saveDataParquet(dirCsv, dirParquet, emptyData=""):
    fileNames = glob.glob(os.path.join(dirCsv, "*.csv"))

    for fileName in fileNames:
        df = pd.read_csv(fileName, index_col=None, header=0, na_values=emptyData)
        fileNameParquet = os.path.basename(fileName.replace("csv", "parquet"))
        df.to_parquet(dirParquet + fileNameParquet, engine="pyarrow")
