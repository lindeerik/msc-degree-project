"""
Generating plots of prediction intervals. May be shown directly or saved to directory. 
"""

import pandas as pd
import matplotlib.pyplot as plt

from visualization.plots import plotBoxplotFromDf, plotLineChartFromDf
from visualization.save_figures import saveFiguresWithDateTime


def main():
    saveDir = "experiments/uncertainty-intervals/"
    modelCol = "Model"
    alphaCol = "Alpha"
    samplesCol = "Training samples"
    trainRatioCol = "Reserved ratio"
    coverageCol = "Empirical coverage"
    widthCol = "Empirical width"

    dfReservedRatio = pd.read_csv(saveDir + "reserved-ratio/20241122T154551Z/data.csv")
    fileNames = [
        "EC_reservedratio_alpha10.png",
        "EW_reservedratio_alpha10.png",
        "EC_reservedratio_alpha20.png",
        "EW_reservedratio_alpha20.png",
    ]
    visualizeReservedRatio(
        dfReservedRatio,
        modelCol,
        alphaCol,
        trainRatioCol,
        coverageCol,
        widthCol,
        show=False,
        savePath="figures/",
        fileNames=fileNames,
    )

    dfTargetCov = pd.read_csv(saveDir + "target-coverage/20241122T135247Z/data.csv")
    fileNames = ["EC_alphas.png", "EW_alphas.png"]
    visualizeTargetCov(
        dfTargetCov,
        modelCol,
        alphaCol,
        coverageCol,
        widthCol,
        show=False,
        savePath="figures/",
        fileNames=fileNames,
    )

    dfSamples = pd.read_csv(saveDir + "samples/20241125T105722Z/data.csv")
    fileNames = ["EC_samples.png", "EW_samples.png"]
    visualizeSamples(
        dfSamples,
        modelCol,
        samplesCol,
        coverageCol,
        widthCol,
        show=False,
        savePath="figures/",
        fileNames=fileNames,
    )

    dfBootstrap = pd.read_csv(saveDir + "bootstrap/20241123T082427Z/data.csv")
    fileNames = [
        "EC_bootstrap_alpha10.png",
        "EW_bootstrap_alpha10.png",
        "EC_bootstrap_alpha20.png",
        "EW_bootstrap_alpha20.png",
    ]
    visualizeBootstrap(
        dfBootstrap,
        modelCol,
        alphaCol,
        coverageCol,
        widthCol,
        show=False,
        savePath="figures/",
        fileNames=fileNames,
    )

    dfBootstrap = pd.read_csv(
        saveDir + "bootstrap/bootstrap_failed_20241114T170915Z/data.csv"
    )
    fileNames = [
        "bootstrap_alpha10_appendix.png",
        "EW_bootstrap_alpha10_appendix.png",
        "bootstrap_alpha20_appendix.png",
        "EW_bootstrap_alpha20_appendix.png",
    ]
    visualizeBootstrap(
        dfBootstrap,
        modelCol,
        alphaCol,
        coverageCol,
        widthCol,
        show=False,
        savePath="figures/",
        fileNames=fileNames,
    )


def visualizeReservedRatio(
    df,
    modelCol,
    alphaCol,
    trainRatioCol,
    coverageCol,
    widthCol,
    show=True,
    savePath=None,
    fileNames=None,
):
    titleWidth = "Empirical Width vs. Reserved Ratio"
    titleCoverage = "Empirical Coverage vs. Reserved Ratio"

    targetCovCol = "Target coverage $(1-\\alpha)$"
    df[targetCovCol] = 1 - df[alphaCol]

    customOrder = ["CQRF", "CQNN", "L-RF", "L-XGB"]
    modelType = pd.CategoricalDtype(categories=customOrder, ordered=True)
    df[modelCol] = df[modelCol].astype(modelType)
    df = df.sort_values(modelCol)

    for alpha in df[alphaCol].unique():
        dfFixedAlpha = df[df[alphaCol] == alpha]
        plotLineChartFromDf(
            dfFixedAlpha,
            trainRatioCol,
            coverageCol,
            modelCol,
            titleCoverage,
        )
        plotLineChartFromDf(
            dfFixedAlpha,
            trainRatioCol,
            widthCol,
            modelCol,
            titleWidth,
        )

    if show:
        plt.show()
    if savePath:
        saveFiguresWithDateTime(savePath, fileNames)


def visualizeTargetCov(
    df,
    modelCol,
    alphaCol,
    coverageCol,
    widthCol,
    show=True,
    savePath=None,
    fileNames=None,
):
    titleWidth = "Empirical Width vs. Target Coverage"
    titleCoverage = "Empirical Coverage vs. Target Coverage"

    targetCovCol = "Target coverage $(1-\\alpha)$"
    df[targetCovCol] = 1 - df[alphaCol]

    customOrder = ["QRF", "QNN", "CQRF", "CQNN", "L-RF", "L-XGB"]
    modelType = pd.CategoricalDtype(categories=customOrder, ordered=True)
    df[modelCol] = df[modelCol].astype(modelType)
    df = df.sort_values(modelCol)

    plotLineChartFromDf(
        df,
        targetCovCol,
        coverageCol,
        modelCol,
        titleCoverage,
        addDiagonal=True,
        diagLabel="Perfectly Calibrated",
    )
    plotLineChartFromDf(
        df,
        targetCovCol,
        widthCol,
        modelCol,
        titleWidth,
    )
    if show:
        plt.show()
    if savePath:
        saveFiguresWithDateTime(savePath, fileNames)


def visualizeSamples(
    df,
    modelCol,
    samplesCol,
    coverageCol,
    widthCol,
    show=True,
    savePath=None,
    fileNames=None,
):
    titleWidth = "Empirical Width vs. Sample Size"
    titleCoverage = "Empirical Coverage vs. Sample Size"

    customOrder = ["QRF", "QNN", "CQRF", "CQNN", "L-RF", "L-XGB"]
    modelType = pd.CategoricalDtype(categories=customOrder, ordered=True)
    df[modelCol] = df[modelCol].astype(modelType)
    df = df.sort_values(modelCol)

    plotLineChartFromDf(
        df,
        samplesCol,
        coverageCol,
        modelCol,
        f"{titleCoverage}",
    )

    plotLineChartFromDf(
        df,
        samplesCol,
        widthCol,
        modelCol,
        f"{titleWidth}",
    )
    if show:
        plt.show()
    if savePath:
        saveFiguresWithDateTime(savePath, fileNames)


def visualizeBootstrap(
    df,
    modelCol,
    alphaCol,
    coverageCol,
    widthCol,
    show=True,
    savePath=None,
    fileNames=None,
):
    titleWidth = "Bootstrap Empirical Width"
    titleCoverage = "Bootstrap Empirical Coverage"

    customOrder = ["QRF", "QNN", "CQRF", "CQNN", "L-RF", "L-XGB"]
    modelType = pd.CategoricalDtype(categories=customOrder, ordered=True)
    df[modelCol] = df[modelCol].astype(modelType)
    df = df.sort_values(modelCol)

    for alpha in df[alphaCol].unique():
        dfFixedAlpha = df[df[alphaCol] == alpha]
        plotBoxplotFromDf(
            dfFixedAlpha,
            f"{titleCoverage}: $\\alpha=$ {alpha:.2f}",
            modelCol,
            coverageCol,
            addHLine=True,
            valHLine=1 - alpha,
        )
        plotBoxplotFromDf(
            dfFixedAlpha, f"{titleWidth}: $\\alpha=$ {alpha:.2f}", modelCol, widthCol
        )
    if show:
        plt.show()
    if savePath:
        saveFiguresWithDateTime(savePath, fileNames)


main()
