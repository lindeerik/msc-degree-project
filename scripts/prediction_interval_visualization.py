"""
Generating plots of prediction intervals. May be shown directly or saved to directory. 
"""

import pandas as pd
import matplotlib.pyplot as plt

from visualization.plots import plotBoxplotFromDf, plotLineChartFromDf
from visualization.save_figures import saveFiguresWithDateTime


def main():
    saveDir = "data/results/conformal-prediction/"
    modelCol = "Model"
    alphaCol = "Alpha"
    trainRatioCol = "Train ratio"
    coverageCol = "Empirical coverage"
    widthCol = "Empirical width"
    df = pd.read_csv(saveDir + "20241017T162004Z.csv")

    visualizeTrials(
        df,
        modelCol,
        alphaCol,
        trainRatioCol,
        coverageCol,
        widthCol,
        show=False,
        savePath="figures/",
    )


def visualizeTrials(
    df,
    modelCol,
    alphaCol,
    trainRatioCol,
    coverageCol,
    widthCol,
    show=True,
    savePath=None,
):
    titleWidth = "Prediction Interval Width"
    titleCoverage = "Prediction Interval Coverage"

    targetCovCol = "Target coverage $(1-\\alpha)$"
    df[targetCovCol] = 1 - df[alphaCol]

    # Fixed train ratio = 0.8
    dfFixedTrainRatio = df[df[trainRatioCol] == 0.8]

    for alpha in df[alphaCol].unique():
        dfRefined = dfFixedTrainRatio[dfFixedTrainRatio[alphaCol] == alpha]
        plotBoxplotFromDf(
            dfRefined,
            f"{titleCoverage}: $\\alpha=$ {alpha:.2f}",
            modelCol,
            coverageCol,
            addHLine=True,
            valHLine=1 - alpha,
        )
        plotBoxplotFromDf(
            dfRefined, f"{titleWidth}: $\\alpha=$ {alpha:.2f}", modelCol, widthCol
        )

    plotLineChartFromDf(
        dfFixedTrainRatio,
        targetCovCol,
        coverageCol,
        modelCol,
        f"{titleCoverage} vs. Target Coverage",
        addDiagonal=True,
        diagLabel="Perfectly Calibrated",
    )
    plotLineChartFromDf(
        dfFixedTrainRatio,
        targetCovCol,
        widthCol,
        modelCol,
        f"{titleWidth} vs. Target Coverage",
    )

    # Fixed alpha = 0.1
    dfFixedAlpha = df[df[alphaCol] == 0.1]
    plotLineChartFromDf(
        dfFixedAlpha,
        trainRatioCol,
        coverageCol,
        modelCol,
        f"{titleCoverage} vs. Training Ratio",
    )
    plotLineChartFromDf(
        dfFixedAlpha,
        trainRatioCol,
        widthCol,
        modelCol,
        f"{titleWidth} vs. Training Ratio",
    )
    if show:
        plt.show()
    if savePath:
        saveFiguresWithDateTime(savePath)


main()
