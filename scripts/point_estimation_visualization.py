"""
Generating plots of point estimation trials. May be shown directly or saved to directory. 
"""

import pandas as pd
import matplotlib.pyplot as plt

from visualization.plots import plotBoxplotFromDf, plotLineChartFromDf
from visualization.save_figures import saveFiguresWithDateTime


def main():
    saveDir = "data/results/point-estimation/"
    modelCol = "Model"
    trainRatioCol = "Train ratio"
    samplesCol = "Number of Samples"
    r2Col = "Coefficient of Determination (R2)"
    mseCol = "Mean Squared Error"
    df = pd.read_csv(saveDir + "20241025T092710Z.csv")

    visualizeTrials(
        df,
        modelCol,
        trainRatioCol,
        samplesCol,
        r2Col,
        mseCol,
        show=False,
        savePath="figures/",
    )


def visualizeTrials(
    df,
    modelCol,
    trainRatioCol,
    samplesCol,
    r2Col,
    mseCol,
    show=True,
    savePath=None,
):
    titleR2 = "Determination of Coefficient ($R^2$)"
    titleMse = "Mean Squared Error"

    # Fixed train ratio = 0.8
    dfFixedTrainRatio = df[df[trainRatioCol] == 0.8]

    plotBoxplotFromDf(
        dfFixedTrainRatio,
        f"{titleR2}: 80% Train Ratio",
        modelCol,
        r2Col,
    )

    plotBoxplotFromDf(
        dfFixedTrainRatio,
        f"{titleMse}: 80% Train Ratio",
        modelCol,
        mseCol,
    )

    plotLineChartFromDf(
        df,
        samplesCol,
        r2Col,
        modelCol,
        f"{titleR2} vs. Number of Samples",
    )

    if show:
        plt.show()
    if savePath:
        saveFiguresWithDateTime(savePath)


main()
