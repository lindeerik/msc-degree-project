"""
Generating plots of point estimation trials. May be shown directly or saved to directory. 
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from visualization.plots import plotLineChartFromDf
from visualization.save_figures import saveFiguresWithDateTime


def main():
    saveDir = "experiments/point-estimation/samples/20241118T160153Z/"
    modelCol = "Model"
    samplesCol = "Number of Samples"
    trainRmseCol = "Training RMSE"
    testRmseCol = "Test RMSE"
    trainMaeCol = "Training MAE"
    testMaeCol = "Test MAE"

    df = pd.read_csv(saveDir + "data.csv")
    fileNames = ["rmse_samples.png", "mae_samples.png"]
    visualizeTrials(
        df,
        modelCol,
        samplesCol,
        trainRmseCol,
        testRmseCol,
        trainMaeCol,
        testMaeCol,
        show=False,
        savePath="figures/",
        fileNames=fileNames,
    )


def visualizeTrials(
    df,
    modelCol,
    samplesCol,
    trainRmseCol,
    testRmseCol,
    trainMaeCol,
    testMaeCol,
    show=True,
    savePath=None,
    fileNames=None,
):
    rmseCol = "RMSE"
    maeCol = "MAE"
    trainTestCol = "Train or Test"
    titleRmse = "Root Mean Squared Error (RMSE) vs. Sample Size"
    titleMae = "Mean Absolute Error (MAE) vs. Sample Size"

    colors = sns.color_palette("Dark2")
    palette = [colors[0], colors[0], colors[1], colors[1], colors[2], colors[2]]
    dashes = [(1, 0), (4, 2), (1, 0), (4, 2), (1, 0), (4, 2)]

    dfRmse = pd.melt(
        df,
        id_vars=[samplesCol, modelCol],
        value_vars=[trainRmseCol, testRmseCol],
        var_name=trainTestCol,
        value_name=rmseCol,
    )

    dfRmse[trainTestCol] = dfRmse[trainTestCol].replace(
        {
            trainRmseCol: "Train",
            testRmseCol: "Test",
        }
    )
    dfRmse[modelCol] = dfRmse[modelCol] + " " + dfRmse[trainTestCol]
    dfRmse = dfRmse.drop(columns=trainTestCol)
    plotLineChartFromDf(
        dfRmse,
        samplesCol,
        rmseCol,
        modelCol,
        titleRmse,
        palette=palette,
        dashes=dashes,
    )

    dfMae = pd.melt(
        df,
        id_vars=[samplesCol, modelCol],
        value_vars=[trainMaeCol, testMaeCol],
        var_name=trainTestCol,
        value_name=maeCol,
    )

    dfMae[trainTestCol] = dfMae[trainTestCol].replace(
        {
            trainMaeCol: "Train",
            testMaeCol: "Test",
        }
    )
    dfMae[modelCol] = dfMae[modelCol] + " " + dfMae[trainTestCol]
    dfMae = dfMae.drop(columns=trainTestCol)
    plotLineChartFromDf(
        dfMae,
        samplesCol,
        maeCol,
        modelCol,
        titleMae,
        palette=palette,
        dashes=dashes,
    )

    if show:
        plt.show()
    if savePath:
        saveFiguresWithDateTime(savePath, fileNames)


main()
