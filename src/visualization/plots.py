"""
Plot visualizations for histograms, boxplots and line charts
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def plotHistogram(
    values,
    bins,
    xLabel,
    yLabel,
    title,
    figSize=(10, 6),
    show=False,
    fitNormalDist=False,
):
    plt.figure(figsize=figSize)
    sns.histplot(
        values, kde=False, bins=bins, color="skyblue", stat="density", label=xLabel
    )

    if fitNormalDist:
        mu, std = norm.fit(values)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, "k", linewidth=2, label=f"Normal fit: μ={mu:.2f}, σ={std:.2f}")
        plt.legend()

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    if show:
        plt.show()


def plotBoxplotFromDf(
    df, title, xLabel, yLabel, figSize=(8, 6), show=False, addHLine=False, valHLine=None
):
    sns.set_theme(style="whitegrid", palette="pastel")
    plt.figure(figsize=figSize)
    sns.boxplot(
        x=xLabel, y=yLabel, data=df, linewidth=1.5, order=sorted(df[xLabel].unique())
    )

    if addHLine and valHLine is not None:
        plt.axhline(y=valHLine, color="red", linestyle="--", linewidth=2)

    sns.despine(left=True)
    plt.title(title, fontsize=14)
    plt.xlabel(xLabel, fontsize=12)
    plt.ylabel(yLabel, fontsize=12)

    if show:
        plt.show()


# pylint: disable-msg=too-many-arguments
def plotLineChartFromDf(
    df,
    xLabel,
    yLabel,
    groupLabel,
    title,
    figSize=(8, 6),
    show=False,
    addDiagonal=False,
    diagLabel=None,
    palette=None,
    dashes=True,
):
    df = df.groupby([xLabel, groupLabel]).mean().reset_index()
    sns.set_theme(style="whitegrid", palette="pastel")
    plt.subplots(figsize=figSize)

    sns.lineplot(
        x=xLabel,
        y=yLabel,
        hue=groupLabel,
        data=df,
        style=groupLabel,
        dashes=dashes,
        palette=palette,
        linewidth=1.5,
        markers=True,
        errorbar=None,
    )

    if addDiagonal:
        minVal = min(df[xLabel].min(), df[yLabel].min())
        maxVal = max(df[xLabel].max(), df[yLabel].max())
        plt.plot([minVal, maxVal], [minVal, maxVal], "r", label=diagLabel, linewidth=1)

    plt.title(title, fontsize=14)
    plt.xlabel(xLabel, fontsize=12)
    plt.ylabel(yLabel, fontsize=12)
    sns.despine(left=True)
    plt.legend(title=groupLabel, title_fontsize="13", fontsize="11")

    if show:
        plt.show()


def plotModelsErrors(errors, models):
    xLabel = "Test Errors"
    yLabel = "Density"
    titlePrefix = "Histogram of Test Errors with Best-Fit Normal Curve: "
    for i, model in enumerate(models):
        # plotModelErrors(errors[i], model.getName())
        plotHistogram(
            errors[i],
            50,
            xLabel,
            yLabel,
            titlePrefix + model.getName(),
            fitNormalDist=True,
        )
    plt.show()
