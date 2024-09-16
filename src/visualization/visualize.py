"""
Visualization for prediction errors
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def plotModelsErrors(errors, models):
    for i, model in enumerate(models):
        plotModelErrors(errors[i], model.getName())
    plt.show()


def plotModelErrors(errors, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(
        errors, kde=False, bins=50, color="skyblue", stat="density", label="Test Errors"
    )

    # Fit a normal distribution to the data
    mu, std = norm.fit(errors)

    # Plot the best-fit normal distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=2, label=f"Normal fit: μ={mu:.2f}, σ={std:.2f}")

    plt.xlabel("Test Error")
    plt.ylabel("Density")
    plt.title("Histogram of Test Errors with Best-Fit Normal Curve: " + title)
    plt.legend()
