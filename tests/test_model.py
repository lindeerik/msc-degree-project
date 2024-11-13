"""
Testing for Model class
"""

from sklearn.ensemble import RandomForestRegressor
import pytest
import pandas as pd
import numpy as np
from models.model import Model


@pytest.fixture
def sampleData():
    X = pd.DataFrame({"feature1": np.random.rand(10), "feature2": np.random.rand(10)})
    Y = pd.DataFrame({"target": np.random.rand(10)})
    return X, Y


@pytest.fixture
def fittedModel(sampleData):
    X, Y = sampleData
    model = RandomForestRegressor()
    m = Model(model, "Random Forest")
    m.fit(X, Y)
    return m


def test_gridSearchFit(sampleData):
    X, Y = sampleData
    param_grid = {"n_estimators": [10, 50]}
    model = RandomForestRegressor()
    m = Model(model, "Random Forest", paramGrid=param_grid)

    m.gridSearchFit(X, Y, folds=3)

    assert (
        m.getBestParams() is not None
    ), "Best parameters were not set after grid search."


def test_predict(sampleData, fittedModel):
    X, _ = sampleData
    model = fittedModel
    predictions = model.predict(X)
    assert predictions is not None, "Predictions should not be None."
    assert (
        len(predictions) == X.shape[0]
    ), "The number of predictions should match the number of samples."


def test_getRmse(sampleData, fittedModel):
    X, Y = sampleData
    model = fittedModel
    rmse = model.getRmse(X, Y)

    assert rmse >= 0, "Mean Squared Error should be non-negative."


def test_getMae(sampleData, fittedModel):
    X, Y = sampleData
    model = fittedModel
    mae = model.getMae(X, Y)

    assert mae >= 0, "Mean Squared Error should be non-negative."


def test_getR2(sampleData, fittedModel):
    X, Y = sampleData
    model = fittedModel
    r2 = model.getR2(X, Y)

    assert -float("inf") < r2 <= 1, "R2 score is not within valid bounds."


def test_getEmptyBestParams():
    model = RandomForestRegressor()
    m = Model(model, "Random Forest")

    bestParams = m.getBestParams()

    assert bestParams is None, "Best parameters should be None before grid search."
