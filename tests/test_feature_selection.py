"""
Testing for feature selection
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pytest

from models.model import Model

from features.sequential_feature_selection import BackwardFeatureSelector, ForwardFeatureSelector


@pytest.fixture
def sampleFloatDf():
    floatCols = ["feature1", "feature2"]
    catCols = []
    dependentCol = "target"

    df = pd.DataFrame(
        {
            floatCols[0]: np.arange(1, 101),
            floatCols[1]: np.random.rand(100),
            dependentCol: np.arange(1, 101),
        }
    )

    return df, floatCols, catCols, dependentCol


@pytest.fixture
def sampleCatDf():
    floatCols = []
    catCols = ["feature1", "feature2"]
    dependentCol = "target"

    df = pd.DataFrame(
        {
            catCols[0]: ["a"] * 50 + ["b"] * 50,
            catCols[1]: ["a", "b"] * 50,
            dependentCol: [1] * 50 + [10] * 50,
        }
    )

    return df, floatCols, catCols, dependentCol


@pytest.fixture
def sampleMixedDfV1():
    floatCols = ["feature1"]
    catCols = ["feature2"]
    dependentCol = "target"

    df = pd.DataFrame(
        {
            floatCols[0]: np.random.rand(100),
            catCols[0]: ["a"] * 50 + ["b"] * 50,
            dependentCol: [1] * 50 + [10] * 50,
        }
    )

    return df, floatCols, catCols, dependentCol


@pytest.fixture
def sampleMixedDfV2():
    floatCols = ["feature1"]
    catCols = ["feature2"]
    dependentCol = "target"

    df = pd.DataFrame(
        {
            floatCols[0]: np.arange(1, 101),
            catCols[0]: ["a", "b"] * 50,
            dependentCol: np.arange(1, 101),
        }
    )

    return df, floatCols, catCols, dependentCol


def test_backwardFeatureSelectionFloat(sampleFloatDf):
    df, floatCols, catCols, dependentCol = sampleFloatDf

    model = Model(LinearRegression(), "Linear Regression")
    selector = BackwardFeatureSelector(model)
    bestFloatCols, _ = selector.getBestFeatures(df, floatCols, catCols, dependentCol)

    assert bestFloatCols == [floatCols[0]]


def test_forwardFeatureSelectionFloat(sampleFloatDf):
    df, floatCols, catCols, dependentCol = sampleFloatDf

    model = Model(LinearRegression(), "Linear Regression")
    selector = ForwardFeatureSelector(model)
    bestFloatCols, _ = selector.getBestFeatures(df, floatCols, catCols, dependentCol)

    assert bestFloatCols == [floatCols[0]]


def test_backwardFeatureSelectionCat(sampleCatDf):
    df, floatCols, catCols, dependentCol = sampleCatDf

    model = Model(LinearRegression(), "Linear Regression")
    selector = BackwardFeatureSelector(model)
    _, bestCatCols = selector.getBestFeatures(df, floatCols, catCols, dependentCol)

    assert bestCatCols == [catCols[0]]


def test_forwardFeatureSelectionCat(sampleCatDf):
    df, floatCols, catCols, dependentCol = sampleCatDf

    model = Model(LinearRegression(), "Linear Regression")
    selector = ForwardFeatureSelector(model)
    _, bestCatCols = selector.getBestFeatures(df, floatCols, catCols, dependentCol)

    assert bestCatCols == [catCols[0]]


def test_backwardFeatureSelectionMixedV1(sampleMixedDfV1):
    df, floatCols, catCols, dependentCol = sampleMixedDfV1

    model = Model(LinearRegression(), "Linear Regression")
    selector = BackwardFeatureSelector(model)
    bestFloatCols, bestCatCols = selector.getBestFeatures(
        df, floatCols, catCols, dependentCol
    )

    assert bestFloatCols == []
    assert bestCatCols == [catCols[0]]


def test_forwardFeatureSelectionMixedV1(sampleMixedDfV1):
    df, floatCols, catCols, dependentCol = sampleMixedDfV1

    model = Model(LinearRegression(), "Linear Regression")
    selector = ForwardFeatureSelector(model)
    bestFloatCols, bestCatCols = selector.getBestFeatures(
        df, floatCols, catCols, dependentCol
    )

    assert bestFloatCols == []
    assert bestCatCols == [catCols[0]]


def test_backwardFeatureSelectionMixedV2(sampleMixedDfV2):
    df, floatCols, catCols, dependentCol = sampleMixedDfV2

    model = Model(LinearRegression(), "Linear Regression")
    selector = BackwardFeatureSelector(model)
    bestFloatCols, bestCatCols = selector.getBestFeatures(
        df, floatCols, catCols, dependentCol
    )

    assert bestFloatCols == [floatCols[0]]
    assert bestCatCols == []


def test_forwardFeatureSelectionMixedV2(sampleMixedDfV2):
    df, floatCols, catCols, dependentCol = sampleMixedDfV2

    model = Model(LinearRegression(), "Linear Regression")
    selector = ForwardFeatureSelector(model)
    bestFloatCols, bestCatCols = selector.getBestFeatures(
        df, floatCols, catCols, dependentCol
    )

    assert bestFloatCols == [floatCols[0]]
    assert bestCatCols == []
