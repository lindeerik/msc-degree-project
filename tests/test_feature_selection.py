"""
Testing for feature selection
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pytest

from models.model import Model
from features.feature_selection import getBestFeatures, dropWorseFeature


@pytest.fixture
def sampleDf():
    floatCols = ["feature1", "feature2"]
    catCols = []
    dependentCol = "target"

    df = pd.DataFrame(
        {
            floatCols[0]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            floatCols[1]: np.random.rand(10),
            dependentCol: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    return df, floatCols, catCols, dependentCol


def test_getBestFeatures(sampleDf):
    df, floatCols, catCols, dependentCol = sampleDf

    model = Model(LinearRegression(), "Linear Regression")
    bestFloatCols = getBestFeatures(df, floatCols, catCols, dependentCol, model)

    assert bestFloatCols == [floatCols[0]]


def test_dropWorseFeature(sampleDf):
    df, floatCols, catCols, dependentCol = sampleDf

    model = Model(LinearRegression(), "Linear Regression")
    dropFloatCol, _ = dropWorseFeature(df, floatCols, catCols, dependentCol, model)

    assert dropFloatCol == floatCols[1]
