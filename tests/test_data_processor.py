"""
Testing for data processing
"""

import pytest
import numpy as np
import pandas as pd

from data.data_processing import processData, getDataProcessor


@pytest.fixture
def sampleData():
    return pd.DataFrame(
        {
            "num_col1": [1.0, 2.0, 3.0, 4.0],
            "num_col2": [4.0, 5.0, 6.0, 7.0],
            "cat_col1": ["A", "B", "A", "B"],
            "cat_col2": ["X", "Y", "X", "Y"],
            "dependent_col": [0, 1, 0, 1],
        }
    )


def test_processDataWithoutProcessor(sampleData):
    floatCols = ["num_col1", "num_col2"]
    catCols = ["cat_col1", "cat_col2"]
    dependentCol = "dependent_col"
    y = sampleData[dependentCol]

    dataX, dataY = processData(sampleData, floatCols, catCols, dependentCol)

    assert y.shape == dataY.shape, "The shapes of y data do not match"
    assert np.allclose(y, dataY), "The y data values have changed"
    assert dataX.shape[0] == sampleData.shape[0], "Number of rows do not match"
    assert dataX.shape[1] > len(floatCols) + len(
        catCols
    ), "Number of columns is too small"


def test_processDataWithProcessor(sampleData):
    floatCols = ["num_col1", "num_col2"]
    catCols = ["cat_col1", "cat_col2"]
    dependentCol = "dependent_col"
    y = sampleData[dependentCol]

    processor = getDataProcessor(floatCols, catCols)
    dataX, dataY = processData(sampleData, floatCols, catCols, dependentCol, processor)

    assert y.shape == dataY.shape, "The shapes of y data do not match"
    assert np.allclose(y, dataY), "The y data values have changed"
    assert dataX.shape[0] == sampleData.shape[0], "Number of rows do not match"
    assert dataX.shape[1] > len(floatCols) + len(catCols), "Too few columns"


def test_processDataNoColumns(sampleData):
    floatCols = []
    catCols = []
    dependentCol = "dependent_col"
    y = sampleData[dependentCol]

    _, dataY = processData(sampleData, floatCols, catCols, dependentCol)

    assert dataY.shape[0] == sampleData.shape[0]
    assert np.allclose(y, dataY), "The y data values have changed"
