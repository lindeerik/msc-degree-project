import pandas as pd
from sklearn.linear_model import LinearRegression
import pytest

from data.data_loader import *
from models.model import Model
from features.feature_selection import *

@pytest.fixture
def sample_df():
    floatCols = ["feature1", "feature2"]
    catCols = []
    dependentCol = "target"

    df = pd.DataFrame({
        floatCols[0]: [1,2,3,4,5,6,7,8,9,10],
        floatCols[1]: np.random.rand(10),
        dependentCol: [1,2,3,4,5,6,7,8,9,10]
    })

    return df, floatCols, catCols, dependentCol

def test_getBestFeatures(sample_df):
    df, floatCols, catCols, dependentCol = sample_df

    model = Model(LinearRegression(), "Linear Regression")
    bestFloatCols = getBestFeatures(df, floatCols, catCols, dependentCol, model)
    
    assert bestFloatCols == [floatCols[0]]

def test_dropWorseFeature(sample_df):
    df, floatCols, catCols, dependentCol = sample_df

    model = Model(LinearRegression(), "Linear Regression")
    dropFloatCol, _ = dropWorseFeature(df, floatCols, catCols, dependentCol, model)

    assert dropFloatCol == floatCols[1]

