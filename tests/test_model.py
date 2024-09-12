from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pytest
import pandas as pd
import numpy as np
from models.model import Model

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10)
    })
    Y = pd.DataFrame({
        'target': np.random.rand(10)
    })
    return X, Y

@pytest.fixture
def fitted_model(sample_data):
    X, Y = sample_data
    model = RandomForestRegressor()
    m = Model(model, "Random Forest")
    m.fit(X, Y) 
    return m

def test_gridSearchFit(sample_data):
    X, Y = sample_data
    param_grid = {'n_estimators': [10, 50]}
    model = RandomForestRegressor()
    m = Model(model, "Random Forest", paramGrid=param_grid)
    
    m.gridSearchFit(X, Y, folds=3)
    
    assert m.getBestParams() is not None, "Best parameters were not set after grid search."
    assert len(m._Model__model.estimators_) > 0, "The model does not have any trees after fitting."

def test_predict(sample_data, fitted_model):
    X, _ = sample_data
    model = fitted_model
    predictions = model.predict(X)
    assert predictions is not None, "Predictions should not be None."
    assert len(predictions) == X.shape[0], "The number of predictions should match the number of samples."

def test_getMse(sample_data, fitted_model):
    X, Y = sample_data
    model = fitted_model
    mse = model.getMse(X, Y)
    
    assert mse >= 0, "Mean Squared Error should be non-negative."

def test_getR2(sample_data, fitted_model):
    X, Y = sample_data
    model = fitted_model
    r2 = model.getR2(X, Y)
    
    assert -float('inf') < r2 <= 1, "R2 score is not within valid bounds."

def test_getEmptyBestParams():
    model = RandomForestRegressor()
    m = Model(model, "Random Forest")
    
    bestParams = m.getBestParams()
    
    assert bestParams is None, "Best parameters should be None before grid search."
