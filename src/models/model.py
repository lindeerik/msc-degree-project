"""
Model wrapper for scikit-learn, xgboost, and skorch
"""

import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


class Model:
    def __init__(self, model, name, paramGrid=None, scorer="neg_mean_squared_error"):
        self.__model = model
        self.__name = name
        self.__paramGrid = paramGrid
        self.__bestParams = None
        self.__scorer = scorer

    def predict(self, X):
        return self.__model.predict(X)

    def fit(self, X, Y):
        # if param grid available, take first combination of params
        if self.__paramGrid:
            params = {key: value[0] for key, value in self.__paramGrid.items()}
            self.__model.set_params(**params)
        self.__model.fit(X, Y.values.ravel())

    def gridSearchFit(self, X, Y, folds):
        if self.__paramGrid is None:
            print(
                "Warning: no parameter grid provided. Defaulting to normal model fit."
            )
            self.__model.fit(X, Y.values.ravel())
        elif getCombinationsOfGridParameters(self.__paramGrid) == 1:
            params = {key: value[0] for key, value in self.__paramGrid.items()}
            self.__model.set_params(**params)
            self.__model.fit(X, Y.values.ravel())
        else:
            grid_search = GridSearchCV(
                estimator=self.__model,
                param_grid=self.__paramGrid,
                cv=folds,
                n_jobs=-1,
                scoring=self.__scorer,
                verbose=1,
            )
            grid_search.fit(X, Y.values.ravel())
            self.__model = grid_search.best_estimator_
            self.__bestParams = grid_search.best_params_

    def getMse(self, X, Y):
        yPred = self.predict(X)
        return mean_squared_error(Y.values, yPred)

    def getR2(self, X, Y):
        yPred = self.predict(X)
        return r2_score(Y.values, yPred)

    def getBestParams(self):
        return self.__bestParams

    def getName(self):
        return self.__name

    def getModel(self):
        return self.__model

    def getMetadata(self):
        metadata = {
            "name": self.__name,
            "type": str(type(self.__model)),
            "parameter_grid": self.__paramGrid,
            "parameters": str(self.__model.get_params()),
        }
        return metadata

    def saveModel(self, modelPath):
        with open(modelPath, "wb") as f:
            pickle.dump(self.__model, f)


def getCombinationsOfGridParameters(paramGrid):
    numCombinations = 1
    for _, value in paramGrid.items():
        numCombinations *= len(value)
    return numCombinations
