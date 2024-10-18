"""
Conformalized quantile regression model classes
"""

from abc import ABC, abstractmethod
import numpy as np


def getConformalQuantileScore(conformalScoreFunc, dataX, dataY, quantile):
    conformalScores = conformalScoreFunc(dataX, dataY)
    return np.quantile(conformalScores, quantile)


class QuantileRegressor(ABC):
    def __init__(self, models, alpha, name=""):
        self._models = models
        self._alpha = alpha
        self._name = name

    def fit(self, X, Y, folds=5):
        for model in self._models:
            model.gridSearchFit(X, Y, folds)

    @abstractmethod
    def predict(self, X):
        pass

    def getCoverageRatio(self, X, Y):
        yPred = self.predict(X)
        return np.mean((yPred[0] <= Y) & (Y <= yPred[1]))

    def getAverageIntervalWidth(self, X):
        yPred = self.predict(X)
        return np.mean(yPred[1] - yPred[0])

    def getAlpha(self):
        return self._alpha

    def getName(self):
        return self._name


class QuantileRegressorNeuralNet(QuantileRegressor):
    def predict(self, X):
        lowerBounds = self._models[0].predict(X)
        upperBounds = self._models[1].predict(X)
        return [lowerBounds, upperBounds]


class QuantileRegressorRandomForest(QuantileRegressor):
    def predict(self, X):
        return self._models[0].predict(X)


class ConformalizedQuantileRegressor:
    def __init__(self, quantileRegressor, scaling=None, name = ""):
        self._quantileRegressor = quantileRegressor
        self._conformalScoreFunc = self.conformalScore
        self._alpha = quantileRegressor.getAlpha()
        self._conformalQuantileScore = None
        self._scalingFunc = self.getScalingFunc(scaling)
        self._name = name

    def fit(self, xTrain, yTrain, xVal, yVal, folds=5):
        self._quantileRegressor.fit(xTrain, yTrain, folds)
        n = xTrain.shape[0]
        conformalQuantile = np.ceil((n + 1) * (1 - self._alpha)) / n
        self._conformalQuantileScore = getConformalQuantileScore(
            self._conformalScoreFunc, xVal, yVal, conformalQuantile
        )

    def predict(self, X):
        intervals = self._quantileRegressor.predict(X)
        scaling = self._scalingFunc(X)
        return [
            intervals[0] - self._conformalQuantileScore * scaling,
            intervals[1] + self._conformalQuantileScore * scaling,
        ]

    def getQuantileRegressor(self):
        return self._quantileRegressor

    def conformalScore(self, X, Y):
        yPred = self._quantileRegressor.predict(X)
        scaling = self._scalingFunc(X)
        return np.maximum((yPred[0] - Y) / scaling, (Y - yPred[1]) / scaling)

    def quantilePredIntervalLength(self, X):
        yPred = self._quantileRegressor.predict(X)
        return yPred[1] - yPred[0]

    def identityScaling(self, _):
        return 1

    def getScalingFunc(self, scaling):
        if scaling == "interval_width_scaling":
            return self.quantilePredIntervalLength
        return self.identityScaling

    def getCoverageRatio(self, X, Y):
        yPred = self.predict(X)
        return np.mean((yPred[0] <= Y) & (Y <= yPred[1]))

    def getAverageIntervalWidth(self, X):
        yPred = self.predict(X)
        return np.mean(yPred[1] - yPred[0])

    def getName(self):
        if self._name != "":
            return self._name
        if self._quantileRegressor.getName() != "":
            return "Conformalized " + self._quantileRegressor.getName()
        return ""
