import numpy as np
from models.conformalprediction.quantile_regression import getConformalQuantileScore


class ConformalizingScalarPredictor:
    def __init__(self, baseModel, errorModel, alpha):
        self._baseModel = baseModel
        self._errorModel = errorModel
        self._conformalScoreFunc = self.getConformalScoreFunc()
        self._alpha = alpha
        self._conformalQuantileScore = None

    def predict(self, X):
        yPred = self._baseModel.predict(X)
        yPredError = self._errorModel.predict(X)
        return [
            yPred - self._conformalQuantileScore * yPredError,
            yPred + self._conformalQuantileScore * yPredError,
        ]

    def fit(self, xTrain, yTrain, xVal, yVal, folds=5):
        self._baseModel.gridSearchFit(xTrain, yTrain, folds)
        # assuming we want to train residual prediction
        residuals = np.abs(self._baseModel.predict(xTrain) - yTrain)
        self._errorModel.gridSearchFit(xTrain, residuals, folds)
        n = xTrain.shape[0]
        conformalQuantile = np.ceil((n + 1) * (1 - self._alpha)) / n
        self._conformalQuantileScore = getConformalQuantileScore(
            self._conformalScoreFunc, xVal, yVal, conformalQuantile
        )

    def getConformalScoreFunc(self):
        conformalScoreFunc = lambda X, Y: np.abs(
            self._baseModel.predict(X) - Y
        ) / self._errorModel.predict(X)
        return conformalScoreFunc

    def getCoverageRatio(self, X, Y):
        yPred = self.predict(X)
        return np.mean((yPred[0] <= Y) & (Y <= yPred[1]))

    def getAverageIntervalWidth(self, X):
        yPred = self.predict(X)
        return np.mean(yPred[1] - yPred[0])

    def getName(self):
        if self._baseModel.getName() == "":
            return ""
        return "Conformalized Scalar " + self._baseModel.getName()
