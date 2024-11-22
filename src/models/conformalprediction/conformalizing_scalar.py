import numpy as np
from models.conformalprediction.quantile_regression import getConformalQuantileScore


class ConformalizingScalarPredictor:
    def __init__(self, baseModel, errorModel, alpha, name="", minVal=None, maxVal=None):
        self._baseModel = baseModel
        self._errorModel = errorModel
        self._conformalScoreFunc = self.getConformalScoreFunc()
        self._alpha = alpha
        self._conformalQuantileScore = None
        self._name = name
        self._minVal = minVal
        self._maxVal = maxVal

    def predict(self, X):
        yPred = self._baseModel.predict(X)
        yPredError = self._errorModel.predict(X)
        lowerBounds = yPred - self._conformalQuantileScore * yPredError
        upperbounds = yPred + self._conformalQuantileScore * yPredError
        if self._minVal is not None:
            lowerBounds = np.maximum(self._minVal, lowerBounds)
        if self._maxVal is not None:
            upperbounds = np.minimum(self._maxVal, upperbounds)
        return [lowerBounds, upperbounds]

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
        if self._name != "":
            return self._name
        if self._baseModel.getName() != "":
            return "Conformalized Scalar " + self._baseModel.getName()
        return ""

    def getMetadata(self):
        metadata = {
            "alpha": self._alpha,
            "base_model": self._baseModel.getMetadata(),
            "error_model": self._errorModel.getMetadata(),
        }
        return metadata

    def saveModel(self, modelPath):
        base, ext = modelPath.rsplit(".", 1)
        self._baseModel.saveModel(f"{base}_base.{ext}")
        self._errorModel.saveModel(f"{base}_error.{ext}")
