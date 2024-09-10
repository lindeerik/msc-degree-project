import numpy as np

def getConformalQuantileScore(conformalScoreFunc, dataX, dataY, quantile):
    conformalScores = conformalScoreFunc(dataX, dataY)
    q = np.quantile(conformalScores, quantile)
    return q 

def pinballConformalScoreFunc(lowerBoundModel, upperBoundModel, X, Y):
    conformalScore = np.maximum(lowerBoundModel.predict(X)-Y, Y-upperBoundModel.predict(X))

    return conformalScore

class QuantileRegressor:
    def __init__(self, lowerBoundModel, upperBoundModel):
        self.lowerBoundModel = lowerBoundModel
        self.upperBoundModel = upperBoundModel

    def fit(self, X, Y, folds=5):
        self.lowerBoundModel.gridSearchFit(X, Y, folds)
        self.upperBoundModel.gridSearchFit(X, Y, folds)

    def predict(self, X):
        lowerBounds = self.lowerBoundModel.predict(X)
        upperBounds = self.upperBoundModel.predict(X)
        return [lowerBounds, upperBounds]
    
    def getCoverageRatio(self, X, Y):
        return np.mean((self.predict(X)[0] <= Y) & (Y <= self.predict(X)[1]))

class ConformalizedQuantileRegressor:
    def __init__(self, quantileRegressor, conformalScoreFunc, alpha):
        self.quantileRegressor = quantileRegressor
        self.conformalScoreFunc = conformalScoreFunc
        self.alpha = alpha
        self.conformalQuantileScore = None

    def fit(self, xTrain, yTrain, xVal, yVal, folds = 5):
        self.quantileRegressor.fit(xTrain, yTrain, folds)
        n = xTrain.shape[0]
        conformalQuantile = np.ceil((n+1)*(1-self.alpha))/n
        self.conformalQuantileScore = getConformalQuantileScore(self.conformalScoreFunc, xVal, yVal, conformalQuantile)

    def predict(self, X):
        intervals = self.quantileRegressor.predict(X)
        return [intervals[0] - self.conformalQuantileScore, intervals[1] + self.conformalQuantileScore]
    
    def getCoverageRatio(self, X, Y):
        return np.mean((self.predict(X)[0] <= Y) & (Y <= self.predict(X)[1]))
    
    def getAverageIntervalWidth(self, X):
        return np.mean(self.predict(X)[1] - self.predict(X)[0])
