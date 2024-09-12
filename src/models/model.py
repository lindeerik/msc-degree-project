from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

class Model:
    def __init__(self, model, name, paramGrid = None, scorer = 'neg_mean_squared_error'):
        self.__model = model
        self.__name = name
        self.__paramGrid = paramGrid
        self.__bestParams = None
        self.__scorer = scorer

    def predict(self, X):
        return self.__model.predict(X.values)

    def fit(self, X, Y):
        self.__model.fit(X.values, Y.values.ravel())

    def gridSearchFit(self, X, Y,folds):
        if self.__paramGrid is None:
            print("Warning: no parameter grid provided. Defaulting to normal model fit.")
            self.fit(X, Y)
            return
        grid_search = GridSearchCV(estimator=self.__model, param_grid=self.__paramGrid, cv=folds, n_jobs=-1, scoring=self.__scorer, verbose=1)
        grid_search.fit(X.values, Y.values.ravel())
        self.__model= grid_search.best_estimator_
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