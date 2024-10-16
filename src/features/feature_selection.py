"""
Functions for drop-out feature selection
"""

from abc import ABC, abstractmethod
import numpy as np
from data.data_processing import processData, getDataProcessor, trainTestSplit


class SequentialFeatureSelector(ABC):
    def __init__(self, model):
        self.model = model
        self.df = None
        self.floatCols = None
        self.catCols = None
        self.dependentCol = None
        self.trainSize = None
        self.isBinaryEncoding = None

    def getBestFeatures(
        self,
        df,
        floatCols,
        catCols,
        dependentCol,
        trainSize=0.7,
        isBinaryEncoding=True,
        eps=1e-3,
    ):
        self.df = df
        self.floatCols = floatCols
        self.catCols = catCols
        self.dependentCol = dependentCol
        self.trainSize = trainSize
        self.isBinaryEncoding = isBinaryEncoding
        prevScore = self.getBaselineScore()
        scoreDiff = 1.0
        # iterate until score does not increase
        availableFloatCols = floatCols.copy()
        availableCatCols = catCols.copy()
        while scoreDiff >= -eps and (
            len(availableFloatCols) + len(availableCatCols) > 1
        ):
            floatCol, catCol, score = self.getNextFeature(
                availableFloatCols,
                availableCatCols,
            )
            scoreDiff = score - prevScore
            prevScore = score
            if scoreDiff >= 0:
                if floatCol is not None:
                    availableFloatCols.remove(floatCol)
                else:
                    availableCatCols.remove(catCol)
        return self.getSelectedCols(availableFloatCols, availableCatCols)

    def getNextFeature(self, availablefFloatCols, availableCatCols):
        # iterate over float features
        scoresFloat = []
        for floatCol in availablefFloatCols:
            xTrain, xTest, yTrain, yTest = self.getModifiedData(
                availablefFloatCols, availableCatCols, selectedFloatCol=floatCol
            )
            # binary encoding may remove columns with only one value
            if xTrain.shape[1] == 0:
                scoresFloat.append(-np.inf)
            else:
                score = self.trainModelAndGetScore(xTrain, xTest, yTrain, yTest)
                scoresFloat.append(score)
        idxFloat = np.argmax(scoresFloat) if len(scoresFloat) > 0 else None
        # iterate over categorical features
        scoresCat = []
        for catCol in availableCatCols:
            xTrain, xTest, yTrain, yTest = self.getModifiedData(
                availablefFloatCols, availableCatCols, selectedCatCol=catCol
            )
            # binary encoding may remove columns with only one value
            if xTrain.shape[1] == 0:
                scoresCat.append(-np.inf)
            else:
                score = self.trainModelAndGetScore(xTrain, xTest, yTrain, yTest)
                scoresCat.append(score)
        idxCat = np.argmax(scoresCat) if len(scoresCat) > 0 else None
        selectedFloatCol, selectedCatCol, selectedScore = self.getFeatureFromScores(
            availablefFloatCols,
            availableCatCols,
            scoresFloat,
            scoresCat,
            idxFloat,
            idxCat,
        )
        return selectedFloatCol, selectedCatCol, selectedScore

    @abstractmethod
    def getBaselineScore(self):
        pass

    @abstractmethod
    def getSelectedCols(
        self, availableFloatCols, availableCatCols, catCol=None, floatCol=None
    ):
        pass

    def getModifiedData(
        self,
        availableFloatCols,
        availableCatCols,
        selectedFloatCol=None,
        selectedCatCol=None,
    ):
        selectedFloatCols, selectedCatCols = self.getSelectedCols(
            availableFloatCols,
            availableCatCols,
            floatCol=selectedFloatCol,
            catCol=selectedCatCol,
        )
        processor = getDataProcessor(
            selectedFloatCols,
            selectedCatCols,
            applyScaler=True,
            binaryEncoding=self.isBinaryEncoding,
        )
        dataX, dataY = processData(
            self.df, selectedFloatCols, selectedCatCols, self.dependentCol, processor
        )
        xTrain, xTest, yTrain, yTest = trainTestSplit(
            dataX, dataY, self.trainSize, randomState=42
        )
        return xTrain, xTest, yTrain, yTest

    def trainModelAndGetScore(self, xTrain, xTest, yTrain, yTest):
        self.model.fit(xTrain, yTrain)
        score = self.model.getR2(xTest, yTest)
        return score

    def getFeatureFromScores(
        self, floatCols, catCols, scoresFloat, scoresCat, idxFloat, idxCat
    ):
        if idxCat is None:
            return floatCols[idxFloat], None, scoresFloat[idxFloat]
        if idxFloat is None or scoresCat[idxCat] > scoresFloat[idxFloat]:
            return None, catCols[idxCat], scoresCat[idxCat]
        return floatCols[idxFloat], None, scoresFloat[idxFloat]


class BackwardFeatureSelector(SequentialFeatureSelector):

    def getBaselineScore(self):
        processor = getDataProcessor(
            self.floatCols,
            self.catCols,
            applyScaler=True,
            binaryEncoding=self.isBinaryEncoding,
        )
        dataX, dataY = processData(
            self.df, self.floatCols, self.catCols, self.dependentCol, processor
        )
        xTrain, xTest, yTrain, yTest = trainTestSplit(
            dataX, dataY, self.trainSize, randomState=42
        )
        return self.trainModelAndGetScore(xTrain, xTest, yTrain, yTest)

    def getSelectedCols(
        self, availableFloatCols, availableCatCols, catCol=None, floatCol=None
    ):
        selectedFloatCols = availableFloatCols.copy()
        selectedCatCols = availableCatCols.copy()
        if floatCol is not None:
            selectedFloatCols.remove(floatCol)
        if catCol is not None:
            selectedCatCols.remove(catCol)
        return selectedFloatCols, selectedCatCols


class ForwardFeatureSelector(SequentialFeatureSelector):

    def getBaselineScore(self):
        return -np.inf

    def getSelectedCols(
        self, availableFloatCols, availableCatCols, catCol=None, floatCol=None
    ):
        selectedFloatCols = [x for x in self.floatCols if x not in availableFloatCols]
        selectedCatCols = [x for x in self.catCols if x not in availableCatCols]
        if floatCol is not None:
            selectedFloatCols.append(floatCol)
        if catCol is not None:
            selectedCatCols.append(catCol)
        return selectedFloatCols, selectedCatCols
