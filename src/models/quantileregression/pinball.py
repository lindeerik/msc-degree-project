import torch
import torch.nn as nn
from sklearn.metrics import make_scorer


class PinballLoss(nn.Module):
    def __init__(self, quantile: float):
        super().__init__()
        self.quantile = quantile

    def forward(self, yPred, yTrue):
        errors = yPred - yTrue
        loss = torch.max(-errors * self.quantile, errors * (1-self.quantile))
        return torch.mean(loss)
    
def negPinballLossValue(yPred, yTrue, quantile):
    yTrueTensor = torch.tensor(yTrue, dtype=torch.float32)
    yPredTensor = torch.tensor(yPred, dtype=torch.float32)
    
    lossFunc = PinballLoss(quantile=quantile) 
    lossValue = lossFunc(yPredTensor, yTrueTensor).item()
    return -lossValue

def pinballLossScorer(quantile):
    scorerFunc = lambda yPred, yTrue: negPinballLossValue(yTrue, yPred, quantile)
    scorer = make_scorer(scorerFunc)
    return scorer

def doublePinballLossScorer(lowerQuantile, upperQuantile):
    scorerFunc = lambda yPred, yTrue: 1/2*(negPinballLossValue(yPred[0], yTrue, lowerQuantile) + negPinballLossValue(yPred[1], yTrue, upperQuantile))
    scorer = make_scorer(scorerFunc)
    return scorer