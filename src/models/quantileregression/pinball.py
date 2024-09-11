import torch
import torch.nn as nn
from sklearn.metrics import make_scorer


class PinballLoss(nn.Module):
    def __init__(self, quantile: float):
        super().__init__()
        self.quantile = quantile

    def forward(self, yPred, yTrue):
        errors = yTrue - yPred
        loss = torch.max(self.quantile * errors, (self.quantile - 1) * errors)
        return torch.mean(loss)
    
def negPinballLossValue(yTrue, yPred, quantile):
    yTrueTensor = torch.tensor(yTrue, dtype=torch.float32)
    yPredTensor = torch.tensor(yPred, dtype=torch.float32)
    
    lossFunc = PinballLoss(quantile=quantile) 
    lossValue = lossFunc(yTrueTensor, yPredTensor).item()
    
    return -lossValue

def pinballLossScorer(quantile):
    scorerFunc = lambda yTrue, yPred: negPinballLossValue(yTrue, yPred, quantile)
    scorer = make_scorer(scorerFunc, greater_is_better=False)
    return scorer

def doublePinballLossScorer(lowerQuantile, upperQuantile):
    scorerFunc = lambda yTrue, yPred: negPinballLossValue(yTrue, yPred[0], lowerQuantile) + negPinballLossValue(yTrue, yPred[1], upperQuantile)
    scorer = make_scorer(scorerFunc, greater_is_better=False)
    return scorer