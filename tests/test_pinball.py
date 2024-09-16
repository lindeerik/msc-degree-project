"""
Testing for Pinball Loss functions
"""

import pytest

# pylint: disable=protected-access

from models.quantileregression.pinball import (
    negPinballLossValue,
    pinballLossScorer,
    doublePinballLossScorer,
)


def test_negPinballLossValue():
    yTrue = [1.0, 2.0, 3.0]
    yPred = [1.5, 1.8, 2.5]
    quantile = 0.5

    expectedLoss = 0.2
    actualLoss = negPinballLossValue(yPred, yTrue, quantile)

    assert actualLoss == pytest.approx(-expectedLoss, rel=1e-2)


def test_pinballLossScorer():
    yTrue = [1.0, 2.0, 3.0]
    yPred = [1.5, 1.8, 2.5]
    quantile = 0.5

    scorer = pinballLossScorer(quantile)
    score = scorer._score_func(yTrue, yPred)

    expectedLoss = 0.2
    assert score == pytest.approx(-expectedLoss, rel=1e-2)


def test_doublePinballLossScorer():
    yTrue = [1.0, 2.0, 3.0]
    yPred = [[1.5, 1.8, 2.5], [0.5, 2.2, 3.5]]
    lowerQuantile = 0.1
    upperQuantile = 0.9

    scorer = doublePinballLossScorer(lowerQuantile, upperQuantile)

    score = scorer._score_func(yTrue, yPred)

    expectedLoss = 0.52 / 3
    assert score == pytest.approx(-expectedLoss, rel=1e-2)
