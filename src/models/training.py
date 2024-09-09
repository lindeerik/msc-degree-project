from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from models.model import Model


def trainModels(models, xTrain, yTrain, xVal, yVal, xTest, yTest):
    errors = []

    for model in models:
        model.gridSearchFit(xTrain, yTrain, 3)
        print(f"Best Parameters for {model.getName()}: {model.getBestParams()}")

        trainMSE = model.getMse(xTrain, yTrain)
        trainR2 = model.getR2(xTrain, yTrain)
        print(f"Training MSE: {trainMSE:.2f}")
        print(f"Training R^2: {trainR2:.2f}")

        valMSE = model.getMse(xVal, yVal)
        valR2 = model.getR2(xVal, yVal)
        print(f"Validation MSE: {valMSE:.2f}")
        print(f"Validation R^2: {valR2:.2f}")

        testMSE = model.getMse(xTest, yTest)
        testR2 = model.getR2(xTest, yTest)
        print(f"Test MSE: {testMSE:.2f}")
        print(f"Test R^2: {testR2:.2f}\n")

        errors.append(yTest - model.predict(xTest))

    return errors
