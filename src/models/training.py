from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score    


def trainModels(models, xTrain, yTrain, xVal, yVal, xTest, yTest):
    errors = []

    for model, paramGrid in models.items():
        grid_search = GridSearchCV(estimator=model, param_grid=paramGrid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(xTrain.values, yTrain.values)
        rfOptimized = grid_search.best_estimator_
        print("Best Parameters:", grid_search.best_params_)

        # Evaluate on the training data
        yTrainPred = rfOptimized.predict(xTrain.values)
        trainMSE = mean_squared_error(yTrain, yTrainPred)
        trainR2 = r2_score(yTrain, yTrainPred)

        print(f"Training MSE: {trainMSE:.2f}")
        print(f"Training R^2: {trainR2:.2f}")


        # Evaluate on the validation data
        yValPred = rfOptimized.predict(xVal.values)
        valMSE = mean_squared_error(yVal, yValPred)
        valR2 = r2_score(yVal, yValPred)

        print(f"Validation MSE: {valMSE:.2f}")
        print(f"Validation R^2: {valR2:.2f}")


        # Finally, evaluate on the test data
        yTestPred = rfOptimized.predict(xTest.values)
        testMSE = mean_squared_error(yTest, yTestPred)
        testR2 = r2_score(yTest, yTestPred)

        print(f"Test MSE: {testMSE:.2f}")
        print(f"Test R^2: {testR2:.2f}")

        errors.append(yTest - yTestPred)

    return errors
