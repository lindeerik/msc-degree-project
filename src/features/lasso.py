"""
Function for LASSO feature selection
"""

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from category_encoders import BinaryEncoder

from models.model import Model


def lassoFeatureSelection(df, floatCols, catCols, dependentCol, cv=5):
    selectedCols = floatCols + catCols + [dependentCol]
    df = df[selectedCols].dropna()
    y = df[dependentCol]

    encoder = BinaryEncoder(cols=catCols)
    X = encoder.fit_transform(df[floatCols + catCols])

    scaler = StandardScaler()
    X[floatCols] = scaler.fit_transform(X[floatCols])

    lasso = LassoCV(cv=cv, max_iter=10000, n_alphas=10000)
    lassoModel = Model(lasso, "Lasso")
    lassoModel.fit(X, y)

    selectedCols = X.columns[(lassoModel.getModel().coef_ != 0)].tolist()
    return selectedCols, lassoModel
