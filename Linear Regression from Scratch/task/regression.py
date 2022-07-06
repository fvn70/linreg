import math
import numpy as np
import pandas as pd


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = 0.0

    def fit(self, X, y):
        if self.fit_intercept:
            I = pd.Series(1, index=X.index)
            X.insert(loc=0, column="I", value=I)
        X = X.to_numpy()
        y = y.to_numpy()
        Xt = X.T
        beta = np.linalg.inv(Xt @ X) @ Xt @ y
        self.coefficient = beta
        if self.fit_intercept:
            self.intercept = beta[0]
            self.coefficient = beta[1:]

    def predict(self, X):
        X = X.to_numpy()
        y = self.intercept + X @ self.coefficient
        return y

    def rmse(self, y, yhat):
        n = y.shape[0]
        return np.sqrt(np.sum((y - yhat)**2) / n)

    def r2_score(self, y, yhat):
        return 1 - np.sum((y - yhat)**2) / np.sum((y - y.mean())**2)


dic = {'x': [0.9, 0.5, 1.75, 2.0, 1.4, 1.5, 3.0, 1.1, 2.6, 1.9],
       'w': [11, 11, 9, 8, 7, 7, 6, 5, 5, 4],
       'y': [21.95, 27.18, 16.9, 15.37, 16.03, 18.15, 14.22, 18.72, 15.4, 14.69]}
df = pd.DataFrame(dic)

reg = CustomLinearRegression()
reg.fit(df[['x', 'w']], df['y'])
y_pred = reg.predict(df[['x', 'w']])
rmse = reg.rmse(df['y'], y_pred)
r2 = reg.r2_score(df['y'], y_pred)

rez = {'Intercept': reg.intercept, 'Coefficient': reg.coefficient, 'R2': r2, 'RMSE': rmse}
print(rez)

