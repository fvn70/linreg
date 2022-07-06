import numpy as np
import pandas as pd


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = 0.0

    def fit(self, X, y):
        X = X.to_numpy()
        y = y.to_numpy()
        if self.fit_intercept:
            X = np.matrix([[1, i] for i in X])
        Xt = X.T
        beta = np.linalg.inv(Xt @ X) @ Xt @ y
        self.coefficient = beta
        if self.fit_intercept:
            self.intercept = beta[0, 0]
            self.coefficient = beta[0, 1]

    def predict(self, X):
        X = X.to_numpy()
        y = X @ self.coefficient
        return y


dic = {'x': [4, 4.5, 5, 5.5, 6, 6.5, 7],
       'w': [1, -3, 2, 5, 0, 3, 6],
       'z': [11, 15, 12, 9, 18, 13, 16],
       'y': [33, 42, 45, 51, 53, 61, 62]}
df = pd.DataFrame(dic)

regCustom = CustomLinearRegression(fit_intercept=False)
regCustom.fit(df[['x', 'w', 'z']], df['y'])
y_pred = regCustom.predict(df[['x', 'w', 'z']])
print(y_pred)
