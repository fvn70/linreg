from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = 0.0

    def fit(self, X, y):
        X0 = X.copy()
        if self.fit_intercept:
            I = pd.Series(1, index=X0.index)
            X0.insert(loc=0, column="I", value=I)
        Xn = X0.to_numpy()
        y = y.to_numpy()
        Xt = Xn.T
        beta = np.linalg.inv(Xt @ Xn) @ Xt @ y
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
df = pd.read_csv('data_stage4.csv')

reg = CustomLinearRegression(fit_intercept=True)
regSci = LinearRegression(fit_intercept=True)

X_train = df[['f1', 'f2', 'f3']]
y_train = df['y']

reg.fit(X_train, y_train)
y_pred = reg.predict(X_train)
rmse = reg.rmse(y_train, y_pred)
r2 = reg.r2_score(y_train, y_pred)

regSci.fit(X_train, y_train)
y_pred = regSci.predict(X_train)
rmse_sci = np.sqrt(mean_squared_error(y_train, y_pred))
r2_sci = r2_score(y_train, y_pred)

rez = {'Intercept': reg.intercept - regSci.intercept_,
       'Coefficient': reg.coefficient - regSci.coef_,
       'R2': r2 - r2_sci, 'RMSE': rmse - rmse_sci}
print(rez)

