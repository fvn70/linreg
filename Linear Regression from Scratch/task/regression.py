import numpy as np


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = 0.0

    def fit(self, X, y):
        Xt = np.transpose(X)
        m2 = np.linalg.inv(np.dot(Xt, X))
        m3 = np.dot(m2, Xt)
        beta = np.dot(m3, y)
        return beta


x = np.array([4.0, 4.5, 5, 5.5, 6.0, 6.5, 7.0])
y = np.array([33, 42, 45, 51, 53, 61, 62])
X = np.matrix([[1, i] for i in x])

model = CustomLinearRegression().fit(X, y)
b0 = model[0, 0]
b1 = model[0, 1]
dic = f"{'{'}'Intercept': {b0}, 'Coefficient': array([{round(b1, 7)}]){'}'}"
print(dic)

