from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

model = LinearRegression()
print(model)


import pandas as pd

from sklearn.datasets import load_boston
boston_dataset = load_boston()
## build a DataFrame
boston = pd.DataFrame(boston_dataset.data, 
                      columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
variates = ['RM', 'LSTAT']
X = boston[variates]
Y = boston['MEDV']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=1)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
print(model.intercept_.round(2))
print(model.coef_.round(2))


import numpy as np
new_RM = np.array([6.5]).reshape(-1,1) # make sure it's 2d
print(model.predict(new_RM))

y_test_predicted = model.predict(X_test)

plt.scatter(X_test, Y_test,
label='testing data');
plt.plot(X_test, y_test_predicted,
label='prediction', linewidth=3)
plt.xlabel('RM'); plt.ylabel('MEDV')
plt.legend(loc='upper left')
plt.savefig("plot.png")
plt.show()

residuals = Y_test - y_test_predicted

# plot the residuals
plt.scatter(X_test, residuals)
# plot a horizontal line at y = 0
plt.hlines(y = 0,
xmin = X_test.min(), xmax=X_test.max(),
linestyle='--')
# set xlim
plt.xlim((4, 9))
plt.xlabel('RM'); plt.ylabel('residuals')
plt.savefig("plot.png")
plt.show()
from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y_test, y_test_predicted))

model.score(X_test, Y_test)
a = (residuals**2).mean()
b = ((Y_test - Y_test.mean())**2).sum()
print(1-a/b)

'''
MAE
MSE
R-Score
RMSE
'''