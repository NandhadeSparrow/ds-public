from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
iris = pd.read_csv('https://sololearn.com/uploads/files/iris.csv')

print(iris.shape)
iris.head()
iris.drop('id', axis=1, inplace=True)
iris.head()
print(iris.groupby('species').size())
print(iris['species'].value_counts())
iris.hist()
plt.show()

import numpy as np
import pandas as pd

iris = pd.read_csv('https://sololearn.com/uploads/files/iris.csv')

# build a dict mapping species to an integer code
inv_name_dict = {'iris-setosa': 0,
'iris-versicolor': 1,
'iris-virginica': 2}

# build integer color code 0/1/2
colors = [inv_name_dict[item] for item in iris['species']]
# scatter plot
scatter = plt.scatter(iris['sepal_len'], iris['sepal_wd'], c = colors)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
## add legend
plt.legend(handles=scatter.legend_elements()[0],
labels = inv_name_dict.keys())
plt.savefig("plot.png")
plt.show()


import matplotlib.pyplot as plt
import pandas as pd

iris = pd.read_csv('https://sololearn.com/uploads/files/iris.csv')

# build a dict mapping species to an integer code
inv_name_dict = {'iris-setosa': 0,
'iris-versicolor': 1,
'iris-virginica': 2}

# build integer color code 0/1/2
colors = [inv_name_dict[item] for item in iris['species']]
# scatter plot
scatter = plt.scatter(iris['petal_len'], iris['petal_wd'],c = colors)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
# add legend
plt.legend(handles= scatter.legend_elements()[0],
labels = inv_name_dict.keys())
plt.savefig("plot.png")
plt.show()
pd.plotting.scatter_matrix()


# https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html

# Preparation

X = iris[['petal_len', 'petal_wd']]
y = iris['species']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)

print(y_train.value_counts())
print(y_test.value_counts())


from sklearn.neighbors import KNeighborsClassifier
## instantiate 
knn = KNeighborsClassifier(n_neighbors=5)
## fit 
print(knn.fit(X_train, y_train))

# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

pred = knn.predict(X_test)
print(pred[:5])
y_pred_prob = knn.predict_proba(X_test)
print(y_pred_prob[10:12])
print(pred[10:12])

y_pred = knn.predict(X_test)

print((y_pred==y_test.values).sum())
print(y_test.size)

print(knn.score(X_test, y_test))
(y_pred==y_test.values).sum()/y_test.size

from sklearn.metrics import confusion_matrix 
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues);
plt.savefig("plot.png")

from sklearn.model_selection import cross_val_score

# create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)
# train model with 5-fold cv
cv_scores = cross_val_score(knn_cv, X, y, cv=5)
# print each cv score (accuracy) 
print(cv_scores)
print(cv_scores.mean())


from sklearn.model_selection import GridSearchCV
# create new a knn model
knn2 = KNeighborsClassifier()
# create a dict of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(2, 10)}
# use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X, y)

knn_final = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])
knn_final.fit(X, y)

y_pred = knn_final.predict(X)
print(knn_final.score(X, y))

new_data = np.array([3.76, 1.20])
new_data = new_data.reshape(1, -1)
new_data = np.array([[3.76, 1.2], 
                     [5.25, 1.2],
                     [1.58, 1.2]])
print(knn_final.predict(new_data))
print(knn_final.predict_proba(new_data))