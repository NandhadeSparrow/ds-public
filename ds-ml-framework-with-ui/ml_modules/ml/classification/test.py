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

