# import pandas as pd


import pandas as pd
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston



## build a DataFrame
# presidents_df = pd.read_csv('https://sololearn.com/uploads/files/president_heights_party.csv', index_col='name')
                                  
# print(presidents_df['party'].describe())
# print(presidents_df['party'].value_counts())
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, 
                      columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

print(boston.shape)
print(boston[['CHAS', 'RM', 'AGE', 'RAD', 'MEDV']].head())
boston.head(5)
print(boston.describe(include = 'all').round(2))

boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
boston.hist(column='CHAS')
plt.savefig("plot1.png")
plt.show()


boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

boston.hist(column='RM', bins=20)
plt.savefig("plot1.png")
plt.show()

corr_matrix = boston.corr().round(2)
boston.plot(kind = 'scatter',
  x = 'RM',
  y = 'MEDV',
  figsize=(8,6))
plt.savefig("plot1.png")
plt.show()
X = boston[['MV']]
Y = boston['MEDV']
print(Y.shape)


# discovering
# structuring
# cleaning
# joining
# validating
# presenting



import numpy as np

