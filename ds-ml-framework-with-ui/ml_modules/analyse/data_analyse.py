# analyse data
print('##### --- head --- #####\n', df.head())
print('##### --- tail --- #####\n', df.tail())
print('##### --- info --- #####\n', df.info())
print('##### --- describe --- #####\n', df.describe())
print('##### --- shape, size --- #####\n', df.shape,df.size)
print('##### --- mean --- #####\n', df.mean())



# descriptive statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(df.head())
print(df.describe())
# range = column.max() - column.min()



# pivot table
df_by_month_plot = df_by_month.pivot('year', 'month', ‘strike level_code')
df_by_month_plot.head()

