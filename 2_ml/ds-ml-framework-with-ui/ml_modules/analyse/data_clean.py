# ask questions
# ask if there can be duplicates
# outliers
# dummy vars
# label encoding
# personal detail columns
# date column
# columns for grouping
# columns for barplot
# columns for boxplot
# column for geoplot
# catagories
# quantities
# date



# clean null
print(df.isnull().sum())
df.fillna(0)
df.fillna({'Annual_Income':1000})
df.dropna()
del df['col_name']


# duplicates
df = df.drop_duplicates().shape




# clean outliers

def addlabels (x,y):
  for i in range (len(x) ):
    plt.text(x(i)-0.5, y[i]+0.05, s = readable_numbers(y[i]))
colors = np.where(df[ 'number_of_strikes'] < lower_limit, 'r', 'b')
fig, ax = plt.subplots(figsize= (16,8))
ax.scatter (df[ 'year'], df[ 'number_of_strikes'],c=colors)
ax.set_xlabel('Year')
ax.set_ylabel('Number of strikes')
ax.set_title('Number of lightning strikes by year')
addlabels(df['year'], df['number_of_strikes'])
for tick in ax.get_xticklabels():
  tick.set_rotation(45)
plt.show()

# Create 2 new columns
df['month'] = df['date'].dt.month
df['month_txt'] = df['date'].dt.month_name().str.slice(stop=3)

# Group by ~“month~ and “month _txt~, sum it, and sort. Assign result to new df
df_by_month = df.groupby(['month', 'month_txt']).sum().sort_values('month', ascending=True) .head()
df_without_outliers = df[df['number_of_strikes'] >= lower_limit]

# Recalculate mean and median values on data without outliers
print("Mean:" + readable_numbers(np.mean(df_without_outliers[ 'number_of_strikes'])))
print("Median:" + readable_numbers(np.median(df_without_outliers['number of _strikes'])))
# outliers
box = sns.boxplot(x=df['number_of_strikes'])
g = plt.gca()
box.set_xticklabels(np.array([readable_numbers(x) for x in g.get_xticks()]))
plt.xlabel('Number of strikes')
plt.title('Yearly number of lightning strikes');

percentile25 = df['number_of_strikes'].quantile(0.25)
percentile75 = df['number_of_strikes'].quantile(0.75)

# Calculate interquartile range
iqr = percentile75 - percentile25

# Calculate upper and lower thresholds for outliers
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr

print('Lower limit is: ', lower_limit)
print(df[df['number_of_strikes'] < lower_limit])

mask = (df['number_of_strikes'] >= lower_limit) & (df['number_of_strikes'] <=
upper_limit)

df = df[mask].copy()
print(df)

# Calculate 10th percentile
tenth_percentile = np.percentile(df['number_of_strikes'], 10)

# Calculate 90th percentile
ninetieth_percentile = np.percentile(df['number_of_strikes'], 90)

# Apply lambda function to replace outliers with thresholds defined above
df['number_of_strikes'] = df['number_of_strikes'].apply(lambda x: (
    tenth_percentile if x < tenth_percentile
    else ninetieth_percentile if x > ninetieth_percentile
    else x))

# Calculate median of all NON-OUTLIER values
median = np.median(df['number_of_strikes'][df['number_of_strikes'] >= lower_limit])

# Impute the median for all values < lower_limit
df['number_of_strikes'] = np.where(df['number_of_strikes'] < lower_limit, median, df['number_of_strikes'] )


# dates
date_column_name = 'date'
df['date'] = pd.to_datetime(df[date_column_name])

df['week'] = df['date'].dt.strftime('%V')
df['weekday'] = df.date.dt.dayname()
df_by_week = df[df['year']=='2019'].groupby(['week']).sum().reset_index()

df['month'] = df['date'].dt.strftime('%m')
df['dt_month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.month_name() #.str.slice(stop=3)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df['month_name'] = pd.Categorical(df['month'], categories = months, ordered = True)
df_by_month = df.groupby(['month', 'month_name']).sum().sort_values('month', ascending=True)
df_by_month = df.groupby(['year', 'month']).sum().reset_index()
df_by_month['strike level'] = pd.qcut(df_by_month[ 'number_of_strikes'], 4, labels = ['mild', 'scattered','Heavy', 'Severe'])
df_by_month['strike_level_code'] = df_by_month['strike_level'].cat.codes

df['quarter'] = df['date'].dt.to_period('Q').dt.strftime('%q')
df['year'] = df['date'].dt.strftime('%Y')


# concat
# union_df = pd.concat([df.drop(['weekday','week'],axis=1),df_2],ignore_index=True)


# merge
# df_joined = df1.merge(df2, how='left', on=['date', 'center_point'])
# df_null_geo = df_joined[pd.isnull(df_joined.state_code)]


# dummies
pd.get_dummies(df_by_month['strike level'])



# data cleaning

pd.isnull(df)
print(df)
print('\n After notnull(): \n')
pd.notnull(df)
print(df)
print('\n After fillna(): \n')

df.fillna(2)
print(df)
print('\n After replace(): \n')

df.replace('Aves', 'bird')
print('Original df: \n \n', df)
print('\n After dropna(axis=0): \n')
print(df.dropna(axis=0))

print('\n After dropna(axis=1): \n')
print(df.dropna(axis=1))
print(df)
print()
df.describe(include = 'all')
print(df)
print('\n Original dtypes of df: \n')

print(df.dtypes)

print('\n dtypes after casting \'class\' column as categorical: \n')

df['class'] = df['class'].astype('category')

print(df.dtypes)
# Cast 'class' column as categorical
df['class'] = df['class'].astype('category')

print('\n \'class\' column: \n')
print(df['class'])

print('\n Category codes of \'class\' column: \n')

df['class'].cat.codes
from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder()
encoder = LabelEncoder()

data = [1, 2, 2, 6]

# Fit to the data
encoder.fit(data)

# Transform the data
transformed = encoder.transform(data)

# Reverse the transformation
inverse = encoder.inverse_transform(transformed)

print('Data =', data)
print('\n Classes: \n', encoder.classes_)
print('\n Encoded (normalized) classes: \n', transformed)
print('\n Reverse from encoded classes to original: \n', inverse)
from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder()
encoder = LabelEncoder()

data = ['paris', 'paris', 'tokyo', 'amsterdam']

# Fit to the data
encoder.fit(data)

# Transform the data
transformed = encoder.transform(data)

# New data
new_data = [0, 2, 1, 1, 2]

# Get classes of new data
inverse = encoder.inverse_transform(new_data)

print('Data =', data)
print('\n Classes: \n', list(encoder.classes_))
print('\n Encoded classes: \n', transformed)
print('\n New data =', new_data)
print('\n Convert new_data to original classes: \n', list(inverse))


# input validation

# Load libraries.
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
# Take a peek at the data.
df .head()
# Display the data type of the columns.
print(df.dtypes)

# Date is currently a string. Let's parse it into a datetime column.
df[{'date'] = pd.to_datetime(df['date'])
# Count the number of missing values in each column.
df.isnull().sum()
# Check ranges for all variables.
df .describe(include = ‘all')
# Find missing dates by comparing all dates in 2018 to dates in
# our ‘date~ column.

full_date_range = pd.date_range(start = '2018-01-01', end = '2018-12-31')
full_date_range.difference(df['date'])
# Make a boxplot to see the range better.

sns.boxplot(y = df['number_of_strikes'])
# Plot again without the outliers to see where the majority of data is.
sns.boxplot(y = df['number_of_strikes'], showfliers = False)
# Plot points on the map to verify data is all from US.
df_points = df[['latitude', 'longitude']].drop_duplicates() # Get unique points.
df_points.head()
# Plot points on the map to verify data is all from US.
df_points = df[['latitude', 'longitude']].drop_duplicates() # Get unique points.
p = px.scatter_geo(df_points, lat = ‘latitude’, lon = ‘longitude')

p.show()



import pandas as pd
import matplotlib.pyplot as plt
import os



# **1. Data Cleaning **
# Code for cleaning (handling missing values, fixing errors, etc.)
# Example:
df.dropna(inplace=True)  # Drop rows with missing values

# **2. Exploratory Data Analysis (EDA)**
# Code for generating distributions, correlations
df.describe()  # Basic summary statistics
df.hist()  # Histograms
df.plot.scatter(x='column1', y='column2')  # Scatterplot (Replace column names)

# **3. Modeling (If applicable)**
# Code for model fitting and prediction
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# ... (split data, fit, prediction logic here)

# **4. Visualization of Results**
# Plotting predicted outcomes or model insights
plt.figure(figsize=(10,6))
plt.scatter(df['column1'], df['column2'])
plt.plot(df['column1'], model.predict(df[['column1']]), color='red')  # Replace column names
plt.title('Data vs Prediction')
plt.show()

# **5. Interpretation (Add Comments Throughout)**
# Add comments to explain code purpose and insights




