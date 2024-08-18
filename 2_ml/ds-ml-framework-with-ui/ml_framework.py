# import modules
from ml_modules.data import data_load, data_to_df, data_export
from ml_modules.analyse import data_analyse, data_clean, eda, hypothesis
from ml_modules.ml import choose_algo, choose_features, model_build, data_predict


data_load()
data_to_df()

data_analyse()
data_clean()
eda()
hypothesis()

choose_algo()
choose_features()
model_build()
data_predict()

data_export()


# linear regression assumption
# linearity
sns.pairplot(chinstrap_penguins)
# normality
import statsmodels.api as sm
import matplotlib.pyplot as plt

residuals = model.resid
fig = sm.qqplot(residuals, line = 's')
plt.show()
fig = sns.histplot(residuals)
fig.set_xlabel("Residual Value")
fig.set_title("Histogram of Residuals")
plt.show()
# independent observations
import matplotlib.pyplot as plt

fig = sns.scatterplot(fitted_values, residuals)
fig.axhline(0)
fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")
plt.show()


# -------------------------------------bala
# understand problem
# data collection
# eda - univariate, bivariate, multivariate
# data cleaning
    #   missing values
    # outlier iqr
    # Encoding
    # diagnosis

# feature selection
    # important fdeatures
    # model.coefficient

# train test split
# algorithm selection
    # category or numericals
    # category get unique labels
    # if 2 class - binary classification
    #     balanced or imbalanced
    #     if imbalanced 
    #         smote eclass to balance
    #         ogistic regression and train
    # if 2+ class - dtree, rfoirrest - 
# model training
    # hyper paramaeter tuning
# model testing
    # validation
# prediction
# metrics


# --------------------------------google
# Imports 
    # Applicable packages and libraries were imported to the code notebook.

# Data Analysis
    # Functions such as head(), describe(), and info() were used to analyze the data. 

# Visualizations
    # Histogram(s) were generated to examine key variables. 
    # Box plot(s) were generated to examine key variables. 
    # Scatter plot(s) were generated to examine key variables. 
    # Bar or pie chart(s) were generated to examine key variables.

# Results and/or Evaluation
    # The executive summary mentioned the tasks completed for this end-of-course project.
    # The executive summary included information regarding the results of the data variable assessment.
    # The executive summary identified recommended next steps in order to build a predictive model.
    # The executive summary included a summary of the results of your exploratory data analysis (EDA). 

# Feature Engineering
    # Categorical variables were encoded as binary variables.
    # A target variable was assigned. 
    # An evaluation metric was chosen. 

# Machine Learning Modeling 
    # The data was split into training and testing sets.

# The following steps were performed for the random forest model:
    # Performed a GridSearch to tune hyperparameters
    # Captured precision, recall, F1 score, and accuracy metrics
    # Obtained validation scores of best model 

# The following steps were performed for the XGBoost model:
    # Performed a GridSearch to tune hyperparameters
    # Captured precision, recall, F1 score, and accuracy metrics
    # Obtained validation scores of best model 

# The random forest model was compared to the XGBoost model.
    # A confusion matrix was plotted.
    # The top 10 most important features of the final model were inspected. 

# Results and/or Evaluation 
    # All questions in the code notebook were answered. 
    # All questions in the PACE strategy document were answered. 
    # The executive summary clearly articulated the challenges presented in this data project. 
    # The executive summary identified the outcome of your work. 
    # The executive summary included recommendations for future work/next steps. 


'''
capstone

Applicable packages and libraries were imported to the code notebook. 
The dataset was imported and read into the notebook using the pd.read_csv() function. 
Basic information and descriptive statistics about the data were gathered using the .info() and .describe() functions, respectively.
Steps to clean the data were considered. 
The dataset was checked for missing values, duplicates, and outliers during the exploratory data analysis process. 
Visualizations were created to examine variables and relationships between variables of interest. 
Box plots and histograms were created to visualize the distributions of variables. 
A scatter plot was created to visualize the correlation between a pair of variables. 
A heatmap was created to examine the correlations between multiple pairs of variables.
The type of prediction task and the types of models most appropriate for this task were identified.
A modeling approach was selected and the corresponding model assumptions were checked if applicable. 
All relevant variables were considered in the feature engineering and encoding process. 
The dataset was split into train/test sets or train/validate/test sets. 
An appropriate machine learning model was constructed, fitted, and used for prediction. 
A confusion matrix and/or a classification report were generated to evaluate the performance of the model. 
Model results were reported using appropriate evaluation metrics. 
All questions in the PACE strategy document were answered. 
An executive summary detailing the work completed, key findings, and business recommendations was created. 







'''