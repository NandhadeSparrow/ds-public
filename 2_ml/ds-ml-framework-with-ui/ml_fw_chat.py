import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import metrics

# Step 1: Data Collection and Cleaning
def collect_data(source):
    file_extension = source.name.split('.')[-1].lower()
    if file_extension in ['csv', 'txt']:
        try:
            return pd.read_csv(source)
        except Exception as e:
            st.error("Error loading data from the uploaded file. Please make sure the file format is correct.")
            return None
    elif file_extension == 'json':
        try:
            return pd.read_json(source)
        except Exception as e:
            st.error("Error loading data from the uploaded file. Please make sure the file format is correct.")
            return None
    else:
        st.error("Unsupported file format. Please upload a CSV, TXT, or JSON file.")
        return None


def clean_data(data, dropna=True, outliers=None):
    """
    Cleans the input DataFrame by applying specified cleaning operations.

    Parameters:
        data (pandas.DataFrame): Input DataFrame.
        dropna (bool): Whether to drop rows with missing values (default: True).
        outliers (dict): Dictionary specifying outlier detection and removal operations. 
                         Example: {'column_name': (lower_bound, upper_bound)}.

    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    cleaned_data = data.copy()  # Create a copy to avoid modifying the original DataFrame

    # Drop rows with missing values
    if dropna:

        cleaned_data = cleaned_data.dropna()

    # Remove outliers
    if outliers:
        for column, (lower_bound, upper_bound) in outliers.items():
            cleaned_data = cleaned_data[(cleaned_data[column] >= lower_bound) & 
                                        (cleaned_data[column] <= upper_bound)]

    return cleaned_data


# Step 2: Exploratory Data Analysis (EDA)
def perform_eda(data):
    st.subheader("Automated Exploratory Data Analysis")
    st.write("## Data Overview")
    st.write(data.head())

    st.write("## Data Info")
    st.write(data.info())

    st.write("## Descriptive Statistics")
    st.write(data.describe())

    st.write("## Missing Values")
    st.write(data.isnull().sum())

    st.write("## Data Correlation")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    st.pyplot()


# Step 3: Feature Engineering (Placeholder implementation)
def engineer_features(data):
    """
    Performs basic feature engineering on the input DataFrame.

    Parameters:
        data (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with engineered features.
    """
    # Example of feature engineering: Creating new features based on existing ones
    data['total_income'] = data['income'] + data['other_income']
    data['is_married'] = (data['marital_status'] == 'Married').astype(int)

    # Example of feature transformation: Scaling numerical features
    numerical_features = ['age', 'income', 'other_income']
    data[numerical_features] = data[numerical_features].apply(lambda x: (x - x.mean()) / x.std())  # Standardization

    return data


def choose_best_model(X_train, y_train):
    """
    Chooses the best model based on cross-validated performance.

    Parameters:
        X_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training target.

    Returns:
        object: Best-performing machine learning model instance.
    """
    # Define candidate models for classification, regression, and clustering tasks
    classification_models = {
        'random_forest': RandomForestClassifier(),
        'logistic_regression': LogisticRegression(),
        'k_neighbors': KNeighborsClassifier(),
        'svm': SVC(),
        'gradient_boosting': GradientBoostingClassifier(),
        'decision_tree': DecisionTreeClassifier(),
        'naive_bayes': GaussianNB(),
        'mlp_classifier': MLPClassifier(),
    }
    regression_models = {
        'random_forest': RandomForestRegressor(),
        'linear_regression': LinearRegression(),
        'k_neighbors': KNeighborsRegressor(),
        'svm': SVR(),
        'gradient_boosting': GradientBoostingRegressor(),
        'lasso_regression': Lasso(),
        'ridge_regression': Ridge(),
        'polynomial_regression': PolynomialFeatures(),
        'mlp_regressor': MLPRegressor(),
    }
    clustering_models = {
        'kmeans': KMeans(),
        'dbscan': DBSCAN(),
        'mean_shift': MeanShift(),
        'agglomerative_clustering': AgglomerativeClustering(),
        'gmm': GaussianMixture(),
    }
    dimensionality_reduction_models = {
        'pca': PCA(),
        'ica': FastICA(),
        'tsne': TSNE(),
    }

    # Choose the best model based on cross-validated performance
    if isinstance(y_train.iloc[0], (int, np.integer)):  # Classification
        models = classification_models
        eval_metric = 'accuracy'
        higher_is_better = True
    elif isinstance(y_train.iloc[0], (float, np.floating)):  # Regression
        models = regression_models
        eval_metric = 'neg_mean_squared_error'
        higher_is_better = False
    else:  # Clustering
        models = clustering_models
        eval_metric = None
        higher_is_better = True

    best_model = None
    best_score = -np.inf if higher_is_better else np.inf

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=eval_metric)
        mean_score = np.mean(scores)
        if (higher_is_better and mean_score > best_score) or \
           (not higher_is_better and mean_score < best_score):
            best_score = mean_score
            best_model = model

    return best_model


def evaluate_clustering(X, labels):
    metrics_dict = {
        'Silhouette Score': metrics.silhouette_score(X, labels),
        'Davies-Bouldin Index': metrics.davies_bouldin_score(X, labels),
        # Add more clustering evaluation metrics as needed
    }
    return metrics_dict


def main():
    st.title('Machine Learning Model Selection')

    # Step 1: Data Collection
    st.header('Step 1: Data Collection')
    data_source = st.radio('Select Data Source', ['Upload File', 'Provide Link'])
    
    if data_source == 'Upload File':
        uploaded_file = st.file_uploader('Upload Data File', type=['csv', 'json', 'txt'])
        if uploaded_file is not None:
            data = collect_data(uploaded_file)
            if data is not None:
                st.write('Data loaded successfully!')
    elif data_source == 'Provide Link':
        link = st.text_input('Enter Link to Data (CSV, JSON, or TXT)')
        if st.button('Load Data'):
            if link:
                data = collect_data(link)
                if data is not None:
                    st.write('Data loaded successfully!')

    if 'data' not in locals():
        st.warning('Please choose a data source and provide the necessary information.')

    if 'data' in locals():

        # Step 2: Data Cleaning
        if data is not None:
            st.header('Step 2: Data Cleaning')
            cleaned_data = clean_data(data)
            st.write(cleaned_data.head())

            # Step 3: Exploratory Data Analysis (EDA)
            st.header('Step 3: Exploratory Data Analysis (EDA)')
            perform_eda(cleaned_data)

            # Step 4: Feature Engineering
            st.header('Step 4: Feature Engineering')
            engineered_data = engineer_features(cleaned_data)
            st.write(engineered_data.head())

            # Step 5: Model Selection and Training
            st.header('Step 5: Model Selection and Training')
            target_column = st.selectbox('Select Target Column', list(engineered_data.columns))
            target = engineered_data[target_column]
            features = engineered_data.drop(columns=[target_column])
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

            # Choose the best model based on cross-validated performance
            best_model = choose_best_model(X_train, y_train)

            # Step 6: Model Evaluation
            st.header('Step 6: Model Evaluation')
            best_model.fit(X_train, y_train)
            if isinstance(best_model, (RandomForestClassifier, GradientBoostingClassifier, LogisticRegression, SVC)):
                predictions = best_model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                st.write("Best Model Accuracy:", accuracy)
            elif isinstance(best_model, (RandomForestRegressor, GradientBoostingRegressor, LinearRegression, SVR)):
                predictions = best_model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                st.write("Best Model Mean Squared Error:", mse)
            else:
                st.header('Step 6: Clustering Evaluation')
                clustering_models = {
                    'KMeans': KMeans(),
                    'DBSCAN': DBSCAN(),
                    'Mean Shift': MeanShift(),
                    'Agglomerative Clustering': AgglomerativeClustering(),
                    'Gaussian Mixture': GaussianMixture(),
                }
                selected_model = st.selectbox('Select Clustering Model', list(clustering_models.keys()))

                if st.button('Evaluate Clustering'):
                    model = clustering_models[selected_model]
                    model.fit(data)
                    labels = model.labels_
                    metrics_dict = evaluate_clustering(data, labels)
                    st.write("Clustering Evaluation Metrics:")
                    for metric, value in metrics_dict.items():
                        st.write(f"- {metric}: {value}")

if __name__ == "__main__":
    main()