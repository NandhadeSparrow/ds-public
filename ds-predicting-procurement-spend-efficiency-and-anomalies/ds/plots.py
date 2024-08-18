import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_histograms(df, num_columns=2):
    num_plots = len(df.columns)
    num_rows = (num_plots + num_columns - 1) // num_columns  # Calculate the number of rows needed

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 4 * num_rows))
    axes = axes.flatten()  # Flatten the array of axes

    # Loop through each numerical column and create a histogram plot
    for i, column in enumerate(df.columns):
        sns.histplot(df[column], kde=True, ax=axes[i])
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True)

    # Remove any unused subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_boxes(df):
    # Set the desired number of columns in the grid
    num_cols = 3  

    # Calculate the number of rows needed
    num_rows = int(np.ceil(len(df.columns) / num_cols))

    # Create subplots in a grid layout
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    # Iterate over columns and axes
    for i, col in enumerate(df.columns):
        row = i // num_cols 
        col_mod = i % num_cols 

        # Create the box plot (no changes here)
        sns.boxplot(y=df[col], ax=axes[row, col_mod], legend=False)
        axes[row, col_mod].set_title(col)
        axes[row, col_mod].set_ylabel('Value')
        
        # Remove the x-axis label (since it's redundant)
        axes[row, col_mod].set_xlabel(None)
    
    # Remove any extra empty subplots
    for i in range(len(df.columns), num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()
    df.columns


def plot_crosstabs(df, target_col, suitable_cols):
    """Creates stacked bar charts for crosstabulations between suitable columns and the target column."""
    if target_col in suitable_cols: suitable_cols.remove(target_col)
    for col in suitable_cols:
        crosstab = pd.crosstab(df[col], df[target_col], normalize='index')

        # Plot the stacked bar chart
        crosstab.plot(kind='bar', stacked=0, figsize=(10, 6))
        plt.title(f'Distribution of {target_col} by {col}')
        plt.ylabel('Proportion')
        plt.xlabel(col)
        plt.legend(title=target_col)
        plt.xticks(rotation=45)  # Rotate x-axis labels if needed
        plt.show()


def plot_categorical_bars(categorical_df, num_columns=2):
    """Plots bar charts for categorical columns in a grid layout.

    Args:
        categorical_df: A pandas DataFrame containing categorical data.
        num_columns: Number of columns in the grid layout (default: 2).
    """
    num_plots = len(categorical_df.columns)
    num_rows = (num_plots + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(10 * num_columns, 5 * num_rows))
    axes = axes.flatten()

    for i, column in enumerate(categorical_df.columns):
        value_counts = categorical_df[column].value_counts()  # Get value counts
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])  # Bar plot
        axes[i].set_title(f'Bar Chart of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        axes[i].tick_params(axis='x', rotation=45)

    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    pass