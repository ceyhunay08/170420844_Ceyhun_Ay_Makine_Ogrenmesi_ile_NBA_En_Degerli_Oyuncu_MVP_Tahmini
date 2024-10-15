import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras 
from keras import layers, models
from keras.losses import MeanSquaredError
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import cross_val_score


import optuna

import matplotlib.pyplot as plt


#imputer for missing values
from sklearn.impute import SimpleImputer

from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

nba_data_regular = pd.read_csv("NBA_Player_Stats_82_22.csv")

nba_data_playoff = pd.read_csv("NBA_Player_Stats_Playoff_82_22.csv")


nba_data = pd.concat([nba_data_regular,nba_data_playoff],axis=0)

def Data_prep():
    print("Data preparing...")
    global nba_data
    
    # Clean the data
    mvps = nba_data.groupby('Season').max('award_share')

# add MVP column to mvps dataframe
    mvps["mvp"] = True

    # join the MVP column to original data, fill the rest with False
    nba_data = nba_data.merge(mvps[["award_share", "mvp"]], on = ["Season", "award_share"], how = "left")
    nba_data["mvp_award"] = nba_data["mvp"].fillna(False)
    
    
    
    nba_data = Data_clean(nba_data)
    
    
    # Fill null and NaN values with appropriate method
    nba_data = Data_handle_categorical(nba_data)
    nba_data = Data_handle_missing_values(nba_data)
    
    
    mvp_counts = nba_data['mvp_award'].value_counts()
    plt.pie(mvp_counts, labels=mvp_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of MVP Award')
    plt.show()
    
    
    mvp_means = nba_data[nba_data['mvp_award'] == True].mean()
    print("Mean of each column for MVPs:")
    print(mvp_means)
    
    # Remove rows where any column value is significantly below the mean (e.g., more than 3 standard deviations below the mean)
    for column in nba_data.columns:
        if nba_data[column].dtype in [np.float64, np.int64]:  # Only apply to numeric columns
            mean = nba_data[column].mean()
            std = nba_data[column].std()
            threshold = mean - 3 * std

            # Count rows significantly below the mean for the column
            rows_below_mean = (nba_data[column] < threshold).sum()
            print(f"Number of rows significantly below the mean for {column}: {rows_below_mean}")
            nba_data = nba_data[~(nba_data[column] < threshold)]

    print("Rows significantly below the mean removed...")
    
    # Remove rows where award_share is zero
    
    
    # Remove outliers using IQR method
    Q1 = nba_data.quantile(0.25)
    Q3 = nba_data.quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier criteria
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers
    nba_data = nba_data[~((nba_data < lower_bound) | (nba_data > upper_bound)).any(axis=1)]

    print("Outliers removed...")
    
    # Show MVP distribution in pie chart form
    mvp_counts = nba_data['mvp_award'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(mvp_counts, labels=mvp_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])
    plt.title('Distribution of MVP Award')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    
    
    
    # Split the dataset by season
    seasons = nba_data['Season'].unique()
    scaled_data = []

    scaler = MinMaxScaler()

    for season in seasons:
        season_data = nba_data[nba_data['Season'] == season]
        season_data_scaled = pd.DataFrame(scaler.fit_transform(season_data.drop('Season', axis=1)), columns=season_data.columns.drop('Season'))
        season_data_scaled['Season'] = season  # Add the season column back
        scaled_data.append(season_data_scaled)

    # Combine the scaled data back into a single DataFrame
    nba_data_scaled = pd.concat(scaled_data, axis=0)

    # Save the scaled dataset to a CSV file
    nba_data_scaled.to_csv('dataset.csv', index=False)
    print("Scaled dataset saved as dataset.csv")
    
def Data_clean(data):
    
    data = data.drop(['Player'], axis=1)
    data = data.drop(["award_share"],axis=1)
    data = data.drop(['mvp'],axis=1)
    
    print("Data cleaning...")
    
    return data

def Data_handle_categorical(data):  
    print("Data encoding...")
    #One-hot encoding
    data_encoded = pd.get_dummies(data, columns=['Tm', 'Pos'], drop_first=True)
    return data_encoded

def Data_handle_missing_values(data):
    
    #Imputing missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    #One-hot encoding
    return data_imputed

Data_prep()