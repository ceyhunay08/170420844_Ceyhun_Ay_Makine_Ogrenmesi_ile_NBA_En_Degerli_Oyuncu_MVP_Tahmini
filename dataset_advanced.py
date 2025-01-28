import json
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras 
from keras import layers, models
from keras.losses import MeanSquaredError
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold
import seaborn as sns
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split,StratifiedKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error,classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import cross_val_score
import time
from models import Model as md
import optuna
import matplotlib.pyplot as plt
#imputer for missing values
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
#Process sürelerini tutmak için kod ekle , Hyperparameter tuning yap. Giriş Literatür.
from models import Model  
from imblearn.over_sampling import SMOTE

nba_data = pd.read_csv("NBA_Regular_Season_Player_Stats_With_Clutch_Score.csv")

columns = []
class_weights = {}
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
    
    # Check for outliers
    print("Checking for outliers...")
    numeric_columns = nba_data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        Q1 = nba_data[column].quantile(0.25)
        Q3 = nba_data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = nba_data[(nba_data[column] < lower_bound) | (nba_data[column] > upper_bound)]
        print(f"Number of outliers in {column}: {len(outliers)}")
        
    
        
    # Fill null and NaN values with appropriate method
    
    
    nba_data = Data_handle_categorical(nba_data)
    nba_data = Data_handle_missing_values(nba_data)

    mvp_means = nba_data[nba_data['mvp_award'] == True].mean()
    print("Mean of each column for MVPs:")
    print(mvp_means)
    # Apply SMOTE to balance the dataset
    
    
    print("Applying SMOTE to balance the dataset...")
    X = nba_data.drop('mvp_award', axis=1)
    y = nba_data['mvp_award']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Combine the resampled data back into a single DataFrame
    nba_data_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    nba_data_resampled['mvp_award'] = y_resampled

    # Update the global nba_data with the resampled data
    nba_data = nba_data_resampled
    #Data_Distribution(nba_data)

    # Split the dataset by season
    seasons = nba_data['Season'].unique()
    scaled_data = []

    # Scale data based on the season
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
    
    global class_weights
    # Assign class weights based on the class distribution
    class_counts = nba_data['mvp_award'].value_counts()
    total_samples = len(nba_data)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    print("Class weights:", class_weights)
    
def Data_clean(data):
    
    data = data.drop(['Player'], axis=1)
    data = data.drop(["award_share"],axis=1)
    data = data.drop(['mvp'],axis=1)
    
    global columns 
    columns = data.columns
    
    print("Data cleaning...")
    
   # 10 Dk dan fazla oynayanlar
    data = data.drop(nba_data[nba_data["MP"] < 10].index)
    
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

def Data_Distribution(nba_data):
    global columns
    
    column_for_removing = columns.drop(['mvp_award'])
    column_for_removing = column_for_removing.drop(['Pos'])
    column_for_removing = column_for_removing.drop(['Tm'])
    column_for_removing = column_for_removing.drop(['Season'])
    
    for column in column_for_removing:
        print(f"Distribution of {column}:")
        print(nba_data[column].describe())
        print(f"Mean of {column}: {nba_data[column].mean()}")
        
        # Plot the distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(nba_data[column], kde=True, bins=30, label='All Players')
        
        # Plot the distribution for MVPs
        sns.histplot(nba_data[nba_data['mvp_award'] == True][column], kde=True, bins=30, color='red', label='MVPs')
        
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


def main():
    Data_prep()
    

if __name__ == "__main__":
    main()