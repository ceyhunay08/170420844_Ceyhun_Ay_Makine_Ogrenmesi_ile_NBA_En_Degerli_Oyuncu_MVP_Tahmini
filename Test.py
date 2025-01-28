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
import joblib
import openpyxl

nba_data = pd.read_csv("datasets/dataset.csv")
# nba_data_test = pd.read_csv("datasets/Test_preprocess.csv")
train_features = pd.read_csv('datasets/train_features.csv').columns.tolist()


def Data_test_prep():
    def Data_clean(data):
        data = data.drop(['Player'], axis=1)
        data = data.drop(["award_share"], axis=1)
        data = data.drop(['Player-additional'], axis=1)
        data = data.drop(['Rk'], axis=1)
        
        # Drop rows where MP is less than 10
        data = data.drop(data[data["MP"] < 10].index)
        
        return data

    def Data_handle_categorical(data):
        # One-hot encoding
        data_encoded = pd.get_dummies(data, columns=['Tm', 'Pos'], drop_first=True)
        return data_encoded

    def Data_handle_missing_values(data):
        # Imputing missing values
        imputer = SimpleImputer(strategy='mean')
        data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        return data_imputed

    # Clean the test data
    global nba_data_test
    nba_data_test = Data_clean(nba_data_test)

    # Handle categorical variables
    nba_data_test = Data_handle_categorical(nba_data_test)

    # Handle missing values
    nba_data_test = Data_handle_missing_values(nba_data_test)

    # Scale the features
    scaler = MinMaxScaler()
    nba_data_test_scaled = scaler.fit_transform(nba_data_test)

    # Convert back to DataFrame
    nba_data_test = pd.DataFrame(nba_data_test_scaled, columns=nba_data_test.columns)

    # Save the final test dataset
    nba_data_test.to_csv("datasets/dataset_test.csv", index=False)

best_models = {
    'RandomForest': joblib.load('best_models/random_forest_model.pkl'),
    'CNN': joblib.load('best_models/cnn_model.pkl'),
    'RNN': joblib.load('best_models/rnn_model.pkl'),
    'KNN': joblib.load('best_models/knn_model.pkl'),
    'GNN': joblib.load('best_models/gnn_model.pkl'),
    'SVM': joblib.load('best_models/svm_model.pkl'),
    'LSTM': joblib.load('best_models/lstm_model.pkl'),
    
}

nba_data['Season'] = nba_data['Season'].astype(str)
seasons = nba_data.groupby('Season')

# Initialize a dictionary to store the results
season_results = {}

# Create a new Excel workbook and select the active sheet
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "MVP Predictions"

# Write the header row
header = ["Season", "MVP", "Model", "Top 10 Predictions", "MVP in Top 10"]
ws.append(header)

for season, data in seasons:
    # Find the MVP for the season
    mvp = data[data['mvp_award'] == True]
    
    # Prepare the data for prediction
    data_test = data.drop(['mvp_award'], axis=1)
    
    # Ensure the test data has the same features as the training data
    missing_features = set(train_features) - set(data_test.columns)
    for feature in missing_features:
        data_test[feature] = 0

    # Remove extra features in the test data
    data_test = data_test[train_features]

    # Convert data to float32
    data_test = data_test.astype(np.float32)

    # Predict using the trained models
    predictions = {}
    for model_name, model in best_models.items():
        if model_name in ['RNN', 'LSTM', 'CNN', 'GNN']:
            data_test_expanded = np.expand_dims(data_test.values, axis=-1)  # Add an extra dimension for RNN, LSTM, CNN, GNN
            predictions_array = model.predict(data_test_expanded)
        else:
            predictions_array = model.predict(data_test.values)
        predictions_series = pd.Series(predictions_array.flatten(), index=data_test.index)
        top_10_predictions = predictions_series.sort_values(ascending=False).head(3)
        predictions[model_name] = top_10_predictions

    # Calculate the probability of being MVP
    season_results[season] = {
        'MVP': mvp.index[0],
        'Predictions': {}
    }
    for model_name, top_10 in predictions.items():
        mvp_in_top_10 = mvp.index[0] in top_10.index
        season_results[season]['Predictions'][model_name] = {
            'Top_10': top_10.index.tolist(),
            'MVP_in_Top_10': mvp_in_top_10
        }

    # Write the results to the Excel sheet
    for model_name, prediction in season_results[season]['Predictions'].items():
        row = [
            season,
            season_results[season]['MVP'],
            model_name,
            ", ".join(map(str, prediction['Top_10'])),
            prediction['MVP_in_Top_10']
        ]
        ws.append(row)

# Save the workbook
wb.save("MVP_Predictions_Top3.xlsx")