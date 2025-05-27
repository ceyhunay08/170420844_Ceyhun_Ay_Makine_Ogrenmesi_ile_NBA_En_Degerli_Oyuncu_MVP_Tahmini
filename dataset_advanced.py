import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
nba_data = pd.read_csv("datasets/updated_NBA_Data_With_Standing.csv")

# Global variables
columns = []
class_weights = {}
playernames = []
sub = pd.DataFrame()
def Data_prep():
    print("Data preparing...")
    global nba_data, playernames, columns, class_weights,sub

    # Clean the data
    mvps = nba_data.groupby('Season').max('award_share')
    mvps["mvp"] = True

    # Ensure all seasons are included

    # Join the MVP column to original data, fill the rest with False
    nba_data = nba_data.merge(mvps[["award_share", "mvp"]], on=["Season", "award_share"], how="left")
    nba_data["mvp_award"] = nba_data["mvp"].fillna(False)

    # Drop unnecessary columns    
    # Create a new DataFrame to hold the required column
    sub = nba_data[['mvp_award', 'Player', 'Season']]
    
    nba_data = Data_handle_missing_values(nba_data)
    nba_data = Data_handle_categorical(nba_data)
    
    nba_data = Data_clean(nba_data)

    # Handle outliers
    print("Handling outliers...")
    
    nba_data = nba_data.drop(['mvp_award', 'Player', 'Season'], axis=1)
    
    
    nba_data = handle_outliers(nba_data)
    # Handle missing values and encode categorical data
    #merge the data
    nba_data = pd.concat([nba_data,sub], axis=1)
    
    nba_data['mvp_award'] = nba_data['mvp_award'].apply(lambda x: 1 if x > 0.5 else 0)
    
    # Scale the data
    print("Scaling the data...")
    scaler = MinMaxScaler()

    nba_data_scaled = scaler.fit_transform(nba_data.drop('Season', axis=1).drop('Player', axis=1))
    
    # Ensure columns variable has the correct number of column names
    columns = nba_data.drop('Season', axis=1).drop('Player', axis=1).columns
    nba_data_scaled = pd.DataFrame(nba_data_scaled, columns=columns)
    nba_data_scaled['Season'] = nba_data['Season']
    nba_data_scaled['Player'] = nba_data['Player']

    
    nba_data_scaled.to_csv("datasets/dataset_extra.csv", index=False)
    print("Scaled datasets saved as train_scaled.csv and test_scaled.csv")

    # Assign class weights based on the class distribution


def Data_clean(data):

    data = data.drop(["award_share"], axis=1)
    data = data.drop(['mvp'], axis=1)

    global columns
    columns = data.columns

    print("Data cleaning...")

    # Feature importance analizi ile önemli sütunları belirle

    # Sadece sayısal ve kategorik olarak encode edilmiş sütunları al
    feature_cols = data.drop(['mvp_award', 'Player', 'Season'], axis=1, errors='ignore').columns
    X = data[feature_cols]
    y = data['mvp_award']

    # RandomForest ile feature importance hesapla
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_

    # Önemli sütunları belirle (ör: importance > 0.01)
    important_features = [col for col, imp in zip(feature_cols, importances) if imp > 0.01]

    print("Önemli sütunlar:", important_features)

    # Sadece önemli sütunları bırak
    cols_to_keep = important_features + ['mvp_award', 'Player', 'Season']
    data = data[cols_to_keep]
    # # Drop players with less than 10 minutes played
    # data = data.drop(data[data["MP"] < 10].index)

    return data

def handle_outliers(data):
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if column == 'mvp_award':
            continue
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
        data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
    return data

def Data_handle_categorical(data):
    print("Data encoding...")

    # Sadece kategorik sütunları seç
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Kategorik sütunlar varsa, bunlara label encoding uygula
    if len(categorical_columns) > 0:
        for col in categorical_columns:
            data[col] = data[col].astype('category').cat.codes


    return data

def Data_handle_missing_values(data):
    print("Handling missing values...")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Impute missing values
    numeric_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
    data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

    return data


def main():
    Data_prep()

if __name__ == "__main__":
    main()