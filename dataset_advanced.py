import pandas as pd
import numpy as np

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
from sklearn.model_selection import GridSearchCV, KFold
import seaborn as sns
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split,StratifiedKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error,classification_report, accuracy_score
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
from sklearn.metrics import classification_report

#Process sürelerini tutmak için kod ekle , Hyperparameter tuning yap. Giriş Literatür.


nba_data = pd.read_csv("NBA_Regular_Season_Player_Stats_With_Clutch_Score.csv")

columns = []

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


def Data_split():
    print("Data splitting...")
    nba_data = pd.read_csv('dataset.csv')
    X = nba_data.drop('mvp_award', axis=1)
    y = nba_data['mvp_award']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y) ## 80 % train, 20 % test
    return X_train, X_test, y_train, y_test

def Model_train():
    X_train, X_test, y_train, y_test = Data_split()
    # Classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred_class = classifier.predict(X_test)
    classifier_accuracy = classifier.score(X_test, y_test)
    classifier_mae = mean_absolute_error(y_test, y_pred_class)
    classifier_report = classification_report(y_test, y_pred_class, output_dict=True)
    
    # Regression
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred_reg = regressor.predict(X_test)
    regressor_r2 = r2_score(y_test, y_pred_reg)
    regressor_mse = mean_squared_error(y_test, y_pred_reg)
    regressor_mae = mean_absolute_error(y_test, y_pred_reg)
    regressor_mae2 = np.mean(np.abs(y_test - y_pred_reg)**2)
    regressor_mape = np.mean(np.abs((y_test - y_pred_reg) / y_test)) * 100
    regressor_median_ae = np.median(np.abs(y_test - y_pred_reg))
    
    # RNN
    model_rnn = keras.Sequential()
    model_rnn.add(layers.SimpleRNN(50, activation='relu', input_shape=(X_train.shape[1], 1)))
    model_rnn.add(layers.Dense(1, activation='sigmoid'))
    model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_rnn.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=32)
    y_pred_rnn = model_rnn.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
    y_pred_rnn = (y_pred_rnn > 0.5).astype(int)
    rnn_accuracy = model_rnn.evaluate(X_test.values.reshape(-1, X_test.shape[1], 1), y_test)[1]
    rnn_report = classification_report(y_test, y_pred_rnn, output_dict=True)
    
    # LSTM
    model_lstm = keras.Sequential()
    model_lstm.add(layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
    model_lstm.add(layers.Dense(1, activation='sigmoid'))
    model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_lstm.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=32, class_weight={0: 1, 1: 10})
    y_pred_lstm = model_lstm.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
    y_pred_lstm = (y_pred_lstm > 0.5).astype(int)
    lstm_accuracy = model_lstm.evaluate(X_test.values.reshape(-1, X_test.shape[1], 1), y_test)[1]
    lstm_report = classification_report(y_test, y_pred_lstm, output_dict=True)
    
    # CNN
    model_cnn = keras.Sequential()
    model_cnn.add(layers.Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
    model_cnn.add(layers.MaxPooling1D(pool_size=2))
    model_cnn.add(layers.Flatten())
    model_cnn.add(layers.Dense(50, activation='relu'))
    model_cnn.add(layers.Dense(1, activation='sigmoid'))
    model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_cnn.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=32)
    y_pred_cnn = model_cnn.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
    y_pred_cnn = (y_pred_cnn > 0.5).astype(int)
    cnn_accuracy = model_cnn.evaluate(X_test.values.reshape(-1, X_test.shape[1], 1), y_test)[1]
    cnn_report = classification_report(y_test, y_pred_cnn, output_dict=True)
    
    # Graph Neural Network (GNN)
    class GraphNeuralNetwork(tf.keras.Model):
        def __init__(self, units):
            super(GraphNeuralNetwork, self).__init__()
            self.dense1 = layers.Dense(units, activation='relu')
            self.dense2 = layers.Dense(1, activation='sigmoid')

        def call(self, inputs):
            x = self.dense1(inputs)
            return self.dense2(x)

    gnn_model = GraphNeuralNetwork(50)
    gnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    gnn_model.fit(X_train, y_train, epochs=10, batch_size=32)
    y_pred_gnn = gnn_model.predict(X_test)
    y_pred_gnn = (y_pred_gnn > 0.5).astype(int)
    gnn_accuracy = gnn_model.evaluate(X_test, y_test)[1]
    gnn_report = classification_report(y_test, y_pred_gnn, output_dict=True)

    # Update metrics data
  

    # Save GNN report to CSV

    
    # Save metrics to CSV
    metrics_data = {
        "Model": ["RandomForestClassifier", "RandomForestRegressor", "RNN", "LSTM", "CNN", "GNN"],
        "Accuracy": [classifier_accuracy, None, rnn_accuracy, lstm_accuracy, cnn_accuracy, gnn_accuracy],
        "MAE": [classifier_mae, regressor_mae, None, None, None, None],
        "R2 Score": [None, regressor_r2, None, None, None, None],
        "MSE": [None, regressor_mse, None, None, None, None],
        "MAE2": [None, regressor_mae2, None, None, None, None],
        "MAPE": [None, regressor_mape, None, None, None, None],
        "Median AE": [None, regressor_median_ae, None, None, None, None]
    }
    
    
    metrics_df = pd.DataFrame(metrics_data)
    classifier_report_df = pd.DataFrame(classifier_report).transpose()
    rnn_report_df = pd.DataFrame(rnn_report).transpose()
    lstm_report_df = pd.DataFrame(lstm_report).transpose()
    cnn_report_df = pd.DataFrame(cnn_report).transpose()  
    gnn_report_df = pd.DataFrame(gnn_report).transpose()

    
    
    
    
    # Function to rank models based on their performance
    def rank_models():
        models = {
            "RandomForestClassifier": classifier,
            "RandomForestRegressor": regressor,
            "RNN": model_rnn,
            "LSTM": model_lstm,
            "CNN": model_cnn,
            "GNN": gnn_model
        }

        # Collecting metrics
        metrics = {
            "RandomForestClassifier": {
                "Accuracy": classifier_accuracy,
                "MAE": classifier_mae
            },
            "RandomForestRegressor": {
                "R2 Score": regressor_r2,
                "MSE": regressor_mse,
                "MAE": regressor_mae
            },
            "RNN": {
                "Accuracy": rnn_accuracy
            },
            "LSTM": {
                "Accuracy": lstm_accuracy
            },
            "CNN": {
                "Accuracy": cnn_accuracy
            },
            "GNN": {
                "Accuracy": gnn_accuracy
            }
        }

        # Ranking models based on accuracy for classification and R2 score for regression
        classification_models = {k: v for k, v in metrics.items() if "Accuracy" in v}
        regression_models = {k: v for k, v in metrics.items() if "R2 Score" in v}

        ranked_classification = sorted(classification_models.items(), key=lambda x: x[1]["Accuracy"], reverse=True)
        ranked_regression = sorted(regression_models.items(), key=lambda x: x[1]["R2 Score"], reverse=True)

        print("Ranked Classification Models:")
        for rank, (model, metric) in enumerate(ranked_classification, 1):
            print(f"{rank}. {model} - Accuracy: {metric['Accuracy']}")

        print("\nRanked Regression Models:")
        for rank, (model, metric) in enumerate(ranked_regression, 1):
            print(f"{rank}. {model} - R2 Score: {metric['R2 Score']}")
            
        # Save metrics to Excel
        with pd.ExcelWriter('model_metrics.xlsx') as writer:
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            classifier_report_df.to_excel(writer, sheet_name='Classifier Report')
            rnn_report_df.to_excel(writer, sheet_name='RNN Report')
            lstm_report_df.to_excel(writer, sheet_name='LSTM Report')
            cnn_report_df.to_excel(writer, sheet_name='CNN Report')
            gnn_report_df.to_excel(writer, sheet_name='GNN Report')

    rank_models()


def Model_Train_Cross():
    nba_data = pd.read_csv('dataset.csv')
    X = nba_data.drop('mvp_award', axis=1)
    y = nba_data['mvp_award']
    
    # Stratified K-Fold Cross-Validation
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Random Forest Classifier
    rf_accuracies = []
    rf_reports = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)
        y_pred_rf = classifier.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
        
        rf_accuracies.append(rf_accuracy)
        rf_reports.append(rf_report)
    
    print(f"Random Forest Mean Accuracy: {np.mean(rf_accuracies)}")
    
    # RNN
    rnn_accuracies = []
    rnn_reports = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model_rnn = keras.Sequential()
        model_rnn.add(layers.SimpleRNN(50, activation='relu', input_shape=(X_train.shape[1], 1)))
        model_rnn.add(layers.Dense(1, activation='sigmoid'))
        model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model_rnn.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=32, verbose=0)
        
        y_pred_rnn = model_rnn.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
        y_pred_rnn = (y_pred_rnn > 0.5).astype(int)
        rnn_accuracy = accuracy_score(y_test, y_pred_rnn)
        rnn_report = classification_report(y_test, y_pred_rnn, output_dict=True)
        
        rnn_accuracies.append(rnn_accuracy)
        rnn_reports.append(rnn_report)
    
    print(f"RNN Mean Accuracy: {np.mean(rnn_accuracies)}")
    
    # LSTM
    lstm_accuracies = []
    lstm_reports = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model_lstm = keras.Sequential()
        model_lstm.add(layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
        model_lstm.add(layers.Dense(1, activation='sigmoid'))
        model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model_lstm.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=32, verbose=0)
        
        y_pred_lstm = model_lstm.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
        y_pred_lstm = (y_pred_lstm > 0.5).astype(int)
        lstm_accuracy = accuracy_score(y_test, y_pred_lstm)
        lstm_report = classification_report(y_test, y_pred_lstm, output_dict=True)
        
        lstm_accuracies.append(lstm_accuracy)
        lstm_reports.append(lstm_report)
    
    print(f"LSTM Mean Accuracy: {np.mean(lstm_accuracies)}")
    
    # CNN
    cnn_accuracies = []
    cnn_reports = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model_cnn = keras.Sequential()
        model_cnn.add(layers.Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
        model_cnn.add(layers.MaxPooling1D(pool_size=2))
        model_cnn.add(layers.Flatten())
        model_cnn.add(layers.Dense(50, activation='relu'))
        model_cnn.add(layers.Dense(1, activation='sigmoid'))
        model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model_cnn.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=32, verbose=0)
        
        y_pred_cnn = model_cnn.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
        y_pred_cnn = (y_pred_cnn > 0.5).astype(int)
        cnn_accuracy = accuracy_score(y_test, y_pred_cnn)
        cnn_report = classification_report(y_test, y_pred_cnn, output_dict=True)
        
        cnn_accuracies.append(cnn_accuracy)
        cnn_reports.append(cnn_report)
    
    print(f"CNN Mean Accuracy: {np.mean(cnn_accuracies)}")
    
    # Graph Neural Network (GNN)
    gnn_accuracies = []
    gnn_reports = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        class GraphNeuralNetwork(tf.keras.Model):
            def __init__(self, units):
                super(GraphNeuralNetwork, self).__init__()
                self.dense1 = layers.Dense(units, activation='relu')
                self.dense2 = layers.Dense(1, activation='sigmoid')

            def call(self, inputs):
                x = self.dense1(inputs)
                return self.dense2(x)

        gnn_model = GraphNeuralNetwork(50)
        gnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        gnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        y_pred_gnn = gnn_model.predict(X_test)
        y_pred_gnn = (y_pred_gnn > 0.5).astype(int)
        gnn_accuracy = accuracy_score(y_test, y_pred_gnn)
        gnn_report = classification_report(y_test, y_pred_gnn, output_dict=True)
        
        gnn_accuracies.append(gnn_accuracy)
        gnn_reports.append(gnn_report)
    
    print(f"GNN Mean Accuracy: {np.mean(gnn_accuracies)}")
    
    # Save metrics to Excel
    with pd.ExcelWriter('model_metrics_cross_validation_kfold.xlsx') as writer:
        for i, report in enumerate(rf_reports):
            pd.DataFrame(report).transpose().to_excel(writer, sheet_name=f'RandomForest Report Fold {i+1}')
        for i, report in enumerate(rnn_reports):
            pd.DataFrame(report).transpose().to_excel(writer, sheet_name=f'RNN Report Fold {i+1}')
        for i, report in enumerate(lstm_reports):
            pd.DataFrame(report).transpose().to_excel(writer, sheet_name=f'LSTM Report Fold {i+1}')
        for i, report in enumerate(cnn_reports):
            pd.DataFrame(report).transpose().to_excel(writer, sheet_name=f'CNN Report Fold {i+1}')
        for i, report in enumerate(gnn_reports):
            pd.DataFrame(report).transpose().to_excel(writer, sheet_name=f'GNN Report Fold {i+1}')
    

Model_Train_Cross()