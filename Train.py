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
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, KFold
import seaborn as sns
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import train_test_split ,cross_val_predict
from sklearn.metrics import mean_absolute_error,classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import time
import matplotlib.pyplot as plt
#imputer for missing values
from sklearn.metrics import classification_report
from models import Model  
import joblib
from tqdm import tqdm

playernames = pd.Series()
def Data_split():
    print("Data splitting...")
    nba_data = pd.read_csv('datasets/dataset_scaled.csv')
    global playernames
    playernames = nba_data['Player']
    nba_data = nba_data.drop('Player', axis=1)
    X = nba_data.drop('mvp_award', axis=1)
    y = nba_data['mvp_award']
    X = X[X['Season'].between(1997, 2017)]
    y = y[X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y) ## 80 % train, 20 % test
    return X_train, X_test, y_train, y_test

def Model_train():
    X_train, X_test, y_train, y_test = Data_split()
    # Reshape input data for RNN
    X_train_rnn = np.expand_dims(X_train, axis=-1)
    X_test_rnn = np.expand_dims(X_test, axis=-1)
    
    # Classifier
    models = Model(x_train=X_train, y_train=y_train, class_weights=None)
    randomForest = models.getModels()['RandomForest']
    cnn = models.getModels()['CNN']
    gnn = models.getModels()['GNN']
    knn = models.getModels()['KNN']
    svm = models.getModels()['SVM']
    ann = models.getModels()['ANN']
    catboost = models.getModels()['CatBoost']
    xgboost  = models.getModels()['XGBoost']
    lightgbm = models.getModels()['LightGBM']
    
    catboost.fit(X_train, y_train)
    catboost_accuracy = catboost.score(X_test, y_test)
    catboost_report = classification_report(y_test, catboost.predict(X_test), output_dict=True)
    # joblib.dump(catboost, 'best_modelc/catboost_model.pkl')
    
    xgboost.fit(X_train, y_train)
    xgboost_accuracy = xgboost.score(X_test, y_test)
    xgboost_report = classification_report(y_test, xgboost.predict(X_test), output_dict=True)
    # joblib.dump(xgboost, 'best_modelc/xgboost_model.pkl')
    
    lightgbm.fit(X_train, y_train)
    lightgbm_accuracy = lightgbm.score(X_test, y_test)
    lightgbm_report = classification_report(y_test, lightgbm.predict(X_test), output_dict=True)
    # joblib.dump(lightgbm, 'best_modelc/lightgbm_model.pkl')
    
    

    randomForest.fit(X_train, y_train)
    y_pred_class = randomForest.predict(X_test)
    randomForest_accuracy = randomForest.score(X_test, y_test)
    randomForest_mae = mean_absolute_error(y_test, y_pred_class)
    randomForest_report = classification_report(y_test, y_pred_class, output_dict=True)
    # Save the Random Forest model
    # joblib.dump(randomForest, 'best_modelc/random_forest_model.pkl')
      
    # CNN
    cnn.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    cnn.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=1, batch_size=32)
    y_pred_cnn = cnn.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
    
    y_pred_cnn = np.zeros_like(y_pred_cnn)
    top_indices = np.argsort(y_pred_cnn, axis=0)[-4:]  # Get indices of top 4 predictions
    y_pred_cnn[top_indices] = 1
    # Identify the real MVPs and predicted MVPs
    real_mvp_indices = y_test[y_test == 1].index
    predicted_mvp_indices = np.where(y_pred_cnn == 1)[0]

    # Map indices to player names
    real_mvps = playernames.iloc[real_mvp_indices]
    predicted_mvps = playernames.iloc[predicted_mvp_indices]

    # Print the results
    print("Real MVPs:")
    print(real_mvps)

    print("\nPredicted MVPs:")
    print(predicted_mvps)
    cnn_accuracy = cnn.evaluate(X_test.values.reshape(-1, X_test.shape[1], 1), y_test)[1]
    
    cnn_report = classification_report(y_test, y_pred_cnn, output_dict=True)
    # cnn.save('best_modelc/cnn_model.h5')
    
    # GNN
    gnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    gnn.fit(X_train, y_train, epochs=100, batch_size=32)
    y_pred_gnn = gnn.predict(X_test)
    
    y_pred_gnn = (y_pred_gnn > 0.5).astype(int)
    gnn_accuracy = gnn.evaluate(X_test, y_test)[1]
    gnn_report = classification_report(y_test, y_pred_gnn, output_dict=True)
    # gnn.save('best_modelc/gnn_model.h5')

    # ANN
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ann.fit(X_train, y_train, epochs=100, batch_size=32)
    ann_accuracy = ann.evaluate(X_test, y_test)[1]
    y_true = ann.predict(X_test)
    y_pred_ann = (y_true > 0.5).astype(int)
    ann_report = classification_report(y_test, y_pred_ann, output_dict=True)
    # ann.save('best_models/ann_model.h5')
    
    knn.fit(X_train, y_train)
    knn_accuracy = knn.score(X_test, y_test)
    knn_report = classification_report(y_test, knn.predict(X_test), output_dict=True)
    # joblib.dump(knn, 'best_modelc/knn_model.pkl')
    
  
    svm = SVC( kernel='linear', probability=True)
    for _ in tqdm(range(1), desc="Fitting SVM"):
        svm.fit(X_train, y_train)
   
    svm_accuracy = svm.score(X_test, y_test)
    svm_report = classification_report(y_test, svm.predict(X_test), output_dict=True)
    # joblib.dump(svm, 'best_modelc/svm_model.pkl')


    

    randomForest_df = pd.DataFrame(randomForest_report).transpose()
    cnn_report_df = pd.DataFrame(cnn_report).transpose()  
    gnn_report_df = pd.DataFrame(gnn_report).transpose()
    knn_report_df = pd.DataFrame(knn_report).transpose()
    svm_report_df = pd.DataFrame(svm_report).transpose()
    ann_report_df = pd.DataFrame(ann_report).transpose()
    catboost_df = pd.DataFrame(catboost_report).transpose()
    xgboost_df = pd.DataFrame(xgboost_report).transpose()
    lightgbm_df = pd.DataFrame(lightgbm_report).transpose()
    
    
    def rank_models():
        models = {
            "RandomForestClassifier": randomForest,
            "CNN": cnn,
            "GNN": gnn,
            "KNN": knn,
            "SVM": svm,
            "ANN": ann,
            "CatBoost": catboost,
            "XGBoost": xgboost,
            "LightGBM": lightgbm
            
        }

        # Collecting metrics
        metrics = {
            "RandomForestClassifier": {
                "Accuracy": randomForest_accuracy,
                "MAE": randomForest_mae
            },
            "CNN": {
                "Accuracy": cnn_accuracy
            },
            "GNN": {
                "Accuracy": gnn_accuracy
            },
            "KNN": {
                "Accuracy": knn_accuracy
            },
            "SVM": {
                "Accuracy": svm_accuracy
            },
            "ANN": {
                "Accuracy": ann_accuracy
            },
            "CatBoost": {
                "Accuracy": catboost_accuracy
            },
            "XGBoost": {
                "Accuracy": xgboost_accuracy
            },
            "LightGBM": {
                "Accuracy": lightgbm_accuracy
            }
        }

        # Ranking models based on accuracy for classification and R2 score for regression
        classification_models = {k: v for k, v in metrics.items() if "Accuracy" in v}
        

        ranked_classification = sorted(classification_models.items(), key=lambda x: float(x[1]["Accuracy"]), reverse=True)

        print("Ranked Classification Models:")
        for rank, (model, metric) in enumerate(ranked_classification, 1):
            print(f"{rank}. {model} - Accuracy: {metric['Accuracy']}")

        # Save metrics to Excel
        with pd.ExcelWriter('results/model_metrics_try3.xlsx') as writer:
            randomForest_df.to_excel(writer, sheet_name='Classifier Report')
            cnn_report_df.to_excel(writer, sheet_name='CNN Report')
            gnn_report_df.to_excel(writer, sheet_name='GNN Report')
            knn_report_df.to_excel(writer, sheet_name='KNN Report')
            svm_report_df.to_excel(writer, sheet_name='SVM Report')
            ann_report_df.to_excel(writer, sheet_name='ANN Report')
            catboost_df.to_excel(writer, sheet_name='CatBoost Report')
            xgboost_df.to_excel(writer, sheet_name='XGBoost Report')
            lightgbm_df.to_excel(writer, sheet_name='LightGBM Report')

    rank_models()

Model_train()


def Model_train_withcross():
    X_train, X_test, y_train, y_test = Data_split()
    # Reshape input data for RNN
    X_train_rnn = np.expand_dims(X_train, axis=-1)
    X_test_rnn = np.expand_dims(X_test, axis=-1)
    
    # Classifier
    models = Model(x_train=X_train, y_train=y_train, class_weights=None)
    randomForest = models.getModels()['RandomForest']
    rnn = models.getModels()['RNN']
    lstm = models.getModels()['LSTM']
    cnn = models.getModels()['CNN']
    gnn = models.getModels()['GNN']
    knn = models.getModels()['KNN']
    svm = models.getModels()['SVM']
    
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    def cross_validate_model(model, X, y):
        accuracies = []
        reports = []
        for train_index, test_index in skf.split(X, y):
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_test_fold)
            accuracy = accuracy_score(y_test_fold, y_pred_fold)
            report = classification_report(y_test_fold, y_pred_fold, output_dict=True)
            accuracies.append(accuracy)
            reports.append(report)
        return np.mean(accuracies), reports
    
    # Random Forest
    randomForest_accuracy, randomForest_reports = cross_validate_model(randomForest, X_train, y_train)
    joblib.dump(randomForest, 'best_modelsc/random_forest_model.pkl')
    
    # RNN
    rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    rnn.fit(X_train_rnn, y_train, epochs=10, batch_size=32)
    rnn_accuracy, rnn_reports = cross_validate_model(rnn, X_train_rnn, y_train)
    joblib.dump(rnn, 'best_modelsc/rnn_model.pkl')
    
    # LSTM
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=32)
    lstm_accuracy, lstm_reports = cross_validate_model(lstm, X_train.values.reshape(-1, X_train.shape[1], 1), y_train)
    joblib.dump(lstm, 'best_modelsc/lstm_model.pkl')
    
    # CNN
    cnn.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    cnn.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=20, batch_size=32)
    cnn_accuracy, cnn_reports = cross_validate_model(cnn, X_train.values.reshape(-1, X_train.shape[1], 1), y_train)
    joblib.dump(cnn, 'best_modelsc/cnn_model.pkl')
    
    # GNN
    gnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    gnn.fit(X_train, y_train, epochs=10, batch_size=32)
    gnn_accuracy, gnn_reports = cross_validate_model(gnn, X_train, y_train)
    joblib.dump(gnn, 'best_modelsc/gnn_model.pkl')
    
    # KNN
    knn_accuracy, knn_reports = cross_validate_model(knn, X_train, y_train)
    joblib.dump(knn, 'best_modelsc/knn_model.pkl')
    
    # SVM
    svm_accuracy, svm_reports = cross_validate_model(svm, X_train, y_train)
    joblib.dump(svm, 'best_modelsc/svm_model.pkl')
    
    # Save metrics to Excel
    with pd.ExcelWriter('crossvalidation/model_metrics_with_cross_validation.xlsx') as writer:
        for i, report in enumerate(randomForest_reports):
            pd.DataFrame(report).transpose().to_excel(writer, sheet_name=f'RandomForest Report Fold {i+1}')
        for i, report in enumerate(rnn_reports):
            pd.DataFrame(report).transpose().to_excel(writer, sheet_name=f'RNN Report Fold {i+1}')
        for i, report in enumerate(lstm_reports):
            pd.DataFrame(report).transpose().to_excel(writer, sheet_name=f'LSTM Report Fold {i+1}')
        for i, report in enumerate(cnn_reports):
            pd.DataFrame(report).transpose().to_excel(writer, sheet_name=f'CNN Report Fold {i+1}')
        for i, report in enumerate(gnn_reports):
            pd.DataFrame(report).transpose().to_excel(writer, sheet_name=f'GNN Report Fold {i+1}')
        for i, report in enumerate(knn_reports):
            pd.DataFrame(report).transpose().to_excel(writer, sheet_name=f'KNN Report Fold {i+1}')
        for i, report in enumerate(svm_reports):
            pd.DataFrame(report).transpose().to_excel(writer, sheet_name=f'SVM Report Fold {i+1}')



def Model_Train_Cross():
    nba_data = pd.read_csv('dataset.csv')
    X = nba_data.drop('mvp_award', axis=1)
    y = nba_data['mvp_award']
    
    # Stratified K-Fold Cross-Validation
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Random Forest Classifier
    rf_accuracies = []
    rf_reports = []
    rf_train_times = []
    rf_pred_times = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        train_time, pred_time, y_pred_rf = measure_time(classifier, X_train, y_train, X_test)
        
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
        
        rf_accuracies.append(rf_accuracy)
        rf_reports.append(rf_report)
        rf_train_times.append(train_time)
        rf_pred_times.append(pred_time)

    print(f"Random Forest Mean Accuracy: {np.mean(rf_accuracies)}")


    rnn_accuracies = []
    rnn_reports = []
    rnn_train_times = []
    rnn_pred_times = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model_rnn = keras.Sequential()
        model_rnn.add(layers.SimpleRNN(50, activation='relu', input_shape=(X_train.shape[1], 1)))
        model_rnn.add(layers.Dense(1, activation='sigmoid'))
        model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        start_train = time.time()
        model_rnn.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=32, verbose=0)
        end_train = time.time()
        
        start_pred = time.time()
        y_pred_rnn = model_rnn.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
        end_pred = time.time()
        
        train_time = end_train - start_train
        pred_time = end_pred - start_pred
        
        y_pred_rnn = (y_pred_rnn > 0.5).astype(int)
        rnn_accuracy = accuracy_score(y_test, y_pred_rnn)
        rnn_report = classification_report(y_test, y_pred_rnn, output_dict=True)
        
        rnn_accuracies.append(rnn_accuracy)
        rnn_reports.append(rnn_report)
        rnn_train_times.append(train_time)
        rnn_pred_times.append(pred_time)

    print(f"RNN Mean Accuracy: {np.mean(rnn_accuracies)}")

    # LSTM
    lstm_accuracies = []
    lstm_reports = []
    lstm_train_times = []
    lstm_pred_times = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model_lstm = keras.Sequential()
        model_lstm.add(layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
        model_lstm.add(layers.Dense(1, activation='sigmoid'))
        model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        start_train = time.time()
        model_lstm.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=32, verbose=0)
        end_train = time.time()
        
        start_pred = time.time()
        y_pred_lstm = model_lstm.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
        end_pred = time.time()
        
        train_time = end_train - start_train
        pred_time = end_pred - start_pred
        
        y_pred_lstm = (y_pred_lstm > 0.5).astype(int)
        lstm_accuracy = accuracy_score(y_test, y_pred_lstm)
        lstm_report = classification_report(y_test, y_pred_lstm, output_dict=True)
        
        lstm_accuracies.append(lstm_accuracy)
        lstm_reports.append(lstm_report)
        lstm_train_times.append(train_time)
        lstm_pred_times.append(pred_time)

    print(f"LSTM Mean Accuracy: {np.mean(lstm_accuracies)}")

    # CNN
    cnn_accuracies = []
    cnn_reports = []
    cnn_train_times = []
    cnn_pred_times = []
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
        
        start_train = time.time()
        model_cnn.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=32, verbose=0)
        end_train = time.time()
        
        start_pred = time.time()
        y_pred_cnn = model_cnn.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
        end_pred = time.time()
        
        train_time = end_train - start_train
        pred_time = end_pred - start_pred
        
        y_pred_cnn = (y_pred_cnn > 0.5).astype(int)
        cnn_accuracy = accuracy_score(y_test, y_pred_cnn)
        cnn_report = classification_report(y_test, y_pred_cnn, output_dict=True)
        
        cnn_accuracies.append(cnn_accuracy)
        cnn_reports.append(cnn_report)
        cnn_train_times.append(train_time)
        cnn_pred_times.append(pred_time)

    print(f"CNN Mean Accuracy: {np.mean(cnn_accuracies)}")

    # Graph Neural Network (GNN)
    gnn_accuracies = []
    gnn_reports = []
    gnn_train_times = []
    gnn_pred_times = []
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
        
        start_train = time.time()
        gnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        end_train = time.time()
        
        start_pred = time.time()
        y_pred_gnn = gnn_model.predict(X_test)
        end_pred = time.time()
        
        train_time = end_train - start_train
        pred_time = end_pred - start_pred
        
        y_pred_gnn = (y_pred_gnn > 0.5).astype(int)
        gnn_accuracy = accuracy_score(y_test, y_pred_gnn)
        gnn_report = classification_report(y_test, y_pred_gnn, output_dict=True)
        
        gnn_accuracies.append(gnn_accuracy)
        gnn_reports.append(gnn_report)
        gnn_train_times.append(train_time)
        gnn_pred_times.append(pred_time)

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
        
        # Save training and prediction times
        times_df = pd.DataFrame({
            'Model': ['RandomForest', 'RNN', 'LSTM', 'CNN', 'GNN'],
            'Mean Train Time': [np.mean(rf_train_times), np.mean(rnn_train_times), np.mean(lstm_train_times), np.mean(cnn_train_times), np.mean(gnn_train_times)],
            'Mean Prediction Time': [np.mean(rf_pred_times), np.mean(rnn_pred_times), np.mean(lstm_pred_times), np.mean(cnn_pred_times), np.mean(gnn_pred_times)]
        })
        times_df.to_excel(writer, sheet_name='Training and Prediction Times')

def plot_and_save_distribution(data, column, filename):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, bins=30, label='All Players')
    sns.histplot(data[data['mvp_award'] == True][column], kde=True, bins=30, color='red', label='MVPs')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'graphics/{filename}.png')
    plt.close()

def augment_data(data):
    data_augmented = data.copy()
    shift_value = np.random.uniform(-0.1, 0.1, data.shape)
    data_augmented.iloc[:, :-1] += shift_value[:, :-1]  # Exclude the last column (mvp_award) from augmentation
    return data_augmented

 

def measure_time(model_name, model, X_train, y_train, X_test, y_test):
    if model_name in ['RNN', 'LSTM', 'CNN', 'GNN']:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Ensure y_train and y_test are binary
    if not np.issubdtype(y_train.dtype, np.integer):
        y_train = (y_train > 0.5).astype(int)
    if not np.issubdtype(y_test.dtype, np.integer):
        y_test = (y_test > 0.5).astype(int)
    
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    
    start_pred = time.time()
    y_pred = model.predict(X_test)
    end_pred = time.time()
    
    train_time = end_train - start_train
    pred_time = end_pred - start_pred
    
    # Convert continuous predictions to binary if necessary
    if model_name in ['RNN', 'LSTM', 'CNN', 'GNN'] or isinstance(model, RandomForestClassifier):
        y_pred = (y_pred > 0.5).astype(int)
    
    return train_time, pred_time, y_pred

class GraphNeuralNetwork(tf.keras.Model):
    def __init__(self, units):
        super(GraphNeuralNetwork, self).__init__()
        self.dense1 = layers.Dense(units, activation='relu')
        self.dense2 = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def Model_train_generic(cross_validate):
    nba_data = pd.read_csv('dataset.csv')
    # nba_data = augment_data(nba_data)
    X = nba_data.drop('mvp_award', axis=1)
    y = nba_data['mvp_award']
    
    models = Model(x_train=X, y_train=y, class_weights=None).getModels()
    
    results = []
    if cross_validate:
        skf = KFold(n_splits=5, shuffle=True, random_state=42)
        for model_name, model in models.items():
            fold = 0
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                # Ensure y_test is binary
                y_test = (y_test > 0.5).astype(int)
                
                train_time, pred_time, y_pred = measure_time(model_name, model, X_train, y_train, X_test, y_test)
                
                # Debugging: Check types of y_test and y_pred
                print(f"Fold {fold} - Model {model_name}")
                print(f"y_test type: {y_test.dtype}, y_pred type: {y_pred.dtype}")
                
                y_pred_binary = (y_pred > 0.5).astype(int)
                accuracy = accuracy_score(y_test, y_pred_binary)
                classification = classification_report(y_test, y_pred_binary, output_dict=True, zero_division=1)
                classification_df = pd.DataFrame(classification).transpose()
                classification_df.to_excel(f'pureexcels/{model_name}_classification_report_fold_{fold}.xlsx', index=True)
                results.append({
                    'Model': model_name,
                    'Fold': fold,
                    'Accuracy': accuracy,
                    'Train Time': train_time,
                    'Prediction Time': pred_time,
                    'Cross-validation': True
                })
                fold += 1
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            train_time, pred_time, y_pred = measure_time(model_name, model, X_train, y_train, X_test, y_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_binary)
            classification = classification_report(y_test, y_pred_binary, output_dict=True, zero_division=1)
            classification_df = pd.DataFrame(classification).transpose()
            classification_df.to_excel(f'pureexcels/{model_name}_classification_report.xlsx', index=True)
            results.append({
                'Model': model_name,
                'Fold': None,
                'Accuracy': accuracy,
                'Train Time': train_time,
                'Prediction Time': pred_time,
                'Cross-validation': False
            })

    results_df = pd.DataFrame(results)
    results_df.to_excel('pureexcels/'+str(cross_validate)+'puremodel_metrics.xlsx', index=False)

    
def hyperparameter_tuning():
        nba_data = pd.read_csv('dataset.csv')
        X = nba_data.drop('mvp_award', axis=1)
        y = nba_data['mvp_award']

        models = {
            'RandomForest': RandomForestClassifier(),
            'XGBoost': xgb.XGBClassifier(),
            'CatBoost': CatBoostClassifier(verbose=0),
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(probability=True),
            'GaussianNB': GaussianNB(),
            'LogisticRegression': LogisticRegression(),
            'RNN': keras.Sequential([
            layers.SimpleRNN(128, activation='relu', input_shape=(X.shape[1], 1)),
            layers.Dense(1, activation='sigmoid')
            ]),
            'CNN': keras.Sequential([
            layers.Conv1D(128, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(64, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
            ]),
            'ANN': keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
            ]),
            'GNN': GraphNeuralNetwork(128),
            'Stacking': StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier()),
                ('xgb', xgb.XGBClassifier()),
                ('cat', CatBoostClassifier(verbose=0))
            ],
            final_estimator=LogisticRegression()
            )
        }

        param_grids = {
            'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
            'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
            'CatBoost': {'iterations': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
            'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
            'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            'GaussianNB': {},
            'LogisticRegression': {'C': [0.1, 1, 10]},
            'RNN': {
            'epochs': [10, 20],
            'batch_size': [32, 64],
            'optimizer': ['adam', 'sgd', 'rmsprop']
            },
            'CNN': {
            'epochs': [10, 20],
            'batch_size': [32, 64],
            'optimizer': ['adam', 'sgd', 'rmsprop']
            },
            'ANN': {
            'epochs': [10, 20],
            'batch_size': [32, 64],
            'optimizer': ['adam', 'sgd', 'rmsprop']
            },
            'GNN': {
            'epochs': [10, 20],
            'batch_size': [32, 64],
            'optimizer': ['adam', 'sgd', 'rmsprop']
            },
            'Stacking': {}
        }

        best_params = {}
        for model_name, model in models.items():
            if model_name in ['RNN', 'CNN', 'ANN', 'GNN']:
               
                for model_name in ['RNN', 'CNN', 'ANN', 'GNN']:
                    print(f"Tuning {model_name}...")
                    best_accuracy = 0
                    best_params[model_name] = {}
                    for epochs in param_grids[model_name]['epochs']:
                        for batch_size in param_grids[model_name]['batch_size']:
                            for optimizer in param_grids[model_name]['optimizer']:
                                model = models[model_name]
                                model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
                                skf = KFold(n_splits=5, shuffle=True, random_state=42)
                                accuracies = []
                                for train_index, test_index in skf.split(X, y):
                                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                                    model.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                                    y_pred = model.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
                                    y_pred = (y_pred > 0.5).astype(int)
                                    accuracy = accuracy_score(y_test, y_pred)
                                    accuracies.append(accuracy)
                                mean_accuracy = np.mean(accuracies)
                                if mean_accuracy > best_accuracy:
                                    best_accuracy = mean_accuracy
                                    best_params[model_name] = {'epochs': epochs, 'batch_size': batch_size, 'optimizer': optimizer}
                    print(f"Best params for {model_name}: {best_params[model_name]}")
            print(f"Tuning {model_name}...")
            grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X, y)
            best_params[model_name] = grid_search.best_params_
            print(f"Best params for {model_name}: {grid_search.best_params_}")

        # Save best parameters
        with open('best_params.json', 'w') as f:
            json.dump(best_params, f)

        # Train and evaluate models with best parameters
        results = []
        skf = KFold(n_splits=5, shuffle=True, random_state=42)
        for model_name, model in models.items():
            if model_name in best_params:
                model.set_params(**best_params[model_name])
            fold = 0
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                if model_name in ['RNN', 'CNN', 'ANN', 'GNN']:
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    model.fit(X_train.values.reshape(-1, X_train.shape[1], 1), y_train, epochs=10, batch_size=32, verbose=0)
                    y_pred = model.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
                    y_pred = (y_pred > 0.5).astype(int)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                classification = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
                classification_df = pd.DataFrame(classification).transpose()
                classification_df.to_excel(f'best_models/{model_name}_classification_report_fold_{fold}.xlsx', index=True)
                results.append({
                    'Model': model_name,
                    'Fold': fold,
                    'Accuracy': accuracy,
                    'Cross-validation': True
                })
                fold += 1

        results_df = pd.DataFrame(results)
        results_df.to_excel('best_models/model_metrics.xlsx', index=False) 