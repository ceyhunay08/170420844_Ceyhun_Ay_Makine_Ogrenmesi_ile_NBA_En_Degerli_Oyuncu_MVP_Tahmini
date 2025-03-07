import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tensorflow import keras
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam, RMSprop, SGD

# Verify GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load dataset
print("Data splitting...")
nba_data = pd.read_csv('datasets/dataset_scaled.csv')
playernames = nba_data['Player']    
nba_data = nba_data.drop('Player', axis=1)
X = nba_data.drop('mvp_award', axis=1)
y = nba_data['mvp_award']
X = X[X['Season'].between(1997, 2017)]
y = y[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # 80% train, 20% test

# Define class weights for imbalanced dataset
class_weights = {0: 1, 1: len(y_train) / sum(y_train)}

ml_param_grid = {
    'CATBOOST': {'iterations': [100], 'depth': [10], 'learning_rate': [ 0.1]},
    'RANDOM_FOREST': {'n_estimators': [300], 'max_depth': [10], 'min_samples_split': [ 10]},
    'DECISION_FOREST': {'n_estimators': [100], 'max_depth': [10], 'min_samples_split': [10]},
    'KNN': {'n_neighbors': [3], 'weights': ['uniform']},
    'SVM': {'C': [ 10], 'kernel': ['linear']},
    'XGBOOST': {'n_estimators': [ 200], 'learning_rate': [0.1], 'max_depth': [6]},
    'LIGHTGBM': {'n_estimators': [ 200], 'learning_rate': [ 0.1], 'num_leaves': [31]}
}

def get_machine_learning_models():
    return {
        'CATBOOST': CatBoostClassifier(verbose=0, class_weights=class_weights),
        'RANDOM_FOREST': RandomForestClassifier(class_weight=class_weights),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(class_weight=class_weights),
        'XGBOOST': XGBClassifier(scale_pos_weight=class_weights[1]),
        'LIGHTGBM': LGBMClassifier(class_weight=class_weights)
    }

def get_neural_network_models():
    def build_rnn(units, optimizer):
        model = Sequential()
        model.add(Dense(units, input_shape=(X_train.shape[1],), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_lstm(units, optimizer):
        model = Sequential()
        model.add(LSTM(units, input_shape=(X_train.shape[1], 1)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_cnn(filters, kernel_size, optimizer):
        model = Sequential()
        model.add(Conv1D(filters, kernel_size, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_ann(layers, units, optimizer):
        model = Sequential()
        for _ in range(layers):
            model.add(Dense(units, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_gnn(units, optimizer):
        model = Sequential()
        model.add(keras.layers.GRU(units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(keras.layers.GRU(units, return_sequences=True))
        model.add(keras.layers.GRU(units))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    return {
        'RNN': build_rnn,
        'LSTM': build_lstm,
        'CNN': build_cnn,
        'ANN': build_ann,
        'GNN': build_gnn
    }

# Compare predictions with actual MVPs
def compare_predictions_with_actual(model, X_test, y_test, playernames):
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)  # Assuming binary classification
    comparison = pd.DataFrame({'Player': playernames.iloc[X_test.index], 'Actual': y_test.values, 'Predicted': y_pred.flatten()})
    comparison['Correct'] = comparison['Actual'] == comparison['Predicted']
    return comparison

# Define parameter grids for neural network models
nn_param_grid = {
    'RNN': {'units': [50], 'optimizer': ['adam']},
    'LSTM': {'units': [50], 'optimizer': ['adam']},
    'CNN': {'filters': [32], 'kernel_size': [3], 'optimizer': ['adam']},
    'GNN': {'units': [50], 'optimizer': ['adam']},
    'ANN': {'layers': [3], 'units': [50], 'optimizer': ['adam']},
}

# Function to perform grid search for neural network models
def perform_nn_grid_search(build_fn, param_grid):
    best_model = None
    best_score = 0
    best_params = None
    for params in ParameterGrid(param_grid):
        print("Training model with parameters: ", params)
        model = build_fn(**params)
        model.fit(X_train, y_train, epochs=1, verbose=10)
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)
        score = accuracy_score(y_test, y_pred)
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params
    return best_model, best_score, best_params

# Find best neural network model
best_nn_model = None
best_nn_score = 0
best_nn_params = None
for model_name, build_fn in get_neural_network_models().items():
    print(model_name + " model is being trained...")
    model, score, params = perform_nn_grid_search(build_fn, nn_param_grid[model_name])
    if score > best_nn_score:
        best_nn_score = score
        best_nn_model = model
        best_nn_params = params
    print(f"Best params for {model_name}: {params}, Score: {score}")

# Compare predictions with actual MVPs
nn_comparison = compare_predictions_with_actual(best_nn_model, X_test, y_test, playernames)

# Function to perform grid search for machine learning models
def perform_ml_grid_search(model, param_grid):
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_

# Find best machine learning model
best_ml_model = None
best_ml_score = 0
best_ml_params = None
for model_name, model in get_machine_learning_models().items():
    print(model_name + " model is being trained...")
    best_model, score, params = perform_ml_grid_search(model, ml_param_grid[model_name])
    if score > best_ml_score:
        best_ml_score = score
        best_ml_model = best_model
        best_ml_params = params
    print(f"Best params for {model_name}: {params}, Score: {score}")

ml_comparison = compare_predictions_with_actual(best_ml_model, X_test, y_test, playernames)

print("Machine Learning Model Comparison:")
print(ml_comparison)

print("Neural Network Model Comparison:")
print(nn_comparison)

# Save the best models and their parameters
with open('best_models.txt', 'w') as f:
    f.write(f"Best Neural Network Model: {best_nn_model}\n")
    f.write(f"Best Neural Network Params: {best_nn_params}\n")
    f.write(f"Best Neural Network Score: {best_nn_score}\n\n")
    f.write(f"Best Machine Learning Model: {best_ml_model}\n")
    f.write(f"Best Machine Learning Params: {best_ml_params}\n")
    f.write(f"Best Machine Learning Score: {best_ml_score}\n")


