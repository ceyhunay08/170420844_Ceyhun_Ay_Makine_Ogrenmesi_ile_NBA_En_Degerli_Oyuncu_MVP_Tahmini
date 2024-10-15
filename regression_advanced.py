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


# Load the dataset
nba_data_regular = pd.read_csv("NBA_Player_Stats_82_22.csv")

nba_data_playoff = pd.read_csv("NBA_Player_Stats_Playoff_82_22.csv")


nba_data = pd.concat([nba_data_regular,nba_data_playoff],axis=0)


X_train, X_test, y_train, y_test = None, None, None, None
importance = None
models_used = {}
results_list = []

def Data_prep():
    print("Data preparing...")
    global nba_data, X_train, X_test, y_train, y_test,importance
    
    # Clean the data
    mvps = nba_data.groupby('Season').max('award_share')

# add MVP column to mvps dataframe
    mvps["mvp"] = True

    # join the MVP column to original data, fill the rest with False
    nba_data = nba_data.merge(mvps[["award_share", "mvp"]], on = ["Season", "award_share"], how = "left")
    nba_data["mvp_award"] = nba_data["mvp"].fillna(False)
    
#     nba_data['MVP'] = (
#     nba_data['PTS'] +
#     (0.4 * nba_data['FG']) -
#     (0.7 * nba_data['FGA']) +
#     (0.5 * nba_data['TRB']) +
#     nba_data['STL'] +
#     (0.7 * nba_data['AST']) +
#     (0.7 * nba_data['BLK']) +
#     (0.5 * nba_data['PF']) -
#     nba_data['TOV']
# )
    nba_data = Data_clean(nba_data)
    nba_data = nba_data.drop(['Season'],axis=1)
    nba_data = nba_data.drop(['mvp'],axis=1)
    
    # Fill null and NaN values with appropriate method
    nba_data = Data_handle_categorical(nba_data)
    nba_data = Data_handle_missing_values(nba_data)
    
    
    mvp_counts = nba_data['mvp_award'].value_counts()
    plt.pie(mvp_counts, labels=mvp_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of MVP Award')
    plt.show()
    

def Data_clean(data):
    
    data = data.drop(['Player'], axis=1)
    data = data.drop(["award_share"],axis=1)
    data = data.drop(['mov'],axis=1)
    data = data.drop(['mov_adj'],axis=1)
    data = data.drop(['win_loss_pct'],axis=1)
    
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
def Data_scale_split():
    
    print("Data splitting...")
    
    global X_train, X_test, y_train, y_test, nba_data
    scaler = MinMaxScaler()
    nba_data_scaled = pd.DataFrame(scaler.fit_transform(nba_data), columns=nba_data.columns)

    X = nba_data_scaled.drop('mvp_award', axis=1)
    
    y = nba_data_scaled['mvp_award']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    correlation_matrix = nba_data.corr()
    
    plt.figure(figsize=(16, 14))  # Increase the figure size
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', annot_kws={"size": 10})  # Adjust font size
    plt.title('Correlation Matrix', fontsize=20)  # Increase title font size
    plt.xticks(fontsize=12)  # Increase x-axis font size
    plt.yticks(fontsize=12) 
    plt.show()

    
def Train_models(X_train, y_train):
    print("Training models...")
    rf_model = RandomForestRegressor(random_state=42, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100,verbose=10)
    rf_model.fit(X_train, y_train)

    
    
    # XGBoost Model
    xgb_model = get_best_xgboost() 
    xgb_model.fit(X_train, y_train)

    # LightGBM Model
    lgb_model = lgb.LGBMRegressor(force_col_wise=True,learning_rate=0.1, min_child_samples=10, max_depth=10, n_estimators=100, random_state=42,verbose=20)
    lgb_model.fit(X_train, y_train)

    # CatBoost Model
    catboost_model = get_best_catboost()
    catboost_model.fit(X_train, y_train)
    
    # KNN Model
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Linear Regression Model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # SVM Model
    svm_model = SVR(kernel='rbf', C=1.0)
    svm_model.fit(X_train, y_train)

    global models_used
    models_used = {'Random Forest': rf_model, 'XGBoost': xgb_model, 'LightGBM': lgb_model, 'CatBoost': catboost_model,'KNN': knn_model, 'Linear Regression': linear_model, 'SVM': svm_model}

def Train_catboost(X_train,y_train,X_test,y_test):
    catboost_params_list = [
    { 'random_state': 42, 'depth': 16, 'iterations': 10, 'learning_rate': 0.03,'l2_leaf_reg' :5,'loss_function':'RMSE' ,'verbose': 1},
    { 'random_state': 42, 'depth': 6, 'iterations': 15, 'learning_rate': 0.03,'bootstrap_type':'Bernoulli', 'verbose': 1},
    { 'random_state': 42, 'depth': 12, 'iterations': 15, 'learning_rate': 0.1 ,'verbose': 1},
    { 'random_state': 42, 'depth': 8, 'iterations': 15, 'learning_rate': 0.1 ,'verbose': 1},
    { 'random_state': 42, 'depth': 10, 'iterations': 15, 'learning_rate': 0.1, 'verbose': 1}
]
    
    cat_models_fitted = {}
    
    for i, params in enumerate(catboost_params_list):
        print(f"Model {i} Fitting...")
        catboost_model = CatBoostRegressor(**params)
        catboost_model.fit(X_train, y_train)
        cat_models_fitted[str(i)] = catboost_model
    
    Evaluate_models(cat_models_fitted,X_test,y_test)


def Optuna_catboost(X_train, y_train, X_test, y_test):
    def objective(trial):
        params = {
            'random_state': 42,
            'depth': trial.suggest_int('depth', 1, 16),
            'iterations': trial.suggest_int('iterations', 10, 100),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
            'loss_function': 'RMSE',
            'verbose': 0
        }
        
        catboost_model = CatBoostRegressor(**params)
        catboost_model.fit(X_train, y_train)
        
        y_pred = catboost_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        return mse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    
    best_params = study.best_params
    best_model = CatBoostRegressor(**best_params)
    best_model.fit(X_train, y_train)
    
    return best_model, best_params
def Train_ensemble_model(X_train,y_train,X_test,y_test):
    kn_model = KNeighborsRegressor(n_neighbors=5)
    catboost_model = get_best_catboost()
    rf_model = RandomForestRegressor(random_state=42, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=1,verbose=10)
    svm_model = SVR(kernel='rbf', C=1.0)
  
    # Stacking model
    stacking_model = StackingRegressor(
        estimators=[('kn',kn_model),('svm',svm_model),('catboost', catboost_model), ('rf', rf_model)],
        final_estimator=LinearRegression()
    )

    # Train the model
    stacking_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = stacking_model.predict(X_test)

    # Evaluate the performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)
    
def Find_best_parameters(X_train, y_train):
    def grid_search(X_train, y_train):
        xgb_param_grid = {
            'n_estimators': [300],
            'max_depth': [7],
            'learning_rate': [0.1],
            'gamma': [0.2],
            'subsample': [0.6],
            'colsample_bytree': [1.0],
            'reg_alpha': [0],
            'reg_lambda': [0.5]
        }
        
        catboost_param_grid = {
            'iterations': [300],
            'depth': [16],
            'learning_rate': [0.1]
        }
        
        best_parameters = {}
        
        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5, verbose=10)
        xgb_grid_search.fit(X_train, y_train)
        best_parameters['XGBoost'] = xgb_grid_search.best_params_
        
        catboost_model = CatBoostRegressor(random_state=42, silent=True)
        catboost_grid_search = GridSearchCV(catboost_model, catboost_param_grid, cv=5,verbose=10)
        catboost_grid_search.fit(X_train, y_train)
        best_parameters['CatBoost'] = catboost_grid_search.best_params_
        
        return best_parameters

    best_parameters = grid_search(X_train, y_train)
    for model, parameters in best_parameters.items():
        print(f"\nBest parameters for {model}:")
        print(parameters)

def Evaluate_models(models_used, X_test, y_test):
    global results_list 
    for name, model in models_used.items():
        y_pred = model.predict(X_test)
        
        print(f"\n{name} - Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
        print(f"{name} - R2 Score:", r2_score(y_test, y_pred))
        
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{name} - Actual vs Predicted')
        plt.show()
        
        # Append the results to the list                                                                                                                                                               
        results_list.append({'Model': name, 'Mean Absolute Error': mean_absolute_error(y_test, y_pred), 'R2 Score': r2_score(y_test, y_pred)})
        
def to_excel():
    global results_list
    # Convert the list of results to a DataFrame
    results = pd.DataFrame(results_list)

    # Save the results DataFrame to an Excel file
    results.to_excel('results_regression.xlsx', index=False)

def standart_run():
    global X_test, X_train, y_test, y_train
    Data_prep()
    Data_scale_split()
    Train_models(X_train, y_train)
    Evaluate_models(models_used, X_test, y_test)
    to_excel()


def grid_search_run():
    Data_prep()
    Data_scale_split()
    Find_best_parameters(X_train, y_train)
    
    
def cat_boost_run():
    Data_prep()
    Data_scale_split()
    Train_catboost(X_train, y_train, X_test, y_test)

def ensemble_run():
    Data_prep()
    Data_scale_split()
    Train_ensemble_model(X_train, y_train, X_test, y_test)


def predict_run():
    Data_prep()
    nba_data_pred = pd.read_csv("NBA_Player_Stats_23_24_with_calculations.csv")
    player = nba_data_pred['Player']
    nba_data_pred = Data_clean(nba_data_pred)

    nba_data_pred = nba_data_pred.drop(['Player-additional'], axis=1)
    nba_data_pred = nba_data_pred.drop(['Rk'],axis=1)

    nba_data_pred = Data_handle_missing_values(nba_data_pred)
    
    
    columns = list(nba_data_pred.columns)
    mp_index = columns.index('mp')
    per_index = columns.index('per')
    
    # MP ve PER sütunlarının yerini değiştir
    columns[mp_index], columns[per_index] = columns[per_index], columns[mp_index]
    nba_data_pred = nba_data_pred.reindex(columns=columns)

    nba_data_pred.replace([np.inf, -np.inf], 0, inplace=True)
    
    Data_scale_split()
    
    global X_train, X_test, y_train, y_test
    print("Predicting..."+"*"*50)
    # Calculate feature importance

    
    model = get_best_xgboost()

    model.fit(X_train, y_train)
    y_pred = model.predict(nba_data_pred)
    
    # Scale the y_pred column
    scaler = MinMaxScaler()
    y_pred_scaled = scaler.fit_transform(y_pred.reshape(-1, 1))
    nba_data_pred['MVP'] = y_pred_scaled
    # nba_data_pred['MVP'] = y_pred
    
    nba_data_pred['Player'] = player
    sorted_nba_data = nba_data_pred.sort_values(by='MVP', ascending=False)
 
    print("Top 10 players with highest predicted MVP probability:")
    print(sorted_nba_data.head(10))
    
    real_mvp = sorted_nba_data[sorted_nba_data['Player'] == 'Nikola Jokić']
    print("Real MVP:")
    print(real_mvp)


def perform_rfe(X, y, n_features_to_select=10):
    print("Performing Recursive Feature Elimination...")
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    fit = rfe.fit(X, y)
    print("Num Features: %s" % (fit.n_features_))
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))
    return fit

def rfe_run():
    nba_data_pred = pd.read_csv("NBA_Player_Stats_23_24_with_calculations.csv")
    player = nba_data_pred['Player']
    nba_data_pred = Data_clean(nba_data_pred)
    nba_data_pred = Data_handle_missing_values(nba_data_pred)
    
    # Identify and handle NaN values
    print("Checking for NaN values...")
    print(nba_data_pred.isna().sum())

    # Option 1: Drop rows with NaN values
    # nba_data_pred = nba_data_pred.dropna()

    # Option 2: Fill NaN values with 0
    nba_data_pred = nba_data_pred.fillna(0)

    # Option 3: Fill NaN values with column mean
    # imputer = SimpleImputer(strategy='mean')
    # nba_data_pred = pd.DataFrame(imputer.fit_transform(nba_data_pred), columns=nba_data_pred.columns)

    # Check for infinite values and replace them
    nba_data_pred.replace([np.inf, -np.inf], np.nan, inplace=True)
    nba_data_pred.fillna(0, inplace=True)

    # Check for very large values and cap them
    nba_data_pred = nba_data_pred.applymap(lambda x: np.nan if np.abs(x) > 1e10 else x)
    nba_data_pred.fillna(0, inplace=True)
    Data_prep()
    Data_scale_split()

    global X_train, X_test, y_train, y_test
    print("Predicting..." + "*" * 50)
    # Calculate feature importance
    print(nba_data_pred.columns)

    # Perform RFE
    rfe_fit = perform_rfe(X_train, y_train, n_features_to_select=10)
    selected_features = X_train.columns[rfe_fit.support_]

    # Create a pipeline with an imputer and SVR
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),
        ('svr', SVR(kernel='rbf', C=1.0))
    ])

    pipeline.fit(X_train[selected_features], y_train)
    y_pred = pipeline.predict(nba_data_pred[selected_features])
    
    # Scale the y_pred column
    scaler = MinMaxScaler()
    y_pred_scaled = scaler.fit_transform(y_pred.reshape(-1, 1))
    nba_data_pred['MVP'] = y_pred_scaled
    # nba_data_pred['MVP'] = y_pred
    
    nba_data_pred['Player'] = player
    sorted_nba_data = nba_data_pred.sort_values(by='MVP', ascending=False)
 
    print("Top 10 players with highest predicted MVP probability:")
    print(sorted_nba_data.head(30))
def get_best_catboost():
    
    return CatBoostRegressor(random_state=42, depth=16, iterations=300, learning_rate=0.1,verbose = 10)

def get_best_ensemble():
    kn_model = KNeighborsRegressor(n_neighbors=5)
    catboost_model = get_best_catboost()
    rf_model = RandomForestRegressor(random_state=42, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=1,verbose=10)
    svm_model = SVR(kernel='rbf', C=1.0)
  
    # Stacking model
    stacking_model = StackingRegressor(
        estimators=[('kn',kn_model),('svm',svm_model),('catboost', catboost_model), ('rf', rf_model)],
        final_estimator=LinearRegression()
    )
    return stacking_model

def get_best_xgboost():
    model = xgb.XGBRegressor(random_state=42,learning_rate=0.1, max_depth=7, n_estimators=300,gamma=0.2,subsample=0.6,colsample_bytree=0.6,reg_alpha=0,reg_lambda=0.5,verbose=10)
    return model


def testdata_fix():
    
    nba_data_pred = pd.read_csv("NBA_Player_Stats_23_24.csv")
    
    nba_data_pred['per'] = (nba_data_pred['PTS'] + nba_data_pred['TRB'] + nba_data_pred['AST'] + nba_data_pred['STL'] + nba_data_pred['BLK'] - nba_data_pred['TOV'] - nba_data_pred['PF']) / nba_data_pred['G']
    nba_data_pred['mp'] = nba_data_pred['MP']
    nba_data_pred['ts_pct'] = nba_data_pred['PTS'] / (2 * (nba_data_pred['FGA'] + 0.44 * nba_data_pred['FTA']))
    nba_data_pred['fg3a_per_fga_pct'] = nba_data_pred['3PA'] / nba_data_pred['FGA']
    nba_data_pred['fta_per_fga_pct'] = nba_data_pred['FTA'] / nba_data_pred['FGA']
    nba_data_pred['orb_pct'] = nba_data_pred['ORB'] / (nba_data_pred['ORB'] + nba_data_pred['DRB'])
    nba_data_pred['drb_pct'] = nba_data_pred['DRB'] / (nba_data_pred['ORB'] + nba_data_pred['DRB'])
    nba_data_pred['trb_pct'] = nba_data_pred['TRB'] / (nba_data_pred['ORB'] + nba_data_pred['DRB'])
    nba_data_pred['ast_pct'] = nba_data_pred['AST'] / nba_data_pred['G']
    nba_data_pred['stl_pct'] = nba_data_pred['STL'] / nba_data_pred['G']
    nba_data_pred['blk_pct'] = nba_data_pred['BLK'] / nba_data_pred['G']
    nba_data_pred['tov_pct'] = nba_data_pred['TOV'] / nba_data_pred['G']
    nba_data_pred['usg_pct'] = nba_data_pred['PTS'] / (nba_data_pred['PTS'] + nba_data_pred['TRB'] + nba_data_pred['AST'] + nba_data_pred['STL'] + nba_data_pred['BLK'] - nba_data_pred['TOV'] - nba_data_pred['PF'])
    nba_data_pred['ows'] = nba_data_pred['PTS'] * 0.1
    nba_data_pred['dws'] = nba_data_pred['TRB'] * 0.1
    nba_data_pred['ws'] = nba_data_pred['ows'] + nba_data_pred['dws']
    nba_data_pred['ws_per_48'] = nba_data_pred['ws'] / nba_data_pred['MP']
    nba_data_pred['obpm'] = nba_data_pred['PTS'] / nba_data_pred['MP']
    nba_data_pred['dbpm'] = nba_data_pred['TRB'] / nba_data_pred['MP']
    nba_data_pred['bpm'] = nba_data_pred['obpm'] + nba_data_pred['dbpm']
    nba_data_pred['vorp'] = nba_data_pred['bpm'] * nba_data_pred['MP'] / 100
    nba_data_pred['award_share'] = nba_data_pred['PTS'] / nba_data_pred['G']
    nba_data_pred['mov'] = nba_data_pred['PTS'] - nba_data_pred['TOV']
    nba_data_pred['mov_adj'] = nba_data_pred['mov'] / nba_data_pred['G']
    nba_data_pred['win_loss_pct'] = nba_data_pred['PTS'] / (nba_data_pred['PTS'] + nba_data_pred['TOV'])

    # Displa_predy the DataFrame with the new columns
    

    # Save the DataFrame to a new CSV file
    nba_data_pred.to_csv('NBA_Player_Stats_23_24_with_calculations.csv', index=False)
    


rfe_run()