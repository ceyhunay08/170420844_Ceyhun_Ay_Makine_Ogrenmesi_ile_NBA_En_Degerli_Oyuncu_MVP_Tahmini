import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras 
from keras import layers, models
from keras.losses import BinaryCrossentropy
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import cross_val_score
import optuna
# Veri setini yükleyin
nba_data = pd.read_csv("NBA_Player_Stats_2.csv")

X_train, X_test, y_train, y_test = None, None, None, None
class_weights = None

models_used = {}
results_list = []

def Data_prep():
    print("Data preparing...")
    global nba_data,class_weights,X_train, X_test, y_train, y_test
    
    # Kategorik sütunları sayısal değerlere dönüştürelim
    nba_data = Data_clean(nba_data)
    nba_data = nba_data.drop(['Season'],axis=1)
    print(nba_data.columns)
    # Fill null and NaN values with appropriate method
    nba_data = nba_data.ffill()
    
    # Calculate correlation matrix
   
def Data_clean(data):
    data = data.drop(['Age'],axis=1)
    data = data.drop(['Rk'],axis=1)
    data = data.drop(['AST'],axis=1)
    data = data.drop(['Pos'],axis=1)
    data = data.drop(['Tm'],axis=1)
    data = data.drop(['BLK'],axis=1)
    data = data.drop(['GS'],axis=1)
    data = data.drop(['3P'],axis=1)
    data = data.drop(['3PA'],axis=1)
    data = data.drop(['3P%'],axis=1)
    data = data.drop(['FT%'],axis=1)
    data = data.drop(['Player'], axis=1)
    
    return data
    
def Data_scale_split():
 
    print("Data splitting...")
    
    global X_train, X_test, y_train, y_test, class_weights,nba_data
    scaler = MinMaxScaler()
    nba_data_scaled = pd.DataFrame(scaler.fit_transform(nba_data), columns=nba_data.columns)


    X = nba_data_scaled.drop('MVP', axis=1)
    
    y = nba_data['MVP']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)


    class_weights = {0: 1, 1: len(y_train) / sum(y_train)}

    correlation_matrix = nba_data.corr()
    
    # Use feature importance algorithms to select features that affect the MVP column
    # Example: Random Forest feature importance
    rf_model = RandomForestClassifier(class_weight=class_weights, random_state=42)
    rf_model.fit(X_train, y_train)
    feature_importance = rf_model.feature_importances_
    
    # Select top features based on importance
    top_features = np.argsort(feature_importance)[::-1][:10]
    
    # Subset the correlation matrix with top features
    correlation_matrix_subset = correlation_matrix.iloc[top_features, top_features]
    
    # Print the correlation matrix subset
    # Plot the correlation matrix subset
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix_subset, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def Train_models(class_weights, X_train, y_train):
    print("Training models...")
    rf_model = RandomForestClassifier(class_weight=class_weights, random_state=42, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100,verbose=10)
    rf_model.fit(X_train, y_train)

    # XGBoost Model
    xgb_model = get_best_xgboost() 
    xgb_model.fit(X_train, y_train)

    # LightGBM Model
    lgb_model = lgb.LGBMClassifier(class_weight=class_weights,force_col_wise=True,learning_rate=0.1, min_child_samples=10, max_depth=10, n_estimators=100, random_state=42,verbose=20)
    lgb_model.fit(X_train, y_train)

    # CatBoost Model
    catboost_model = get_best_catboost()
    catboost_model.fit(X_train, y_train)
    
    # KNN Model
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Logistic Regression Model
    logistic_model = LogisticRegression(solver='liblinear', penalty='l1')
    logistic_model.fit(X_train, y_train)

    # SVM Model
    svm_model = SVC(class_weight=class_weights, random_state=42, kernel='rbf', C=1.0,probability=True)
    svm_model.fit(X_train, y_train)

    # Naive Bayes Model
    nb_model = get_best_naive_bias()
    nb_model.fit(X_train, y_train)

    global models_used
    models_used = {'Random Forest': rf_model, 'XGBoost': xgb_model, 'LightGBM': lgb_model, 'CatBoost': catboost_model,'KNN': knn_model, 'Logistic Regression': logistic_model, 'SVM': svm_model, 'Naive Bayes': nb_model}

def Train_catboost(class_weights,X_train,y_train,X_test,y_test):
    catboost_params_list = [
    { 'random_state': 42, 'depth': 16, 'iterations': 10, 'learning_rate': 0.03,'l2_leaf_reg' :5,'loss_function':'CrossEntropy' ,'verbose': 1},
    { 'random_state': 42, 'depth': 6, 'iterations': 15, 'learning_rate': 0.03,'bootstrap_type':'Bernoulli', 'verbose': 1},
    { 'random_state': 42, 'depth': 12, 'iterations': 15, 'learning_rate': 0.1 ,'verbose': 1},
    { 'random_state': 42, 'depth': 8, 'iterations': 15, 'learning_rate': 0.1 ,'verbose': 1},
    { 'random_state': 42, 'depth': 10, 'iterations': 15, 'learning_rate': 0.1, 'verbose': 1}
]
    
    cat_models_fitted = {}
    
    for i, params in enumerate(catboost_params_list):
        print(f"Model {i} Fitting...")
        catboost_model = CatBoostClassifier(**params)
        catboost_model.fit(X_train, y_train)
        cat_models_fitted[str(i)] = catboost_model
    
    Evaluate_models(cat_models_fitted,X_test,y_test)

def Train_Naive_Bayes(X_train,y_train,X_test,y_test):
    
    def objective(trial):
        var_smoothing = trial.suggest_loguniform('var_smoothing', 1e-9, 1e-3)
        model = GaussianNB(var_smoothing=var_smoothing)
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("Best parameters found: ", study.best_params)
    print("Best accuracy score: ", study.best_value)
    
    best_model = GaussianNB(var_smoothing=study.best_params['var_smoothing'])
    best_model.fit(X_train, y_train)
    accuracy = best_model.score(X_test, y_test)
    print("Test accuracy: ", accuracy)
    
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    threshold = 0.9995
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    cf = confusion_matrix(y_test, y_pred)
    print(cf)
    cr = classification_report(y_test, y_pred)
    print(cr)
    
def Train_ensembled_model(X_train,y_train,X_test,y_test):
    kn_model = KNeighborsClassifier(n_neighbors=5,p=3,weights='distance')
    catboost_model = get_best_catboost()
    rf_model = RandomForestClassifier(class_weight=class_weights, random_state=42, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=1,verbose=10)
    svm_model = SVC(class_weight=class_weights, random_state=42, kernel='rbf', C=1.0)
  
    # Stacking model
    stacking_model = StackingClassifier(
        estimators=[('kn',kn_model),('svm',svm_model),('catboost', catboost_model), ('rf', rf_model)],
        final_estimator=LogisticRegression(solver='liblinear', penalty='l1'),
        verbose=5
    )

    # Modeli eğitin
    stacking_model.fit(X_train, y_train)

    # Doğrulama seti üzerinde tahmin yapın
    y_pred_prob = stacking_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.3).astype(int)

    # Performansı ölçün
    print("Accuracy:", accuracy_score(y_test, y_pred))  
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
def Find_best_parameters(class_weights,X_train, y_train):
    def grid_search(class_weights, X_train, y_train):
        # Define the parameter grid for each model
        # rf_param_grid = {
        #     'n_estimators': [100, 200, 300],
        #     'max_depth': [None, 5, 10],
        #     'min_samples_split': [2, 5, 10],
        #     'min_samples_leaf': [1, 2, 4]
        # }
        
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
        
        # lgb_param_grid = {
        #     'n_estimators': [100, 200, 300],
        #     'max_depth': [3, 5, 7],
        #     'learning_rate': [0.1, 0.01, 0.001]
        # }
        
        catboost_param_grid = {
            'iterations': [300],
            'depth': [16],
            'learning_rate': [0.1]
        }
        
        # Create a dictionary to store the best parameters for each model
        best_parameters = {}
        
        # Random Forest
        # rf_model = RandomForestClassifier(class_weight=class_weights, random_state=42)
        # rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5)
        # rf_grid_search.fit(X_train, y_train)
        # best_parameters['Random Forest'] = rf_grid_search.best_params_
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(scale_pos_weight=len(y_train) / sum(y_train), random_state=42)
        xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=5, verbose=10)
        xgb_grid_search.fit(X_train, y_train)
        best_parameters['XGBoost'] = xgb_grid_search.best_params_
        
        # # LightGBM
        # lgb_model = lgb.LGBMClassifier(class_weight=class_weights, random_state=42)
        # lgb_grid_search = GridSearchCV(lgb_model, lgb_param_grid, cv=5)
        # lgb_grid_search.fit(X_train, y_train)
        # best_parameters['LightGBM'] = lgb_grid_search.best_params_
        
        # CatBoost
        catboost_model = CatBoostClassifier(class_weights=class_weights, random_state=42, silent=True)
        catboost_grid_search = GridSearchCV(catboost_model, catboost_param_grid, cv=5,verbose=10)
        catboost_grid_search.fit(X_train, y_train)
        best_parameters['CatBoost'] = catboost_grid_search.best_params_
        
        return best_parameters

    best_parameters = grid_search(class_weights,X_train, y_train)
    # Print the best parameters for each model
    for model, parameters in best_parameters.items():
        print(f"\nBest parameters for {model}:")
        print(parameters)

def focal_loss():
    pass
    # # Focal Loss için Keras Modeli
    # def focal_loss(gamma=2., alpha=0.25):
    #     def focal_loss_fixed(y_true, y_pred):
    #         epsilon = tf.keras.backend.epsilon()
    #         y_true = tf.cast(y_true, tf.float32)
    #         y_pred = tf.cast(y_pred, tf.float32)
    #         y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    #         cross_entropy = -y_true * tf.math.log(y_pred)
    #         loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
    #         return tf.reduce_mean(loss, axis=-1)
    #     return focal_loss_fixed

    # focal_loss_fn = focal_loss()

    # model = models.Sequential([
    #     layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    #     layers.Dense(32, activation='relu'),
    #     layers.Dense(1, activation='sigmoid')
    # ])

    # model.compile(optimizer='adam', loss=focal_loss_fn, metrics=['accuracy'])

    # model.fit(X_train, y_train, epochs=20, batch_size=32, class_weight=class_weights, validation_data=(X_test, y_test))

    # models = {'Random Forest': rf_model, 'XGBoost': xgb_model, 'LightGBM': lgb_model, 'CatBoost': catboost_model, 'Focal Loss Model': model}

def Evaluate_models(models_used, X_test, y_test):
    global results_list 
    for name, model in models_used.items():
        # if name == 'Focal Loss Model':
        #     y_prob = model.predict(X_test).ravel()
        # else:
        y_prob = model.predict_proba(X_test)
        
        print(y_prob)
        
        threshold = 0.3
        y_pred_threshold = (y_prob >= threshold).astype(int)
        
        print(f"\n{name} - Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_threshold))
        print(f"{name} - Classification Report:")
        print(classification_report(y_test, y_pred_threshold))
        
        cm = confusion_matrix(y_test, y_pred_threshold)
        

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
        recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        f1 = 2 * (precision * recall) / (precision + recall)
        
        # Append the results to the list
        results_list.append({'Model': name, 'Confusion Matrix': cm, 'Precision': precision, 'Recall': recall, 'F1 Score': f1})
        
def to_excel():
    global results_list
    # Convert the list of results to a DataFrame
    results = pd.DataFrame(results_list)

    # Save the results DataFrame to an Excel file
    results.to_excel('results1.xlsx', index=False)


def standart_run():
    global class_weights, X_train, y_train

    Data_prep()
    Data_scale_split()
    Train_models(class_weights, X_train, y_train)
    Evaluate_models(models_used, X_test, y_test)
    to_excel()


def grid_search_run():
    global class_weights, X_train, y_train
    Data_prep()
    Data_scale_split()
    Find_best_parameters(class_weights, X_train, y_train)
    
    
def cat_boost_run():
    Data_prep()
    Data_scale_split()
    Train_catboost(class_weights,X_train,y_train,X_test,y_test)

def ensemble_run():
    Data_prep()
    Data_scale_split()
    Train_ensembled_model(X_train,y_train,X_test,y_test)


def naive_bayes_run():
    global class_weights, X_train, y_train, X_test, y_test
    Data_prep()
    Data_scale_split()
    Train_Naive_Bayes(X_train,y_train,X_test,y_test)
    
def predict_run():
    nba_data_pred = pd.read_csv("NBA_Player_Stats_22_23.csv")
    player = nba_data_pred['Player']
    nba_data_pred = Data_clean(nba_data_pred)
    nba_data_pred = nba_data_pred.drop(['Player-additional'],axis=1)
    nba_data_pred = nba_data_pred.ffill()
    
    Data_prep()
    Data_scale_split()
    
    global X_train, X_test, y_train, y_test
    
    best_model = get_best_xgboost()
    best_model.fit(X_train, y_train)
    accuracy = best_model.score(X_test, y_test)
    print("Test accuracy: ", accuracy)
    
    nba_data_pred = pd.DataFrame(MinMaxScaler().fit_transform(nba_data_pred), columns=nba_data_pred.columns)
    y_pred_prob = best_model.predict_proba(nba_data_pred)[:, 1]
    
    
    y_pred_prob = MinMaxScaler().fit_transform(y_pred_prob.reshape(-1, 1))
    nba_data_pred['MVP'] = y_pred_prob
    
    nba_data_pred['Player'] = player
    sorted_nba_data = nba_data_pred.sort_values(by='MVP', ascending=False)
    print("Top 10 players with highest probability:")
    print(sorted_nba_data.head(10))
    
    
    real_mvp = sorted_nba_data[sorted_nba_data['Player'] == 'Joel Embiid']
    print(real_mvp)

def get_best_naive_bias():
    return   GaussianNB(var_smoothing=9.748288799431293e-07)

def get_best_catboost():
    global class_weights
    return CatBoostClassifier(class_weights=class_weights, random_state=42, depth=16, iterations=300, learning_rate=0.1,verbose = 10)

def get_best_ensemble():
    kn_model = KNeighborsClassifier(n_neighbors=5,p=3,weights='distance')
    catboost_model = CatBoostClassifier(class_weights='balanced', random_state=42, depth=16, iterations=100, learning_rate=0.1,verbose = 10,l2_leaf_reg=3)
    rf_model = RandomForestClassifier(class_weight=class_weights, random_state=42, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=1,verbose=10)
    svm_model = SVC(class_weight=class_weights, random_state=42, kernel='rbf', C=1.0)
  
    # Stacking model
    stacking_model = StackingClassifier(
        estimators=[('kn',kn_model),('svm',svm_model),('catboost', catboost_model), ('rf', rf_model)],
        final_estimator=LogisticRegression(solver='liblinear', penalty='l1'),
        verbose=5
    )
    return stacking_model

def get_best_xgboost():
    return xgb.XGBClassifier(scale_pos_weight=len(y_train) / sum(y_train), random_state=42,learning_rate=0.1, max_depth=7, n_estimators=300,gamma=0.2,subsample=0.6,colsample_bytree=0.6,reg_alpha=0,reg_lambda=0.5,verbose=10)
predict_run()