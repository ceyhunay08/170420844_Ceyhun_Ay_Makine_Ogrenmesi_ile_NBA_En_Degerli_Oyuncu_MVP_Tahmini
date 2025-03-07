from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import tensorflow as tf
import lightgbm as lgb
from tensorflow import keras
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.models import Sequential

import numpy as np

class Model():
    def __init__(self, x_train, y_train, class_weights):
    
                
        self.model_xgb = xgb.XGBClassifier(
            scale_pos_weight=len(y_train) / sum(y_train),
            random_state=42,
            learning_rate=0.1,  # Daha düşük learning rate
            max_depth=6,  # Daha büyük max_depth
            n_estimators=1000,  # Daha fazla n_estimators
            gamma=0.2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.5,
            reg_lambda=1.5,
            verbose=10
        )
        # self.model_stack = StackingClassifier(
#         estimators=[('kn',KNeighborsClassifier(n_neighbors=5,p=3,weights='distance')),
#                     ('svm',SVC(class_weight=class_weights, random_state=42, kernel='rbf', C=1.0)),
#                     ('catboost', CatBoostClassifier(class_weights='balanced', random_state=42, depth=16, iterations=100, learning_rate=0.1,verbose = 10,l2_leaf_reg=3)
#   ), ('rf', RandomForestClassifier(class_weight=class_weights, random_state=42, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=1,verbose=10)
# )],
#         final_estimator=LogisticRegression(solver='liblinear', penalty='l1'),
#         verbose=5
#     )   
        self.model_catboost = CatBoostClassifier(
            class_weights=class_weights,
            random_state=42,
            depth=10,  # Daha küçük depth
            iterations=500,  # Daha fazla iterations
            learning_rate=0.1,  # Daha düşük learning rate
            verbose=10,
            l2_leaf_reg=5
        )  
        
        self.model_gaussian = GaussianNB(var_smoothing=9.748288799431293e-07)
            
        self.model_randomforest = RandomForestClassifier(
            class_weight=class_weights,
            random_state=42,
            max_depth=20,  # Daha büyük max_depth
            min_samples_leaf=1,  # Daha küçük min_samples_leaf
            min_samples_split=10,  # Daha küçük min_samples_split
            n_estimators=1000,  # Daha fazla n_estimators
            verbose=10
        )
        self.model_rnn = Sequential(
             [
            LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            Dropout(0.3),  # Increase dropout to reduce overfitting
            LSTM(50, return_sequences=True),
            Dropout(0.3),
            LSTM(50),
            Dense(1, activation='sigmoid')
            ]
        )
     
      
        self.model_lstm = Sequential(
            [
            LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            Dropout(0.3),  # Increase dropout to reduce overfitting
            LSTM(50, return_sequences=True),
            Dropout(0.3),
            LSTM(50),
            Dense(1, activation='sigmoid')
            ]
        )
        
        # Ensure x_train is reshaped correctly
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        
        
        self.model_cnn = Sequential(
            [
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Dropout(0.4),  # Increase dropout to reduce overfitting
            Conv1D(filters=32, kernel_size=3, activation='relu'),  # Increase filters
            MaxPooling1D(pool_size=2),
            Dropout(0.4),
            Flatten(),
            Dense(32, activation='relu'),  # Increase units
            Dropout(0.4),
            Dense(1, activation='sigmoid')
            ]
        )
        
        self.model_gnn = keras.Sequential(
            [
            keras.layers.GRU(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            Dropout(0.3),  # Increase dropout to reduce overfitting
            keras.layers.GRU(50, return_sequences=True),
            Dropout(0.3),
            keras.layers.GRU(50),
            Dense(1, activation='sigmoid')
            ]
        )
      
        self.model_knn = KNeighborsClassifier(n_neighbors=3,p=3,weights='uniform')
        self.model_svm = SVC(random_state=42, kernel='linear', C=10)
        self.model_logistic = LogisticRegression(class_weight=class_weights, solver='liblinear', penalty='l1')
        
        self.model_ann = Sequential(
            [
            Dense(50, activation='relu', input_shape=(x_train.shape[1],)),
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dense(1, activation='sigmoid')
            ]
        )
       
        self.model_lightgbm = lgb.LGBMClassifier(
            class_weight=class_weights,
            random_state=42,
            max_depth=10,
            min_child_samples=20,
            n_estimators=200,
            num_leaves=31,
            reg_alpha=0.0,
            reg_lambda=0.0,
            verbose=10,
            learning_rate=0.1
        )
        
    def getModels(self):
        return {
            'CatBoost': self.model_catboost,
            # 'GaussianNB': self.model_gaussian,
            'RandomForest': self.model_randomforest,
            'RNN': self.model_rnn,
            'LSTM': self.model_lstm,
            'CNN': self.model_cnn,
            'GNN': self.model_gnn,
            'KNN': self.model_knn,
            'SVM': self.model_svm,
            'ANN': self.model_ann,
            'LightGBM': self.model_lightgbm,
            # 'LogisticRegression': self.model_logistic,
            # 'Stacking': self.model_stack,
            'XGBoost': self.model_xgb
            # '
        }
        
