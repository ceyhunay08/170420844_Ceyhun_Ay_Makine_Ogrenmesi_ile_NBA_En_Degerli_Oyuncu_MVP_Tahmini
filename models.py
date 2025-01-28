
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential


class Model():
    def __init__(self,x_train,y_train,class_weights):
#         self.model_xgb = xgb.XGBClassifier(scale_pos_weight=len(y_train) / sum(y_train), random_state=42,learning_rate=0.2, max_depth=7, n_estimators=200,gamma=0.2,subsample=0.6,colsample_bytree=0.6,reg_alpha=0,reg_lambda=0.5,verbose=10)
#         self.model_stack = StackingClassifier(
#         estimators=[('kn',KNeighborsClassifier(n_neighbors=5,p=3,weights='distance')),
#                     ('svm',SVC(class_weight=class_weights, random_state=42, kernel='rbf', C=1.0)),
#                     ('catboost', CatBoostClassifier(class_weights='balanced', random_state=42, depth=16, iterations=100, learning_rate=0.1,verbose = 10,l2_leaf_reg=3)
#   ), ('rf', RandomForestClassifier(class_weight=class_weights, random_state=42, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=1,verbose=10)
# )],
#         final_estimator=LogisticRegression(solver='liblinear', penalty='l1'),
#         verbose=5
#     )   
#       self.model_catboost = CatBoostClassifier(class_weights=class_weights, random_state=42, depth=16, iterations=100, learning_rate=0.2,verbose = 10)
        self.model_gaussian = GaussianNB(var_smoothing=9.748288799431293e-07)
        self.model_randomforest = RandomForestClassifier(class_weight=class_weights, random_state=42, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=50,verbose=10)
        
        self.model_rnn = Sequential(
            [
                LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                LSTM(128),
                Dense(1, activation='sigmoid')
            ]
        )
        
        self.model_lstm = Sequential(
            [
            LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(1, activation='sigmoid')
            ]
        )
        self.model_lstm.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        
        self.model_cnn = Sequential(
            [
            Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
            ]
        )
        self.model_gnn = keras.Sequential(
            [
                keras.layers.GRU(128, return_sequences=True, input_shape=(x_train.shape[1],1)),
                keras.layers.GRU(128),
                keras.layers.Dense(1, activation='sigmoid')
            ]
        )
        self.model_knn = KNeighborsClassifier(n_neighbors=3,p=3,weights='uniform')
        self.model_svm = SVC(class_weight=class_weights, random_state=42, kernel='linear', C=1.0)
        self.model_logistic = LogisticRegression(class_weight=class_weights, solver='liblinear', penalty='l1')
        
        
        
    def getModels(self):
        return {
            # 'CatBoost': self.model_catboost,
            # 'GaussianNB': self.model_gaussian,
            'RandomForest': self.model_randomforest,
            'RNN': self.model_rnn,
            'LSTM': self.model_lstm,
            'CNN': self.model_cnn,
            'GNN': self.model_gnn,
            'KNN': self.model_knn,
            'SVM': self.model_svm,
            # 'LogisticRegression': self.model_logistic,
            # 'Stacking': self.model_stack,
            # 'XGBoost': self.model_xgb
            # '
        }
        
# Best params for RandomForest: {'max_depth': 10, 'n_estimators': 50}
# Tuning XGBoost...
# Best params for XGBoost: {'learning_rate': 0.2, 'n_estimators': 200}
# Tuning CatBoost...
# Best params for CatBoost: {'iterations': 100, 'learning_rate': 0.2}
# Tuning KNN...
# Best params for KNN: {'n_neighbors': 3, 'weights': 'uniform'}
# Tuning SVM...
# Best params for SVM: {'C': 1, 'kernel': 'linear'}
#  Best params for RNN: {'epochs': 10, 'batch_size': 32, 'optimizer': 'adam'}
#  Best params for CNN: {'epochs': 20, 'batch_size': 32, 'optimizer': 'sgd'}
#  Best params for ANN: {'epochs': 10, 'batch_size': 32, 'optimizer': 'adam'}
    