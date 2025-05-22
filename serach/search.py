from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pickle
from datetime import datetime
import numpy as np
from imblearn.over_sampling import SMOTE

# Define the parameter grid
param_dist = {
    'n_estimators': randint(300, 600),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'gamma': uniform(0, 5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1),
    'scale_pos_weight': uniform(1, 10),  # Adding scale_pos_weight parameter
}

# Initialize the XGBClassifier
xgb = XGBClassifier(eval_metric='auc')

# Perform Bayesian search using RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=500,
    scoring='roc_auc',
    cv=3,
    verbose=3,
    random_state=42,
    n_jobs=-1
)

# Example dataset (replace with your own data)

path='ehr/data/Train-test-dataset_Ver4_median_mode.csv'
data = pd.read_csv(path)
# 分离特征和目标变量
target_column = 'have_stone'
X = data.drop(columns=[target_column])
Y = data[target_column]
Y = Y.astype('int')  # 确保标签是整数类型
# 划分训练集和测试集
test_size = 0.2
random_state = 42
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_size, random_state=random_state)
# 应用SMOTE
# X_train_resampled, Y_train_resampled = apply_smote(X_train, Y_train)
# if 1:
#     return X_train_resampled, X_test, Y_train_resampled, Y_test
# return X_train, X_test, Y_train, Y_test
# X_train, X_test, y_train, y_test = train_test_split(data, test_size=0.2, random_state=42)

# Fit the model
random_search.fit(X_train, Y_train)

# Output the best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)