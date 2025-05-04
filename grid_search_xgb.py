import logging
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from datetime import datetime
import utils

from XGB import XGBClassifier

# 配置日志记录器 记录超参数
logging.basicConfig(filename='logs/grid_search.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 创建一个捕获标准输出的处理程序
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# 将处理程序添加到日志记录器
logging.getLogger().addHandler(stdout_handler)


def grid_search(X_train, y_train, param_grid, model, name):
    # 创建GridSearchCV对象
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=6, verbose=5)
    grid_search.fit(X_train, y_train)
    logging.info(f"{name}Best parameters found: {grid_search.best_params_}")
    logging.info(f"{name}Best accuracy found: {grid_search.best_score_}")
    return grid_search.best_params_


def Xgbsearch(X_train, y_train):
    # 定义要搜索的参数网格
    param_grid = {
        'max_depth': [15, 20, 25, 30],
        'learning_rate': [0.1, 0.2, 0.3],
        'n_estimators': [100,  200, 300],
        'gamma': [0.25, 0.3],
    }
    # 创建XGBoost分类器
    model = XGBClassifier(objective='binary:logistic',
                          booster='gbtree',
                          min_child_weight=1,
                          subsample=0.8,
                          colsample_bytree=1,
                          reg_alpha=1,
                          reg_lambda=1,
                          seed=0)
    grid_search(X_train, y_train, param_grid, model, "XGB")


def RFsearch(X_train, y_train):
    # 定义要搜索的参数网格
    param_grid = {
        'n_estimators': [150, 500, 800, 1200],
        # 'criterion': ['gini', 'entropy'],
        'max_depth': [15, 25, 30],
        # 'min_samples_leaf': [1,10, 200, 500],
        # 'min_samples_split': [1,  100, 500],
        # 'class_weight': [None, 'balanced']
    }
    # 创建随机森林分类器
    model = RandomForestClassifier()
    grid_search(X_train, y_train, param_grid, model, "RF")


if __name__ == "__main__":
    df = pd.read_excel("./data/all_data.xlsx")
    # df = pd.read_excel("./data/demo_data.xlsx")
    X_train, X_test, y_train, y_test = utils.get_data(df)
    Xgbsearch(X_train, y_train)
    # RFsearch(X_train, y_train)
    # best_params = grid_search(X_train, y_train)
