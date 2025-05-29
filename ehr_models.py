from datetime import datetime
from math import e
import os
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def init_adaBoost(default_parm=True):
    """
    This function initializes the AdaBoost classifier model.
    """
    if default_parm:
        model_adaboost = AdaBoostClassifier()
    else:
        model_adaboost = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
    return model_adaboost


def init_decisionTree(default_parm=True):
    """
    This function initializes the Decision Tree classifier model.
    """
    if default_parm:
        model_decisionTree = DecisionTreeClassifier()
    else:
        model_decisionTree = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2)
    return model_decisionTree


def init_gaussianNB(default_parm=True):
    """
    This function initializes the Gaussian Naive Bayes classifier model.
    """
    if default_parm:
        model_gaussianNB = GaussianNB()
    else:
        model_gaussianNB = GaussianNB(var_smoothing=1e-9)
    return model_gaussianNB


def init_gradientBoosting(default_parm=True):
    """
    This function initializes the Gradient Boosting classifier model.
    """
    if default_parm:
        model_gradientBoosting = GradientBoostingClassifier()
    else:
        model_gradientBoosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    return model_gradientBoosting


def init_lightGBM(default_parm=True):
    """
    This function initializes the LightGBM classifier model.
    """
    if default_parm:
        model_lgbm = LGBMClassifier()
    else:
        model_lgbm = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=30, learning_rate=0.1, n_estimators=300)
    return model_lgbm


def init_linearDiscriminantAnalysis(default_parm=True):
    """
    This function initializes the Linear Discriminant Analysis classifier model.
    """
    if default_parm:
        model_linearDiscriminantAnalysis = LinearDiscriminantAnalysis()
    else:
        model_linearDiscriminantAnalysis = LinearDiscriminantAnalysis(solver='svd', shrinkage=None)
    return model_linearDiscriminantAnalysis


def init_logisticRegression(default_parm=True):
    """
    This function initializes the Logistic Regression classifier model.
    """
    if default_parm:
        model_logisticRegression = LogisticRegression()
    else:
        model_logisticRegression = LogisticRegression(solver='lbfgs', max_iter=100, random_state=42)
    return model_logisticRegression


def init_MLPClassifier(default_parm=True):
    """
    This function initializes the Multi-layer Perceptron classifier model.
    """
    if default_parm:
        model_MLPClassifier = MLPClassifier()
    else:
        model_MLPClassifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200)
    return model_MLPClassifier


def init_randomForest(default_parm=True):
    """
    This function initializes the Random Forest classifier model.
    """
    if default_parm:
        model_randomForest = RandomForestClassifier()
    else:
        model_randomForest = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    return model_randomForest


def init_XGBoost(default_parm=True):
    """
    This function initializes the XGBoost classifier model.
    """
    if default_parm:
        model_XGBoost = XGBClassifier()
    else:
        model_XGBoost = XGBClassifier(n_estimators=500,
                                      max_depth=8,
                                      learning_rate=0.01,
                                      eval_metric='auc',
                                      device='gpu',
                                      subsample=0.2,
                                      colsample_bytree=0.5,
                                      reg_lambda=0.9953311090514885,
                                      reg_alpha=0.4581043332068245,
                                      scale_pos_weight=3.0)

    return model_XGBoost


def save_model(model, model_name, method='default', auc=None):
    """
    This function saves the trained model to a file.
    """
    method_dir = os.path.join('trained_models', method)
    os.makedirs(method_dir, exist_ok=True)
    # 保存模型
    model_path = os.path.join(method_dir, f'{model_name}_{auc:0.2f}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
