import os
import pickle
from re import X
from typing import Any
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

#对外暴露的函数
__all__ = [
    "init_adaBoost", "init_decisionTree", "init_gaussianNB", "init_gradientBoosting", "init_lightGBM", "init_linearDiscriminantAnalysis", "init_logisticRegression",
    "init_MLPClassifier", "init_randomForest", "init_XGBoost"
]


def init_adaBoost(default_parm: bool = True) -> AdaBoostClassifier:
    """
    用于初始化AdaBoost分类器模型的函数。
    default_parm: 是否使用默认参数，默认为True。
    如果为True，则使用默认参数初始化模型；如果为False，则使用自定义参数初始化模型。
    """
    if default_parm:
        model_adaboost = AdaBoostClassifier()
    else:
        model_adaboost = AdaBoostClassifier(estimator=RandomForestClassifier(max_depth=1, n_jobs=-1))
    return model_adaboost


def init_decisionTree(default_parm: bool = True) -> DecisionTreeClassifier:
    """
    用于初始化决策树分类器模型的函数。
    default_parm: 是否使用默认参数，默认为True。
    如果为True，则使用默认参数初始化模型；如果为False，则使用自定义参数初始化模型。
    """
    if default_parm:
        model_decisionTree = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
    else:
        model_decisionTree = DecisionTreeClassifier(
            class_weight='balanced',
            criterion='gini',
            max_depth=12,
            min_samples_leaf=1,
            min_samples_split=10,
        )
    return model_decisionTree


def init_gaussianNB(default_parm: bool = True) -> GaussianNB:
    """
    用于初始化高斯朴素贝叶斯分类器模型的函数。
    default_parm: 是否使用默认参数，默认为True。
    如果为True，则使用默认参数初始化模型；如果为False，则使用自定义参数初始化模型。
    """
    if default_parm:
        model_gaussianNB = GaussianNB(priors=[0.04, 0.96])
    else:
        model_gaussianNB = GaussianNB(priors=[0.04, 0.96], var_smoothing=1e-11)
    return model_gaussianNB


def init_gradientBoosting(default_parm: bool = True) -> GradientBoostingClassifier:
    """
    用于初始化梯度提升分类器模型的函数。
    default_parm: 是否使用默认参数，默认为True。
    如果为True，则使用默认参数初始化模型；如果为False，则使用自定义参数初始化模型。
    """
    if default_parm:
        model_gradientBoosting = GradientBoostingClassifier()
    else:
        model_gradientBoosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    return model_gradientBoosting


def init_lightGBM(default_parm: bool = True) -> LGBMClassifier:
    """
    用于初始化LightGBM分类器模型的函数。
    default_parm: 是否使用默认参数，默认为True。
    如果为True，则使用默认参数初始化模型；如果为False，则使用自定义参数初始化模型。
    """
    if default_parm:
        model_lgbm = LGBMClassifier(class_weight={0: 1, 1: 30})
    else:
        model_lgbm = LGBMClassifier(boosting_type='gbdt', num_leaves=96, max_depth=3, learning_rate=0.11, n_estimators=400, class_weight={0: 1, 1: 30})
    return model_lgbm


def init_linearDiscriminantAnalysis(default_parm: bool = True) -> LinearDiscriminantAnalysis:
    """
    用于初始化线性判别分析分类器模型的函数。
    default_parm: 是否使用默认参数，默认为True。
    如果为True，则使用默认参数初始化模型；如果为False，则使用自定义参数初始化模型。
    """
    if default_parm:
        model_linearDiscriminantAnalysis = LinearDiscriminantAnalysis(priors=[0.04, 0.96])
    else:
        model_linearDiscriminantAnalysis = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=[0.04, 0.96])
    return model_linearDiscriminantAnalysis


def init_logisticRegression(default_parm: bool = True) -> LogisticRegression:
    """
    用于初始化逻辑回归分类器模型的函数。
    default_parm: 是否使用默认参数，默认为True。
    如果为True，则使用默认参数初始化模型；如果为False，则使用自定义参数初始化模型。
    """
    if default_parm:
        model_logisticRegression = LogisticRegression(class_weight='balanced')
    else:
        model_logisticRegression = LogisticRegression(C=8.07, solver='liblinear', max_iter=100, random_state=42, class_weight='balanced')
    return model_logisticRegression


def init_MLPClassifier(default_parm: bool = True) -> MLPClassifier:
    """
    用于初始化多层感知机分类器模型的函数。
    default_parm: 是否使用默认参数，默认为True。
    如果为True，则使用默认参数初始化模型；如果为False，则使用自定义参数初始化模型。
    """
    if default_parm:
        model_MLPClassifier = MLPClassifier()
    else:
        model_MLPClassifier = MLPClassifier(
            hidden_layer_sizes=(200, 50),
            activation='relu',
            solver='adam',
        )
    return model_MLPClassifier


def init_randomForest(default_parm: bool = True) -> RandomForestClassifier:
    """
    用于初始化随机森林分类器模型的函数。
    default_parm: 是否使用默认参数，默认为True。
    如果为True，则使用默认参数初始化模型；如果为False，则使用自定义参数初始化模型。
    """
    if default_parm:
        model_randomForest = RandomForestClassifier(n_jobs=-1,class_weight='balanced')
    else:
        model_randomForest = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_leaf=5, min_samples_split=10, random_state=42, class_weight='balanced', n_jobs=-1)
    return model_randomForest


def init_XGBoost(default_parm: bool = True,) -> XGBClassifier:
    """
    用于初始化XGBoost分类器模型的函数。
    default_parm: 是否使用默认参数，默认为True。
    如果为True，则使用默认参数初始化模型；如果为False，则使用自定义参数初始化模型。
    """
    if default_parm:
        model_XGBoost = XGBClassifier(scale_pos_weight=30.0,device='gpu',)
    else:
        model_XGBoost = XGBClassifier(n_estimators=500,
                                      max_depth=3,
                                      gamma=3.0,
                                      learning_rate=0.13,
                                      eval_metric='auc',
                                      device='gpu',
                                      min_child_weight=5,
                                      subsample=0.9,
                                      colsample_bytree=1,
                                      random_state=42,
                                      reg_lambda=8,
                                      reg_alpha=0.9,
                                      scale_pos_weight=30.0)

    return model_XGBoost


def save_model(model: Any, model_name: str, auc: float = 0, method: str = 'mean', opt: str = 'opt') -> None:
    """
    model: 需要保存的模型
    model_name: 模型名称
    auc: 模型的AUC值
    method: 数据清洗方法, 默认为'mean'，可选值包括'mean'、'median'、'mode'等
    opt: 优化选项，默认为'opt'，可选值包括'opt'和'default'。分别为优化参数的模型和默认参数的模型
    该函数将模型保存到指定目录下，目录结构为 trained_models_{opt}/{method}/
    """
    method_dir = os.path.join(f'trained_models_{opt}', method)
    os.makedirs(method_dir, exist_ok=True)
    # 保存模型
    model_path = os.path.join(method_dir, f'{model_name}_{auc:0.2f}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
