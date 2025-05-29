from typing import List
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV

def search_params(param_space, model, X_train, Y_train):
    """
    使用贝叶斯优化搜索超参数
    :param param_space: 超参数空间
    :param model: 机器学习模型
    :param X_train: 训练特征
    :param Y_train: 训练标签
    :return: 最佳参数和最佳分数
    """
    # 初始化贝叶斯优化器
    opt_random_forest = BayesSearchCV(model, param_space, n_iter=50, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1, verbose=10)
    # 拟合贝叶斯优化器
    opt_random_forest.fit(X_train, Y_train)
    # 输出最佳参数
    print("Best parameters for Random Forest: ", opt_random_forest.best_params_) # type: ignore
    print("Best cross-validation score: ", opt_random_forest.best_score_) # type: ignore
    
def get_train_test(df, name='roc_pr', train_size=0.8, random_state=42) -> List:
    """
    自定义划分函数,将df直接划分为训练集和测试集。
    """
    y = df.pop('have_stone')
    x = df
    return train_test_split(x, y, train_size=train_size, random_state=random_state, stratify=y)