from datetime import datetime
import os
import pickle
from pyexpat import model
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

import os
import pickle
import torch
from scipy.stats import sem, t

import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score, \
    precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt


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
        model_XGBoost = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=42)
    return model_XGBoost


def save_model(model, model_name, method):
    """
    This function saves the trained model to a file.
    """
    method_dir = os.path.join('models', method)
    if not os.path.exists(method_dir):
        os.makedirs(method_dir)
    current_time = datetime.now().strftime("%m-%d_%H%M")
    # 保存模型
    model_path = os.path.join(method_dir, f'{model_name}_{current_time}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


class Models():
    """
    This class is used to define custom models for the EHR system.
    """

    def __init__(self):
        self.models_list = []
        self.models_names_list = ("model_adab", "model_dt", "model_gb", "model_gnb", "model_lgbm", "model_lda", "model_lr", "model_mlp", "model_rf", "model_xgb")

        # 初始化所有模型
        self.model_adab = init_adaBoost()
        self.model_dt = init_decisionTree()
        self.model_gb = init_gradientBoosting()
        self.model_gnb = init_gaussianNB()
        self.model_lgbm = init_lightGBM()
        self.model_lda = init_linearDiscriminantAnalysis()
        self.model_lr = init_logisticRegression()
        self.model_mlp = init_MLPClassifier()
        self.model_rf = init_randomForest()
        self.model_xgb = init_XGBoost()

        # 将模型和名称添加到列表
        for name in self.models_names_list:
            # 从 self 的属性中获取模型
            model = getattr(self, name)
            self.models_list.append(model)

    def train_all_models(self, X_train, Y_train, method):
        """
        This method trains all models.
        """
        for model in tqdm(self.models_list):
            model.fit(X_train, Y_train)
            save_model(model, model.__class__.__name__, method)

    def train_model(self, model, X_train, Y_train):
        """
        This method trains a specific model.
        """
        model.fit(X_train, Y_train)

    def custom_model(self):
        """
        This method defines a custom model.
        """
        pass

    def get_model(self, model_name):
        """
        This method retrieves the custom model.
        """
        return self.models_list[model_name]


def calculate_metrics(model, X_test, Y_test) -> list:
    # 预测概率
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(Y_test, y_pred_prob)
    accuracy = accuracy_score(Y_test, model.predict(X_test))
    sensitivity = recall_score(Y_test, model.predict(X_test))
    specificity = recall_score(Y_test, model.predict(X_test), pos_label=0)
    ppv = precision_score(Y_test, model.predict(X_test))
    npv = precision_score(Y_test, model.predict(X_test), pos_label=0)
    f1 = f1_score(Y_test, model.predict(X_test))
    # 计算95%置信区间
    auc_se = sem(y_pred_prob)
    # 计算t分布的临界值，自由度为n-2（n是样本数量）
    n = len(Y_test)
    t_critical = t.ppf((1 + 0.95) / 2, n - 2)
    # 计算AUC的95%置信区间
    auc_ci_low = auc_score - t_critical * auc_se
    auc_ci_high = auc_score + t_critical * auc_se

    return [auc_score, auc_ci_low, auc_ci_high, accuracy, sensitivity, specificity, ppv, npv, f1]


class Models_trained():

    def __init__(self,path='models'):
        self.models_dict = {}  # 字典，用于存储方法名和对应的模型列表
        # 遍历 models 文件夹及其所有子文件夹
        for root, dirs, files in os.walk(path):
            for filename in files:
                # 检查文件是否为 .pkl 文件
                if filename.endswith('.pkl'):
                    # 构建完整的文件路径
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, 'rb') as file:
                            model = pickle.load(file)
                            # 根据文件所在的子目录确定方法
                            method = os.path.basename(root)  # 获取子目录名作为方法名
                            # 将模型添加到对应方法的列表中
                            if method in self.models_dict:
                                self.models_dict[method].append(model)
                            else:
                                self.models_dict[method] = [model]
                    except Exception as e:
                        print(f"Error loading model from {filepath}: {e}")
            print(self.models_dict)

    def eval(self, model, X_test, Y_test):
        """
        This method evaluates the trained models.
        """
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if model is None:
            print(self.models_dict)
            # 如果没有指定模型，则对所有模型进行评估
            for method in tqdm(sorted(self.models_dict.keys())):  # 对方法进行排序
                print(f"Evaluating method: {method}")
                for model in sorted(self.models_dict[method], key=lambda x: x.__class__.__name__):  # 对模型按名称排序
                    # 评估每个模型
                    # print(f"Evaluating model: {model.__class__.__name__}")
                    metrics = calculate_metrics(model, X_test, Y_test)
                    # 构建 DataFrame 行
                    metrics_names = ['AUC', 'AUC_CI_Low', 'AUC_CI_High', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1']
                    model_name = model.__class__.__name__
                    row = [method, model_name] + metrics
                    # 收集所有结果
                    if not hasattr(self, 'all_results'):
                        self.all_results = []
                    self.all_results.append(row)
            # 所有模型评估完后，保存到一个总表格
            if hasattr(self, 'all_results'):
                all_metrics_names = ['Method', 'Model'] + metrics_names
                df = pd.DataFrame(self.all_results, columns=all_metrics_names)
                results_path = os.path.join(results_dir, 'all_metrics.xlsx')
                df.to_excel(results_path, index=False)
        else:
            metrics = calculate_metrics(model, X_test, Y_test)
            # 保存指标到 results 目录下的 Excel 表
            results_path = os.path.join(results_dir, f'metrics_{model.__class__.__name__}.xlsx')
            # 构建 DataFrame
            metrics_names = ['AUC', 'AUC_CI_Low', 'AUC_CI_High', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1']
            df = pd.DataFrame([metrics], columns=metrics_names)

            # 如果文件已存在则追加，否则新建
            if os.path.exists(results_path):
                with pd.ExcelWriter(results_path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
            else:
                df.to_excel(results_path, index=False)
            return metrics


if __name__ == "__main__":
    # 测试 Models 类
    # 这里可以根据需要添加测试代码
    models = Models()
    print(models.get_models())
    print(models.models_names_list)
