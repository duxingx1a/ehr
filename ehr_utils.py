import pickle
from typing import Any, List
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score, \
    precision_score, recall_score, accuracy_score, auc
import pandas as pd
from sklearn.utils import resample


def get_train_test(data_path, train_size=0.8, random_state=42) -> List:
    """
    自定义划分函数,将df直接划分为训练集和测试集。
    
    - data_path: 数据文件的路径
    - train_size: 训练集的比例，默认为0.8
    - random_state: 随机种子，默认为42
    
    返回X_train,X_test,Y_train,Y_test。
    """
    df = pd.read_csv(data_path)
    y = df.pop('have_stone')
    x = df
    return train_test_split(x, y, train_size=train_size, random_state=random_state, stratify=y)


def get_prediction_results(model: Any, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    返回模型的预测概率和预测结果。
    
    - model: 训练好的模型
    - X_test: 测试集特征
    
    返回预测概率和预测标签。
    """
    Y_prob = model.predict_proba(X_test)[:, 1]
    Y_pred = model.predict(X_test)
    return Y_prob, Y_pred


def plot_roc_pr_curves(y_true: np.ndarray, y_proba: np.ndarray, model_name: str = 'model', fig_name: str = 'test') -> None:
    """
    绘制ROC曲线和Precision-Recall曲线。
    
    - y_true: 真实标签
    - y_proba: 预测概率
    - model_name: 模型名称，用于保存图像文件
    - fig_name: 图像标题，用于标识数据是训练集还是测试集
    
    返回None，绘制的图像将保存到results_fig/roc_pr_curve_{model_name}.png目录下。
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    #计算指标
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auroc = roc_auc_score(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)

    # 绘制 ROC 曲线
    axs[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.2f})')
    axs[0].plot([0, 1], [0, 1], color='#666666', lw=1, linestyle='--')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title(f'Receiver Operating Characteristic (ROC) Curve from {fig_name}')
    axs[0].legend(loc="lower right")
    axs[0].set_xlim([0, 1])
    axs[0].set_ylim([0, 1])

    # 绘制 PR 曲线
    axs[1].plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {auprc:.2f})')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].set_title(f'Precision-Recall (PR) Curve from {fig_name}')
    axs[1].legend(loc="lower left")
    axs[1].set_xlim([0, 1])
    axs[1].set_ylim([0, 1])

    os.makedirs('results_fig', exist_ok=True)
    plt.savefig(f'results_fig/roc_pr_curve_{model_name}.png')
    plt.show()
    # 打印 AUROC、AUPRC 和 F1 分数
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")


def eval_model(y_true, y_prob, y_pred) -> list:
    """
    评估模型性能，返回各项指标的值包括AUC、准确率、灵敏度、特异度、PPV、NPV和F1分数。
    
    - y_true: 真实标签
    - y_prob: 预测概率
    - y_pred: 预测标签
    
    返回一个包含各项指标的列表。
    """
    # 计算指标
    auc_score = roc_auc_score(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    ppv = precision_score(y_true, y_pred)
    npv = precision_score(y_true, y_pred, pos_label=0)
    f1 = f1_score(y_true, y_pred)
    y_true = np.array(y_true)
    # Bootstrap 计算 AUC 的 95% 置信区间
    n_iterations = 20
    auc_scores = []
    for _ in range(n_iterations):
        # 生成随机索引（有放回抽样）
        indices = resample(np.arange(len(y_true)), n_samples=int(len(y_true) * 0.5), replace=True)
        score = roc_auc_score(y_true[indices], y_prob[indices])
        auc_scores.append(score)

    # 排序后计算置信区间
    auc_scores = sorted(auc_scores)
    auc_ci_low = auc_scores[int(0.025 * len(auc_scores))]
    auc_ci_high = auc_scores[int(0.975 * len(auc_scores))]

    return [auc_score, auc_ci_low, auc_ci_high, accuracy, sensitivity, specificity, ppv, npv, f1]


def test_plot():
    """
    测试绘图函数。
    """
    # 创建一个示例数据集
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=10, n_redundant=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    df['have_stone'] = y
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = get_train_test(df)
    # 训练一个简单的模型
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    plot_roc_pr_curves(y_test, y_proba, model_name='test')


def load_model(path: str) -> Any:
    """
    从指定路径加载模型。
    path: 模型文件的路径
    返回加载的模型对象。
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def get_all_trained_models(path: str) -> List:
    """
    获取所有训练好的模型。
    path: 模型存储的目录路径
    返回一个包含所有加载模型的列表。
    """
    #获取path下的所有模型

    model_files = os.listdir(path)
    model_paths = sorted([os.path.join(path, f) for f in model_files])
    print(f"检测到 {len(model_paths)} 个模型：{model_paths}")

    load_models_list = []
    for model_path in model_paths:
        model = load_model(model_path)
        load_models_list.append(model)
    return load_models_list


if __name__ == "__main__":
    # test_plot()
    get_all_trained_models('trained_models_default/drop')  # 假设模型存储在 'models' 目录中
