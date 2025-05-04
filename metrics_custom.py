import os
import pickle
import torch
from scipy.stats import sem, t
from tqdm import tqdm

# from CNN import CNN
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score, \
    precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

# import utils


def plot_roc_pr_curves(y_true, y_proba, predicted):
    # 计算 ROC 曲线和 AUROC
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auroc = roc_auc_score(y_true, y_proba)

    # 计算 PR 曲线和 AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)

    # 计算 F1 分数
    f1 = f1_score(y_true, predicted)

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # 绘制 PR 曲线
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {auprc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc="lower left")
    plt.show()

    # 打印 AUROC、AUPRC 和 F1 分数
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1 Score: {f1:.4f}")


def show(model, deepmodel=False):
    df = pd.read_excel("./data/all_data.xlsx")
    # df = pd.read_excel("./data/demo_data.xlsx")
    X_train, X_test, y_train, Y_test = utils.get_data(df)

    # 深度模型用model()
    if deepmodel:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 将 NumPy 数组转换为 PyTorch 张量
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
        Y_test_numpy = Y_test.to_numpy()
        Y_test_tensor = torch.tensor(Y_test_numpy, dtype=torch.float32).view(-1, 1).to(device)
        with torch.no_grad():
            outputs = model(X_test_tensor)
            predicted = (outputs > 0.5).float()
        plot_roc_pr_curves(Y_test_tensor.cpu().numpy(), outputs.cpu().numpy(), predicted.cpu().numpy())
    # 预测测试集
    else:
        y_proba = model.predict_proba(X_test)
        predicted = model.predict(X_test)
        acc = model.score(X_test, Y_test)
        print(f"测试集准确率: {acc}")
        plot_roc_pr_curves(Y_test, y_proba[:, 1], predicted)


def CNNx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load("models/CNN_04-29_1703_75.70%.pth", map_location=device))
    show(model, deepmodel=True)


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


def save_to_excel(metrics_list, filename):
    metrics_df = pd.DataFrame(metrics_list, columns=['Model', 'AUC', 'AUC CI_low', 'AUC CI_high', 'Accuracy',
                                                     'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1 score'])
    # 保存DataFrame到Excel文件
    metrics_df.to_excel(filename, index=False)


def main():
    models_names_list = ("AdaBoost", "DecisionTree", "GradientBoosting", "GaussianNB", "LinearDiscriminantAnalysis",
                         "LightGBM", "LogisticRegression", "MultilayerPerceptron", "RandomForest", "SVM", "XGBoost")
    models_path = 'models/'
    models_list = []
    # 遍历models文件夹中的所有文件
    for filename in os.listdir(models_path):
        # 检查文件是否为.pkl文件
        if filename.endswith('.pkl'):
            # 构建完整的文件路径
            filepath = os.path.join(models_path, filename)
            try:
                with open(filepath, 'rb') as file:
                    model = pickle.load(file)
                    models_list.append(model)
            except Exception as e:
                print(f"Error loading model from {filepath}: {e}")
    if len(models_list) != len(models_names_list):
        print("模型数量不对")
    df = pd.read_excel("./data/all_data.xlsx")
    # df = pd.read_excel("./data/demo_data.xlsx")
    _, X_test, _, Y_test = utils.get_data(df)
    metrics_list = []
    for i, model in enumerate(models_list):
        metrics = calculate_metrics(model, X_test, Y_test)
        # 添加模型名称到指标列表
        metrics_list.append([models_names_list[i]] + metrics)
        print("模型:", models_names_list[i], "指标计算完成")

    # 保存结果到Excel文件
    save_to_excel(metrics_list, 'metrics.xlsx')


if __name__ == '__main__':
    # CNNx()
    main()
