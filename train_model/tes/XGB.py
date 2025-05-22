from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from datetime import datetime
import pandas as pd
# import utils
import pickle
from scipy.stats import sem, t
# import custom_log
import sys
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.metrics import roc_curve, average_precision_score
from scipy import stats
import pandas as pd
from datetime import datetime
import sys
import pickle
from imblearn.over_sampling import SMOTE
sys.path.append('..')
def apply_smote(X_train, Y_train):
    """
    使用SMOTE对训练数据进行过采样。

    参数:
        X_train (pd.DataFrame): 训练集特征。
        Y_train (pd.Series): 训练集标签。

    返回:
        X_train_resampled (pd.DataFrame): 过采样后的训练集特征。
        Y_train_resampled (pd.Series): 过采样后的训练集标签。
    """
    smote = SMOTE(random_state=42)
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

    return X_train_resampled, Y_train_resampled

def get_train_test_data(df=None,
             target_column='have_stone',
             test_size=0.2,
             random_state=0):
    """
    从 DataFrame 中提取训练集和测试集。  

    参数:
        df (pd.DataFrame): 输入的 DataFrame。
        target_column (str): 目标变量（标签）的列名。
        test_size (float): 测试集的比例，默认为 0.2。
        random_state (int): 随机种子，确保结果可复现。

    返回:
        X_train (pd.DataFrame): 训练集特征。
        X_test (pd.DataFrame): 测试集特征。
        Y_train (pd.Series): 训练集标签。
        Y_test (pd.Series): 测试集标签。
    """
    # 分离特征和目标变量
    X = df.drop(columns=[target_column])
    Y = df[target_column]
    Y = Y.astype('int')  # 确保标签是整数类型
    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state)
    
    # if 1:
    #     # 应用SMOTE
    #     X_train_resampled, Y_train_resampled = apply_smote(X_train, Y_train)
    #     return X_train_resampled, X_test, Y_train_resampled, Y_test
        # pass
    return X_train, X_test, Y_train, Y_test

def init_xgb():
    model_xgb = xgb.XGBClassifier(learning_rate=0.02,
                                  n_estimators=600,
                                  max_depth=7,
                                  min_child_weight=2,
                                  gamma=5,
                                  subsample=0.8,
                                  colsample_bytree=0.5,
                                  objective='binary:logistic',
                                  nthread=-1,
                                  scale_pos_weight=5,
                                  seed=42,
                                  reg_alpha=0,
                                  reg_lambda=1,
                                  eval_metric='auc')
    # model_xgb = xgb.XGBClassifier(nthread=-1,device='gpu')
    return model_xgb

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
    plt.plot(fpr,
             tpr,
             color='darkorange',
             lw=2,
             label=f'ROC curve (area = {auroc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('ROC_curve.png')
    # 绘制 PR 曲线
    plt.figure()
    plt.plot(recall,
             precision,
             color='darkorange',
             lw=2,
             label=f'PR curve (area = {auprc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc="lower left")
    plt.savefig('PR_curve.png')
    plt.show()

    # 打印 AUROC、AUPRC 和 F1 分数
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1 Score: {f1:.4f}")
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

    return [
        auc_score, auc_ci_low, auc_ci_high, accuracy, sensitivity, specificity,
        ppv, npv, f1
    ]
def train_xgb():
    # dataset = '/home/mclab/ydl/ehr/data/Train-test-dataset_Ver4_median_mode.csv'
    dataset = '/home/mclab/ydl/ehr/data/updated_data.csv'
    df = pd.read_csv(dataset)
    X_train, X_test, Y_train, Y_test = get_train_test_data(df)
    feature_names = X_train.columns.tolist()
    
    model_xgb = init_xgb()
    model_xgb.fit(X_train, Y_train)
    acc = model_xgb.score(X_test, Y_test)
    print(f"测试集准确率: {acc}")

    # 计算评估指标
    metrics = calculate_metrics(model_xgb, X_test, Y_test)
    print(f"AUC: {metrics[0]:.4f}, 95% CI: [{metrics[1]:.4f}, {metrics[2]:.4f}]")
    print(f"Accuracy: {metrics[3]:.4f}")
    print(f"F1 Score: {metrics[4]:.4f}")

    # 生成预测结果
    y_pred_prob = model_xgb.predict_proba(X_test)[:, 1]
    y_pred = model_xgb.predict(X_test)

    # 绘制ROC和PR曲线
    plot_roc_pr_curves(Y_test, y_pred_prob, y_pred)

if __name__ == '__main__':
    train_xgb()