from datetime import datetime
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from train_model.AdB import init_adb
from train_model.DT import init_dt
from train_model.GB import init_gb
from train_model.GNB import init_gnb
from train_model.LDA import init_lda
from train_model.LGBM import init_lgbm
from train_model.LR import init_lr
from train_model.MLP import init_mlp
from train_model.RF import init_rf
from train_model.XGB import init_xgb
from tqdm import tqdm
import utils
import pickle
import custom_log

logger = custom_log.setup_logging('train_model')
pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 80)


def init_models():
    models_list = []
    models_names_list = ("model_adb", "model_dt", "model_gb", "model_gnb",
                         "model_lgbm", "model_lda", "model_lr", "model_mlp",
                         "model_rf", "model_xgb")
    model_adb = init_adb()
    model_dt = init_dt()
    model_gb = init_gb()
    model_gnb = init_gnb()
    model_lgbm = init_lgbm()
    model_lda = init_lda()
    model_lr = init_lr()
    model_mlp = init_mlp()
    model_rf = init_rf()
    model_xgb = init_xgb()

    for name in models_names_list:
        model = locals()[name]
        models_list.append(model)
    return models_list, models_names_list


def train_model(models, names, X_train, X_test, Y_train, Y_test, feature_names,
                method):
    #清空之前的模型
    # 创建方法对应的目录
    method_dir = os.path.join('models', method)
    if os.path.exists(method_dir):
        # 清空之前的模型
        for file in os.listdir(method_dir):
            os.remove(os.path.join(method_dir, file))
    else:
        os.makedirs(method_dir)
    for model, name in tqdm(zip(models, names)):
        # 主要是为了绑定特征名
        X_train = pd.DataFrame(X_train, columns=feature_names)
        model.fit(X_train, Y_train)
        # 在测试集上简单评估模型准确率，确认模型没有出大问题
        acc = model.score(X_test, Y_test)
        print(f"{name:<10}:测试集准确率: {acc:.3f}")
        current_time = datetime.now().strftime("%m-%d_%H%M")
        # 保存模型
        model_path = os.path.join(
            method_dir, f'{name}_{current_time}_{acc * 100:.2f}%.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    return models


def metrics():
    pass


def main():
    datasets = utils.get_all_dataset()

    for dataset in tqdm(datasets):
        # 获取该数据集使用的方法，方便后续区分
        method = dataset.split('dataset_')[1].split('.')[0]
        df = pd.read_csv(dataset)
        X_train, X_test, Y_train, Y_test = utils.get_train_test_data(df)
        feature_names = X_train.columns
        feature_names = list(feature_names)
        logger.info(feature_names)
        models_set, models_names_set = init_models()
        trained_models = train_model(models_set, models_names_set, X_train,
                                     X_test, Y_train, Y_test, feature_names,
                                     method)
        # metrics(trained_models)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # 颜色表
        C = [
            '#f98e62', '#d6eef4', '#f0c184', '#f6ebb1', '#929fc9', '#f8fbcb',
            '#f8fbcb', '#ef8c67', '#8074ca', '#a5d954', '#b44763'
        ]
        # 为了输出对齐,出此下策， 后期再想解决办法
        names = [
            "ADB   ", "DT     ", "GB     ", "GNB   ", "RF      ", "LGBM ",
            "LDA    ", "LR      ", "MLP    ", "XGB   "
        ]
        # 遍历每个模型，计算AUC和PRC，并绘制曲线
        for i, model in tqdm(enumerate(trained_models[:-1])):
            # 预测概率
            y_pred_prob = model.predict_proba(X_test)[:, 1]

            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
            auc_score = auc(fpr, tpr)
            # 绘制AUC曲线
            ax[0].plot(fpr,
                       tpr,
                       label=f'{names[i]}({auc_score:.2f})',
                       color=C[i],
                       alpha=0.7,
                       linewidth=1)

            # 计算PRC曲线
            precision, recall, _ = precision_recall_curve(Y_test, y_pred_prob)
            prc_score = average_precision_score(Y_test, y_pred_prob)
            # 绘制PRC曲线
            name_width = 5
            ax[1].plot(recall,
                       precision,
                       label=f'{names[i]}({prc_score:.2f})',
                       color=C[i],
                       alpha=0.7,
                       linewidth=1)
        y_pred_prob = trained_models[-1].predict_proba(X_test)[:, 1]
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
        auc_score = auc(fpr, tpr)
        # 绘制AUC曲线
        ax[0].plot(fpr,
                   tpr,
                   label=f'{names[-1]}({auc_score:.2f})',
                   color=C[-1],
                   alpha=0.9,
                   linewidth=2)

        # 计算PRC曲线
        precision, recall, _ = precision_recall_curve(Y_test, y_pred_prob)
        prc_score = average_precision_score(Y_test, y_pred_prob)
        # 绘制PRC曲线
        ax[1].plot(recall,
                   precision,
                   label=f'{names[-1]}({prc_score:.2f})',
                   color=C[-1],
                   alpha=0.9,
                   linewidth=2)
        # 显示图表
        ax[0].set_ylim(0, 1)
        ax[0].set_xlim(0, 1)
        ax[1].set_ylim(0, 1)
        ax[1].set_xlim(0, 1)
        ax[0].set_title('ROC Curves')
        ax[0].set_xlabel('False Positive Rate')
        ax[0].set_ylabel('True Positive Rate')
        ax[0].legend(loc='lower right')
        ax[1].set_title('PRC Curves')
        ax[1].set_xlabel('Recall')
        ax[1].set_ylabel('Precision')
        ax[1].legend(loc='lower left')
        plt.tight_layout()
        # 创建结果保存目录
        if not os.path.exists('result'):
            os.makedirs('result')
        plt.savefig(f'result/result_{method}.png')
        plt.show()


if __name__ == '__main__':
    main()
    # train_svm()
