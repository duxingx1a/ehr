"""
读取trained目录下所有模型的预测结果，并计算每个模型各个指标。保存到result_tables/model_metrics.xlsx文件中。
"""
from typing import Any, List
from matplotlib import pyplot as plt
import pandas as pd
import os
import datetime
import ehr_utils
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

plt.rcParams['font.family'] = ['sans-serif'] # 设置字体为无衬线体
plt.rcParams['font.sans-serif'] = 'Times New Roman' # 设置字体为Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def plot_all_models_roc_pr(trained_models: List[Any], X_test: pd.DataFrame, Y_test: pd.Series, fill_method_name: str,opt: str = 'opt') -> None:
    """
    绘制所有模型的ROC曲线和PR曲线，并将图片保存到result_fig下的roc_pr_curves_all_{月日}.png。
    
    - trained_models: 训练好的模型列表
    - X_test: 测试集特征
    - Y_test: 测试集标签
    - fill_method_name: 填充方法名称，用于区分不同的数据集
    - opt: 优化选项，默认为'opt'，可选值包括'opt'和'default'。分别为优化参数的模型和默认参数的模型
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    # 颜色表
    C = ['#f98e62', '#d6eef4', '#f0c184', '#f6ebb1', '#929fc9', '#f8fbcb', '#ef8c67', '#8074ca', '#a5d954', '#b44763']
    # 为了在图中对齐指标
    names = ["ADB   ", "DT      ", "GNB   ", "GB      ", "LGBM", "LDA    ", "LR       ", "MLP    ", "RF       ", "XGB    "]
    # 遍历除xgb外每个模型，计算AUC和PRC，并绘制曲线
    for i, model in enumerate(trained_models[:-1]):
        # 预测概率
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
        auc_score = auc(fpr, tpr)
        # 绘制AUC曲线
        ax[0].plot(fpr, tpr, label=f'{names[i]}({auc_score:.2f})', color=C[i], alpha=0.7, linewidth=1)

        # 计算PRC曲线
        precision, recall, _ = precision_recall_curve(Y_test, y_pred_prob)
        prc_score = average_precision_score(Y_test, y_pred_prob)
        # 绘制PRC曲线
        ax[1].plot(recall, precision, label=f'{names[i]}({prc_score:.2f})', color=C[i], alpha=0.7, linewidth=1)
    #xgb单独拎出来plot
    y_pred_prob = trained_models[-1].predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
    auc_score = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(Y_test, y_pred_prob)
    prc_score = average_precision_score(Y_test, y_pred_prob)
    ax[0].plot(fpr, tpr, label=f'{names[-1]}({auc_score:.2f})', color=C[-1], alpha=0.9, linewidth=2)
    ax[1].plot(recall, precision, label=f'{names[-1]}({prc_score:.2f})', color=C[-1], alpha=0.9, linewidth=2)
    
    fig.suptitle(f'ROC and PRC Curves for All Models', fontsize=16, fontweight='bold')
    

    # 显示图表
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(0, 1)
    ax[0].set_title('ROC Curves of the Optimized Models(Mean)')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].legend(loc='lower right')

    ax[1].set_ylim(0, 1)
    ax[1].set_xlim(0, 1)
    ax[1].set_title('PRC Curves of the Optimized Models(Mean)')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].legend(loc='best')
    plt.tight_layout()
    fig.savefig(f'results_fig/roc_pr_curves_all_{opt}_{fill_method_name}_{datetime.datetime.now().strftime("%m%d")}.png')
    # plt.show()


def eval_all_models_to_excel(trained_models: List[Any], X_test: pd.DataFrame, Y_test: pd.Series, fill_method_type: str, opt: str = 'opt') -> None:
    """
    评估所有模型的性能，并将结果保存到result_tables目录下model_metrics_{opt}_{月日}.xlsx文件中。
    
    - trained_models: 训练好的模型列表
    - X_test: 测试集特征
    - Y_test: 测试集标签
    - fill_method_type: 填充方法类型，用于区分不同的数据集
    - opt: 优化选项，默认为'opt'，可选值包括'opt'和'default'。分别为优化参数的模型和默认参数的模型
    """
    metrics = []
    names = ["ADB", "DT", "GNB", "GB", "LGBM", "LDA", "LR", "MLP", "RF", "XGB"]
    for model in trained_models:
        Y_prob, Y_pred = ehr_utils.get_prediction_results(model, X_test)
        model_metrics = ehr_utils.eval_model(Y_test, Y_prob, Y_pred)
        metrics.append(model_metrics)
    # 创建一个 DataFrame
    df = pd.DataFrame(metrics, index=names, columns=['AUC', 'AUC_CI_Low', 'AUC_CI_High', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1'])
    os.makedirs('result_tables', exist_ok=True)

    output_file = f'result_tables/model_metrics_{opt}_{datetime.datetime.now().strftime("%m%d")}.xlsx'
    #如果不存在就创建新表
    if not os.path.exists(output_file):
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, sheet_name=fill_method_type, index=True)
    #存在旧表就用追加模式
    else:
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=fill_method_type, index=True)


if __name__ == '__main__':
    #使用什么数据  可选 mean drop  bayesian
    fill_methods = ['drop', 'mean', 'bayesian']
    #使用什么优化选项 可选 opt default
    opt = 'default'
    #是否重新计算结果
    re_calculate = False
    re_plot = True
    # 评估所有模型
    for fill_method in fill_methods:
        # 评估trained_models_{默认or优化参数}/{控制填充方法}下的所有模型，将其保存到xlsx中的一个sheet中
        models_list = ehr_utils.get_all_trained_models(f'trained_models_{opt}/{fill_method}')
        file_path = f'data_cleaned/ver-noelectrolyte-logtransform_{fill_method}.csv'
        X_train, X_test, Y_train, Y_test = ehr_utils.get_train_test(file_path)
        if re_calculate:
            eval_all_models_to_excel(models_list, X_test, Y_test, fill_method, opt=opt)
            print(f"\n{fill_method}方法表格评估结果完成，保存到result_tables/model_metrics_{opt}.xlsx中")
        else:
            print(f"\n跳过表格评估结果")
        if re_plot:
            # 绘制ROC和PR曲线
            plot_all_models_roc_pr(models_list, X_test, Y_test, fill_method, opt=opt)
            print(f"\n绘制ROC和PR曲线并保存到results_fig/roc_pr_curves_all_{opt}_{fill_method}.png")
        else:
            print(f"\n跳过ROC和PR曲线绘制")
    print("\n所有模型评估完成！")
    print(f"请查看result_tables/model_metrics_{opt}.xlsx和results_fig/roc_pr_curves_all_{opt}.png文件。")
