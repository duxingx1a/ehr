import os
from turtle import st
from utils import custom_dataset, custom_log, custom_models
import pandas as pd

if __name__ == "__main__":
    # 设置日志
    logger = custom_log.setup_logging('train_model')
    pd.set_option('display.max_columns', 80)
    pd.set_option('display.max_rows', 80)

    # 初始化模型
    models_hub = custom_models.Models()
    # 获取数据集路径
    datasets = custom_dataset.EHR_dataset_hub()
    # 创建保存模型的目录
    if not os.path.exists('models'):
        os.makedirs('models')
    # 清除 models 目录下所有文件
    for root, dirs, files in os.walk('models'):
        for file in files:
            os.remove(os.path.join(root, file))

    for method, dataset in datasets.get_all_dataset().items():

        # 获取训练集和测试集
        X_train, X_test, Y_train, Y_test = dataset.get_train_test_data()
        # 获取特征名称
        feature_names = X_train.columns.tolist()
        # 训练模型
        models_hub.train_all_models(X_train, Y_train, method)
