import os
import pickle
from utils import custom_dataset, custom_log, custom_models
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 设置日志
    logger = custom_log.setup_logging('eval_model')
    pd.set_option('display.max_columns', 80)
    pd.set_option('display.max_rows', 80)
    # 获取数据集路径
    datasets = custom_dataset.EHR_dataset_hub()
    X_train, X_test, Y_train, Y_test = datasets.get_dataset_from_method('drop').get_train_test_data(smote=True)
    # 获取训练后的模型
    
    
    # with open('/home/mclab/ydl/ehr_new/models/mean_mode/XGBClassifier_05-22_1704.pkl', 'rb') as f:
    #     model_xgb = pickle.load(f)
    custom_models.Models_trained('models').eval(None, X_test, Y_test)
    # 假设有一个数据集
    # df = pd.read_csv('data/demo_data.csv')  # 假设有demo_data.csv
    # X = df.drop('label', axis=1)
    # y = df['label']

#
