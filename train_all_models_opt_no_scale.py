from datetime import datetime
from tqdm import tqdm
# 导入模型初始化函数
from ehr_models import *
import ehr_utils
import ehr_models
from sklearn.preprocessing import RobustScaler
import numpy as np


def train_and_save_all_models(file_path, method='default'):
    X_train, X_test, Y_train, Y_test = ehr_utils.get_train_test(file_path)
    # 定义所有模型初始化函数
    model_initializers = {
        'AdaBoost': init_adaBoost,
        'DecisionTree': init_decisionTree,
        'GaussianNB': init_gaussianNB,
        'GradientBoosting': init_gradientBoosting,
        'LightGBM': init_lightGBM,
        'LinearDiscriminantAnalysis': init_linearDiscriminantAnalysis,
        'LogisticRegression': init_logisticRegression,
        'MLPClassifier': init_MLPClassifier,
        'RandomForest': init_randomForest,
        'XGBoost': init_XGBoost
    }

    # 遍历每个模型，训练并保存
    for model_name, init_func in tqdm(model_initializers.items()):
        print(f"Training {model_name}...")
        model = init_func(default_parm=False)
        model.fit(X_train, Y_train)
        print(f"Model training completed for {model_name}.")

        # 获取预测结果
        Y_prob, Y_pred = ehr_utils.get_prediction_results(model, X_test)
        # 评估模型
        metrics = ehr_utils.eval_model(Y_test, Y_prob, Y_pred)
        print(f"Metrics for {model_name}: {metrics}")
        # 保存模型
        ehr_models.save_model(model, model_name, auc=metrics[0], method=method, opt='opt_no_scale')


# 调用主函数
if __name__ == "__main__":
    file_path_drop = 'data_cleaned/ver-noelectrolyte-crystal_removed_history_removed_drop.csv'
    file_path_mean = 'data_cleaned/ver-noelectrolyte-crystal_removed_history_removed_mean.csv'
    file_path_median = 'data_cleaned/ver-noelectrolyte-crystal_removed_history_removed_median.csv'
    file_path_bayesian = 'data_cleaned/ver-noelectrolyte-crystal_removed_history_removed_bayesian.csv'
    train_and_save_all_models(file_path_drop, method='drop')
    train_and_save_all_models(file_path_mean, method='mean')
    train_and_save_all_models(file_path_median, method='median')
    train_and_save_all_models(file_path_bayesian, method='bayesian')
