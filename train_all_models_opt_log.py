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
    num_cols = [col for col in X_train.select_dtypes(include=[np.number]).columns if X_train[col].nunique() > 3]
    print("数值型变量（排除one-hot编码）:", num_cols)
    print("总特征数:", len(X_train.columns))
    print("需要log1p的特征数:", len(num_cols))
    
    X_train[num_cols] = np.log1p(X_train[num_cols])
    X_test[num_cols] = np.log1p(X_test[num_cols])
    # robust_train=RobustScaler()
    # robust_test=RobustScaler()
    # X_train = robust_train.fit_transform(X_train)  # 使用RobustScaler进行特征缩放
    # X_test = robust_test.fit_transform(X_test)  # 使用RobustScaler进行特征缩放
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
        ehr_models.save_model(model, model_name, auc=metrics[0], method=method, opt='opt_log')


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
