import time
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

import custom_log
logger = custom_log.setup_logging("clean_data")
def clean_data(df, method='mean', fill_value=None):
    """
    清洗DataFrame中的空值。
    
    参数:
        df (pd.DataFrame): 输入的原始DataFrame
        method (str): 填充方法 ('mean', 'median', 'most_frequent', 'constant', 'iterative', 'random_forest', 'polynomial')
        fill_value: 如果method是'constant'，则使用此值填充
        
    返回:
        pd.DataFrame: 缺失值处理后的DataFrame
    """
    df_clean = df.copy()

    # 分离标签列
    target_column = 'have_stone'
    if target_column in df_clean.columns:
        y = df_clean.pop(target_column)
    else:
        y = None

    # 方法一：均值填充
    if method == 'mean':
        imputer_num = SimpleImputer(strategy='mean')
        df_clean = pd.DataFrame(imputer_num.fit_transform(df_clean), columns=df_clean.columns)
    # 方法二：中位数填充
    elif method == 'median':
        imputer_num = SimpleImputer(strategy='median')
        df_clean = pd.DataFrame(imputer_num.fit_transform(df_clean), columns=df_clean.columns)
    # 方法三：删除有缺失值的行
    elif method == 'drop':
        df_clean.dropna(inplace=True)

    # 方法四：随机森林模型填充缺失值
    elif method == 'random_forest':
        estimator_rf = RandomForestRegressor(n_estimators=100, random_state=0,n_jobs=-1)
        imputer_rf = IterativeImputer(estimator=estimator_rf, random_state=0, max_iter=10,verbose=2)
        df_clean = pd.DataFrame(imputer_rf.fit_transform(df_clean), columns=df_clean.columns)

    # 方法五：回归填充缺失值
    elif method == 'iterative':
        imputer = IterativeImputer(max_iter=10, random_state=0)
        df_clean = pd.DataFrame(imputer.fit_transform(df_clean), columns=df_clean.columns)

    else:
        raise ValueError(f"Unsupported method: {method}")

    # 将标签列重新加回去
    if y is not None:
        df_clean[target_column] = y

    return df_clean


if __name__ == "__main__":
    # 读取源数据
    file_path = 'data/Train-test-dataset_Ver-electronic-nooutliers.csv'
    df = pd.read_csv(file_path)
    df['have_stone'] = df['have_stone'].astype('int')  # 确保标签是整数类型

    # 定义填充方法和对应的文件名
    methods = {
        'drop': f'{file_path[:-4]}_drop.csv',
        'mean': f'{file_path[:-4]}_mean_mode.csv',
        'median': f'{file_path[:-4]}_median_mode.csv',
        # 'random_forest': f'{file_path[:-4]}_random_forest.csv',
        'iterative': f'{file_path[:-4]}_iterative.csv'
    }

    # 使用循环进行数据清洗和保存
    for method, file_path in methods.items():
        start_time = time.time()
        df_cleaned = clean_data(df, method=method)
        df_cleaned.to_csv(file_path, index=False)
        logger.info(f"✅ 已保存：{method}填充数据")
        logger.info(f"{method}填充数据已保存，耗时：{time.time() - start_time:.2f}秒")