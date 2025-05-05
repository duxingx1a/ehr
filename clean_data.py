import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import numpy as np

def clean_data(df, method='mean', fill_value=None):
    """
    清洗DataFrame中的空值。
    
    参数:
        df (pd.DataFrame): 输入的原始DataFrame
        method (str): 填充方法 ('mean', 'median', 'most_frequent', 'constant', 'knn', 'iterative')
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
    
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    categorical_columns = df_clean.select_dtypes(exclude=[np.number]).columns
    
    # 特定的类别型特征
    specific_categorical_columns = ['Alcohol', 'Smoke', 'Uric_bacteria']
    
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
        df_clean[numeric_columns] = imputer.fit_transform(df_clean[numeric_columns])
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
        df_clean[numeric_columns] = imputer.fit_transform(df_clean[numeric_columns])
    elif method == 'most_frequent':
        # 使用众数填充所有类别型特征
        imputer_all_cat = SimpleImputer(strategy='most_frequent')
        df_clean[categorical_columns] = imputer_all_cat.fit_transform(df_clean[categorical_columns])
        
        # 使用众数填充特定的类别型特征
        imputer_specific_cat = SimpleImputer(strategy='most_frequent')
        df_clean[specific_categorical_columns] = imputer_specific_cat.fit_transform(df_clean[specific_categorical_columns])
    elif method == 'constant':
        imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
        df_clean[numeric_columns] = imputer.fit_transform(df_clean[numeric_columns])
        df_clean[categorical_columns] = imputer.fit_transform(df_clean[categorical_columns])
    elif method == 'iterative':
        imputer = IterativeImputer(max_iter=10, random_state=0, verbose=2)
        df_clean = pd.DataFrame(imputer.fit_transform(df_clean), columns=df_clean.columns)
    else:
        raise ValueError("Unsupported method")
    
    # 将标签列重新添加回去
    if y is not None:
        df_clean[target_column] = y
    
    return df_clean

if __name__ == "__main__":
    # 读取数据
    df = pd.read_csv('data/Train-test-dataset_Ver3.csv')
    # 清洗数据
    df_cleaned = clean_data(df, method='iterative')
    # 保存清洗后的数据
    df_cleaned.to_csv('data/Train-test-dataset_Ver3_iterative.csv', index=False)