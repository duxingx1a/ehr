import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

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
    
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    categorical_columns = df_clean.select_dtypes(exclude=[np.number]).columns
    
    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
        df_clean[numeric_columns] = imputer.fit_transform(df_clean[numeric_columns])
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
        df_clean[numeric_columns] = imputer.fit_transform(df_clean[numeric_columns])
    elif method == 'most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
        df_clean[numeric_columns] = imputer.fit_transform(df_clean[numeric_columns])
        df_clean[categorical_columns] = imputer.fit_transform(df_clean[categorical_columns])
    elif method == 'constant':
        imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
        df_clean[numeric_columns] = imputer.fit_transform(df_clean[numeric_columns])
        df_clean[categorical_columns] = imputer.fit_transform(df_clean[categorical_columns])
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        df_clean[numeric_columns] = imputer.fit_transform(df_clean[numeric_columns])
    elif method == 'iterative':
        imputer = IterativeImputer(max_iter=10, random_state=0)
        df_clean[numeric_columns] = imputer.fit_transform(df_clean[numeric_columns])
    else:
        raise ValueError("Unsupported method")
    
    return df_clean