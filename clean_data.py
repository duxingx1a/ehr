import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression


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

    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    categorical_columns = df_clean.select_dtypes(exclude=[np.number]).columns

    specific_categorical_columns = ['Alcohol', 'Smoke', 'Uric_bacteria']
    print(f"特定分类列: {specific_categorical_columns}")
    # 方法一：均值填充
    if method == 'mean':
        imputer_num = SimpleImputer(strategy='mean')
        df_clean[numeric_columns] = imputer_num.fit_transform(df_clean[numeric_columns])

        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_clean[specific_categorical_columns] = imputer_cat.fit_transform(df_clean[specific_categorical_columns])

    # 方法二：中位数填充
    elif method == 'median':
        imputer_num = SimpleImputer(strategy='median')
        df_clean[numeric_columns] = imputer_num.fit_transform(df_clean[numeric_columns])

        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_clean[specific_categorical_columns] = imputer_cat.fit_transform(df_clean[specific_categorical_columns])

    # 方法三：删除有缺失值的行
    elif method == 'drop':
        df_clean.dropna(inplace=True)

    # 方法四：随机森林模型填充缺失值
    elif method == 'random_forest':
        estimator_rf = RandomForestRegressor(n_estimators=100, random_state=0,n_jobs=-1)
        imputer_rf = IterativeImputer(estimator=estimator_rf, random_state=0, max_iter=10,verbose=2)
        df_clean = pd.DataFrame(imputer_rf.fit_transform(df_clean), columns=df_clean.columns)

    # 方法五：多项式回归插值填充（适合连续数值）
    elif method == 'polynomial':
        for col in numeric_columns:
            other_cols = [c for c in numeric_columns if c != col]
            if len(other_cols) > 0 and df_clean[col].isnull().any():
                poly = PolynomialFeatures(degree=2, include_bias=False)
                model = make_pipeline(poly, LinearRegression())
                mask = df_clean[col].notna()
                model.fit(df_clean.loc[mask, other_cols], df_clean.loc[mask, col])
                df_clean.loc[~mask, col] = model.predict(df_clean.loc[~mask, other_cols])

        imputer_cat = SimpleImputer(strategy='most_frequent')
        df_clean[specific_categorical_columns] = imputer_cat.fit_transform(df_clean[specific_categorical_columns])

    # 其他方法
    elif method == 'constant':
        imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
        df_clean[numeric_columns] = imputer.fit_transform(df_clean[numeric_columns])
        df_clean[categorical_columns] = imputer.fit_transform(df_clean[categorical_columns])

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
    df = pd.read_csv('data/Train-test-dataset_Ver3.csv')

    # # 1. 删除有缺失值的行
    # df_drop = clean_data(df, method='drop')
    # df_drop.to_csv('data/Train-test-dataset_Ver3_drop.csv', index=False)
    # print("✅ 已保存：删除缺失值行的数据")

    # # 2. 均值 + 众数填充
    # df_mean_mode = clean_data(df, method='mean')
    # df_mean_mode.to_csv('data/Train-test-dataset_Ver3_mean_mode.csv', index=False)
    # print("✅ 已保存：均值+众数填充数据")

    # # 3. 中位数 + 众数填充
    # df_median_mode = clean_data(df, method='median')
    # df_median_mode.to_csv('data/Train-test-dataset_Ver3_median_mode.csv', index=False)
    # print("✅ 已保存：中位数+众数填充数据")

    # 4. 随机森林填充
    df_rf = clean_data(df, method='random_forest')
    df_rf.to_csv('data/Train-test-dataset_Ver3_random_forest.csv', index=False)
    print("✅ 已保存：随机森林填充数据")

    # 5. 多项式核回归填充
    df_poly = clean_data(df, method='polynomial')
    df_poly.to_csv('data/Train-test-dataset_Ver3_polynomial.csv', index=False)
    print("✅ 已保存：多项式回归填充数据")