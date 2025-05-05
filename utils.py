import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 80)


def get_data(df=None, target_column='have_stone', test_size=0.2, random_state=42):
    """
    从 DataFrame 中提取训练集和测试集。

    参数:
        df (pd.DataFrame): 输入的 DataFrame。
        target_column (str): 目标变量（标签）的列名。
        test_size (float): 测试集的比例，默认为 0.2。
        random_state (int): 随机种子，确保结果可复现。

    返回:
        X_train (pd.DataFrame): 训练集特征。
        X_test (pd.DataFrame): 测试集特征。
        Y_train (pd.Series): 训练集标签。
        Y_test (pd.Series): 测试集标签。
    """
    #统一数据
    df = pd.read_csv("./data/Train-test-dataset_Ver3.csv")
    # 类别型变量众数填充，提前处理
    categorical_columns = ['Alcohol', 'Smoke', 'Uric_bacteria','Uric_epithelium']
    print(df[categorical_columns].isnull().sum())
    for col in categorical_columns:
        # 计算每列的众数
        mode_value = df[col].mode()[0]  # 使用 mode()[0] 获取第一个众数
        # 使用众数填充该列的空值
        df[col] = df[col].fillna(mode_value)  # 直接在原始 DataFrame 上操作
    df.to_csv('selected_samples.csv', index=False)
    print(df[categorical_columns].isnull().sum())
    # 分离特征和目标变量
    X = df.drop(columns=[target_column])
    Y = df[target_column]
    Y = Y.astype('int')  # 确保标签是整数类型
    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state)
    # 保存到新的CSV文件
    
    return X_train, X_test, Y_train, Y_test


def handle_missing_values(df,
                          method='simple',
                          strategy='mean',
                          n_neighbors=5,
                          max_iter=10,
                          random_state=42):
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.pipeline import FeatureUnion
    from sklearn.impute import MissingIndicator
    """
    处理缺失值的通用函数。

    参数:
        df (pd.DataFrame): 输入的 DataFrame。
        method (str): 处理缺失值的方法，可选 'simple'、'knn'、'iterative'、'drop' 或 'indicator'。
        strategy (str): 填充策略，用于 'simple' 和 'indicator' 方法，可选 'mean'、'median'、'most_frequent' 或 'constant'。
        n_neighbors (int): 用于 'knn' 方法的最近邻数量。
        max_iter (int): 用于 'iterative' 方法的最大迭代次数。
        random_state (int): 用于 'iterative' 方法的随机种子。

    返回:
        pd.DataFrame: 处理后的 DataFrame。
    """
    if method == 'simple':
        # 使用 SimpleImputer 填充缺失值
        imputer = SimpleImputer(strategy=strategy)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    elif method == 'knn':
        # 使用 KNNImputer 填充缺失值
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    elif method == 'iterative':
        # 使用 IterativeImputer 填充缺失值
        imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    elif method == 'drop':
        # 删除包含缺失值的行
        df_imputed = df.dropna()
    elif method == 'indicator':
        # 使用 MissingIndicator 标记缺失值，并用 SimpleImputer 填充
        indicator = MissingIndicator()
        imputer = SimpleImputer(strategy=strategy)
        transformer = FeatureUnion([("indicators", indicator),
                                    ("imputer", imputer)])
        df_imputed = pd.DataFrame(transformer.fit_transform(df),
                                  columns=df.columns)
    else:
        raise ValueError(
            "Invalid method. Choose from 'simple', 'knn', 'iterative', 'drop', or 'indicator'."
        )
    return df_imputed


if __name__ == '__main__':
    df = pd.read_csv("./data/Train-test-dataset_Ver3.csv")
    X_train, X_test, Y_train, Y_test = get_data(df)
    # print(X_train.shape, Y_train.shape)
    # print(X_test.shape, Y_test.shape)
    # print(X_train.isna().sum())
