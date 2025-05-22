import pandas as pd
from sklearn.model_selection import train_test_split
import os
from imblearn.over_sampling import SMOTE
pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 80)

# 在get_train_test_data函数后添加以下内容以应用SMOTE
def apply_smote(X_train, Y_train):
    """
    使用SMOTE对训练数据进行过采样。

    参数:
        X_train (pd.DataFrame): 训练集特征。
        Y_train (pd.Series): 训练集标签。

    返回:
        X_train_resampled (pd.DataFrame): 过采样后的训练集特征。
        Y_train_resampled (pd.Series): 过采样后的训练集标签。
    """
    smote = SMOTE(random_state=42)
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

    return X_train_resampled, Y_train_resampled

def get_all_dataset():
    """
    获取所有数据集的路径列表。

    返回:
        list: 包含所有数据集路径的列表。
    """
    
    ## 手动
    dataset_paths = [
        'data/Train-test-dataset_Ver-electronic-nooutliers_drop.csv',
        'data/Train-test-dataset_Ver-electronic-nooutliers_mean_mode.csv',
        'data/Train-test-dataset_Ver-electronic-nooutliers_median_mode.csv'
        'data/Train-test-dataset_Ver-electronic-nooutliers_iterative.csv'
        # "data/Train-test-dataset_Ver4_drop.csv",
        # "data/Train-test-dataset_Ver4_iterative.csv",
        # "data/Train-test-dataset_Ver4_mean_mode.csv",
        # "data/Train-test-dataset_Ver4_median_mode.csv"
    ]
    #自动读取data目录下所有数据文件
    # dataset_paths = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".csv")]
    return dataset_paths

def get_train_test_data(df=None,
             target_column='have_stone',
             test_size=0.2,
             random_state=42):
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
    # 应用SMOTE
    X_train_resampled, Y_train_resampled = apply_smote(X_train, Y_train)
    if 1:
        return X_train_resampled, X_test, Y_train_resampled, Y_test
    return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
    dataset=get_all_dataset()
    print(dataset)
    df = pd.read_csv("data/Train-test-dataset_Ver4_process.csv_drop.csv")
    X_train, X_test, Y_train, Y_test = get_train_test_data(df)
    # print(X_train.shape, Y_train.shape)
    # print(X_test.shape, Y_test.shape)
    # print(X_train.isna().sum())
