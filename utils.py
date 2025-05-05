import pandas as pd
from sklearn.model_selection import train_test_split
import os

pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 80)


def get_all_dataset():
    """
    获取所有数据集的路径列表。

    返回:
        list: 包含所有数据集路径的列表。
    """
    
    ## 手动
    # dataset_paths = [
    #     "data/Train-test-dataset_Ver3.csv",
    #     "data/Train-test-dataset_Ver4.csv",
    #     "data/Train-test-dataset_Ver4_process.csv",
    #     "data/Train-test-dataset_Ver4_process.csv_drop.csv"
    # ]
    #自动
    dataset_paths = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".csv")]
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
    # 分离特征和目标变量
    X = df.drop(columns=[target_column])
    Y = df[target_column]
    Y = Y.astype('int')  # 确保标签是整数类型
    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state)
    # 保存到新的CSV文件

    return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
    dataset=get_all_dataset()
    print(dataset)
    df = pd.read_csv("data/Train-test-dataset_Ver4_process.csv_drop.csv")
    X_train, X_test, Y_train, Y_test = get_train_test_data(df)
    # print(X_train.shape, Y_train.shape)
    # print(X_test.shape, Y_test.shape)
    # print(X_train.isna().sum())
