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
        "data/Train-test-dataset_Ver4_drop.csv",
        "data/Train-test-dataset_Ver4_iterative.csv",
        "data/Train-test-dataset_Ver4_mean_mode.csv",
        "data/Train-test-dataset_Ver4_median_mode.csv"
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


class EHR_dataset():
    """
    This class is used to define custom models for the EHR system.
    """

    def __init__(self):
        # 读取数据集并存储为字典
        self.datasets = {}
        #手动或自动
        dataset_paths = [
            "data/Train-test-dataset_Ver4_drop.csv",
            "data/Train-test-dataset_Ver4_iterative.csv",
            "data/Train-test-dataset_Ver4_mean_mode.csv",
            "data/Train-test-dataset_Ver4_median_mode.csv"
        ]
        #自动读取data目录下所有数据文件
        # dataset_paths = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".csv")]
        dataset_names = ['drop', 'iterative', 'mean_mode', 'median_mode']
        
        for name, path in zip(dataset_names, dataset_paths):
            self.datasets[name] = pd.read_csv(path)    
        
    def custom_model(self):
        """
        This method defines a custom model.
        """
        pass

    def get_models(self):
        """
        This method retrieves the custom model.
        """
        return self.models_list 

if __name__ == "__main__":
    # 测试 Models 类
    # 这里可以根据需要添加测试代码
    ehr_datasets= EHR_dataset()
    # 获取所有数据集
    datasets = ehr_datasets.datasets
    # 打印数据集名称和路径
    for name, path in datasets.items():
        print(f"Dataset Name: {name}, Path: {path}")
    # 获取训练集和测试集
    for name, df in datasets.items():
        X_train, X_test, Y_train, Y_test = get_train_test_data(df)
        print(f"Dataset Name: {name}, X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"Dataset Name: {name}, Y_train shape: {Y_train.shape}, Y_test shape: {Y_test.shape}")

