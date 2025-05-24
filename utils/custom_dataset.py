from ast import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 80)


class EHR_dataset():

    def __init__(self, path='data'):
        self.dataset = pd.read_csv(path)

    from typing import Tuple

    def get_train_test_data(self, target_column='have_stone', test_size=0.2, random_state=42, smote=False, std=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        This method retrieves the custom model.
        """
        # 读取数据集
        df = self.dataset
        # 分离特征和目标变量
        X = df.drop(columns=[target_column])
        Y = df[target_column]
        Y = Y.astype('int')  # 确保标签是整数类型
        # 保存特征名称
        feature_names = X.columns.tolist()
        # 划分训练集和测试集
        X_train: pd.DataFrame
        X_test: pd.DataFrame
        Y_train: pd.Series
        Y_test: pd.Series
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
        # print(f"X_train的类别{type(X_train)}")
        # print(f"Y_train的类别{type(Y_train)}")
        # print('-------------')
        # 应用SMOTE
        smote = SMOTE(random_state=42)
        scaler = StandardScaler().fit(X_train)
        #将训练集中的数据处理
        if smote:
            X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train) # type: ignore
            X_train_smote = pd.DataFrame(X_train_smote, columns=feature_names)
            return X_train_smote, X_test, Y_train_smote, Y_test # type: ignore
        if std:
            X_train_std = scaler.transform(X_train)
            X_test_std = scaler.transform(X_test)
            # 将特征名称重新赋予给数据集
            X_train_std = pd.DataFrame(X_train_std, columns=feature_names)
            X_test_std = pd.DataFrame(X_test_std, columns=feature_names)
            return X_train_std, X_test_std, Y_train, Y_test
        # 如果同时使用SMOTE和标准,返回smote和标准化后的训练集
        if smote and std:
            X_train_std = scaler.transform(X_train)
            X_train_std_smote, Y_train_smote = smote.fit_resample(X_train_std, Y_train) # type: ignore
            X_train_std_smote = pd.DataFrame(X_train_std_smote, columns=X.columns)
            X_test_std = pd.DataFrame(X_test_std, columns=X.columns) # type: ignore
            return X_train_std_smote, X_test_std, Y_train_smote, Y_test  # type: ignore
        return X_train, X_test, Y_train, Y_test


class EHR_dataset_hub():
    """
    This class is used to define custom models for the EHR system.
    """

    def __init__(self, path='data'):
        # 读取数据集并存储为字典
        self.datasets = {}
        #手动或自动
        dataset_paths = [
            "data/Train-test-dataset_Ver4_drop.csv", "data/Train-test-dataset_Ver4_iterative.csv", "data/Train-test-dataset_Ver4_mean_mode.csv",
            "data/Train-test-dataset_Ver4_median_mode.csv"
        ]
        #自动读取data目录下所有数据文件
        dataset_paths = [os.path.join(f"{path}", f) for f in os.listdir(f"{path}") if f.endswith(".csv")]
        dataset_names = ['drop', 'iterative', 'mean_mode', 'median_mode']

        for name, path in zip(dataset_names, dataset_paths):
            self.datasets[name] = EHR_dataset(path)

    def get_all_dataset(self) -> dict[str, EHR_dataset]:
        """
        return all dataset paths
        """
        return self.datasets

    def get_dataset_from_method(self, method='mean_mode') -> EHR_dataset:
        """
        This method retrieves the custom model.
        """
        return self.datasets[method]


if __name__ == "__main__":
    # 测试 Models 类
    # 这里可以根据需要添加测试代码
    ehr_datasets = EHR_dataset_hub('ehr_new/data')
    # 获取所有数据集
    dataset = ehr_datasets.get_dataset_from_method(method='mean_mode')
    # 打印数据集名称和路径
    print(dataset)
    # for name, path in datasets.():
    #     print(f"Dataset Name: {name}, Path: {path}")
    # 获取训练集和测试集
    # for name, df in datasets.items():
    #     X_train, X_test, Y_train, Y_test = get_train_test_data(df)
    # print(f"Dataset Name: {name}, X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    # print(f"Dataset Name: {name}, Y_train shape: {Y_train.shape}, Y_test shape: {Y_test.shape}")
