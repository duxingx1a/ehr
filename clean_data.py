"""
清洗数据
"""
import time
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import os


def clean_data(df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
    """
    清洗输入的dataframe中的空值，并且将标签转化为数值。清洗方式有：删除有缺失值的行、均值填充、贝叶斯岭回归填充缺失值。
    pd.DataFrame: 缺失值处理后的DataFrame
    method: 清洗方法，默认为'mean'。可选值为'drop'、'mean'、'bayesian'。
    """
    df_clean = df.copy()
    df_clean['have_stone'] = df_clean['have_stone'].astype('int')  # 确保标签是整数类型
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

    # 方法二：删除有缺失值的行
    elif method == 'drop':
        df_clean.dropna(inplace=True)
    elif method == 'median':
        # 方法二：中位数填充
        imputer_num = SimpleImputer(strategy='median')
        df_clean = pd.DataFrame(imputer_num.fit_transform(df_clean), columns=df_clean.columns)
    # 方法三：回归填充缺失值
    elif method == 'bayesian':
        imputer = IterativeImputer(max_iter=10, random_state=0)
        df_clean = pd.DataFrame(imputer.fit_transform(df_clean), columns=df_clean.columns)
    else:
        raise ValueError(f"Unsupported method: {method}")
    # 将标签列重新加回去
    if y is not None:
        df_clean[target_column] = y
    return df_clean


def is_already_cleaned(output_file_path: str) -> bool:
    """
    检查输出文件是否已经存在。
    如果文件已存在，返回 True；否则返回 False。
    """
    return os.path.exists(output_file_path)


def main():
    #列出所有需要清洗的原始数据
    original_data = os.listdir('data_original')

    print(f"读取到{len(original_data)}个文件：Original data files: {original_data}")
    for file in original_data:
        if not file.lower().endswith('.csv'):
            print(f"Skipping non-csv file: {file}")
            continue
        file_path = os.path.join('data_original', file)
        file_name = os.path.basename(file_path)
        # 去掉文件的扩展名
        file_name_without_extension = os.path.splitext(file_name)[0]
        print(f"Processing file: {file_name}")
        df = pd.read_csv(file_path)
        # 确保输出目录存在
        output_dir = f'data_cleaned/'
        os.makedirs(output_dir, exist_ok=True)
        # 定义填充方法和对应的文件名
        drop_name = f'{file_name_without_extension}_drop.csv'
        mean_name = f'{file_name_without_extension}_mean.csv'
        median_name = f'{file_name_without_extension}_median.csv'
        bayesian_name = f'{file_name_without_extension}_bayesian.csv'
        methods = {
            'drop': drop_name,
            'mean': mean_name,
            'median': median_name,
            'bayesian': bayesian_name,
        }

        # 遍历三种方法进行数据清洗和保存
        for method, output_file_name in methods.items():
            # 拼接完整的输出文件路径
            output_file_path = os.path.join(output_dir, output_file_name)
            # 检查是否已清洗
            if is_already_cleaned(output_file_path):
                print(f"File '{output_file_path}' already cleaned. Skipping...")
                continue
            start_time = time.time()
            df_cleaned = clean_data(df, method=method)
            df_cleaned.to_csv(output_file_path, index=False)
            print(f"Data cleaned using {method} method and saved to '{output_file_path}'. Time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
    print('清洗完成')
