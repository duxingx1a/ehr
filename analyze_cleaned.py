import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data= pd.read_csv('data/Train-test-dataset_Ver3.csv')
data_cleaned = pd.read_csv('data/Train-test-dataset_Ver3_cleaned.csv')

print(data.info())

print(data.describe())

from pandas.plotting import scatter_matrix

scatter_matrix(data_cleaned.select_dtypes(include=['float64', 'int64']).iloc[:, :5], figsize=(12, 8))
plt.suptitle('Scatter Matrix of First 5 Features in Cleaned Data')
plt.show()

# 计算相关系数并绘制热图
corr = data_cleaned.select_dtypes(include=['float64', 'int64']).corr()
plt.figure(figsize=(12, 8))
plt.title('Correlation Heatmap of Cleaned Data')
sns.heatmap(corr, annot=True, fmt=".2f")
plt.show()