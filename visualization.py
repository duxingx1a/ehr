import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import shap
# import utils
# from XGB import XGBClassifier


def show_reduction():
    df = pd.read_csv("./data/Train-test-dataset_Ver1.csv")
    # df = pd.read_excel("./data/all_data.xlsx")
    # df_processed = df.drop(df.columns[:5], axis=1)
    
    # df_encoded = pd.get_dummies(df, columns=['Sex'])
    df = df.fillna(df.mean())
    df_processed = df
    # 选择需要降维的特征列
    features = df_processed.drop(['have_stone'], axis=1)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # 随机选1000个数据降维
    random_indices = np.random.choice(len(features_scaled), size=100, replace=False)

    # 使用 PCA 进行降维
    pca_result = PCA(n_components=2).fit_transform(features_scaled)
    # 使用 t-SNE 进行降维
    tsne_result = TSNE(n_components=2, random_state=42).fit_transform(features_scaled)
    # 使用 UMAP 进行降维
    umap_result = umap.UMAP(n_components=2).fit_transform(features_scaled)

    pca_result_sampled = pca_result[random_indices]
    tsne_result_sampled = tsne_result[random_indices]
    umap_result_sampled = umap_result[random_indices]
    # 创建三个绘图窗口
    labels_sampled = df['have_stone'].iloc[random_indices].values

    # 设置颜色映射
    colors = ['#a5d954', '#f98e62']
    cmap = ListedColormap(colors)

    # 创建一个包含三个子图的图形
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # 绘制 PCA 图
    axs[0].scatter(pca_result_sampled[:, 0], pca_result_sampled[:, 1], c=labels_sampled, cmap=cmap, alpha=1, s=5)
    axs[0].set_title('PCA 2D')
    axs[0].set_xlabel('Principal Component 1')
    axs[0].set_ylabel('Principal Component 2')

    # 绘制 t-SNE 图
    axs[1].scatter(tsne_result_sampled[:, 0], tsne_result_sampled[:, 1], c=labels_sampled, cmap=cmap, alpha=1, s=5)
    axs[1].set_title('t-SNE 2D')
    axs[1].set_xlabel('t-SNE Feature 1')
    axs[1].set_ylabel('t-SNE Feature 2')

    # 绘制 UMAP 图
    axs[2].scatter(umap_result_sampled[:, 0], umap_result_sampled[:, 1], c=labels_sampled, cmap=cmap, alpha=1, s=5)
    axs[2].set_title('UMAP 2D')
    axs[2].set_xlabel('UMAP Feature 1')
    axs[2].set_ylabel('UMAP Feature 2')

    # 调整子图间距
    plt.tight_layout()
    plt.savefig('./classification_pr.png')
    plt.show()


def show_shap():
    # df = pd.read_excel("./data/all_data.xlsx")
    df = pd.read_csv(".data/Train-test-dataset_Ver1.csv")
    # df = pd.read_excel("./data/demo_data.xlsx")
    X_train, X_test, y_train, y_test = utils.get_data(df)

    feature_names = df.columns.tolist()
    feature_names = feature_names[4:-1]
    model = XGBClassifier()
    model.load_model("models/XBG_04-19_1500_79.69%.json")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print(f"base value: {explainer.expected_value:.2f}")
    # shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    sample_index = 0
    shap.force_plot(explainer.expected_value, shap_values[sample_index, :], X_test[sample_index, :],
                    feature_names=feature_names, matplotlib=True)

if __name__ == '__main__':
    show_reduction()
    # show_shap()
