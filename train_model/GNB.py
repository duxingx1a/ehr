from datetime import datetime
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import utils
import pickle

def init_gnb():
    model_gaussian_bayes = GaussianNB()
    return model_gaussian_bayes

def train_gaussian_bayes():
    df = pd.read_excel("../data/all_data.xlsx")
    # df = pd.read_excel("../data/demo_data.xlsx")

    X_train, X_test, Y_train, Y_test = utils.get_data(df)
    feature_names = df.columns
    feature_names = list(feature_names[3:-1])

    model_gaussian_bayes = init_gnb()
    # 将数据转换为 DataFrame,绑定特征名
    X_train = pd.DataFrame(X_train, columns=feature_names)
    model_gaussian_bayes.fit(X_train, Y_train)
    acc = model_gaussian_bayes.score(X_test, Y_test)
    print(f"测试集准确率: {acc}")
    current_time = datetime.now().strftime("%m-%d_%H%M")
    with open(f'../models/GNB_{current_time}_{acc * 100:.2f}%.pkl', 'wb') as f:
        pickle.dump(model_gaussian_bayes, f)
    return model_gaussian_bayes, X_test, Y_test

if __name__ == '__main__':
    train_gaussian_bayes()