from datetime import datetime
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import utils
import pickle


def init_lda():
    model_lda = LinearDiscriminantAnalysis(solver='lsqr',
                                           shrinkage='auto',
                                           priors=None)
    return model_lda


def train_LDA():
    df = pd.read_excel("../data/all_data.xlsx")
    # df = pd.read_excel("../data/demo_data.xlsx")

    X_train, X_test, Y_train, Y_test = utils.get_data(df)
    feature_names = df.columns
    feature_names = list(feature_names[3:-1])

    model_lda = init_lda()
    # 将数据转换为 DataFrame,绑定特征名
    X_train = pd.DataFrame(X_train, columns=feature_names)
    model_lda.fit(X_train, Y_train)
    acc = model_lda.score(X_test, Y_test)
    print(f"测试集准确率: {acc}")
    current_time = datetime.now().strftime("%m-%d_%H%M")
    with open(f'../models/LDA_{current_time}_{acc * 100:.2f}%.pkl', 'wb') as f:
        pickle.dump(model_lda, f)
    return model_lda, X_test, Y_test


if __name__ == '__main__':
    train_LDA()
