from datetime import datetime
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
import utils
import pickle


def init_adb():
    # model_adaboost = AdaBoostClassifier(n_estimators=100,
    #                                     learning_rate=1.0,
    #                                     random_state=42)
    model_adaboost = AdaBoostClassifier()
    return model_adaboost


def train_adaboost():
    df = pd.read_excel("../data/all_data.xlsx")
    # df = pd.read_excel("../data/demo_data.xlsx")

    X_train, X_test, Y_train, Y_test = utils.get_data(df)
    feature_names = df.columns
    feature_names = list(feature_names[3:-1])

    model_adaboost = init_adb()
    # 将数据转换为 DataFrame,绑定特征名
    X_train = pd.DataFrame(X_train, columns=feature_names)
    model_adaboost.fit(X_train, Y_train)
    acc = model_adaboost.score(X_test, Y_test)
    print(f"测试集准确率: {acc}")
    current_time = datetime.now().strftime("%m-%d_%H%M")
    with open(f'../models/AdB_{current_time}_{acc * 100:.2f}%.pkl', 'wb') as f:
        pickle.dump(model_adaboost, f)
    return model_adaboost, X_test, Y_test


if __name__ == '__main__':
    train_adaboost()
