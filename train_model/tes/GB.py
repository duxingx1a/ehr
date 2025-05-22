from datetime import datetime
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import utils
import pickle


def init_gb():
    # model_gb = GradientBoostingClassifier(n_estimators=100,
    #                                       learning_rate=0.1,
    #                                       max_depth=3,
    #                                       random_state=42)
    model_gb = GradientBoostingClassifier()
    return model_gb


def train_gb():
    df = pd.read_excel("../data/all_data.xlsx")
    # df = pd.read_excel("../data/demo_data.xlsx")

    X_train, X_test, Y_train, Y_test = utils.get_data(df)
    feature_names = df.columns
    feature_names = list(feature_names[3:-1])

    model_gb = init_gb()
    # 将数据转换为 DataFrame,绑定特征名
    X_train = pd.DataFrame(X_train, columns=feature_names)
    model_gb.fit(X_train, Y_train)
    acc = model_gb.score(X_test, Y_test)
    print(f"测试集准确率: {acc}")
    current_time = datetime.now().strftime("%m-%d_%H%M")
    with open(f'../models/GB_{current_time}_{acc * 100:.2f}%.pkl', 'wb') as f:
        pickle.dump(model_gb, f)
    return model_gb, X_test, Y_test


if __name__ == '__main__':
    train_gb()
