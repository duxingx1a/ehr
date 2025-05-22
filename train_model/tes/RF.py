from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import utils
import pickle


def init_rf():
    # model_rf = RandomForestClassifier(n_estimators=300,
    #                                   oob_score=True,
    #                                   max_depth=30,
    #                                   n_jobs=8,
    #                                   random_state=42)
    model_rf = RandomForestClassifier(n_jobs=-1,)
    return model_rf


def train_random_forest():
    df = pd.read_excel("../data/all_data.xlsx")
    # df = pd.read_excel("../data/demo_data.xlsx")

    X_train, X_test, Y_train, Y_test = utils.get_data(df)
    feature_names = df.columns
    feature_names = list(feature_names[3:-1])

    model_rf = init_rf()
    # 将数据转换为 DataFrame,绑定特征名
    X_train = pd.DataFrame(X_train, columns=feature_names)
    model_rf.fit(X_train, Y_train)
    acc = model_rf.score(X_test, Y_test)
    print(f"测试集准确率: {acc}")
    current_time = datetime.now().strftime("%m-%d_%H%M")
    with open(f'../models/RF_{current_time}_{acc * 100:.2f}%.pkl', 'wb') as f:
        pickle.dump(model_rf, f)
    return model_rf, X_test, Y_test


if __name__ == '__main__':
    train_random_forest()
