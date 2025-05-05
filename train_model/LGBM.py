from datetime import datetime
import pandas as pd
from lightgbm import LGBMClassifier
import pickle
import utils


def init_lgbm():
    model_lgbm = LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=30,
        learning_rate=0.1,
        n_estimators=300,
        objective='binary',  # 默认是二分类
        min_split_gain=0.0,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=1,
        reg_lambda=1,
        random_state=0,
        verbosity=-1
    )
    return model_lgbm


def train_lightgbm():
    # df = pd.read_excel("../data/all_data.xlsx")
    df = pd.read_excel("../data/demo_data.xlsx")

    X_train, X_test, Y_train, Y_test = utils.get_data(df)
    feature_names = df.columns
    feature_names = list(feature_names[3:-1])

    model_lgbm = init_lgbm()
    # 将数据转换为 DataFrame,绑定特征名
    X_train = pd.DataFrame(X_train, columns=feature_names)
    model_lgbm.fit(X_train, Y_train)
    acc = model_lgbm.score(X_test, Y_test)
    print(f"测试集准确率: {acc}")
    current_time = datetime.now().strftime("%m-%d_%H%M")
    with open(f'../models/LGBM_{current_time}_{acc * 100:.2f}%.pkl', 'wb') as f:
        pickle.dump(model_lgbm, f)
    return model_lgbm, X_test, Y_test


if __name__ == '__main__':
    train_lightgbm()
