import xgboost as xgb
from datetime import datetime
import pandas as pd
import utils
import pickle

def init_xgb():
    # model_xgb = xgb.XGBClassifier(learning_rate=0.05,
    #                               n_estimators=300,
    #                               max_depth=5,
    #                               min_child_weight=2,
    #                               gamma=0.2,
    #                               subsample=0.8,
    #                               colsample_bytree=1,
    #                               objective='binary:logistic',
    #                               nthread=-1,
    #                               scale_pos_weight=10,
    #                               seed=42,
    #                               reg_alpha=0,
    #                               reg_lambda=1,
    #                               eval_metric='auc')
    model_xgb = xgb.XGBClassifier(nthread=-1,device='gpu')
    return model_xgb

def train_xgb():
    df = pd.read_excel("../data/all_data.xlsx")
    # df = pd.read_excel("../data/demo_data.xlsx")

    X_train, X_test, Y_train, Y_test = utils.get_data(df)
    feature_names = df.columns
    feature_names = list(feature_names[3:-1])

    model_xgb = init_xgb()
    # 将数据转换为 DataFrame,绑定特征名
    X_train = pd.DataFrame(X_train, columns=feature_names)
    model_xgb.fit(X_train, Y_train)
    acc = model_xgb.score(X_test, Y_test)
    print(f"测试集准确率: {acc}")
    current_time = datetime.now().strftime("%m-%d_%H%M")
    with open(f'../models/XGB_{current_time}_{acc * 100:.2f}%.pkl', 'wb') as f:
        pickle.dump(model_xgb, f)
    return model_xgb, X_test, Y_test

if __name__ == '__main__':
    train_xgb()