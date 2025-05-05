import warnings
from datetime import datetime
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import utils
import pickle
# 忽略ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
def init_lr():
    model_lr = LogisticRegression(max_iter=1000,
                                  random_state=42,
                                  solver='lbfgs',
                                  penalty='l2',
                                  C=1.0,
                                  )
    return model_lr

def train_lr():
    df = pd.read_excel("../data/all_data.xlsx")
    # df = pd.read_excel("../data/demo_data.xlsx")

    X_train, X_test, Y_train, Y_test = utils.get_data(df)
    feature_names = df.columns
    feature_names = list(feature_names[3:-1])

    model_lr = init_lr()
    # 将数据转换为 DataFrame,绑定特征名
    X_train = pd.DataFrame(X_train, columns=feature_names)
    model_lr.fit(X_train, Y_train)
    acc = model_lr.score(X_test, Y_test)
    print(f"测试集准确率: {acc}")
    current_time = datetime.now().strftime("%m-%d_%H%M")
    with open(f'../models/LR_{current_time}_{acc * 100:.2f}%.pkl', 'wb') as f:
        pickle.dump(model_lr, f)
    return model_lr, X_test, Y_test

if __name__ == '__main__':
    train_lr()