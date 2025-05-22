from datetime import datetime
import pandas as pd
from sklearn.svm import SVC
import utils
import pickle

def init_svm():
    model_svm = SVC(kernel="rbf",
                    probability=True,
                    C=5.0,
                    degree=4)
    return model_svm
def train_svm():
    df = pd.read_excel("../data/all_data.xlsx")
    # df = pd.read_excel("../data/demo_data.xlsx")

    X_train, X_test, Y_train, Y_test = utils.get_data(df)
    feature_names = df.columns
    feature_names = list(feature_names[3:-1])

    model_svm = init_svm()
    # 将数据转换为 DataFrame,绑定特征名
    X_train = pd.DataFrame(X_train, columns=feature_names)
    model_svm.fit(X_train, Y_train)
    acc = model_svm.score(X_test, Y_test)
    print(f"测试集准确率: {acc}")
    current_time = datetime.now().strftime("%m-%d_%H%M")
    with open(f'../models/SVM_{current_time}_{acc * 100:.2f}%.pkl', 'wb') as f:
        pickle.dump(model_svm, f)
    return model_svm, X_test, Y_test

if __name__ == '__main__':
    train_svm()


