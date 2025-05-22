import warnings
from datetime import datetime
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
import utils
import pickle
# 忽略ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def init_mlp():
    # model_mlp = MLPClassifier(hidden_layer_sizes=(50, 200),
    #                           activation='logistic',
    #                           solver='adam',
    #                           alpha=0.001,
    #                           batch_size='auto',
    #                           learning_rate='constant',
    #                           learning_rate_init=0.001,
    #                           power_t=0.5,
    #                           max_iter=200,
    #                           random_state=42,
    #                           tol=0.0001,
    #                           verbose=False,
    #                           warm_start=False,
    #                           momentum=0.9,
    #                           nesterovs_momentum=True,
    #                           early_stopping=False,
    #                           validation_fraction=0.1,
    #                           n_iter_no_change=10)
    model_mlp = MLPClassifier()
    return model_mlp


def train_mlp():
    df = pd.read_excel("../data/all_data.xlsx")
    # df = pd.read_excel("../data/demo_data.xlsx")

    X_train, X_test, Y_train, Y_test = utils.get_data(df)
    feature_names = df.columns
    feature_names = list(feature_names[3:-1])

    model_mlp = init_mlp()
    # 将数据转换为 DataFrame,绑定特征名
    X_train = pd.DataFrame(X_train, columns=feature_names)
    model_mlp.fit(X_train, Y_train)
    acc = model_mlp.score(X_test, Y_test)
    print(f"测试集准确率: {acc}")
    current_time = datetime.now().strftime("%m-%d_%H%M")
    with open(f'../models/MLP_{current_time}_{acc * 100:.2f}%.pkl', 'wb') as f:
        pickle.dump(model_mlp, f)
    return model_mlp, X_test, Y_test


if __name__ == '__main__':
    train_mlp()
