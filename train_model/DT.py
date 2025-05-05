from datetime import datetime
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import utils
import pickle

def init_dt():
    model_dt = DecisionTreeClassifier(criterion="gini",
                                      splitter="best",
                                      max_depth=20,
                                      min_samples_split=4,
                                      min_samples_leaf=2,
                                      min_weight_fraction_leaf=0.,
                                      max_features=None,
                                      random_state=None,
                                      max_leaf_nodes=None,
                                      min_impurity_decrease=0.,
                                      # min_impurity_split=None,
                                      class_weight=None,
                                      # presort='deprecated',
                                      ccp_alpha=0.0)
    return model_dt
def train_decision_tree():
    df = pd.read_excel("../data/all_data.xlsx")
    # df = pd.read_excel("../data/demo_data.xlsx")

    X_train, X_test, Y_train, Y_test = utils.get_data(df)
    feature_names = df.columns
    feature_names = list(feature_names[3:-1])

    model_dt = init_dt()
    # 将数据转换为 DataFrame,绑定特征名
    X_train = pd.DataFrame(X_train, columns=feature_names)
    model_dt.fit(X_train, Y_train)
    acc = model_dt.score(X_test, Y_test)
    print(f"测试集准确率: {acc}")
    current_time = datetime.now().strftime("%m-%d_%H%M")
    with open(f'models/DT_{current_time}_{acc * 100:.2f}%.pkl', 'wb') as f:
        pickle.dump(model_dt, f)
    return model_dt, X_test, Y_test


if __name__ == '__main__':
    train_decision_tree()
