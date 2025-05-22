from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from xgboost import XGBClassifier


def init_adaBoost(default_parm=True):
    """
    This function initializes the AdaBoost classifier model.
    """
    if default_parm:
        model_adaboost = AdaBoostClassifier()
    else:
        model_adaboost = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
    return model_adaboost


def init_decisionTree(default_parm=True):
    """
    This function initializes the Decision Tree classifier model.
    """
    if default_parm:
        model_decisionTree = DecisionTreeClassifier()
    else:
        model_decisionTree = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2)
    return model_decisionTree


def init_gaussianNB(default_parm=True):
    """
    This function initializes the Gaussian Naive Bayes classifier model.
    """
    if default_parm:
        model_gaussianNB = GaussianNB()
    else:
        model_gaussianNB = GaussianNB(var_smoothing=1e-9)
    return model_gaussianNB


def init_gradientBoosting(default_parm=True):
    """
    This function initializes the Gradient Boosting classifier model.
    """
    if default_parm:
        model_gradientBoosting = GradientBoostingClassifier()
    else:
        model_gradientBoosting = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    return model_gradientBoosting


def init_lightGBM(default_parm=True):
    """
    This function initializes the LightGBM classifier model.
    """
    if default_parm:
        model_lgbm = LGBMClassifier()
    else:
        model_lgbm = LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=30, learning_rate=0.1, n_estimators=300)
    return model_lgbm


def init_linearDiscriminantAnalysis(default_parm=True):
    """
    This function initializes the Linear Discriminant Analysis classifier model.
    """
    if default_parm:
        model_linearDiscriminantAnalysis = LinearDiscriminantAnalysis()
    else:
        model_linearDiscriminantAnalysis = LinearDiscriminantAnalysis(solver='svd', shrinkage=None)
    return model_linearDiscriminantAnalysis


def init_logisticRegression(default_parm=True):
    """
    This function initializes the Logistic Regression classifier model.
    """
    if default_parm:
        model_logisticRegression = LogisticRegression()
    else:
        model_logisticRegression = LogisticRegression(solver='lbfgs', max_iter=100, random_state=42)
    return model_logisticRegression


def init_MLPClassifier(default_parm=True):
    """
    This function initializes the Multi-layer Perceptron classifier model.
    """
    if default_parm:
        model_MLPClassifier = MLPClassifier()
    else:
        model_MLPClassifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200)
    return model_MLPClassifier


def init_randomForest(default_parm=True):
    """
    This function initializes the Random Forest classifier model.
    """
    if default_parm:
        model_randomForest = RandomForestClassifier()
    else:
        model_randomForest = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    return model_randomForest


def init_XGBoost(default_parm=True):
    """
    This function initializes the XGBoost classifier model.
    """
    if default_parm:
        model_XGBoost = XGBClassifier()
    else:
        model_XGBoost = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=42)
    return model_XGBoost


class Models():
    """
    This class is used to define custom models for the EHR system.
    """

    def __init__(self):
        self.models_list = []
        self.models_names_list = ("model_adab", "model_dt", "model_gb", "model_gnb", "model_lgbm", "model_lda", "model_lr", "model_mlp", "model_rf", "model_xgb")

        # 初始化所有模型
        self.model_adab = init_adaBoost()
        self.model_dt = init_decisionTree()
        self.model_gb = init_gradientBoosting()
        self.model_gnb = init_gaussianNB()
        self.model_lgbm = init_lightGBM()
        self.model_lda = init_linearDiscriminantAnalysis()
        self.model_lr = init_logisticRegression()
        self.model_mlp = init_MLPClassifier()
        self.model_rf = init_randomForest()
        self.model_xgb = init_XGBoost()

        # 将模型和名称添加到列表
        for name in self.models_names_list:
            # 从 self 的属性中获取模型
            model = getattr(self, name)
            self.models_list.append(model)

    def custom_model(self):
        """
        This method defines a custom model.
        """
        pass

    def get_models(self):
        """
        This method retrieves the custom model.
        """
        return self.models_list 

if __name__ == "__main__":
    # 测试 Models 类
    # 这里可以根据需要添加测试代码
    models = Models()
    print(models.get_models())
    print(models.models_names_list)
