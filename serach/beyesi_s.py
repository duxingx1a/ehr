from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# 定义贝叶斯搜索的参数空间
param_space = {
    'n_estimators': Integer(50, 600),
    'max_depth': Integer(3, 15),
    'learning_rate': Real(0.01, 0.3, prior='uniform'),
    'subsample': Real(0.5, 1.0, prior='uniform'),
    'colsample_bytree': Real(0.5, 1.0, prior='uniform'),
    'gamma': Real(0, 5, prior='uniform'),
    'reg_alpha': Real(0, 1, prior='uniform'),
    'reg_lambda': Real(0, 1, prior='uniform'),
    'scale_pos_weight': Real(1, 10, prior='uniform'),  # 添加 scale_pos_weight 参数
}

# 初始化 XGBoost 分类器
xgb = XGBClassifier(eval_metric='logloss')

# 初始化贝叶斯搜索
bayes_search = BayesSearchCV(
    estimator=xgb,
    search_spaces=param_space,
    n_iter=300,  # 贝叶斯优化的迭代次数
    cv=5,        # 交叉验证的折数
    scoring='roc_auc',  # 使用 ROC AUC 作为评分标准
    verbose=3,
    random_state=42,
    n_jobs=-1
)

# 加载数据
path = 'ehr/data/Train-test-dataset_Ver4_median_mode.csv'
data = pd.read_csv(path)
target_column = 'have_stone'
X = data.drop(columns=[target_column])
Y = data[target_column]
Y = Y.astype('int')  # 确保标签是整数类型

# 划分训练集和测试集
test_size = 0.2
random_state = 42
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_size, random_state=random_state
)

# 执行贝叶斯搜索
bayes_search.fit(X_train, Y_train)

# 输出最佳参数和得分
print("Best Parameters:", bayes_search.best_params_)
print("Best Score:", bayes_search.best_score_)