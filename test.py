import pandas as pd
import ehr_utils
# test_file_path = 'data_clean/Train-test-dataset_Ver-electronic-haveoutliers_xgb.csv'
test_file_path = 'data_clean/Train-test-dataset_Ver-electronic-haveoutliers-delete8cols_mean_mode.csv'
df = pd.read_csv(test_file_path)
X_train, X_test, Y_train, Y_test = ehr_utils.get_train_test(df, train_size=0.8, random_state=42)
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc, f1_score

model = xgb.XGBClassifier(n_estimators=800,
                          max_depth=8,
                          learning_rate=0.01,
                          eval_metric='auc',
                          device='gpu',
                          subsample=1.0,
                          colsample_bytree=0.5,
                          reg_lambda=0.9953311090514885,
                          reg_alpha=0.4581043332068245,
                          scale_pos_weight=8.0)

model.fit(X_train, Y_train)
Y_pred_proba,Y_pred = ehr_utils.get_prediction_results(model,X_test)
print(ehr_utils.eval_model(Y_test, Y_pred, Y_pred_proba))
