import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score

#Load data
train_df = pd.read_csv('Data/train.csv')
test_df  = pd.read_csv('Data/test.csv')

X_train = train_df.drop(columns=['Drafted'])
y_train = train_df['Drafted']
X_test  = test_df.drop(columns=['Drafted'])
y_test  = test_df['Drafted']

#Scale weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

#make classifier
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=67,
)

#Train model
model.fit(X_train, y_train,)

#Test
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

print(f"ROC-AUC Score : {roc_auc_score(y_test, y_pred):.4f}")
print(f"Classification report : \n {classification_report(y_test, y_pred)}")
print(f"Confusion Matrix : \n {confusion_matrix(y_test, y_pred)}")