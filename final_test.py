import xgboost as xgb

#Load model
model = xgb.XGBClassifier()
model.load_model('xgb_final.json')

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd

#Load test data
test_df  = pd.read_csv('Data/test.csv')

X_test  = test_df.drop(columns=['Drafted'])
y_test  = test_df['Drafted']


#Find predicted vals
y_pred = model.predict(X_test)

#Roc-auc
print((f"ROC-AUC Score : {roc_auc_score(y_test, y_pred):.4f}"))

#scores
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"Classification report : \n {classification_report(y_test, y_pred)}")