import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import numpy as np

#Load data
train_df = pd.read_csv('Data/train.csv')
test_df  = pd.read_csv('Data/test.csv')

X_train = train_df.drop(columns=['Drafted'])
y_train = train_df['Drafted']
X_test  = test_df.drop(columns=['Drafted'])
y_test  = test_df['Drafted']

#Scale weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() * 1.2

#Finding Params
param_grid = {
    'n_estimators': [350, 300, 400, 450, 500, 250, 200],
    'max_depth': [4, 5, 3, 6, 7],
    'learning_rate': [0.01, 0.05, 0.025, 0.1, 0.02],
    'subsample': [0.7, 0.5, 0.9],
    'colsample_bytree': [0.3, 0.6, 0.75, 0.9],
    'gamma': [0, 0.1, 0.5],
    'min_child_weight': [1, 3, 5, 7, 9],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [0, 0.1, 0.5]
}

search = GridSearchCV(
    estimator=xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=67,
    ),
    param_grid=param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
)
#Use random search to save time when testing param_grid
rand_search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=67,
    ),
    param_distributions=param_grid,  # Same param_grid as before
    n_iter=400,  # Test 200 random combinations instead of all
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=67
)

#This part below is commented out to expedite running the code, uncomment to tune the model

#Fit model

#rand_search.fit(X_train, y_train)

#Print best params

#print("Best params:", rand_search.best_params_)

#FInd roc auc of best params

#best_model = rand_search.best_estimator_
#y_pred_best = best_model.predict(X_test)
#test_roc_auc = roc_auc_score(y_test, y_pred_best)
#print(f"Test ROC-AUC Score: {test_roc_auc:.4f}")

#make classifier
model = xgb.XGBClassifier(
    n_estimators=450,  #450
    max_depth=5,  #5
    learning_rate=0.01, #0.01
    subsample=0.5,  #0.5
    colsample_bytree=0.9, #0.9
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=67,
    gamma = 0, #0.0
    min_child_weight = 7,#7
    reg_alpha = 0.01, #0.01
    reg_lambda = 0 #0.5
)


#Train model
model.fit(X_train, y_train,)

#Test
y_pred = model.predict(X_test)

#Roc-auc
print((f"ROC-AUC Score : {roc_auc_score(y_test, y_pred):.4f}"))


#Feature importance 
import matplotlib.pyplot as plt

importances = model.get_booster().get_score(importance_type='weight')
importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

plt.figure(figsize=(10, 6))
plt.barh(list(importances.keys())[:15], list(importances.values())[:15])
plt.xlabel('Weight')
plt.title('Top 15 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

model.save_model('xgb_final.json')
print("Model saved to xgb_final.json")