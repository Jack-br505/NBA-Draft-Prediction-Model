import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
import numpy as np

#Load data
train_df = pd.read_csv('Data/train.csv')
test_df  = pd.read_csv('Data/test.csv')

X_train = train_df.drop(columns=['Drafted'])
y_train = train_df['Drafted']
X_test  = test_df.drop(columns=['Drafted'])
y_test  = test_df['Drafted']

#Scale weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

#Finding Params
param_grid = {
    'n_estimators': [300], #Moving params around to find best for each param
    'max_depth': [5],
    'learning_rate': [0.1],
    'subsample': [.75, 0.76, 0.77],
    'colsample_bytree': [0.7, 0.75, .77]
}

search = GridSearchCV(
    estimator=xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=67,
    ),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
)

search.fit(X_train, y_train)

print("Best params:", search.best_params_)
print("Best val accuracy", search.best_score_)

#make classifier
model = xgb.XGBClassifier(
    n_estimators=300, #Best is 300
    max_depth=5, #Best param is 5, originally 4
    learning_rate=0.1, #Best is 0.1, first was .05
    subsample=0.75, #Best = 0.75, og = 0.8
    colsample_bytree=0.75, #Best = 0.75, og = 0.8
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42,
)

#Train model
model.fit(X_train, y_train,)


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