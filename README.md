# NBA-Draft-Prediction-Model
Using machine learning to predict which players will get drafted, Guided project through MSU AI club. 

### Final Model parameters:
n_estimators=450, max_depth=5, learning_rate=0.01, subsample=0.5, 
colsample_bytree=0.9, scale_pos_weight= 10.956923076923076, eval_metric='auc', random_state=67,
gamma = 0, min_child_weight = 7, reg_alpha = 0.01, reg_lambda = 0 

### Featured Columns in model
['GP','FGA','ORB', 'DRB', 'STL', 'TOV', 
'FG%', '2P%', '3P%', 'TS%', 'eFG%', 'POS_enc', 'Class_enc',
'PPG', 'APG', 'RPG', 'BPG', 'SPG', 'MPG', '3PAr', 'FTr']