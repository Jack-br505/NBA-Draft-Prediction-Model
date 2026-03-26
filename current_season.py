import pandas as pd

curr = pd.read_csv('Data/current_season.csv')


# Fill na values
for col in curr.columns.to_list():
    curr[col] = curr[col].fillna(0)

# Add per game cols
curr['PPG'] = curr['PTS'] / curr['GP']
curr['RPG'] = curr['TRB'] / curr['GP']
curr['APG'] = curr['AST'] / curr['GP']
curr['BPG'] = curr['BLK'] / curr['GP']
curr['SPG'] = curr['STL'] / curr['GP']
curr['MPG'] = curr['MP'] / curr['GP']
curr['GS_rate'] = curr['GS'] / curr['GP']

# Add ratio of shot attempt cols
curr['3PAr'] = curr['3PA'] / curr['FGA']
curr['FTr'] = curr['FTA'] / curr['FGA']

# Change characters to numeric
pos_map = {'G': 0, 'F': 1, 'C': 2}
curr['POS_enc'] = curr['Pos'].map(pos_map)

yr_map = {'FR': 0, 'SO': 1, 'JR': 2, 'SR': 3}
curr['Class_enc'] = curr['Class'].map(yr_map)

feature_cols = [
    # your full list of columns here
      'GP','FGA','ORB', 
      'DRB', 'STL', 'TOV', 
      'FG%', '2P%', '3P%', 'FT%', 'TS%', 'eFG%', 'POS_enc', 'Class_enc',
      'PPG', 'APG', 'RPG', 'BPG', 'SPG', 'MPG', '3PAr', 'FTr',
]

#Import model
import xgboost as xgb

model = xgb.XGBClassifier()
model.load_model('xgb_final.json')

pred_proba = model.predict_proba(curr[feature_cols])[:, 1]
curr['Draft_Prob'] = pred_proba

#Print top 10 players in draft probability
#print(curr.nlargest(n=10, columns = ['Draft_Prob']))

#Print Michigan State players probability
#print(curr[['Player', 'Draft_Prob']][curr['Team'] == "Michigan State"])

import duckdb


con = duckdb.connect(database='curr_season.duckdb')
con.execute("CREATE TABLE IF NOT EXISTS curr AS SELECT * FROM curr")
con.close()

con = duckdb.connect(database='curr_season.duckdb')
#result = con.execute("SELECT Player FROM curr ORDER BY Draft_Prob DESC LIMIT 10;").fetchdf()
result = con.execute("SELECT Player, Draft_Prob FROM curr WHERE Team = 'Michigan State';").fetchdf()
print(result)
con.close()