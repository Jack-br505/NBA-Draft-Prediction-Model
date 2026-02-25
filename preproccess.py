#source .venv/bin/activate
#Create dataframe
import pandas as pd

ncaa = pd.read_csv('Data/ncaa_data.csv')

#Fill na values
for col in ncaa.columns.to_list():
    ncaa[col] = ncaa[col].fillna(0)

#Add per game cols
ncaa['PPG']  = ncaa['PTS'] / ncaa['GP']
ncaa['RPG']  = ncaa['TRB'] / ncaa['GP']
ncaa['APG']  = ncaa['AST'] / ncaa['GP']
ncaa['BPG'] = ncaa['BLK'] / ncaa['GP']
ncaa['SPG'] = ncaa['STL'] / ncaa['GP']
ncaa['MPG'] = ncaa['MP'] / ncaa['GP']
ncaa['GS_rate'] = ncaa['GS'] / ncaa['GP']

#Add ratio of shot attempt cols
ncaa['3PAr'] = ncaa['3PA'] / ncaa['FGA']
ncaa['FTr'] = ncaa['FTA'] / ncaa['FGA']

#Change charcters to numeric
pos_map = {'G': 0, 'F': 1, 'C': 2}
ncaa['POS_enc'] = ncaa['POS'].map(pos_map)

yr_map = {'FR':0, 'SO':1, 'JR':2, 'SR':3}
ncaa['Class_enc'] = ncaa['Class '].map(yr_map)

feature_cols = [
    # your full list of columns here
      'GP', 'GS', 'FG',
      'FGA', '2PA', '3PA', 'FT', 'FTA', 'ORB', 
      'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS', 
      'FG%', '2P%', '3P%', 'FT%', 'TS%', 'eFG%', 'POS_enc', 'Class_enc',
      'PPG', 'APG', 'RPG', 'BPG', 'SPG', 'MPG', '3PAr', 'FTr',
]

X = ncaa[feature_cols]
y = ncaa['Drafted']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67, stratify=y
)

# Save training data to CSV
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv('Data/train.csv', index=False)

#Save test dat to csv
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv('Data/test.csv', index=False)