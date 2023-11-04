import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

data = pd.read_csv('matches_champ.csv', index_col=0)

# print(data.head, data.dtypes) - reveals that date, venue, opponent, result are all object types and need to be numeric (float or integer) to be used by ML engine.

# some data cleansing
data['date'] = pd.to_datetime(data['date'])
data['venue_code'] = data['venue'].astype('category').cat.codes
data['opp_code'] = data['opponent'].astype('category').cat.codes
# target also needs to be coded
data['target'] = (data['result'] == 'W').astype('int')

# setup ML model
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = data[data['date'] < '2022-01-01']
test = data[data['date'] > '2022-01-01']
predictors = ['venue_code', 'opp_code']
rf.fit(train[predictors], train['target'])
prediction = rf.predict(test[predictors])

# We need to work out how good the model is.
acc = accuracy_score(test['target'], prediction) # answer = 0.5980980980980981. (Correct Pred / All Preds)
combined = pd.DataFrame(dict(actual=test['target'], predicted=prediction))
comb = pd.crosstab(index=combined['actual'], columns=combined['predicted'])
precision = precision_score(test['target'], prediction) # answer = 0.42592592592592593. (True Preds / True Preds + False Preds)
recall = recall_score(test['target'], prediction) # answer = 0.21552878179384202. (Correct Positive Preds / All Positive Instances)


# improving the data set.
grouped_data = data.groupby('team')
group = grouped_data.get_group('Norwich City')

def rolling_averages(group, cols, new_cols):
    group = group.sort_values('date')
    rolling_stats = group[cols].rolling(3, closed='left').mean() # closed=left disregards future data (which the dataset knows but we don't want the model to use).
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols) # most ML models can't handle NA data 
    return group

cols = ['sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
new_cols = [f'{c}_rolling' for c in cols]

# apply to all matches
matches_rolling = data.groupby('team').apply(lambda x: rolling_averages(x, cols, new_cols))
# drop index level
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])

# let's make a new set of predictions
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    error = precision_score(test["target"], preds)
    return combined, error

combined, error = make_predictions(matches_rolling, predictors + new_cols)
# error = 0.44542772861356933 which is up from 42.5% earlier 
# calling combined is unhelpful because it doesn't give us full picture - so we will add some context:
combined = combined.merge(matches_rolling[['date', 'team', 'opponent', 'result']], left_index=True, right_index=True)

class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    'Sheffield United': 'Sheffield United',
    'P\'borough Utd': 'Peterborough United',
    'Queens Park Rangers': 'QPR',
    'Rotherham Utd': 'Rotherham United',
    'West Bromwich Albion': 'West Brom',
    'Nott\'ham Forest': 'Nottingham Forest'
}

mapping = MissingDict(**map_values)
combined['new_team'] = combined['team'].map(mapping)
merged = combined.merge(combined, left_on=['date', 'new_team'], right_on=['date', 'opponent'])

print(merged[(merged['predicted_x'] == 1) & (merged['predicted_y'] == 0)]['actual_x'].value_counts())
# actual_x
# 0    123
# 1    100
print(100 / 223) # 0.4484304932735426
