import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

matches = pd.read_csv("matches2.csv", index_col=0)
# ML engine requires data to be numeric (float or int). So some conversions must take place. 
# print(matches.dtypes) # highlights some columns that we'd like to use as predictors are objects. Some conversion must occur.
# convert date column
matches['date'] = pd.to_datetime(matches['date'])
# convert home or away to int.
matches['venue_code'] = matches['venue'].astype('category').cat.codes
# give each opponent their own code 
matches['opp_code'] = matches['opponent'].astype('category').cat.codes
# split out each hour of the day
matches['hour'] = matches['time'].str.replace(':.+', '', regex=True).astype('int')
# split out days of the week
matches['day_code'] = matches['date'].dt.dayofweek
# set target by coding wins
matches['target'] = (matches['result'] == 'W').astype('int')

# set up prediction model. RandomForest is chosen because a linear model will not work - the higher the opp_code does not correlate with opp difficulty.
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
# split training and test data
train = matches[matches['date'] < '2022-01-01']
test = matches[matches['date'] > '2022-01-01']
# make predictions
predictors = ['venue_code', 'opp_code', 'hour', 'day_code']
rf.fit(train[predictors], train['target'])
preds = rf.predict(test[predictors])
acc = accuracy_score(test['target'], preds)
# answer is 0.6123188405797102 meaning 61% of time what we predicted came true. 

# create combined df with actual and predicted values to see in which situations our accuracy was high vs. low. 
combined = pd.DataFrame(dict(actual=test['target'], predicted=preds))
comb = pd.crosstab(index=combined['actual'], columns=combined['predicted'])

# when we predicted a win, how often did the team win. 
precision = precision_score(test['target'], preds)
# answer is 0.4745762711864407 meaning we only correctly predicted a win 47% of the time - not great.

# let's improve the model by using rolling averages (if a team isn't playing well they will lose multiple games in a row - this is relevant). 
# split data set up by team. 
grouped_matches = matches.groupby('team')
group = grouped_matches.get_group('Manchester City')

def rolling_averages(group, cols, new_cols):
    group = group.sort_values('date')
    rolling_stats = group[cols].rolling(3, closed='left').mean() # closed=left disregards future data (which the dataset knows but we don't want the model to use).
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols) # most ML models can't handle NA data 
    return group

cols = ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
new_cols = [f'{c}_rolling' for c in cols]

# apply to all matches
matches_rolling = matches.groupby('team').apply(lambda x: rolling_averages(x, cols, new_cols))
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
# error = 0.625 which is up from 47% earlier 
# calling combined is unhelpful because it doesn't give us full picture - so we will add some context:
combined = combined.merge(matches_rolling[['date', 'team', 'opponent', 'result']], left_index=True, right_index=True)

class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    'Brighton and Hove Albion': 'Brighton',
    'Manchester United': 'Manchester Utd',
    'Newcastle United': 'Newcastle Utd',
    'Tottenham Hotspur': 'Tottenham',
    'West Ham United': 'West Ham',
    'Wolverhampton Wanderers': 'Wolves'
}
mapping = MissingDict(**map_values)
combined['new_team'] = combined['team'].map(mapping)
merged = combined.merge(combined, left_on=['date', 'new_team'], right_on=['date', 'opponent'])
# merged[(merged['predicted_x'] == 1) & (merged['predicted_y'] == 0)]['actual_x'].value_counts()

print(merged[(merged['predicted_x'] == 1) & (merged['predicted_y'] == 0)]['actual_x'].value_counts())
# actual_x
# 1    27
# 0    13
# accuracy proven as 0.675

print(27 / 40)