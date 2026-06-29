import os
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

path = './data'

# read the data used for training
train_data_path = os.path.join(path, 'train.csv')
train_data = pd.read_csv(train_data_path)

# filter the not available columns from training data
filtered_train_data = train_data.dropna(axis=1)

# read the data used for submission
test_data_path = os.path.join(path, 'test.csv')
test_data = pd.read_csv(test_data_path)

best_k = 0
lowest_value = float('inf')

# remove the 'Id' column which do not contain any relevant information regarding the sale prices
# and 'SalePrice' column which is the target
train_data_columns = list(filtered_train_data.columns)
remove_columns = ['SalePrice', 'Id']

for column in remove_columns:
    train_data_columns.remove(column)

y = filtered_train_data.SalePrice
X = filtered_train_data[train_data_columns]

# let Pandas automatically translate all text into 1s and 0s
X = pd.get_dummies(X)

# split data into training data and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# find the best features to use for the model
n = len(train_data_columns)
for k in range(1, n):
    # filter the data for the current 'k' best features
    feature_judge = SelectKBest(score_func=f_regression, k=k)
    train_X_k = feature_judge.fit_transform(train_X, train_y)
    val_X_k = feature_judge.transform(val_X)

    # train the model and make prediction
    model = RandomForestRegressor(random_state=1)
    model.fit(train_X_k, train_y)
    prediction = model.predict(val_X_k)

    # calculate the error
    current_error = mean_absolute_error(prediction, val_y)

    if current_error < lowest_value:
        best_k = k
        lowest_value = current_error

print(lowest_value)
print(best_k)

features_judge = SelectKBest(score_func=f_regression, k=best_k)
train_X_k = features_judge.fit_transform(train_X, train_y)

prediction_model = RandomForestRegressor(random_state=1)
prediction_model.fit(train_X_k, train_y)

# extract best k features obtained from features_judge
best_features = features_judge.get_feature_names_out()

X = pd.get_dummies(test_data)

# align the data with the training data so the column match
_, X = train_X.align(X, join='left', axis=1, fill_value=0)

X = X[best_features]

# transform the not available values into the average values
X = X.fillna(X.mean())

prediction = prediction_model.predict(X.values)

solution = pd.DataFrame({'Id': test_data.Id, 'SalePrice': prediction})
solution.to_csv('submission.csv', index=False)