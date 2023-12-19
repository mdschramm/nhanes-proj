import numpy as np
from preprocessing import process_data
from dataload import NHANES_13_14, select_features
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


def drop_bad_y_rows(df, y_col, bad_values):
    return df[~df[y_col].isin(bad_values)]


# Select features and drop rows with missing/bad labels

'''Question to answer'''
# y_col = 'DPQ030'
y_col = 'BPQ020'
'''Year-specific questionnaires to include as features'''
# features = select_features(['INQ_H', 'PAQ_H', 'SMQ_H'])
features = select_features(["ALQ_H", "CDQ_H", "HSQ_H", "DIQ_H",
                            "IMQ_H", "INQ_H", "MCQ_H", "PAQ_H", "DPQ_H", "SMQ_H", "SMQSHS_H"])
# raw dataframe from nhanes
data = NHANES_13_14.filter(list(features | set([y_col])))
data = drop_bad_y_rows(data, y_col,
                       bad_values=[7, 9, np.nan])

X_raw = data[list(features)]
y = np.round(data[y_col].to_numpy()).astype(int)
X = process_data(X_raw)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=123)

z = list(splitter.split(X, y))
(train_idx, test_idx) = z[0]
X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]


def test_model(model, params):
    clf = model(random_state=123, **params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    print(f'accuracy: {acc}')


# Mean Squared Error: 0.9227709616185261
# R^2 Score: -0.0007543302727242018
# params = {'fit_intercept': True, }
# test_model(Lasso, params)


# Mean Squared Error: 0.22910216718266255
# R^2 Score: -0.025850668598022875
# accuracy: 0.7708978328173375
# params = {'max_iter': 500, 'penalty': 'l1', 'solver': 'liblinear'}
# test_model(LogisticRegression, params)

# Mean Squared Error: 0.22213622291021673
# R^2 Score: 0.005340736866106255
# accuracy: 0.7778637770897833
params = {'n_estimators': 100, 'learning_rate': .1,
          'max_depth': 6}
test_model(GradientBoostingClassifier, params)
