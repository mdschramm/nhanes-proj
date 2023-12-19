import numpy as np
import pandas as pd
from functools import reduce
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from dataload import NHANES_13_14, questionaire_to_questions_13_14, questionaire_to_questions_15_16, questionaire_to_questions_17_18


REFUSED_MAGIC_NUM, IDK_MAGIC_NUM, MISSING_MAGIC_NUM = (
    np.random.uniform(), np.random.uniform(), np.random.uniform())

# Adds indicators for all refused, idk, and missing columns


def missing_indicator_transformer(df):
    refused_values = [7, 77, 777, 7777, 77777, 777777]
    idk_values = [9, 99, 999, 9999, 99999, 999999]

    df.replace(refused_values, REFUSED_MAGIC_NUM, inplace=True)
    df.replace(idk_values, IDK_MAGIC_NUM, inplace=True)
    df.replace(np.nan, MISSING_MAGIC_NUM, inplace=True)

    return ColumnTransformer(
        [
            ('missing', MissingIndicator(
                missing_values=MISSING_MAGIC_NUM, features='all'), df.columns),
            ('refused', MissingIndicator(
                missing_values=REFUSED_MAGIC_NUM, features='all'), df.columns),
            ('idk', MissingIndicator(
             missing_values=IDK_MAGIC_NUM, features='all'), df.columns),
        ]
    )


class ColumnFilterTransfomer():
    def __init__(self, threshold=.3):
        self.threshold = threshold

    def transform(self, X, y=None):
        n_rows, n_cols = X.shape
        threshold = self.threshold*n_rows
        remove_cols = []
        for i in range(n_cols):
            counts_map = dict(zip(*np.unique(X[:, i], return_counts=True)))
            missing_counts = counts_map[MISSING_MAGIC_NUM] if MISSING_MAGIC_NUM in counts_map else 0
            refused_counts = counts_map[REFUSED_MAGIC_NUM] if REFUSED_MAGIC_NUM in counts_map else 0
            idk_counts = counts_map[IDK_MAGIC_NUM] if IDK_MAGIC_NUM in counts_map else 0
            n_bad = missing_counts + refused_counts + idk_counts
            if n_bad > threshold:
                remove_cols.append(i)

        print('num columns to remove:', len(remove_cols))
        X = np.delete(X, remove_cols, axis=1)
        return X

    def fit(self, df, y=None):
        return self


'''Imputes magic numbers assigned to the refused, idk, and missing columns'''


def impute_workaround():
    unique_val = np.random.uniform()
    return Pipeline(
        [
            # replace all the magic numbers with single magic number
            ('nan_refused', SimpleImputer(strategy='constant',
             fill_value=unique_val, missing_values=REFUSED_MAGIC_NUM)),
            ('nan_idk', SimpleImputer(strategy='constant',
             fill_value=unique_val, missing_values=IDK_MAGIC_NUM)),
            ('nan_missing', SimpleImputer(strategy='constant',
                                          fill_value=unique_val, missing_values=MISSING_MAGIC_NUM)),
            # replace unique value with nan so it can be imputed all at once
            ('nan_11', SimpleImputer(strategy='constant',
                                     fill_value=np.nan, missing_values=unique_val)),
            ('impute', SimpleImputer(strategy='mean', missing_values=np.nan))
        ]
    )


def select_features(df, features, feature_map=questionaire_to_questions_13_14):
    # for 13
    all_features = set()
    for f in features:
        all_features = all_features | feature_map[f]
    return all_features


def process_data(df):
    # drop string valued columns
    df.drop(inplace=True, columns=(df.select_dtypes(
        exclude=['int64', 'float64']).columns))

    features = select_features(df, ['ALQ_H', 'PAQ_H', 'SMQ_H'])
    target = set(['DPQ030'])

    df = df.filter(list(features | target))
    print(df.columns)
    process_pipe = Pipeline(
        [
            # Adds missing indicators for refused, idk, and missing,
            # replaces each of those value types with a magic number for later processing
            ('missing_indicators', FeatureUnion([
                ('existing_data', 'passthrough'),
                ('missing_indicators', missing_indicator_transformer(
                    df)),
            ])),

            ('drop_underfilled_columns', ColumnFilterTransfomer()),
            ('average_impute_magic_numbers', impute_workaround())
        ]
    )
    res = process_pipe.fit_transform(df)
    return res


process_data(NHANES_13_14)
