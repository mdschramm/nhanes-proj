import numpy as np
import pandas as pd
from dataload import NHANES_13_14, NHANES_15_16, NHANES_17_18

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator


assert (len(NHANES_13_14.columns) == len(NHANES_15_16.columns))
assert (len(NHANES_13_14.columns) == len(NHANES_17_18.columns))


REFUSED_MAGIC_NUM, IDK_MAGIC_NUM, MISSING_MAGIC_NUM = (
    np.random.uniform(), np.random.uniform(), np.random.uniform())

# Adds indicators for all refused, idk, and missing columns
print('REFUSED_MAGIC_NUM', REFUSED_MAGIC_NUM, 'IDK_MAGIC_NUM',
      IDK_MAGIC_NUM, 'MISSING_MAGIC_NUM', MISSING_MAGIC_NUM)


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


def process_data(df):
    # replace all strings in dataset
    df.drop(inplace=True, columns=(df.select_dtypes(
        exclude=['int64', 'float64']).columns))
    process_pipe = Pipeline(
        [
            # Adds missing indicators for refused, idk, and missing,
            # replaces each of those value types with a magic number for later processing
            ('missing_indicators', FeatureUnion([
                ('existing_data', 'passthrough'),
                ('missing_indicators', missing_indicator_transformer(
                    df)),
            ])),

            ('drop_underfilled_columns', ColumnFilterTransfomer())
        ]
    )
    res = process_pipe.fit_transform(df)
    return res


res = process_data(NHANES_13_14)
print(res.shape)
