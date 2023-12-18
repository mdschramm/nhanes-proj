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


def missing_indicator_transformer(df):
    refused_values = [7, 77, 777, 7777, 77777, 777777]
    idk_values = [9, 99, 999, 9999, 99999, 999999]

    df.replace(refused_values, REFUSED_MAGIC_NUM, inplace=True)
    df.replace(idk_values, IDK_MAGIC_NUM, inplace=True)
    df.replace(np.nan, MISSING_MAGIC_NUM, inplace=True)
    print(df['SEQN'])

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


def process_data(df):
    process_pipe = Pipeline(
        [
            # Adds missing indicators for refused, idk, and missing,
            # replaces each of those value types with a magic number for later processing
            ('missing_indicators', FeatureUnion([
                ('existing_data', 'passthrough'),
                ('missing_indicators', missing_indicator_transformer(
                    df)),
            ]))
        ]
    )
    res = process_pipe.fit_transform(df)
    return res


print(NHANES_13_14.shape)
res = process_data(NHANES_13_14)
print(res.shape)
