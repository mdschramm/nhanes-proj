import numpy as np
import pandas as pd
from dataload import NHANES_13_14, NHANES_15_16, NHANES_17_18

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator


print(NHANES_13_14.shape)
assert (len(NHANES_13_14.columns) == len(NHANES_15_16.columns))
assert (len(NHANES_13_14.columns) == len(NHANES_17_18.columns))


# Adds indicators for all refused, idk, and missing columns
def missing_indicators(df):
    refused_values = [7, 77, 777, 7777, 77777, 777777]
    idk_values = [9, 99, 999, 9999, 99999, 999999]
    refused_magic_num, idk_magic_num, missing_magic_num = (
        np.random.uniform(), np.random.uniform(), np.random.uniform())
    df = df.replace(refused_values, refused_magic_num)
    df = df.replace(idk_values, idk_magic_num)
    df = df.replace(np.nan, missing_magic_num)

    return df, ColumnTransformer(
        [
            ('missing', MissingIndicator(
                missing_values=missing_magic_num, features='all'), df.columns),
            ('refused', MissingIndicator(
                missing_values=refused_magic_num, features='all'), df.columns),
            ('idk', MissingIndicator(
             missing_values=idk_magic_num, features='all'), df.columns),

        ]
    )


def preprocess(df):
    new_df, missing_transformer = missing_indicators(df)
    union = FeatureUnion([
        ('existing_data', 'passthrough'),
        ('missing_indicators', missing_transformer),
    ])

    # now need to impute magic numbers with something

    return union.fit_transform(new_df)


processed_data = preprocess(NHANES_13_14)
print(processed_data.shape)
