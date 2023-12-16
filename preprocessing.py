import numpy as np
from dataload import NHANES_13_14, NHANES_15_16, NHANES_17_18

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator


print(NHANES_13_14.shape)
assert (len(NHANES_13_14.columns) == len(NHANES_15_16.columns))
assert (len(NHANES_13_14.columns) == len(NHANES_17_18.columns))

refused_values = [7, 77, 777, 7777, 77777, 777777]
idk_values = [9, 99, 999, 9999, 99999, 999999]

missing_values = refused_values + idk_values + [np.nan, '']


def preprocess(df):
    return df.replace(missing_values, np.nan)


processed_data = preprocess(NHANES_13_14)
print(processed_data.shape)
