# This is bureau script!

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Defining the directory
DATA_PATH = r'C:\Users\AC037628\Desktop\CEBD1260-dev\Data'
# DATA_PATH = '/Users/ferdinand/Desktop/data'
bureau = os.path.join(DATA_PATH, 'bureau.csv')

def preprocessing_bureau():
    train_df = pd.read_csv(bureau)

    # identifying the Nan
    nas = train_df.isna()
    nacnt = {}
    for f in train_df.columns:
        n = nas[f].sum()
        nacnt[n]=[]
    for f in train_df.columns:
        n = nas[f].sum()
        nacnt[n].append(f)

    # replacing the Nan by the mean. When data is normally distributed, we can use average if the data has a skewed
    # distribution, we should use median instead

    train_df['DAYS_CREDIT_ENDDATE'].fillna(train_df['DAYS_CREDIT_ENDDATE'].mean(), inplace=True)
    train_df['DAYS_ENDDATE_FACT'].fillna(train_df['DAYS_ENDDATE_FACT'].mean(), inplace=True)
    train_df['AMT_CREDIT_MAX_OVERDUE'].fillna(train_df['AMT_CREDIT_MAX_OVERDUE'].mean(), inplace=True)
    train_df['AMT_CREDIT_SUM'].fillna(train_df['AMT_CREDIT_SUM'].mean(), inplace=True)
    train_df['AMT_CREDIT_SUM_DEBT'].fillna(train_df['AMT_CREDIT_SUM_DEBT'].mean(), inplace=True)
    train_df['AMT_CREDIT_SUM_LIMIT'].fillna(train_df['AMT_CREDIT_SUM_LIMIT'].mean(), inplace=True)
    train_df['AMT_ANNUITY'].fillna(train_df['AMT_ANNUITY'].mean(), inplace=True)
    train_df.drop(['CREDIT_ACTIVE', 'CREDIT_CURRENCY','CREDIT_TYPE'], axis=1)

    # Converting to lower format to optimize memory usage

    train_df['SK_ID_CURR'] = train_df['SK_ID_CURR'].astype(np.int32)
    train_df['SK_ID_BUREAU'] = train_df['SK_ID_BUREAU'].astype(np.int32)
    train_df['DAYS_CREDIT'] = train_df['DAYS_CREDIT'].astype(np.int16)
    train_df['CREDIT_DAY_OVERDUE'] = train_df['CREDIT_DAY_OVERDUE'].astype(np.int16)
    train_df['DAYS_CREDIT_ENDDATE'] = train_df['DAYS_CREDIT_ENDDATE'].astype(np.float16)
    train_df['DAYS_ENDDATE_FACT'] = train_df['DAYS_ENDDATE_FACT'].astype(np.float16)
    train_df['AMT_CREDIT_MAX_OVERDUE'] = train_df['AMT_CREDIT_MAX_OVERDUE'].astype(np.float32)
    train_df['CNT_CREDIT_PROLONG'] = train_df['CNT_CREDIT_PROLONG'].astype(np.int8)
    train_df['AMT_CREDIT_SUM'] = train_df['AMT_CREDIT_SUM'].astype(np.float32)
    train_df['AMT_CREDIT_SUM_DEBT'] = train_df['AMT_CREDIT_SUM_DEBT'].astype(np.float32)
    train_df['AMT_CREDIT_SUM_LIMIT'] = train_df['AMT_CREDIT_SUM_LIMIT'].astype(np.float32)
    train_df['AMT_CREDIT_SUM_OVERDUE'] = train_df['AMT_CREDIT_SUM_OVERDUE'].astype(np.float32)
    train_df['DAYS_CREDIT_UPDATE'] = train_df['DAYS_CREDIT_UPDATE'].astype(np.int32)
    train_df['AMT_ANNUITY'] = train_df['AMT_ANNUITY'].astype(np.float32)

    train_df.drop(['CREDIT_ACTIVE', 'CREDIT_CURRENCY','CREDIT_TYPE'], axis=1)
    bureau_df = train_df
    
    return bureau_df