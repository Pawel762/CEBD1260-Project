# Script for bureau_balance.csv
import numpy as np  
import pandas as pd
import os
import glob
# Defining the directory

def preprocessing_bureau_balance():
    path = r'C:\Users\AC037628\Desktop\CEBD1260-dev\Data'
    # path = '/Users/ferdinand/Desktop/data'
    file_name = os.path.join(path, 'bureau_balance.csv')
    train_df = pd.read_csv(file_name)

    # identifying the Nan
    nas = train_df.isna()
    nacnt = {}
    for f in train_df.columns:
        n = nas[f].sum()
        nacnt[n] = []
    for f in train_df.columns:
        n = nas[f].sum()
        nacnt[n].append(f)

    train_df.drop(['STATUS'], axis=1)

    # Converting to lower format to optimize memory usage
    train_df['MONTHS_BALANCE'] = train_df['MONTHS_BALANCE'].astype(np.int8)

    bureau_balance_df = train_df
    return bureau_balance_df