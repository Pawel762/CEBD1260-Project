# This is for POS Cash balance
# Import library
import numpy as np  
import pandas as pd
import os
import glob

def preprocessing_POS_CASH_balance():
    # Read data
    path = r'C:\Users\AC037628\Desktop\CEBD1260-dev\Data'
    # path = '/Users/ferdinand/Desktop/data'
    file_name = os.path.join(path,'POS_CASH_balance.csv')
    POS_CASH_balance_df = pd.read_csv(file_name)

    # Get non object features
    features = [f for f in POS_CASH_balance_df.columns.values]

    nb_features = []
    for f in features:
        if str(POS_CASH_balance_df[f].dtype) in ['int64','float64']:
            nb_features.append(f)

    # Fill missing values
    POS_CASH_balance_df['CNT_INSTALMENT'].fillna(POS_CASH_balance_df['CNT_INSTALMENT'].mean(),inplace=True)
    POS_CASH_balance_df['CNT_INSTALMENT_FUTURE'].fillna(POS_CASH_balance_df['CNT_INSTALMENT_FUTURE'].mean(),inplace=True)

    # Change datatype for memory optimization
    POS_CASH_balance_df['MONTHS_BALANCE'] = POS_CASH_balance_df['MONTHS_BALANCE'].astype(np.int8)
    POS_CASH_balance_df['CNT_INSTALMENT'] = POS_CASH_balance_df['CNT_INSTALMENT'].astype(np.int8)
    POS_CASH_balance_df['CNT_INSTALMENT_FUTURE'] = POS_CASH_balance_df['CNT_INSTALMENT_FUTURE'].astype(np.int8)
    POS_CASH_balance_df['SK_DPD'] = POS_CASH_balance_df['SK_DPD'].astype(np.int16)
    POS_CASH_balance_df['SK_DPD_DEF'] = POS_CASH_balance_df['SK_DPD_DEF'].astype(np.int16)

    # Remove object feature from dataset
    POS_CASH_balance_df = POS_CASH_balance_df.drop(columns=['NAME_CONTRACT_STATUS'])

    return POS_CASH_balance_df