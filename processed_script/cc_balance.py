# this is for credit card balance dataset
# Import library
import numpy as np  
import pandas as pd
import os
import glob

def preprocessing_cc_balance():
    # Read data
    path = r'C:\Users\AC037628\Desktop\CEBD1260-dev\Data'
    # path = '/Users/ferdinand/Desktop/data'
    file_name = os.path.join(path,'credit_card_balance.csv')
    cc_balance_df = pd.read_csv(file_name)

    #identify features with int & float
    features = [f for f in cc_balance_df.columns.values]
    nb_features = []
    for f in features:
        if str(cc_balance_df[f].dtype) in ['int64','float64']:
            nb_features.append(f)

    # fill na with mean value
    cc_balance_df['AMT_DRAWINGS_ATM_CURRENT'].fillna(cc_balance_df['AMT_DRAWINGS_ATM_CURRENT'].mean(),inplace=True)
    cc_balance_df['AMT_DRAWINGS_OTHER_CURRENT'].fillna(cc_balance_df['AMT_DRAWINGS_OTHER_CURRENT'].mean(),inplace=True)
    cc_balance_df['AMT_DRAWINGS_POS_CURRENT'].fillna(cc_balance_df['AMT_DRAWINGS_POS_CURRENT'].mean(),inplace=True)
    cc_balance_df['AMT_INST_MIN_REGULARITY'].fillna(cc_balance_df['AMT_INST_MIN_REGULARITY'].mean(),inplace=True)
    cc_balance_df['AMT_PAYMENT_CURRENT'].fillna(cc_balance_df['AMT_PAYMENT_CURRENT'].mean(),inplace=True)
    cc_balance_df['CNT_DRAWINGS_ATM_CURRENT'].fillna(cc_balance_df['CNT_DRAWINGS_ATM_CURRENT'].mean(),inplace=True)
    cc_balance_df['CNT_DRAWINGS_OTHER_CURRENT'].fillna(cc_balance_df['CNT_DRAWINGS_OTHER_CURRENT'].mean(),inplace=True)
    cc_balance_df['CNT_DRAWINGS_POS_CURRENT'].fillna(cc_balance_df['CNT_DRAWINGS_POS_CURRENT'].mean(),inplace=True)
    cc_balance_df['CNT_INSTALMENT_MATURE_CUM'].fillna(cc_balance_df['CNT_INSTALMENT_MATURE_CUM'].mean(),inplace=True)

    # change to appropriate dtypes
    cc_balance_df['MONTHS_BALANCE'] = cc_balance_df['MONTHS_BALANCE'].astype(np.int8)
    cc_balance_df['AMT_BALANCE'] = cc_balance_df['AMT_BALANCE'].astype(np.int32)
    cc_balance_df['AMT_CREDIT_LIMIT_ACTUAL'] = cc_balance_df['AMT_CREDIT_LIMIT_ACTUAL'].astype(np.int32)
    cc_balance_df['AMT_DRAWINGS_ATM_CURRENT'] = cc_balance_df['AMT_DRAWINGS_ATM_CURRENT'].astype(np.int32)
    cc_balance_df['AMT_DRAWINGS_CURRENT'] = cc_balance_df['AMT_DRAWINGS_CURRENT'].astype(np.int32)
    cc_balance_df['AMT_DRAWINGS_OTHER_CURRENT']= cc_balance_df['AMT_DRAWINGS_OTHER_CURRENT'].astype(np.int32)
    cc_balance_df['AMT_DRAWINGS_POS_CURRENT']= cc_balance_df['AMT_DRAWINGS_POS_CURRENT'].astype(np.int32)
    cc_balance_df['AMT_INST_MIN_REGULARITY'] = cc_balance_df['AMT_INST_MIN_REGULARITY'].astype(np.int32)
    cc_balance_df['AMT_PAYMENT_CURRENT'] = cc_balance_df['AMT_PAYMENT_CURRENT'].astype(np.int32)
    cc_balance_df['AMT_PAYMENT_TOTAL_CURRENT'] = cc_balance_df['AMT_PAYMENT_TOTAL_CURRENT'].astype(np.int32)
    cc_balance_df['AMT_RECEIVABLE_PRINCIPAL'] = cc_balance_df['AMT_RECEIVABLE_PRINCIPAL'].astype(np.int32)
    cc_balance_df['AMT_RECIVABLE'] = cc_balance_df['AMT_RECIVABLE'].astype(np.int32)
    cc_balance_df['AMT_TOTAL_RECEIVABLE'] = cc_balance_df['AMT_TOTAL_RECEIVABLE'].astype(np.int32)
    cc_balance_df['CNT_DRAWINGS_ATM_CURRENT'] = cc_balance_df['CNT_DRAWINGS_ATM_CURRENT'].astype(np.int8)
    cc_balance_df['CNT_DRAWINGS_CURRENT'] = cc_balance_df['CNT_DRAWINGS_CURRENT'].astype(np.int16)
    cc_balance_df['CNT_DRAWINGS_OTHER_CURRENT'] = cc_balance_df['CNT_DRAWINGS_OTHER_CURRENT'].astype(np.int8)
    cc_balance_df['CNT_DRAWINGS_POS_CURRENT'] = cc_balance_df['CNT_DRAWINGS_POS_CURRENT'].astype(np.int16)
    cc_balance_df['CNT_INSTALMENT_MATURE_CUM'] = cc_balance_df['CNT_INSTALMENT_MATURE_CUM'].astype(np.int8)
    cc_balance_df['SK_DPD'] = cc_balance_df['SK_DPD'].astype(np.int16)
    cc_balance_df['SK_DPD_DEF'] = cc_balance_df['SK_DPD_DEF'].astype(np.int16)

    # modify object datatype
    cc_balance_df['NAME_CONTRACT_STATUS'] = cc_balance_df['NAME_CONTRACT_STATUS'].astype('category')
    cc_balance_df['CAT_NAME_CONTRACT_STATUS'] = cc_balance_df['NAME_CONTRACT_STATUS'].cat.codes
    cc_balance_df = cc_balance_df.drop(columns=['NAME_CONTRACT_STATUS'])

    return cc_balance_df

