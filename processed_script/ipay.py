# This is for instalment payments dataset
# Import library
import numpy as np  
import pandas as pd
import os
import glob

def preprocessing_ipay():
    # Read data
    path = r'C:\Users\AC037628\Desktop\CEBD1260-dev\Data'
    # path = '/Users/ferdinand/Desktop/data'
    file_name = os.path.join(path,'installments_payments.csv')
    ipay_df = pd.read_csv(file_name)
    
    # fill na as requried
    ipay_df['DAYS_ENTRY_PAYMENT'].fillna(ipay_df['DAYS_ENTRY_PAYMENT'].mean(),inplace=True)
    ipay_df['AMT_PAYMENT'].fillna(ipay_df['AMT_PAYMENT'].mean(),inplace=True)

    # change datatype for optimization 
    ipay_df['NUM_INSTALMENT_VERSION'] = ipay_df['NUM_INSTALMENT_VERSION'].astype(np.int16)
    ipay_df['NUM_INSTALMENT_NUMBER'] = ipay_df['NUM_INSTALMENT_NUMBER'].astype(np.int16)
    ipay_df['DAYS_INSTALMENT'] = ipay_df['DAYS_INSTALMENT'].astype(np.int16)
    ipay_df['DAYS_ENTRY_PAYMENT'] = ipay_df['DAYS_ENTRY_PAYMENT'].astype(np.int16)
    ipay_df['AMT_INSTALMENT'] = ipay_df['AMT_INSTALMENT'].astype(np.int32)
    ipay_df['AMT_PAYMENT'] = ipay_df['AMT_PAYMENT'].astype(np.int32)

    return ipay_df



