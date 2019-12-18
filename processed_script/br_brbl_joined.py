# This is the script to merge Left join bureau_bl.csv to bureau.csv

import pandas as pd
import numpy as np
import os
import glob


# Defining the directory

def preprocessing_br_brbl():
    # Importing multiple .csv

    path = 'C:\\Users\\AC037628\\Desktop\\CEBD1260-dev\\Data\\'
    files = glob.glob(path + '*.csv')

    print('Importing files...')

    # This references the bureau balanced csv
    brbl_df = pd.read_csv(files[2])

    # This references the bureau csv
    br_df = pd.read_csv(files[1])

    print('Starting with bureau_balance.csv')
    # ------------------------------------------------------------------------------------------------------------
    # This is the br_balance script with aggregation

    # Defining the directory
    #
    # DATA_PATH = 'C:\\Users\\KALPAW01\\Dropbox\\Data_for_ML_course'
    # br_balance = os.path.join(DATA_PATH, 'bureau_balance.csv')
    #
    # br_balance_df = pd.read_csv(br_balance)

    #
    # Creating the group by
    gp = brbl_df.groupby('SK_ID_BUREAU')

    # Aggregating the MONTHS_BALANCE by using median
    add1_df = pd.DataFrame(gp['MONTHS_BALANCE'].median())

    # Renamed the column to make it easier to ID after joining to Main table
    add1_df = add1_df.rename({'MONTHS_BALANCE': 'BRBL_MONTHS_BALANCE_median'}, axis=1)

    print('Getting the frequency of every STATUS')
    # Using a pivot table to get the frequency of every STATUS
    STS_Agg = brbl_df.pivot_table(index="SK_ID_BUREAU", columns="STATUS", fill_value=0, aggfunc="count")
    # Arrange column names to be more readable
    STS_Agg.columns = ['BURO_{}_{}'.format(x[0], x[1]) for x in STS_Agg.columns.tolist()]
    STS_Agg

    # Joining the pivot table (STS_Agg) with the aggregated MONTH_BALANCE (add_df)
    add2_df = add1_df.merge(STS_Agg, on='SK_ID_BUREAU', how='left')

    print('Renaming columns')
    # Renaming the columns created by the pivot table
    add2_df.rename({'BURO_MONTHS_BALANCE_0': 'BRBL_STATUS_0', 'BURO_MONTHS_BALANCE_1': 'BRBL_STATUS_1',
                    'BURO_MONTHS_BALANCE_2': 'BRBL_STATUS_2', 'BURO_MONTHS_BALANCE_3': 'BRBL_STATUS_3',
                    'BURO_MONTHS_BALANCE_4': 'BRBL_STATUS_4', 'BURO_MONTHS_BALANCE_5': 'BRBL_STATUS_5',
                    'BURO_MONTHS_BALANCE_C': 'BRBL_STATUS_C', 'BURO_MONTHS_BALANCE_X': 'BRBL_STATUS_X'}, axis=1,
                   inplace=True)

    # Converting the pivot to Dataframe
    add3_df = pd.DataFrame(add2_df.to_records())
    # ------------------------------------------------------------------------------------------------------------
    print('Joining aggregated bureau balance to bureau...')

    # Joining the aggregated bureau balance to the bureau csv
    br_brbl_mrg = br_df.merge(add3_df, on='SK_ID_BUREAU', how='left')

    # ------------------------------------------------------------------------------------------------------------

    # replacing the Nan by the mean. When data is normally distributed, we can use average if the data has a skewed
    # distribution, we should use median instead

    print('Replacing the NAN...')

    br_brbl_mrg['DAYS_CREDIT_ENDDATE'].fillna(br_brbl_mrg['DAYS_CREDIT_ENDDATE'].median(), inplace=True)
    br_brbl_mrg['DAYS_ENDDATE_FACT'].fillna(br_brbl_mrg['DAYS_ENDDATE_FACT'].median(), inplace=True)
    br_brbl_mrg['AMT_CREDIT_MAX_OVERDUE'].fillna(br_brbl_mrg['AMT_CREDIT_MAX_OVERDUE'].median(), inplace=True)
    br_brbl_mrg['AMT_CREDIT_SUM'].fillna(br_brbl_mrg['AMT_CREDIT_SUM'].median(), inplace=True)
    br_brbl_mrg['AMT_CREDIT_SUM_DEBT'].fillna(br_brbl_mrg['AMT_CREDIT_SUM_DEBT'].median(), inplace=True)
    br_brbl_mrg['AMT_CREDIT_SUM_LIMIT'].fillna(br_brbl_mrg['AMT_CREDIT_SUM_LIMIT'].median(), inplace=True)
    br_brbl_mrg['AMT_ANNUITY'].fillna(br_brbl_mrg['AMT_ANNUITY'].median(), inplace=True)

    # Memory usage prior to optimization

    print('Memory usage prior to optimization')

    mem_use = br_brbl_mrg.memory_usage().sum() / 1024 ** 3
    print('Memory usage of dataframe is {:.2f} GB'.format(mem_use))

    print('Converting to lower format to optimize memory usage')

    br_brbl_mrg['SK_ID_CURR'] = br_brbl_mrg['SK_ID_CURR'].astype(np.int32)
    br_brbl_mrg['SK_ID_BUREAU'] = br_brbl_mrg['SK_ID_BUREAU'].astype(np.int32)
    br_brbl_mrg['DAYS_CREDIT'] = br_brbl_mrg['DAYS_CREDIT'].astype(np.int16)
    br_brbl_mrg['CREDIT_DAY_OVERDUE'] = br_brbl_mrg['CREDIT_DAY_OVERDUE'].astype(np.int16)
    br_brbl_mrg['DAYS_CREDIT_ENDDATE'] = br_brbl_mrg['DAYS_CREDIT_ENDDATE'].astype(np.float16)
    br_brbl_mrg['DAYS_ENDDATE_FACT'] = br_brbl_mrg['DAYS_ENDDATE_FACT'].astype(np.float16)
    br_brbl_mrg['AMT_CREDIT_MAX_OVERDUE'] = br_brbl_mrg['AMT_CREDIT_MAX_OVERDUE'].astype(np.float32)
    br_brbl_mrg['CNT_CREDIT_PROLONG'] = br_brbl_mrg['CNT_CREDIT_PROLONG'].astype(np.int8)
    br_brbl_mrg['AMT_CREDIT_SUM'] = br_brbl_mrg['AMT_CREDIT_SUM'].astype(np.float32)
    br_brbl_mrg['AMT_CREDIT_SUM_DEBT'] = br_brbl_mrg['AMT_CREDIT_SUM_DEBT'].astype(np.float32)
    br_brbl_mrg['AMT_CREDIT_SUM_LIMIT'] = br_brbl_mrg['AMT_CREDIT_SUM_LIMIT'].astype(np.float32)
    br_brbl_mrg['AMT_CREDIT_SUM_OVERDUE'] = br_brbl_mrg['AMT_CREDIT_SUM_OVERDUE'].astype(np.float32)
    br_brbl_mrg['DAYS_CREDIT_UPDATE'] = br_brbl_mrg['DAYS_CREDIT_UPDATE'].astype(np.int32)
    br_brbl_mrg['AMT_ANNUITY'] = br_brbl_mrg['AMT_ANNUITY'].astype(np.float32)

    # Memory usage after optimization

    print('Memory usage after optimization')

    mem_use = br_brbl_mrg.memory_usage().sum() / 1024 ** 3
    print('Memory usage of dataframe is {:.2f} GB'.format(mem_use))

    # Encoding the objects by using one-hot-coding

    print('Encoding the objects by using one-hot-coding')
    temp_df = pd.get_dummies(br_brbl_mrg['CREDIT_ACTIVE'], prefix='CREDIT_ACTIVE')
    br_brbl_mrg = pd.concat([br_brbl_mrg, temp_df], axis=1)

    temp_df = pd.get_dummies(br_brbl_mrg['CREDIT_CURRENCY'], prefix='CREDIT_CURRENCY')
    br_brbl_mrg = pd.concat([br_brbl_mrg, temp_df], axis=1)

    temp_df = pd.get_dummies(br_brbl_mrg['CREDIT_TYPE'], prefix='CREDIT_TYPE')
    br_brbl_mrg = pd.concat([br_brbl_mrg, temp_df], axis=1)

    # Dropping the one-hot-coded columns (CREDIT_ACTIVE AND CREDIT_CURRENCY)
    br_brbl_mrg.drop(['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE'], axis=1)

    # Aggregating the features
    print('Aggregating features...')
    client_ids = list(br_brbl_mrg['SK_ID_CURR'].unique())
    temp_df1 = br_brbl_mrg[br_brbl_mrg['SK_ID_CURR'].isin(client_ids)]

    agg_dict = {
        'CREDIT_ACTIVE_Active': ['count'],
        'CREDIT_ACTIVE_Bad debt': ['count'],
        'CREDIT_ACTIVE_Closed': ['count'],
        'CREDIT_ACTIVE_Sold': ['count'],
        'CREDIT_CURRENCY_currency 1': ['count'],
        'CREDIT_CURRENCY_currency 2': ['count'],
        'CREDIT_CURRENCY_currency 3': ['count'],
        'CREDIT_CURRENCY_currency 4': ['count'],
        'CREDIT_TYPE_Another type of loan': ['count'],
        'CREDIT_TYPE_Car loan': ['count'],
        'CREDIT_TYPE_Cash loan (non-earmarked)': ['count'],
        'CREDIT_TYPE_Consumer credit': ['count'],
        'CREDIT_TYPE_Credit card': ['count'],
        'CREDIT_TYPE_Interbank credit': ['count'],
        'CREDIT_TYPE_Loan for business development': ['count'],
        'CREDIT_TYPE_Loan for purchase of shares (margin lending)': ['count'],
        'CREDIT_TYPE_Loan for the purchase of equipment': ['count'],
        'CREDIT_TYPE_Loan for working capital replenishment': ['count'],
        'CREDIT_TYPE_Microloan': ['count'],
        'CREDIT_TYPE_Mobile operator loan': ['count'],
        'CREDIT_TYPE_Real estate loan': ['count'],
        'CREDIT_TYPE_Unknown type of loan': ['count'],
        'DAYS_CREDIT': ['mean', 'median', 'std', 'max', 'min'],
        'CREDIT_DAY_OVERDUE': ['mean', 'median', 'std', 'max', 'min'],
        'DAYS_CREDIT_ENDDATE': ['mean', 'median', 'std', 'max', 'min'],
        'DAYS_ENDDATE_FACT': ['mean', 'median', 'std', 'max', 'min'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'median', 'std', 'max', 'min'],
        'CNT_CREDIT_PROLONG': ['mean', 'median', 'std', 'max', 'min'],
        'AMT_CREDIT_SUM': ['mean', 'median', 'std', 'max', 'min'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'median', 'std', 'max', 'min'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'median', 'std', 'max', 'min'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean', 'median', 'std', 'max', 'min'],
        'DAYS_CREDIT_UPDATE': ['mean', 'median', 'std', 'max', 'min'],
        'AMT_ANNUITY': ['mean', 'median', 'std', 'max', 'min'],
        'BRBL_MONTHS_BALANCE_median': ['median'],
        'BRBL_STATUS_0': ['count'],
        'BRBL_STATUS_1': ['count'],
        'BRBL_STATUS_2': ['count'],
        'BRBL_STATUS_3': ['count'],
        'BRBL_STATUS_4': ['count'],
        'BRBL_STATUS_5': ['count'],
        'BRBL_STATUS_C': ['count'],
        'BRBL_STATUS_X': ['count']
    }

    agg_df = temp_df1.groupby('SK_ID_CURR').agg(agg_dict)
    # Arrange columns names to be more readable
    agg_df.columns = ['BURO_{}_{}'.format(x[0], x[1]) for x in agg_df.columns.tolist()]

    agg1_df = pd.DataFrame(agg_df.to_records())

    print(agg1_df.shape)

    return agg1_df
