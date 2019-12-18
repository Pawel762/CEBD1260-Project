# This is master script
# Import library
import numpy as np
import pandas as pd
import os
import glob
import sys
import time

start = time.time()

sys.path.append('..')
from processed_script.main import preprocessing_main
from processed_script.prev_app import preprocessing_prev_app
from processed_script.cc_balance import preprocessing_cc_balance
# from processed_script.bureau import preprocessing_bureau
from processed_script.POS_CASH_balance import preprocessing_POS_CASH_balance
from processed_script.ipay import preprocessing_ipay
from processed_script.br_brbl_joined import preprocessing_br_brbl
# from processed_script.br_balance import preprocessing_bureau_balance


main_df = preprocessing_main()
prev_app_df = preprocessing_prev_app()
cc_balance_df = preprocessing_cc_balance()
# bureau_df = preprocessing_bureau()
POS_CASH_balance_df = preprocessing_POS_CASH_balance()
ipay_df = preprocessing_ipay()
# bureau_balance_df = preprocessing_bureau_balance()
agg1_df = preprocessing_br_brbl()
# *************** Data Joining *************** #

# 1. Join Main table to previous applicaltion table
# Find common ids to main tables
main_ids = list(main_df['MAIN_SK_ID_CURR'].unique())
prev_ids = list(prev_app_df['SK_ID_CURR'].unique())
common1_ids = set(main_ids).intersection(set(prev_ids))  # see how many ids are common to main tables

filtered_prev_df = prev_app_df.loc[prev_app_df.SK_ID_CURR.isin(main_ids)]
agg_prev_dict = {'AMT_ANNUITY': ['median'],
                 'AMT_APPLICATION': ['median'],
                 'AMT_CREDIT': ['median'],
                 'AMT_DOWN_PAYMENT': ['median'],
                 'AMT_GOODS_PRICE': ['median'],
                 'HOUR_APPR_PROCESS_START': ['median'],
                 'NFLAG_LAST_APPL_IN_DAY': ['median'],
                 'RATE_DOWN_PAYMENT': ['median'],
                 'RATE_INTEREST_PRIMARY': ['median'],
                 'RATE_INTEREST_PRIVILEGED': ['median'],
                 'DAYS_DECISION': ['median'],
                 'SELLERPLACE_AREA': ['median'],
                 'CNT_PAYMENT': ['median'],
                 'DAYS_FIRST_DRAWING': ['median'],
                 'DAYS_FIRST_DUE': ['median'],
                 'DAYS_LAST_DUE_1ST_VERSION': ['median'],
                 'DAYS_LAST_DUE': ['median'],
                 'DAYS_TERMINATION': ['median'],
                 'NFLAG_INSURED_ON_APPROVAL': ['median']
                 }

prev_num_df = filtered_prev_df.groupby('SK_ID_CURR').agg(agg_prev_dict)
agg_prev_df = prev_num_df.reset_index()
agg_prev_df.columns = ['PREV_APP_{}_{}'.format(x[0], x[1]) for x in agg_prev_df.columns.tolist()]
agg_prev_df.rename({'PREV_APP_SK_ID_CURR_': 'PREV_APP_SK_ID_CURR'}, axis=1, inplace=True)

main_df.rename({'MAIN_SK_ID_CURR': 'SK_ID_CURR'}, axis=1, inplace=True)
agg_prev_df.rename({'PREV_APP_SK_ID_CURR': 'SK_ID_CURR'}, axis=1, inplace=True)

final_temp1_df = main_df.merge(agg_prev_df, on='SK_ID_CURR', how='left')

# 2. Join Main table to credit card balance table
cc_ids = list(cc_balance_df['SK_ID_CURR'].unique())
common2_ids = set(main_ids).intersection(set(cc_ids))  # see how many ids are common to main tables

filtered_cc_df = cc_balance_df.loc[cc_balance_df.SK_ID_CURR.isin(main_ids)]
# aggregate cc dataset
agg_cc_dict = {'MONTHS_BALANCE': ['median'],
               'AMT_BALANCE': ['median'],
               'AMT_CREDIT_LIMIT_ACTUAL': ['median'],
               'AMT_DRAWINGS_ATM_CURRENT': ['median'],
               'AMT_DRAWINGS_CURRENT': ['median'],
               'AMT_DRAWINGS_OTHER_CURRENT': ['median'],
               'AMT_DRAWINGS_POS_CURRENT': ['median'],
               'AMT_INST_MIN_REGULARITY': ['median'],
               'AMT_PAYMENT_CURRENT': ['median'],
               'AMT_PAYMENT_TOTAL_CURRENT': ['median'],
               'AMT_RECEIVABLE_PRINCIPAL': ['median'],
               'AMT_RECIVABLE': ['median'],
               'AMT_TOTAL_RECEIVABLE': ['median'],
               'CNT_DRAWINGS_ATM_CURRENT': ['median'],
               'CNT_DRAWINGS_CURRENT': ['median'],
               'CNT_DRAWINGS_OTHER_CURRENT': ['median'],
               'CNT_DRAWINGS_POS_CURRENT': ['median'],
               'CNT_INSTALMENT_MATURE_CUM': ['median']
               }
cc_num_df = filtered_cc_df.groupby('SK_ID_CURR').agg(agg_cc_dict)
agg_cc_df = cc_num_df.reset_index()
agg_cc_df.columns = ['CC_BAL_{}_{}'.format(x[0], x[1]) for x in agg_cc_df.columns.tolist()]
agg_cc_df.rename({'CC_BAL_SK_ID_CURR_': 'SK_ID_CURR'}, axis=1, inplace=True)

# connect main to cc balance
final_temp2_df = final_temp1_df.merge(agg_cc_df, on='SK_ID_CURR', how='left')

# # 3. Join Main table to bureau table
# bureau_ids = list(bureau_df['SK_ID_CURR'].unique())
# common3_ids = set(main_ids).intersection(set(bureau_ids))  # see how many ids are common to main tables
# filtered_bureau_df = bureau_df.loc[bureau_df.SK_ID_CURR.isin(main_ids)]
#
# agg_bureau_dict = {
#     'DAYS_CREDIT': ['median'],
#     'CREDIT_DAY_OVERDUE': ['median'],
#     'DAYS_CREDIT_ENDDATE': ['median'],
#     'DAYS_ENDDATE_FACT': ['median'],
#     'AMT_CREDIT_MAX_OVERDUE': ['median'],
#     'CNT_CREDIT_PROLONG': ['median'],
#     'AMT_CREDIT_SUM': ['median'],
#     'AMT_CREDIT_SUM_DEBT': ['median'],
#     'AMT_CREDIT_SUM_LIMIT': ['median'],
#     'AMT_CREDIT_SUM_OVERDUE': ['median'],
#     'DAYS_CREDIT_UPDATE': ['median'],
#     'AMT_ANNUITY': ['median']
# }
#
# bureau_num_df = filtered_bureau_df.groupby('SK_ID_CURR').agg(agg_bureau_dict)
# agg_bureau_df = bureau_num_df.reset_index()
# agg_bureau_df.columns = ['BUR_{}_{}'.format(x[0], x[1]) for x in agg_bureau_df.columns.tolist()]
# agg_bureau_df.rename({'BUR_SK_ID_CURR_': 'SK_ID_CURR'}, axis=1, inplace=True)

# connect main to bureau balance
# final_temp3_df = final_temp2_df.merge(agg_bureau_df, on='SK_ID_CURR', how='left')

# 4. Join Main table to POS Cash Balance table
POS_ids = list(POS_CASH_balance_df['SK_ID_CURR'].unique())
common4_ids = set(main_ids).intersection(set(POS_ids))  # see how many ids are common to main tables
filtered_POS_df = POS_CASH_balance_df.loc[POS_CASH_balance_df.SK_ID_CURR.isin(main_ids)]

agg_POS_dict = {'MONTHS_BALANCE': ['median'],
                'CNT_INSTALMENT': ['median'],
                'CNT_INSTALMENT_FUTURE': ['median'],
                'SK_DPD': ['median'],
                'SK_DPD_DEF': ['median']
                }

POS_num_df = filtered_POS_df.groupby('SK_ID_CURR').agg(agg_POS_dict)
agg_POS_df = POS_num_df.reset_index()
agg_POS_df.columns = ['POS_{}_{}'.format(x[0], x[1]) for x in agg_POS_df.columns.tolist()]
agg_POS_df.rename({'POS_SK_ID_CURR_': 'SK_ID_CURR'}, axis=1, inplace=True)

# connect main to bureau balance
final_temp4_df = final_temp2_df.merge(agg_POS_df, on='SK_ID_CURR', how='left')

# 5. Join Main table to installement payment table
ipay_ids = list(ipay_df['SK_ID_CURR'].unique())
common5_ids = set(main_ids).intersection(set(ipay_ids))  # see how many ids are common to main tables
filtered_ipay_df = ipay_df.loc[ipay_df.SK_ID_CURR.isin(main_ids)]

agg_ipay_dict = {'NUM_INSTALMENT_VERSION': ['median'],
                 'NUM_INSTALMENT_NUMBER': ['median'],
                 'DAYS_INSTALMENT': ['median'],
                 'DAYS_ENTRY_PAYMENT': ['median'],
                 'AMT_INSTALMENT': ['median'],
                 'AMT_PAYMENT': ['median']
                 }

ipay_num_df = filtered_ipay_df.groupby('SK_ID_CURR').agg(agg_ipay_dict)
agg_ipay_df = ipay_num_df.reset_index()
agg_ipay_df.columns = ['IPAY_{}_{}'.format(x[0], x[1]) for x in agg_ipay_df.columns.tolist()]
agg_ipay_df.rename({'IPAY_SK_ID_CURR_': 'SK_ID_CURR'}, axis=1, inplace=True)

# connect main to bureau /bureau balance
final_temp5_df = final_temp4_df.merge(agg1_df, on='SK_ID_CURR', how='left')

# 6. Final dataset
final_df = final_temp5_df
final_df.fillna(0)
print(final_df.shape)
# *************** Algorithm *************** #

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier

# import lightgbm
kf = KFold(n_splits=5)

X_train = final_df.drop(['TARGET'], axis=1)
y_train = final_df['TARGET']

# Split dataframe by index
for i, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print('Fold :{}'.format(i))
    tr_X = X_train.loc[tr_idx]  # training for this loop
    tr_y = y_train[tr_idx]
    val_X = X_train.loc[val_idx]  # validation data for this loop
    val_y = y_train[val_idx]

    model = LGBMClassifier(
        n_jobs=4,
        n_estimators=100000,
        boost_from_average='false',
        learning_rate=0.01,
        num_leaves=64,
        num_threads=4,
        max_depth=10,
        feature_fraction=0.7,
        bagging_freq=5,
        bagging_fraction=0.5,
        silent=-1,
        verbose=-1
    )
    model.fit(tr_X, tr_y, eval_set=[(tr_X, tr_y), (val_X, val_y)], eval_metric='binary_logloss', verbose=100,
              early_stopping_rounds=200)
    pred_val_y = model.predict_proba(val_X, num_iteration=model.best_iteration_)[:, 1]

end = time.time()
print((end - start)/60)
