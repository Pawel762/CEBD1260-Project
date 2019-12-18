# This is for previous application
# Import library
import numpy as np  
import pandas as pd
import os
import glob

def preprocessing_prev_app():
    # Read data
    path = r'C:\Users\AC037628\Desktop\CEBD1260-dev\Data'
    # path = '/Users/ferdinand/Desktop/data'
    file_name = os.path.join(path,'previous_application.csv')
    prev_app_df = pd.read_csv(file_name)
    
    # split into obj and non obj features
    features = [f for f in prev_app_df.columns.values]

    # for non object features
    nb_features = []
    obj_features = []
    for f in features:
        if str(prev_app_df[f].dtype) in ('int64','float64'):
            nb_features.append(f)
        elif str(prev_app_df[f].dtype) in ('object'):
            obj_features.append(f)

    # fill na as required
    prev_app_df['AMT_ANNUITY'].fillna(prev_app_df['AMT_ANNUITY'].mean(),inplace=True)
    prev_app_df['AMT_CREDIT'].fillna(prev_app_df['AMT_CREDIT'].mean(),inplace=True)
    prev_app_df['AMT_DOWN_PAYMENT'].fillna(prev_app_df['AMT_DOWN_PAYMENT'].mean(),inplace=True)
    prev_app_df['AMT_GOODS_PRICE'].fillna(prev_app_df['AMT_GOODS_PRICE'].mean(),inplace=True)
    prev_app_df['RATE_DOWN_PAYMENT'].fillna(prev_app_df['RATE_DOWN_PAYMENT'].mean(),inplace=True)
    prev_app_df['RATE_INTEREST_PRIMARY'].fillna(prev_app_df['RATE_INTEREST_PRIMARY'].mean(),inplace=True)
    prev_app_df['RATE_INTEREST_PRIVILEGED'].fillna(prev_app_df['RATE_INTEREST_PRIVILEGED'].mean(),inplace=True)
    prev_app_df['CNT_PAYMENT'].fillna(prev_app_df['CNT_PAYMENT'].mean(),inplace=True)
    prev_app_df['DAYS_FIRST_DRAWING'].fillna(prev_app_df['DAYS_FIRST_DRAWING'].mean(),inplace=True)
    prev_app_df['DAYS_FIRST_DUE'].fillna(prev_app_df['DAYS_FIRST_DUE'].mean(),inplace=True)
    prev_app_df['DAYS_LAST_DUE_1ST_VERSION'].fillna(prev_app_df['DAYS_LAST_DUE_1ST_VERSION'].mean(),inplace=True)
    prev_app_df['DAYS_LAST_DUE'].fillna(prev_app_df['DAYS_LAST_DUE'].mean(),inplace=True)
    prev_app_df['DAYS_TERMINATION'].fillna(prev_app_df['DAYS_TERMINATION'].mean(),inplace=True)
    prev_app_df['NFLAG_INSURED_ON_APPROVAL'].fillna(-1,inplace=True)

    # change to lighter dtypes
    prev_app_df['AMT_ANNUITY'] = prev_app_df['AMT_ANNUITY'].astype('int32')
    prev_app_df['AMT_APPLICATION'] = prev_app_df['AMT_APPLICATION'].astype('int32')
    prev_app_df['AMT_CREDIT'] = prev_app_df['AMT_CREDIT'].astype('int32')
    prev_app_df['AMT_DOWN_PAYMENT'] = prev_app_df['AMT_DOWN_PAYMENT'].astype('int32')
    prev_app_df['AMT_GOODS_PRICE'] = prev_app_df['AMT_GOODS_PRICE'].astype('int32')
    prev_app_df['HOUR_APPR_PROCESS_START'] = prev_app_df['HOUR_APPR_PROCESS_START'].astype('int8')
    prev_app_df['NFLAG_LAST_APPL_IN_DAY'] = prev_app_df['NFLAG_LAST_APPL_IN_DAY'].astype('int8')
    prev_app_df['DAYS_DECISION'] = prev_app_df['DAYS_DECISION'].astype('int16')
    prev_app_df['SELLERPLACE_AREA'] = prev_app_df['SELLERPLACE_AREA'].astype('int32')
    prev_app_df['CNT_PAYMENT'] = prev_app_df['CNT_PAYMENT'].astype('int8')
    prev_app_df['DAYS_FIRST_DRAWING'] = prev_app_df['DAYS_FIRST_DRAWING'].astype('int32')
    prev_app_df['DAYS_FIRST_DUE'] = prev_app_df['DAYS_FIRST_DUE'].astype('int32')
    prev_app_df['DAYS_LAST_DUE_1ST_VERSION'] = prev_app_df['DAYS_LAST_DUE_1ST_VERSION'].astype('int32')
    prev_app_df['DAYS_LAST_DUE'] = prev_app_df['DAYS_LAST_DUE'].astype('int32')
    prev_app_df['DAYS_TERMINATION'] = prev_app_df['DAYS_TERMINATION'].astype('int32')
    prev_app_df['NFLAG_INSURED_ON_APPROVAL'] = prev_app_df['NFLAG_INSURED_ON_APPROVAL'].astype('int8')

    # replace value to int for obj features
    val = {'NAME_CONTRACT_TYPE':{'Cash loans':0,
                                'Consumer loans':1, 
                                'Revolving loans':2, 
                                'XNA':3},
        'FLAG_LAST_APPL_PER_CONTRACT':{'Y':1,
                                        'N':0},
        'WEEKDAY_APPR_PROCESS_START':{'MONDAY':0, 
                                        'TUESDAY':1, 
                                        'WEDNESDAY':2, 
                                        'THURSDAY':3, 
                                        'FRIDAY':4, 
                                        'SATURDAY':5, 
                                        'SUNDAY':6},
        'NAME_CASH_LOAN_PURPOSE':{'XAP':0, 
                                    'XNA':1,
                                    'Repairs':2,
                                    'Other':3,
                                    'Urgent needs':4,
                                    'Buying a used car':5,
                                    'Building a house or an annex':6,
                                    'Everyday expenses':7,
                                    'Medicine':8,
                                    'Payments on other loans':9,
                                    'Education':10,
                                    'Journey':11,
                                    'Purchase of electronic equipment':12,
                                    'Buying a new car':13,
                                    'Wedding / gift / holiday':14,
                                    'Buying a home':15,
                                    'Car repairs':16,
                                    'Furniture':17,
                                    'Buying a holiday home / land':18,
                                    'Business development':19,
                                    'Gasification / water supply':20,
                                    'Buying a garage':21,
                                    'Hobby':22,
                                    'Money for a third person':23,
                                    'Refusal to name the goal':24
                                    },
            'NAME_CONTRACT_STATUS':{'Refused':0, 
                                'Approved':1,
                                'Canceled':2,
                                    'Unused offer':3
                                },
        'NAME_PAYMENT_TYPE':{'XNA':0,
                                'Non-cash from your account':1,
                                'Cashless from the account of the employer':2,
                            },
        
        'CODE_REJECT_REASON':{'XNA':0, 
                                'HC':1,
                                'LIMIT':2,
                                'SCO':3,
                                'CLIENT':4,
                                'SCOFR':5,
                                'XAP':6,
                                'VERIF':7,
                                'SYSTEM':8 
                                },
        'NAME_TYPE_SUITE':{'Unaccompanied':0, 
                            'Family':1,
                            'Spouse, partner':2,
                            'Children':3,
                            'Other_B':4,
                            'Other_A':5,
                            'Group of people':6
                            },
        'NAME_CLIENT_TYPE':{'XNA':0, 
                            'Repeater':1,
                            'New':2,
                            'Refreshed':3                           
                            },
        'NAME_GOODS_CATEGORY':{'XNA':0, 
                                'Mobile':1, 
                                'Consumer Electronics':2,
                                'Computers':3,
                                'Audio/Video':4, 
                                'Furniture':5,
                                'Photo / Cinema Equipment':6,
                                'Construction Materials':7,
                                'Clothing and Accessories':8,
                                'Auto Accessories':9,
                                'Jewelry':10,
                                'Homewares':11,
                                'Medical Supplies':12,
                                'Vehicles':13,
                                'Sport and Leisure':14,
                                'Gardening':15,
                                'Other':16,
                                'Office Appliances':17,
                                'Tourism':18,
                                'Medicine':19,
                                'Direct Sales':20,
                                'Fitness':21,
                                'Additional Service':22,
                                'Education':23,
                                'Weapon':24,
                                'Insurance':25,
                                'House Construction':26,
                                'Animals':27
                                },
        'NAME_PORTFOLIO':{'XNA':0, 
                            'POS':1, 
                            'Cash':2,
                            'Cards':3,
                            'Cars':4
                            },
        
        'NAME_PRODUCT_TYPE':{'XNA':0,
                            'x-sell':1,
                            'walk-in':2
                            },
        
        'CHANNEL_TYPE':{'Credit and cash offices':0,
                        'Country-wide':1,
                        'Stone':2,
                        'Regional / Local':3,
                        'Contact center':4,
                        'AP+ (Cash loan)':5,
                        'Channel of corporate sales':6,
                        'Car dealer':7
                        },
        'NAME_SELLER_INDUSTRY':{'XNA':0,
                                'Consumer electronics':1,
                                'Connectivity':2,
                                'Furniture':3,
                                'Construction':4,
                                'Clothing':5,
                                'Industry':6,
                                'Auto technology':7,
                                'Jewelry':8,
                                'MLM partners':9,
                                'Tourism':10
                                },
        'NAME_YIELD_GROUP':{'XNA':0, 
                            'middle':1,
                            'high':2,
                            'low_normal':3,
                            'low_action':4
                            },
        'PRODUCT_COMBINATION':{'Cash':0,
                                'POS household with interest':1,
                                'POS mobile with interest':2, 
                                'Cash X-Sell: middle':3,
                                'Cash X-Sell: low':4,
                                'Card Street':5,
                                'POS industry with interest':6,
                                'POS household without interest':7,
                                'Card X-Sell':8,
                                'Cash Street: high':9,
                                'Cash X-Sell: high':10,
                                'Cash Street: middle':11,
                                'Cash Street: low':12,
                                'POS mobile without interest':13,
                                'POS other with interest':14,
                                'POS industry without interest':15,
                                'POS others without interest':16
                                }
        }
    # for object dtype, use categorical
    prev_app_df['NAME_TYPE_SUITE'].fillna(value=-1,inplace=True)
    prev_app_df['PRODUCT_COMBINATION'].fillna(value=-1,inplace=True)
    
    prev_app_df.replace(val,inplace=True)

    return prev_app_df










