# Import library
import numpy as np  
import pandas as pd
import os
import glob

def preprocessing_main():
    # Read data
    path = r'C:\Users\AC037628\Desktop\CEBD1260-dev\Data'
    # path = '/Users/ferdinand/Desktop/data'
    file_name = os.path.join(path,'application_train.csv')
    main_df = pd.read_csv(file_name)

    # split into obj and non obj features
    features = [f for f in main_df.columns.values]

    # for non object features
    nb_features = []
    obj_features = []
    for f in features:
        if str(main_df[f].dtype) in ('int64','float64'):
            nb_features.append(f)
        elif str(main_df[f].dtype) in ('object'):
            obj_features.append(f)

    # fill na as required
    main_df['AMT_ANNUITY'].fillna(main_df['AMT_ANNUITY'].mean(),inplace=True)
    main_df['AMT_GOODS_PRICE'].fillna(main_df['AMT_GOODS_PRICE'].mean(),inplace=True)
    main_df['OWN_CAR_AGE'].fillna(main_df['OWN_CAR_AGE'].mean(),inplace=True)
    main_df['CNT_FAM_MEMBERS'].fillna(main_df['CNT_FAM_MEMBERS'].mean(),inplace=True)
    main_df['APARTMENTS_AVG'].fillna(main_df['APARTMENTS_AVG'].mean(),inplace=True)
    main_df['BASEMENTAREA_AVG'].fillna(main_df['BASEMENTAREA_AVG'].mean(),inplace=True)
    main_df['YEARS_BEGINEXPLUATATION_AVG'].fillna(main_df['YEARS_BEGINEXPLUATATION_AVG'].mean(),inplace=True)
    main_df['YEARS_BUILD_AVG'].fillna(main_df['YEARS_BUILD_AVG'].mean(),inplace=True)
    main_df['COMMONAREA_AVG'].fillna(main_df['COMMONAREA_AVG'].mean(),inplace=True)
    main_df['ELEVATORS_AVG'].fillna(main_df['ELEVATORS_AVG'].mean(),inplace=True)
    main_df['ENTRANCES_AVG'].fillna(main_df['ENTRANCES_AVG'].mean(),inplace=True)
    main_df['FLOORSMAX_AVG'].fillna(main_df['FLOORSMAX_AVG'].mean(),inplace=True)
    main_df['FLOORSMIN_AVG'].fillna(main_df['FLOORSMIN_AVG'].mean(),inplace=True)
    main_df['LANDAREA_AVG'].fillna(main_df['LANDAREA_AVG'].mean(),inplace=True)
    main_df['LIVINGAPARTMENTS_AVG'].fillna(main_df['LIVINGAPARTMENTS_AVG'].mean(),inplace=True)
    main_df['LIVINGAREA_AVG'].fillna(main_df['LIVINGAREA_AVG'].mean(),inplace=True)
    main_df['NONLIVINGAPARTMENTS_AVG'].fillna(main_df['NONLIVINGAPARTMENTS_AVG'].mean(),inplace=True)
    main_df['NONLIVINGAREA_AVG'].fillna(main_df['NONLIVINGAREA_AVG'].mean(),inplace=True)
    main_df['APARTMENTS_MODE'].fillna(main_df['APARTMENTS_MODE'].mean(),inplace=True)
    main_df['BASEMENTAREA_MODE'].fillna(main_df['BASEMENTAREA_MODE'].mean(),inplace=True)
    main_df['YEARS_BEGINEXPLUATATION_MODE'].fillna(main_df['YEARS_BEGINEXPLUATATION_MODE'].mean(),inplace=True)
    main_df['YEARS_BUILD_MODE'].fillna(main_df['YEARS_BUILD_MODE'].mean(),inplace=True)
    main_df['COMMONAREA_MODE'].fillna(main_df['COMMONAREA_MODE'].mean(),inplace=True)
    main_df['ELEVATORS_MODE'].fillna(main_df['ELEVATORS_MODE'].mean(),inplace=True)
    main_df['ENTRANCES_MODE'].fillna(main_df['ENTRANCES_MODE'].mean(),inplace=True)
    main_df['FLOORSMAX_MODE'].fillna(main_df['FLOORSMAX_MODE'].mean(),inplace=True)
    main_df['FLOORSMIN_MODE'].fillna(main_df['FLOORSMIN_MODE'].mean(),inplace=True)
    main_df['LANDAREA_MODE'].fillna(main_df['LANDAREA_MODE'].mean(),inplace=True)
    main_df['LIVINGAPARTMENTS_MODE'].fillna(main_df['LIVINGAPARTMENTS_MODE'].mean(),inplace=True)
    main_df['LIVINGAREA_MODE'].fillna(main_df['LIVINGAREA_MODE'].mean(),inplace=True)
    main_df['NONLIVINGAPARTMENTS_MODE'].fillna(main_df['NONLIVINGAPARTMENTS_MODE'].mean(),inplace=True)
    main_df['NONLIVINGAREA_MODE'].fillna(main_df['NONLIVINGAREA_MODE'].mean(),inplace=True)
    main_df['APARTMENTS_MEDI'].fillna(main_df['APARTMENTS_MEDI'].mean(),inplace=True)
    main_df['BASEMENTAREA_MEDI'].fillna(main_df['BASEMENTAREA_MEDI'].mean(),inplace=True)
    main_df['YEARS_BEGINEXPLUATATION_MEDI'].fillna(main_df['YEARS_BEGINEXPLUATATION_MEDI'].mean(),inplace=True)
    main_df['YEARS_BUILD_MEDI'].fillna(main_df['YEARS_BUILD_MEDI'].mean(),inplace=True)
    main_df['COMMONAREA_MEDI'].fillna(main_df['COMMONAREA_MEDI'].mean(),inplace=True)
    main_df['ELEVATORS_MEDI'].fillna(main_df['ELEVATORS_MEDI'].mean(),inplace=True)
    main_df['ENTRANCES_MEDI'].fillna(main_df['ENTRANCES_MEDI'].mean(),inplace=True)
    main_df['FLOORSMAX_MEDI'].fillna(main_df['FLOORSMAX_MEDI'].mean(),inplace=True)
    main_df['FLOORSMIN_MEDI'].fillna(main_df['FLOORSMIN_MEDI'].mean(),inplace=True)
    main_df['LANDAREA_MEDI'].fillna(main_df['LANDAREA_MEDI'].mean(),inplace=True)
    main_df['LIVINGAPARTMENTS_MEDI'].fillna(main_df['LIVINGAPARTMENTS_MEDI'].mean(),inplace=True)
    main_df['LIVINGAREA_MEDI'].fillna(main_df['LIVINGAREA_MEDI'].mean(),inplace=True)
    main_df['NONLIVINGAPARTMENTS_MEDI'].fillna(main_df['NONLIVINGAPARTMENTS_MEDI'].mean(),inplace=True)
    main_df['NONLIVINGAREA_MEDI'].fillna(main_df['NONLIVINGAREA_MEDI'].mean(),inplace=True)
    main_df['TOTALAREA_MODE'].fillna(main_df['TOTALAREA_MODE'].mean(),inplace=True)
    main_df['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(main_df['OBS_30_CNT_SOCIAL_CIRCLE'].mean(),inplace=True)
    main_df['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(main_df['DEF_30_CNT_SOCIAL_CIRCLE'].mean(),inplace=True)
    main_df['OBS_60_CNT_SOCIAL_CIRCLE'].fillna(main_df['OBS_60_CNT_SOCIAL_CIRCLE'].mean(),inplace=True)
    main_df['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(main_df['DEF_60_CNT_SOCIAL_CIRCLE'].mean(),inplace=True)
    main_df['DAYS_LAST_PHONE_CHANGE'].fillna(main_df['DAYS_LAST_PHONE_CHANGE'].mean(),inplace=True)
    main_df['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(main_df['AMT_REQ_CREDIT_BUREAU_HOUR'].mean(),inplace=True)
    main_df['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(main_df['AMT_REQ_CREDIT_BUREAU_DAY'].mean(),inplace=True)
    main_df['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(main_df['AMT_REQ_CREDIT_BUREAU_WEEK'].mean(),inplace=True)
    main_df['AMT_REQ_CREDIT_BUREAU_MON'].fillna(main_df['AMT_REQ_CREDIT_BUREAU_MON'].mean(),inplace=True)
    main_df['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(main_df['AMT_REQ_CREDIT_BUREAU_QRT'].mean(),inplace=True)
    main_df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(main_df['AMT_REQ_CREDIT_BUREAU_YEAR'].mean(),inplace=True)

    # change dtype
    main_df['TARGET'] = main_df['TARGET'].astype('int8') 
    main_df['CNT_CHILDREN'] = main_df['CNT_CHILDREN'].astype('int8') 
    main_df['AMT_INCOME_TOTAL'] = main_df['AMT_INCOME_TOTAL'].astype('int32') 
    main_df['AMT_CREDIT'] = main_df['AMT_CREDIT'].astype('int32') 
    main_df['AMT_ANNUITY'] = main_df['AMT_ANNUITY'].astype('int32') 
    main_df['AMT_GOODS_PRICE'] = main_df['AMT_GOODS_PRICE'].astype('int32') 
    main_df['REGION_POPULATION_RELATIVE'] = main_df['REGION_POPULATION_RELATIVE'].astype('float16') 
    main_df['DAYS_BIRTH'] = main_df['DAYS_BIRTH'].astype('int32') 
    main_df['DAYS_EMPLOYED'] = main_df['DAYS_EMPLOYED'].astype('int32') 
    main_df['DAYS_REGISTRATION'] = main_df['DAYS_REGISTRATION'].astype('int32') 
    main_df['DAYS_ID_PUBLISH'] = main_df['DAYS_ID_PUBLISH'].astype('int32') 
    main_df['OWN_CAR_AGE'] = main_df['OWN_CAR_AGE'].astype('int8') 
    main_df['FLAG_MOBIL'] = main_df['FLAG_MOBIL'].astype('int8') 
    main_df['FLAG_EMP_PHONE'] = main_df['FLAG_EMP_PHONE'].astype('int8') 
    main_df['FLAG_WORK_PHONE'] = main_df['FLAG_WORK_PHONE'].astype('int8') 
    main_df['FLAG_CONT_MOBILE'] = main_df['FLAG_CONT_MOBILE'].astype('int8') 
    main_df['FLAG_PHONE'] = main_df['FLAG_PHONE'].astype('int8') 
    main_df['FLAG_EMAIL'] = main_df['FLAG_EMAIL'].astype('int8') 
    main_df['CNT_FAM_MEMBERS'] = main_df['CNT_FAM_MEMBERS'].astype('int8') 
    main_df['REGION_RATING_CLIENT'] = main_df['REGION_RATING_CLIENT'].astype('int8') 
    main_df['REGION_RATING_CLIENT_W_CITY'] = main_df['REGION_RATING_CLIENT_W_CITY'].astype('int8') 
    main_df['HOUR_APPR_PROCESS_START'] = main_df['HOUR_APPR_PROCESS_START'].astype('int8') 
    main_df['REG_REGION_NOT_LIVE_REGION'] = main_df['REG_REGION_NOT_LIVE_REGION'].astype('int8') 
    main_df['REG_REGION_NOT_WORK_REGION'] = main_df['REG_REGION_NOT_WORK_REGION'].astype('int8') 
    main_df['LIVE_REGION_NOT_WORK_REGION'] = main_df['LIVE_REGION_NOT_WORK_REGION'].astype('int8') 
    main_df['REG_CITY_NOT_LIVE_CITY'] = main_df['REG_CITY_NOT_LIVE_CITY'].astype('int8') 
    main_df['REG_CITY_NOT_WORK_CITY'] = main_df['REG_CITY_NOT_WORK_CITY'].astype('int8') 
    main_df['LIVE_CITY_NOT_WORK_CITY'] = main_df['LIVE_CITY_NOT_WORK_CITY'].astype('int8') 
    main_df['FLAG_DOCUMENT_2'] = main_df['FLAG_DOCUMENT_2'].astype('int8') 
    main_df['FLAG_DOCUMENT_2'] = main_df['FLAG_DOCUMENT_2'].astype('int8') 
    main_df['FLAG_DOCUMENT_3'] = main_df['FLAG_DOCUMENT_3'].astype('int8') 
    main_df['FLAG_DOCUMENT_4'] = main_df['FLAG_DOCUMENT_4'].astype('int8') 
    main_df['FLAG_DOCUMENT_5'] = main_df['FLAG_DOCUMENT_5'].astype('int8') 
    main_df['FLAG_DOCUMENT_6'] = main_df['FLAG_DOCUMENT_6'].astype('int8') 
    main_df['FLAG_DOCUMENT_7'] = main_df['FLAG_DOCUMENT_7'].astype('int8') 
    main_df['FLAG_DOCUMENT_8'] = main_df['FLAG_DOCUMENT_8'].astype('int8') 
    main_df['FLAG_DOCUMENT_9'] = main_df['FLAG_DOCUMENT_9'].astype('int8') 
    main_df['FLAG_DOCUMENT_10'] = main_df['FLAG_DOCUMENT_10'].astype('int8') 
    main_df['FLAG_DOCUMENT_11'] = main_df['FLAG_DOCUMENT_11'].astype('int8') 
    main_df['FLAG_DOCUMENT_12'] = main_df['FLAG_DOCUMENT_12'].astype('int8') 
    main_df['FLAG_DOCUMENT_13'] = main_df['FLAG_DOCUMENT_13'].astype('int8') 
    main_df['FLAG_DOCUMENT_14'] = main_df['FLAG_DOCUMENT_14'].astype('int8') 
    main_df['FLAG_DOCUMENT_15'] = main_df['FLAG_DOCUMENT_15'].astype('int8') 
    main_df['FLAG_DOCUMENT_16'] = main_df['FLAG_DOCUMENT_16'].astype('int8') 
    main_df['FLAG_DOCUMENT_17'] = main_df['FLAG_DOCUMENT_17'].astype('int8') 
    main_df['FLAG_DOCUMENT_18'] = main_df['FLAG_DOCUMENT_18'].astype('int8') 
    main_df['FLAG_DOCUMENT_19'] = main_df['FLAG_DOCUMENT_19'].astype('int8') 
    main_df['FLAG_DOCUMENT_20'] = main_df['FLAG_DOCUMENT_20'].astype('int8') 
    main_df['FLAG_DOCUMENT_21'] = main_df['FLAG_DOCUMENT_21'].astype('int8') 
    main_df['AMT_REQ_CREDIT_BUREAU_HOUR'] = main_df['AMT_REQ_CREDIT_BUREAU_HOUR'].astype('float16') 
    main_df['AMT_REQ_CREDIT_BUREAU_DAY'] = main_df['AMT_REQ_CREDIT_BUREAU_DAY'].astype('float16') 
    main_df['AMT_REQ_CREDIT_BUREAU_WEEK'] = main_df['AMT_REQ_CREDIT_BUREAU_WEEK'].astype('float16') 
    main_df['AMT_REQ_CREDIT_BUREAU_MON'] = main_df['AMT_REQ_CREDIT_BUREAU_MON'].astype('float16') 
    main_df['AMT_REQ_CREDIT_BUREAU_QRT'] = main_df['AMT_REQ_CREDIT_BUREAU_QRT'].astype('float16') 
    main_df['AMT_REQ_CREDIT_BUREAU_YEAR'] = main_df['AMT_REQ_CREDIT_BUREAU_YEAR'].astype('float16') 

    val = {'NAME_CONTRACT_TYPE':{'Cash loans':0, 
                             'Revolving loans':1},
       'CODE_GENDER':{'XNA':0, 
                      'M':1, 
                      'F':2},
       'FLAG_OWN_CAR':{'N':0, 
                       'Y':1},
       'FLAG_OWN_REALTY':{'N':0, 
                          'Y':1},
       'NAME_TYPE_SUITE':{'Unaccompanied':0,
                          'Family':1,
                          'Spouse, partner':2,
                          'Children':3,
                          'Other_A':4,
                          'Other_B':5,
                          'Group of people':6},
       'NAME_INCOME_TYPE':{'Working':0,
                           'Commercial associate':1,
                           'Pensioner':2,
                           'State servant':3,
                           'Unemployed':4,
                           'Student':5,
                           'Businessman':6,
                           'Maternity leave':7},
       'NAME_EDUCATION_TYPE':{'Secondary / secondary special':0,
                              'Higher education':1,
                              'Incomplete higher':2,
                              'Lower secondary':3,
                              'Academic degree':4},
       'NAME_FAMILY_STATUS':{'Married':0,
                             'Single / not married':1,
                             'Civil marriage':2,
                             'Separated':3,
                             'Widow':4,
                             'Unknown':5},
       'NAME_HOUSING_TYPE':{'House / apartment':0,
                            'With parents':1,
                            'Municipal apartment':2,
                            'Rented apartment':3,
                            'Office apartment':4,
                            'Co-op apartment':5},
       'OCCUPATION_TYPE':{'Laborers':0,
                          'Sales staff':1,
                          'Core staff':2,
                          'Managers':3,
                          'Drivers':4,
                          'High skill tech staff':5,
                          'Accountants':6,
                          'Medicine staff':7,
                          'Security staff':8,
                          'Cooking staff':9,
                          'Cleaning staff':10,
                          'Private service staff':11,
                          'Low-skill Laborers':12,
                          'Waiters/barmen staff':13,
                          'Secretaries':14,
                          'Realty agents':15,
                          'HR staff':16,
                          'IT staff':17},
       'WEEKDAY_APPR_PROCESS_START':{'MONDAY':0,
                                     'TUESDAY':1,
                                     'WEDNESDAY':2,
                                     'THURSDAY':3,
                                     'FRIDAY':4,
                                     'SATURDAY':5,
                                     'SUNDAY':6},
       'ORGANIZATION_TYPE':{'XNA':0,
                            'Business Entity Type 3':1,
                            'Self-employed':2,
                            'Other':3,
                            'Medicine':4,
                            'Business Entity Type 2':5,
                            'Government':6,
                            'School':7,
                            'Trade: type 7':8,
                            'Kindergarten':9,
                            'Construction':10,
                            'Business Entity Type 1':11,
                            'Transport: type 4':12,
                            'Trade: type 3':13,
                            'Industry: type 9':14,
                            'Industry: type 3':15,
                            'Security':16,
                            'Housing':17,
                            'Industry: type 11':18,
                            'Military':19,
                            'Bank':20,
                            'Agriculture':21,
                            'Police':22,
                            'Transport: type 2':23,
                            'Postal':24,
                            'Security Ministries':25,
                            'Trade: type 2':26,
                            'Restaurant':27,
                            'Services':28,
                            'University':29,
                            'Industry: type 7':30,
                            'Transport: type 3':31,
                            'Industry: type 1':32,
                            'Hotel':33,
                            'Electricity':34,
                            'Industry: type 4':35,
                            'Trade: type 6':36,
                            'Industry: type 5':37,
                            'Insurance':38,
                            'Telecom':39,
                            'Emergency':40,
                            'Industry: type 2':41,
                            'Advertising':42,
                            'Realtor':43,
                            'Culture':44,
                            'Industry: type 12':45,
                            'Trade: type 1':46,
                            'Mobile':47,
                            'Legal Services':48,
                            'Cleaning':49,
                            'Transport: type 1':50,
                            'Industry: type 6':51,
                            'Industry: type 10':52,
                            'Religion':53,
                            'Industry: type 13':54,
                            'Trade: type 4':55,
                            'Trade: type 5':56,
                            'Industry: type 8':57},
       'FONDKAPREMONT_MODE':{'reg oper account':0,
                             'reg oper spec account':1,
                             'not specified':2,
                             'org spec account':3},
       'HOUSETYPE_MODE':{'block of flats':0,
                         'specific housing':1,
                         'terraced house':2
                        },
       
       'WALLSMATERIAL_MODE':{'Panel':0,
                             'Stone, brick':1,
                             'Block':2,
                             'Wooden':3,
                             'Mixed':4,
                             'Monolithic':5,
                             'Others':6
                            },
       'EMERGENCYSTATE_MODE':{'No':0,
                              'Yes':1}
    }
    main_df.replace(val,inplace=True)

    # fillna for categorical variable
    main_df['NAME_TYPE_SUITE'].fillna(value=-1,inplace=True)
    main_df['OCCUPATION_TYPE'].fillna(value=-1,inplace=True)
    main_df['FONDKAPREMONT_MODE'].fillna(value=-1,inplace=True)
    main_df['HOUSETYPE_MODE'].fillna(value=-1,inplace=True)
    main_df['WALLSMATERIAL_MODE'].fillna(value=-1,inplace=True)
    main_df['EMERGENCYSTATE_MODE'].fillna(value=-1,inplace=True)

    # data aggregation for numerical features
    agg_dict = {'CNT_CHILDREN':['mean'],
                'AMT_INCOME_TOTAL':['mean'],
                'AMT_CREDIT':['mean'],
                'AMT_ANNUITY':['mean'],
                'AMT_GOODS_PRICE':['mean'],
                'REGION_POPULATION_RELATIVE':['mean'],
                'DAYS_BIRTH':['mean'],
                'DAYS_EMPLOYED':['mean'],
                'DAYS_REGISTRATION':['mean'],
                'DAYS_ID_PUBLISH':['mean'],
                'OWN_CAR_AGE':['mean'],
                'FLAG_MOBIL':['mean'],
                'FLAG_EMP_PHONE':['mean'],
                'FLAG_CONT_MOBILE':['mean'],
                'FLAG_PHONE':['mean'],
                'FLAG_EMAIL':['mean'],
                'CNT_FAM_MEMBERS':['mean'],
                'REGION_RATING_CLIENT':['mean'],
                'REGION_RATING_CLIENT_W_CITY':['mean'],
                'HOUR_APPR_PROCESS_START':['mean'],
                'REG_REGION_NOT_LIVE_REGION':['mean'],
                'REG_REGION_NOT_WORK_REGION':['mean'],
                'LIVE_REGION_NOT_WORK_REGION':['mean'],
                'REG_CITY_NOT_LIVE_CITY':['mean'],
                'REG_CITY_NOT_WORK_CITY':['mean'],
                'LIVE_CITY_NOT_WORK_CITY':['mean'],
                'EXT_SOURCE_1':['mean'],
                'EXT_SOURCE_2':['mean'],
                'EXT_SOURCE_3':['mean'],
                'APARTMENTS_AVG':['mean'],
                'BASEMENTAREA_AVG':['mean'],
                'YEARS_BEGINEXPLUATATION_AVG':['mean'],
                'YEARS_BUILD_AVG':['mean'],
                'COMMONAREA_AVG':['mean'],
                'ELEVATORS_AVG':['mean'],
                'ENTRANCES_AVG':['mean'],
                'FLOORSMAX_AVG':['mean'],
                'FLOORSMIN_AVG':['mean'],
                'LANDAREA_AVG':['mean'],
                'LIVINGAPARTMENTS_AVG':['mean'],
                'LIVINGAREA_AVG':['mean'],
                'NONLIVINGAPARTMENTS_AVG':['mean'],
                'NONLIVINGAREA_AVG':['mean'],
                'APARTMENTS_MODE':['mean'],
                'BASEMENTAREA_MODE':['mean'],
                'YEARS_BEGINEXPLUATATION_MODE':['mean'],
                'YEARS_BUILD_MODE':['mean'],
                'COMMONAREA_MODE':['mean'],
                'ELEVATORS_MODE':['mean'],
                'ENTRANCES_MODE':['mean'],
                'FLOORSMAX_MODE':['mean'],
                'FLOORSMIN_MODE':['mean'],
                'LANDAREA_MODE':['mean'],
                'LIVINGAPARTMENTS_MODE':['mean'],
                'LIVINGAREA_MODE':['mean'],
                'NONLIVINGAPARTMENTS_MODE':['mean'],
                'NONLIVINGAREA_MODE':['mean'],
                'APARTMENTS_MEDI':['mean'],
                'BASEMENTAREA_MEDI':['mean'],
                'YEARS_BEGINEXPLUATATION_MEDI':['mean'],
                'YEARS_BUILD_MEDI':['mean'],
                'COMMONAREA_MEDI':['mean'],
                'ELEVATORS_MEDI':['mean'],
                'ENTRANCES_MEDI':['mean'],
                'FLOORSMAX_MEDI':['mean'],
                'FLOORSMIN_MEDI':['mean'],
                'LANDAREA_MEDI':['mean'],
                'LIVINGAPARTMENTS_MEDI':['mean'],
                'LIVINGAREA_MEDI':['mean'],
                'NONLIVINGAPARTMENTS_MEDI':['mean'],
                'NONLIVINGAREA_MEDI':['mean'],
                'TOTALAREA_MODE':['mean'],
                'OBS_30_CNT_SOCIAL_CIRCLE':['mean'],
                'DEF_30_CNT_SOCIAL_CIRCLE':['mean'],
                'OBS_60_CNT_SOCIAL_CIRCLE':['mean'],
                'DEF_60_CNT_SOCIAL_CIRCLE':['mean'],
                'DAYS_LAST_PHONE_CHANGE':['mean'],
                'FLAG_DOCUMENT_2':['mean'],
                'FLAG_DOCUMENT_3':['mean'],
                'FLAG_DOCUMENT_4':['mean'],
                'FLAG_DOCUMENT_5':['mean'],
                'FLAG_DOCUMENT_6':['mean'],
                'FLAG_DOCUMENT_7':['mean'],
                'FLAG_DOCUMENT_8':['mean'],
                'FLAG_DOCUMENT_9':['mean'],
                'FLAG_DOCUMENT_10':['mean'],
                'FLAG_DOCUMENT_11':['mean'],
                'FLAG_DOCUMENT_12':['mean'],
                'FLAG_DOCUMENT_13':['mean'],
                'FLAG_DOCUMENT_14':['mean'],
                'FLAG_DOCUMENT_15':['mean'],
                'FLAG_DOCUMENT_16':['mean'],
                'FLAG_DOCUMENT_17':['mean'],
                'FLAG_DOCUMENT_18':['mean'],
                'FLAG_DOCUMENT_19':['mean'],
                'FLAG_DOCUMENT_20':['mean'],
                'FLAG_DOCUMENT_21':['mean'],
                'AMT_REQ_CREDIT_BUREAU_HOUR':['mean'],
                'AMT_REQ_CREDIT_BUREAU_DAY':['mean'],
                'AMT_REQ_CREDIT_BUREAU_WEEK':['mean'],
                'AMT_REQ_CREDIT_BUREAU_MON':['mean'],
                'AMT_REQ_CREDIT_BUREAU_QRT':['mean'],
                'AMT_REQ_CREDIT_BUREAU_YEAR':['mean']}

    num_df = main_df.groupby('SK_ID_CURR').agg(agg_dict)
    agg_num_df = num_df.reset_index()
    agg_num_df.columns = ['MAIN_{}_{}'.format(x[0],x[1]) for x in agg_num_df.columns.tolist()]
    agg_num_df.rename({'MAIN_SK_ID_CURR_':'MAIN_SK_ID_CURR'},axis=1,inplace=True)

    # data aggregation for obj features
    agg_obj_df = main_df.loc[:,['SK_ID_CURR',
                                'NAME_CONTRACT_TYPE',
                                'CODE_GENDER',
                                'FLAG_OWN_CAR',
                                'FLAG_OWN_REALTY',
                                'NAME_TYPE_SUITE',
                                'NAME_INCOME_TYPE',
                                'NAME_EDUCATION_TYPE',
                                'NAME_FAMILY_STATUS',
                                'NAME_HOUSING_TYPE',
                                'OCCUPATION_TYPE',
                                'WEEKDAY_APPR_PROCESS_START',
                                'ORGANIZATION_TYPE',
                                'FONDKAPREMONT_MODE',
                                'HOUSETYPE_MODE',
                                'WALLSMATERIAL_MODE',
                                'EMERGENCYSTATE_MODE']]

    agg_obj_df.columns = ['MAIN_{}'.format(x) for x in agg_obj_df.columns.tolist()]
    agg_obj_df.rename({'MAIN_SK_ID_CURR':'MAIN_SK_ID_CURR'},axis=1,inplace=True)
    
    # merge all dataframe
    # target variable dataframe
    target_df = main_df.loc[:,['SK_ID_CURR','TARGET']]
    target_df.rename({'SK_ID_CURR':'MAIN_SK_ID_CURR'},axis=1,inplace=True)
    
    all_features_df = agg_num_df.merge(agg_obj_df, on='MAIN_SK_ID_CURR', how='left')
    main_agg_df = target_df.merge(all_features_df, on='MAIN_SK_ID_CURR', how='left')
    
    return main_agg_df


