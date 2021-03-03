import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm
import gc
import random
import lightgbm as lgb
import re
from sklearn.metrics import *
from sklearn.model_selection import KFold
import warnings
from collections import Counter, defaultdict
warnings.filterwarnings(action='ignore')

from gensim.models import KeyedVectors

from scipy.stats import skew
from scipy.stats import norm, kurtosis

def make_datetime_day(x):
    # string 타입의 Time column을 datetime 타입으로 변경
    x = str(x)
    # print(x)
    year = int(x[:4])
    month = int(x[4:6])
    day = int(x[6:8])
    hour = int(x[8:10])
    #min  = int(x[10:12])
    #sec  = int(x[12:])
    return dt.datetime(year, month, day, hour)

def dataset_trans2(df, types, Num_df_user, Num_errtype, First_index, fwver_total):
    num_df_user = Num_df_user
    num_errtype =Num_errtype
    first_index = First_index
    
    df2 = df.copy()
    
    df2['time_second'] = df2['time'].apply(make_datetime_second)
    
    
    
    df2['hour'] =df2.time_second.dt.hour


    df2 = df2.loc[(df2.time_second >=pd.to_datetime('2020-11-01 00:00:00')) & (df2.time_second <=pd.to_datetime('2020-11-30 23:59:59'))]
    #datas = df[['user_id','errtype','hour']]
    #df_=datas[['user_id','hour','errtype']].count().to_frame().reset_index()
    df_=df2.groupby(['user_id','hour','errtype']).count().reset_index()
    #df_ =df_.sort_values(['user_id','hour']).rename(columns = {0:'counts'}).reset_index(drop=True)


    day_data = np.zeros((num_df_user,42,24))
    for i in range(24):
        # time 변수가 결국 count 수.
        dfa = df_.loc[df_['hour']==i][['user_id','errtype','time']].values
        for inx , val1 ,val2 in tqdm(dfa):
            day_data[:,:,i][inx-first_index,val1-1] = val2

    m2=day_data.mean(axis=2)
    std2=day_data.std(axis=2)       
    m2_max = day_data.max(axis=2)
    #m2_1=day_data.max(axis=2)
    
    df2['day'] =df2.time_second.dt.day


    df2 = df2.loc[(df2.time_second >=pd.to_datetime('2020-11-01 00:00:00')) & (df2.time_second <=pd.to_datetime('2020-11-30 23:59:59'))]
    #datas = df[['user_id','errtype','day']]
    #df_=datas[['user_id','day','errtype']].count().to_frame().reset_index()
    df_=df2.groupby(['user_id','day','errtype']).count().reset_index()
    #df_ =df_.sort_values(['user_id','day']).rename(columns = {0:'counts'}).reset_index(drop=True)


    day_data = np.zeros((num_df_user,42,30))
    for i in range(30):
        dfa = df_.loc[df_['day']==(i+1)][['user_id','errtype','time']].values
        for inx , val1 ,val2 in tqdm(dfa):
            day_data[:,:,i][inx-first_index,val1-1] = val2

    m3=day_data.mean(axis=2)
    std3=day_data.std(axis=2)       
    m3_max = day_data.max(axis=2)
    #m3_1=day_data.max(axis=2)
    
    return [m2, std2,m2_max, m3, std3,m3_max]



def f_pr_auc(probas_pred, y_true):
    labels=y_true.get_label()
    p, r, _ = precision_recall_curve(labels, probas_pred)
    score=auc(r,p) 
    return "pr_auc", score, True


def mk_err_feature(df,user_num,user_min,complainer_48h_errcode_unique_testtrain,no_complainer_48h_errcode_unique_testtrain):
    model = KeyedVectors.load_word2vec_format('errtype_w2v')
    df['typecode'] = df.errtype.astype(str) + df.errcode.astype(str)
    id_err_var = df[['user_id','typecode','errtype','fwver','errcode','model_nm']].values

    # 빈 array 생성
    typecode_arr = np.zeros((user_num,3))
    type_arr = np.zeros((user_num,47))
    fwver_arr = np.zeros((user_num,14))
    code_arr = np.zeros((user_num, 17))
    type_w2v_arr = np.zeros((user_num, 32))

    for idx, typecode,type, fwver, code, model in tqdm(id_err_var):

        # type + code
        if typecode in ['101','23connection fail to establish']:
            typecode_arr[idx - user_min,0] += 1
        elif typecode in ['40','332','261','141','151','161','111','121']:
            typecode_arr[idx - user_min,1] += 1

        typecode_arr[idx - user_min,2] = (typecode_arr[idx - user_min,0])/(typecode_arr[idx - user_min,1]+1)
        
        # errtype
        type_arr[idx - user_min,type - 1] += 1

        if type in [25,18,20,19,21]:
            type_arr[idx - user_min,42] += 1
        elif type in [34,10,35,13,30,27,28]:
            type_arr[idx - user_min,43] += 1
        elif type in [2,4,42,26]:
            type_arr[idx - user_min,44] += 1
        elif type in [1,8]:
            type_arr[idx - user_min,45] += 1
        type_arr[idx - user_min,46] = (type_arr[idx - user_min,42])/(type_arr[idx - user_min,45]+1)  

        # lst = []
        # for type in id_err_var[id_err_var.user_id==idx].errtype:
        #     lst.append(model.wv.get_vector(str(type)))

        # type_w2v_arr[idx-user_min] = np.array(lst).mean(axis=0)

        # fwver
        fwver_dict = {'05.15.2138':0,'04.22.1750':1,'04.33.1261':2,'04.16.3553':3,'03.11.1167':4,'04.22.1778':5,'04.22.1684':6,'04.33.1185':7,'04.16.3571':8}
        try:
            fwver_arr[idx-user_min,fwver_dict[fwver]] += 1
        except:
            fwver_arr[idx-user_min,9] += 1

        if fwver in ['04.33.1149','04.73.2571','04.16.3571']:
            fwver_arr[idx-user_min,10] += 1
        elif fwver in ['05.15.2120','10']:
            fwver_arr[idx-user_min,11] += 1
        elif fwver in ['04.73.2237','04.22.1684','05.15.2138']:
            fwver_arr[idx-user_min,12] += 1
        fwver_arr[idx - user_min,13] = (fwver_arr[idx-user_min,10])/(fwver_arr[idx-user_min,12]+1)

        # errcode
        errcode_top14 = ['1', '0', 'connection timeout', 'B-A8002', '80', '79', '14', 'active','2', '84', '85', 'standby', 'NFANDROID2','connection fail to establish']
        if code in errcode_top14:
            code_arr[idx-user_min,errcode_top14.index(code)] += 1
        elif code in list(complainer_48h_errcode_unique_testtrain)+['5','6','V-21008','terminate by peer user']:
            code_arr[idx-user_min,14] += 1
        # elif code in ['H-51042','connection fail to establish','4','14','13','83','3','connection timeout']:
        #     code_arr[idx-user_min,15] += 1
        elif code in list(no_complainer_48h_errcode_unique_testtrain)+['Q-64002','S-65002','0']:
            code_arr[idx-user_min,15] += 1
        code_arr[idx-user_min,16] = (code_arr[idx-user_min,14])/(code_arr[idx-user_min,15]+1)


    # 변수 평균 분산 추가
    type_mean = type_arr[:,42:].mean(axis=1)
    type_std = type_arr[:,42:].std(axis=1)

    typecode_mean = typecode_arr.mean(axis=1)
    typecode_std = typecode_arr.std(axis=1)

    fwver_arr_mean = fwver_arr[:,9:].mean(axis=1)
    fwver_arr_std = fwver_arr[:,9:].std(axis=1)

    code_mean = code_arr[:,:14].mean(axis=1)
    code_std = code_arr[:,:14].std(axis=1)

    mean_var = np.concatenate((type_mean.reshape(-1,1),type_std.reshape(-1,1),typecode_mean.reshape(-1,1),typecode_std.reshape(-1,1),fwver_arr_mean.reshape(-1,1),fwver_arr_std.reshape(-1,1),code_mean.reshape(-1,1),code_std.reshape(-1,1)),axis=1)

    return np.concatenate((typecode_arr,type_arr,fwver_arr,code_arr),axis=1)


def mk_qt_feature(df,vars,user_num,user_min):

    for qual_num in list(map(lambda x: 'quality_'+ x, [str(i) for i in range(13)])):
        df[qual_num] = df[qual_num].apply(lambda x: float(x.replace(",","")) if type(x) == str else x)

    q1 = np.zeros((user_num,6))
    q2 = np.zeros((user_num,6))
    q3 = np.zeros((user_num,1))
    qt_cnt = df.groupby('user_id').count()['time']/12
    dict = {key:value for key,value in zip(qt_cnt.index,qt_cnt.values)}
    for i in range(user_num):
        if i+user_min in dict.keys():
            q3[i,0] = dict[i+user_min]

    # 0,1,2,6,8,11,12 거의 비슷, 5,7,9,10 거의 비슷, 각각 평균 내서 사용
    for i, var in enumerate(vars):
        id_q = df[['user_id',var]].values
        res = np.zeros((user_num,6))

        for idx, num in tqdm(id_q):
            if num == 0:
                res[int(idx)-user_min,0] += 1
            elif num == -1:
                res[int(idx)-user_min,1] += 1
            elif num == 1:
                res[int(idx)-user_min,2] += 1
            elif num == 2:
                res[int(idx)-user_min,3] += 1
            elif num == 3:
                res[int(idx)-user_min,4] += 1
            else:
                res[int(idx)-user_min,5] += 1
        q1 += res

        qt_mean = q1.mean(axis=1)
        qt_var = q1.std(axis=1)

        # q1 = q1/q1.sum(axis=1).shape(-1,1)
        
    return np.concatenate((q1/11,q3,qt_mean.reshape(-1,1),qt_var.reshape(-1,1)),axis=1)



def make_datetime(x):
    # string 타입의 Time column을 datetime 타입으로 변경
    x = str(x)
    # print(x)
    year = int(x[:4])
    month = int(x[4:6])
    day = int(x[6:8])
    hour = int(x[8:10])
    # min  = int(x[10:12])
    # sec  = int(x[12:])
    return dt.datetime(year, month, day, hour)



def mk_time_feature(df, user_num, user_min,err_mode=True):
    # hour 구간  count 4개 비율 4개 총 8개
    # day 구간 count 4개 비율  4개 총 8개
    # err은 일자별 error statics 6개
    # qual: 16개/err : 22개
    df['time'] = df['time'].map(lambda x: make_datetime(x))

    # df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek

    # # hour
    # hour_error = df[['user_id', 'hour']].values
    # hour = np.zeros((user_num, 24))
    #
    # for person_idx, hr in tqdm(hour_error):
    #     hour[person_idx - user_min, hr - 1] += 1

    df["hour"] = df["time"].dt.hour
    conditionlist = [
        (df['hour'] >= 11) & (df['hour'] < 14),
        (df['hour'] >= 14) & (df['hour'] < 20),
        (df['hour'] >= 20) & (df['hour'] < 24) | (df['hour'] == 0)]

    choicelist = [0, 1, 2]  # lunch :0, Afternoon:1 , Night : 2, others : 3
    df['hour_segment'] = np.select(conditionlist, choicelist, default=3)

    df_time_err = pd.concat([df['user_id'], df['hour_segment']], axis=1).values

    hour_err = np.zeros((user_num, 8))

    print('hour_Err shape', hour_err.shape)
    print('train_time_err shape', df_time_err.shape)

    for person_idx, hr in tqdm(df_time_err):
        hour_err[person_idx - user_min, hr - 1] += 1

    hour_err_sum = np.sum(hour_err, axis=1)

    for num in range(4):
        hour_err[:, num + 4] = hour_err[:, num] / hour_err_sum
    
    df_hour = pd.DataFrame(hour_err)
    df_hour = df_hour.fillna(0)

    hour_err = df_hour.values

    # day
    day_error = df[['user_id', 'dayofweek']].values
    day = np.zeros((user_num, 4))

    for person_idx, d in tqdm(day_error):
        if d == 1:
            day[person_idx - user_min, 0] += 1
        if d == 5:
            day[person_idx - user_min, 1] += 1
        if d == 6:
            day[person_idx - user_min, 2] += 1
        else:
            day[person_idx - user_min, 3] += 1

    df_day = pd.DataFrame(day, columns=['Mon', 'Sat', 'Sun', 'others'])
    df_day['all'] = df_day['Mon'] + df_day['Sat'] + df_day['Sun'] + df_day['others']

    for var in ['Mon', 'Sat', 'Sun', 'others']:
        df_day[var + '_pct'] = df_day[var] / df_day['all']

    del df_day['all']
    df_day = df_day.fillna(0)

    df_day_val = df_day.values
    if err_mode :
        err_date = df.groupby([df['user_id'],df['time'].dt.date]).size().reset_index(name='counts')
        err_time_stat = err_date.groupby('user_id').agg({'counts': [np.min, np.max, np.mean, np.std, skew,np.size]}).reset_index()
        err_time_stat.columns = ['user_id', 'time_min', 'time_max', 'time_mean', 'time_std', 'time_skew','time_count']
        err_time_stat.time_std = err_time_stat.time_std.fillna(0)
        if user_min>10000:
            err_time_stat.loc[-1] = [43262, 0, 0, 0,0,0,0]  # adding a row
            err_time_stat.index = err_time_stat.index + 1  # shifting index
            err_time_stat = err_time_stat.sort_values(by='user_id')  # sorting by index
        err_time_stat.drop('user_id', axis=1, inplace=True)
        err_time_val = err_time_stat.values

        return np.concatenate((hour_err, df_day_val,err_time_val), axis=1)
    else:
        return np.concatenate((hour_err, df_day_val), axis=1)



## fwver_count
def mk_fwver_feature(df,user_num,user_min):
    df = df.groupby(['user_id', 'model_nm'])
    user_id_fwver_count = df['fwver'].describe()
    fwver_array = np.array(user_id_fwver_count.unique)
    fwver_count = np.zeros((user_num, 1))
    
    id = 0
    for user_id, model_nm in tqdm(user_id_fwver_count.index):
        fwver_count[user_id-user_min,0] += fwver_array[id]
        id +=1
        
    return fwver_count






def make_date(x):
    # string 타입의 Time column을 datetime 타입으로 변경
    x     = str(x)
    year  = int(x[:4])
    month = int(x[4:6])
    day   = int(x[6:8])
    return dt.datetime(year, month, day)



def fill_quality_missing(df_err, df_quality):
    # df_err['time_day']  = df_err['time'].map(lambda x : make_date(x))
    # df_quality['time_day']  = df_err['time'].map(lambda x : make_date(x))

    # #fwver 채우기
    # for i in len(df_quality[df_quality['fwver'].isna()]):
    #     df_quality[df_quality['fwver'].isna()][i]['fwver'] =  df_err[(df_err['user_id'] == df_quality[df_quality['fwver'].isna()][i]['user_id']) & (df_err['time_day'] ==df_quality[df_quality['fwver'].isna()][i]['time_day'])]['fwver'][0]


    #quality_n 채우기
    qual_list = ['quality_0', 'quality_1', 'quality_2', 'quality_5', 'quality_6', 'quality_7', 'quality_8', 'quality_9', 'quality_11', 'quality_12']
    for i in qual_list:
        df_quality[i].fillna(0)

    df_quality['qulity_10'].fillna(3)
    
    return df_quality


def err_count(df,user_num, df_cat):
    if df_cat == 'train':
        n_total_train = df.groupby('user_id')['user_id'].count()
        #print(n_total_train.shape)
        output= np.array(n_total_train).reshape(user_num,1)
    else:
        n_total_test = df.groupby('user_id')['user_id'].count()
        total_test_list = n_total_test.tolist()
        total_test_list.insert(13262,0)
        output= np.array(total_test_list).reshape(user_num,1)
        #test_x3.shape
        
    return output


def qua_count(df,user_num, user_min,qt_id, noqt_id):
    qua_count = df.groupby('user_id')['user_id'].count()/12
    qua_count_mean = qua_count.mean()
    qua_count_list = [0 for i in range(user_num)]
    
    id=0
    for i in qt_id:
        i = i-user_min
        qua_count_list[i] = qua_count.iloc[id]
        id+=1
    for i in noqt_id:
        i = i-user_min
        qua_count_list[i] = qua_count_mean
    return np.array(qua_count_list).reshape(user_num,1)


def tfidf(train=True):
    if train:
        with open("train_errtype_Text.pickle","rb") as fr:
            lst = pickle.load(fr)
    else:
        with open("test_errtype_Text.pickle","rb") as fr:
            lst = pickle.load(fr)

    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(lst)

    return X.toarray()


def make_datetime_second(x):
    # string 타입의 Time column을 datetime 타입으로 변경
    x = str(x)
    # print(x)
    year = int(x[:4])
    month = int(x[4:6])
    day = int(x[6:8])
    hour = int(x[8:10])
    min  = int(x[10:12])
    sec  = int(x[12:])
    return dt.datetime(year, month, day, hour, min, sec)



def dataset_trans(df, types, Num_df_user, Num_errtype, First_index, fwver_total):
    num_df_user = Num_df_user
    num_errtype =Num_errtype
    first_index = First_index
        
    
    fwver_total_dic ={}
    for v in range(len(fwver_total)):
        fwver_total_dic[sorted(list(fwver_total))[v]] = v+1

    def fwver_tran(x):
        return fwver_total_dic[x]


    df['ver_num'] = df['fwver'].apply(fwver_tran)
    #
    fwver_np = np.zeros((num_df_user,5))

    v3=df[['user_id','ver_num']]
    getdf =~(v3 == v3.shift(1))
    logical =(getdf.user_id.apply(int) + getdf.ver_num.apply(int)) > 0
    fwver_num=v3[logical]

    fwver_num = fwver_num.reset_index(drop=True)
    count =np.zeros(len(fwver_num),dtype=int)

    for v in range(1,len(fwver_num)):
        if fwver_num.user_id.values[v-1] ==fwver_num.user_id.values[v]:
            count[v] = count[v-1] +1


    fwver_num['count'] =count
    fw_v = fwver_num.loc[fwver_num['count'].isin([0,1,2,3,4])].pivot(index='user_id',columns='count').reset_index().fillna(0).values
    fw_v =fw_v.astype('int64')

    
    for inx, v1,v2,v3,v4,v5 in tqdm(fw_v):
        fwver_np[inx-first_index,0] =v1
        fwver_np[inx-first_index,1] =v2
        fwver_np[inx-first_index,2] =v3
        fwver_np[inx-first_index,3] =v4
        fwver_np[inx-first_index,4] =v5
    #print(fwver_np.shape)
    #print(model_n.shape)
    
    target_df = df
    first_num = first_index
    count_num =num_df_user

    dp = target_df[['user_id','model_nm','fwver']]
    unique_data =target_df[(dp !=dp.shift(1)).sum(axis=1)>0]

    dp2 = target_df[['user_id','model_nm']]
    unique_data2 =target_df[(dp2 !=dp2.shift(1)).sum(axis=1)>0]

    fwver_total_dic ={}
    for v in range(len(fwver_total)):
        fwver_total_dic[sorted(list(fwver_total))[v]] = v+1
        
    def fwver_tran(x):
        return fwver_total_dic[x]

    fwver = np.zeros((count_num,24))
    for idx in tqdm(unique_data.user_id.unique()):
        df_md =unique_data2.loc[unique_data2.user_id==idx].model_nm.values
        df_fw = unique_data.loc[unique_data.user_id==idx].fwver.values

        for md in range(len(df_md)):
            fwver[idx-first_num,md] = int(df_md[md][-1])+1

        for l in range(3,len(df_fw)+3):
            fwver[idx-first_num,l] =fwver_total_dic[df_fw[l-3]]

    fw_df = pd.DataFrame(fwver).reset_index().rename(columns={'index':'user_id'})

    fwver_total_dic_rev = {v: k for k, v in fwver_total_dic.items()}
    fwver_total_dic_rev2 = fwver_total_dic_rev.copy()
    fwver_total_dic_rev[0] =0
    fwver_total_dic_rev2[0] = '04.22.1750'  #max 값


    def fwver_tras_reverse(x):
        return fwver_total_dic_rev[x]

    def fwver_tras_reverse2(x):
        return fwver_total_dic_rev2[x]

    fw_df[3] =fw_df[3].apply(fwver_tras_reverse2)
    fw_df[4] =fw_df[4].apply(fwver_tras_reverse)
    fw_df[5] =fw_df[5].apply(fwver_tras_reverse)
    fw_df[6] =fw_df[6].apply(fwver_tras_reverse)
    fw_df[7] =fw_df[7].apply(fwver_tras_reverse)


    fw_df = fw_df.rename(columns={0:'md1',1:'md2',2:'md3',3:'fw1',4:'fw2',5:'fw3',6:'fw4',7:'fw5'})
    fw_df['user_id'] =fw_df['user_id']+10000

    pre_df=fw_df.iloc[:,:9]

    md_flow = {str(x.astype("int")):(i+1) for i,x in enumerate(pre_df[['md1','md2','md3']].drop_duplicates().reset_index(drop=True).values)}
    fw_flow = {str(x):(i+1) for i,x in enumerate(pre_df[['fw1','fw2','fw3','fw4','fw5']].drop_duplicates().reset_index(drop=True).values)}

    
    def fw_change_counter(x):
        fwlst = []
        for v in ['fw1','fw2','fw3','fw4','fw5']:
            if x[v] ==0:
                pass
            else:
                fwlst +=[x[v]]

        if len(fwlst) ==len(list(set(fwlst))):
            return 0
        else:
            return 1


    def md_flow_change(x):
        return md_flow[str(x[['md1','md2','md3']].values.astype("int"))]

    def fw_flow_change(x):
        return fw_flow[str(x[['fw1','fw2','fw3','fw4','fw5']].values)]
    
    def mean_str_fw_dum(x):
        fwlst = []
        for v in ['fw1','fw2','fw3','fw4','fw5']:
            if x[v] ==0:
                pass
            else:
                fwlst +=[int(x[v].replace('.',""))]
        return np.array(fwlst).mean()



    def std_str_fw_dum(x):
        fwlst = []
        for v in ['fw1','fw2','fw3','fw4','fw5']:
            if x[v] ==0:
                pass
            else:
                fwlst +=[int(x[v].replace('.',""))]
        return np.array(fwlst).std()

    pre_df=fw_df.iloc[:,:9]
    #pre_df['md_counts'] = pre_df[['md1','md2','md3']].astype('bool').sum(axis=1)
    #pre_df['fw_counts'] = pre_df[['fw1','fw2','fw3','fw4','fw5']].astype('bool').sum(axis=1)

    pre_df['fw_change'] = pre_df.apply(fw_change_counter,axis=1)
    pre_df['fw_flows'] = pre_df.apply(fw_flow_change,axis=1)
    pre_df['md_flows'] = pre_df.apply(md_flow_change,axis=1)
    
    ## mean, std 추가해볼만하다.
    pre_df['fw_mean'] = pre_df.apply(mean_str_fw_dum,axis=1)
    pre_df['fw_std'] = pre_df.apply(std_str_fw_dum,axis=1)
    fw_model_flow =pre_df.iloc[:,9:].values
    
    
    first_num = first_index
    count_num =num_df_user
    
    time_term = np.zeros((count_num,4))
    
    df['time_second'] = df['time'].apply(make_datetime_second)
    tre_t =df[['user_id','time_second']].drop_duplicates()

    for v in tqdm(tre_t.user_id.unique()):
        test =tre_t.loc[tre_t.user_id ==v].time_second
        if len(test) <=2:
            time_term[v-first_num,0] = 0
            time_term[v-first_num,1] = 0
            time_term[v-first_num,2] = test.values[-1]-test.values[0]
            time_term[v-first_num,3] = len(test)
        else:
            time_term[v-first_num,0] = (test -test.shift(1)).max().total_seconds()
            time_term[v-first_num,1] = (test -test.shift(1)).min().total_seconds()
            time_term[v-first_num,2] = test.values[-1]-test.values[0]  
            time_term[v-first_num,3] = len(test)

    dft = pd.DataFrame(time_term).copy()

    dft[0] =dft[0]/3600
    dft[2] =dft[2]/3600/24/10e8
    dft[2] =np.where(dft[2].values==0,1,dft[2].values)
    dft[4] =dft[0]/dft[3]
    dft[4] = dft[0]/dft[3]*3600
    dft[5] = dft[0]/24/dft[2]
    time_term = dft.fillna(0).values
    
    
    tsed = df.dropna(axis=0).reset_index(drop=True)[['user_id','time','fwver']]
    dfw = tsed[['user_id','fwver']]
    fw_d =dfw.loc[(dfw !=dfw.shift(1)).sum(axis=1)>0]

    main_fw_ar = np.zeros((num_df_user,6))
    for i,tgid in enumerate(tqdm(range(first_index,first_index+num_df_user))):

        tgdf =fw_d.loc[fw_d.user_id ==tgid].iloc[1:,:]
        tgidtotal = tsed.loc[tsed.user_id ==tgid]
        try:
            data =tgidtotal.loc[sorted([tgidtotal.index[0]] + [x-1 for x in tgdf.index]+[x for x in tgdf.index] + [tgidtotal.index[-1]] )]
            t1 =data.time_second
            if len(t1) %2 !=0:
                print('lenth error')
            time_delta = (t1-t1.shift(1)).dt.total_seconds()

            main_fwver =data.loc[time_delta.loc[time_delta==time_delta.max()].index].fwver.values[0]
            main_fw_ar[i,0] = fwver_total_dic[main_fwver]
            main_fw_ar[i,1] =(time_delta[1::2].values).max().astype('float')/(time_delta.values[1:]).sum().astype('float')  #target fw workingtime / total
            if len(time_delta) ==1:
                main_fw_ar[i,2] =0  #min of change fwver time==0
                main_fw_ar[i,3] =0  #std of change fwver time ==0
                main_fw_ar[i,4] =0  #std
                main_fw_ar[i,5] =0  #variance
            else:
                main_fw_ar[i,2] =time_delta[::2].min()/3600 # min hours
                main_fw_ar[i,3] =time_delta[::2].std()/3600
                main_fw_ar[i,4] =time_delta[1::2].values.astype('float').std()/3600  #std running time of fw
                main_fw_ar[i,5] =(time_delta[1::2].values.astype('float')/3600).var()  #std running time of fw
        except:
            main_fw_ar[i,0] =0
            main_fw_ar[i,1] =0
            main_fw_ar[i,2] =0
            main_fw_ar[i,3] =0
            main_fw_ar[i,4] =0
            main_fw_ar[i,5] =0
   
    #5, 5, 6, 6
    return [fwver_np, fw_model_flow, time_term, main_fw_ar]

def check_unique(col,df1,df2):
    def change_len(x):
        if len(x) ==10:
            return x[:5]
        else:
            return x
    print("about",col)
    if col !='fwvers':
        train_c = set(df1[col].unique())
        test_c  = set(df2[col].unique())
        total = (train_c | test_c)
    else:
        train_c = set(df1[col].apply(change_len).unique())
        test_c  = set(df2[col].apply(change_len).unique())
        total = (train_c | test_c)        
        
    print()
        
    return total

def qual_change(df, user_num, user_min):
    tmp = df.groupby('user_id')[['quality_' + str(i) for i in range(13)]].nunique() - 1
    tmp2 = tmp.sum(axis=1)
    qual_dic = defaultdict(lambda: 0, zip(tmp2.index, tmp2))
    qaul_num = pd.DataFrame(data={'user_id': [num for num in range(user_min, user_min+user_num)]})
    qaul_num['n_qualchange'] = qaul_num['user_id'].map(qual_dic)

    return qaul_num['n_qualchange'].values


def model_ft(df,user_num):
    # model_nm
    id_model = df[['user_id','model_nm']]
    #user_num = 14999
    #model = np.zeros((user_num,9))
    id_model_not_dup = id_model.drop_duplicates()
    id_model_count = id_model_not_dup.groupby('user_id').count()
    id_model_count_np = id_model_count.values

    if user_num == 15000 :
        id_model_count_np[13991 - 10000][0] = 3
        id_model_count_np[18525 - 10000][0] = 3
        id_model_count_np[20921 - 10000][0] = 3

    elif user_num == 14999:
        id_model_count_np=np.insert(id_model_count_np,13262,1)
        id_model_count_np[34758 - 30000] = 3

    return id_model_count_np


def qual_statics(df, user_count, user_min):
    # quality 11개 별 4개의 statics : 44개변수
    # -1의 갯수와 비율 2개
    # 12개, 24개, 24/12 비율
    # 총 49개
    for x in range(0,13):
        if x == 3 or x==4:
            pass
        else:
            qual_df = df.groupby('user_id')['quality_'+str(x)].agg(['mean', 'std', 'min', 'max'])
            qual_df = qual_df.reset_index()
            qaul_num = pd.DataFrame(data={'user_id': [num for num in range(user_min, user_min+user_count)]})
            ql_mg = pd.merge(qaul_num,qual_df,on='user_id',how='left')
            ql_mg.drop('user_id',axis=1,inplace=True)
            ql_val = ql_mg.fillna(0).values
            if x == 0:
                qual_val_all = ql_val
            else:
                qual_val_all = np.concatenate((qual_val_all,ql_val),axis=1)

    qual_num = pd.DataFrame(data={'user_id': [num for num in range(user_min, user_min+user_count)]})


        
    for x in range(0,13):
        if x == 3 or x==4:
            pass
        else:
            qual_i_mean = df['quality_'+str(x)].agg(['mean']) #값 하나

            user_qual_i_mean = df.groupby('user_id')['quality_'+str(x)].agg(['mean'])
            user_qual_i_mean = user_qual_i_mean.reset_index()
            user_qual_i_std = df.groupby('user_id')['quality_'+str(x)].agg(['std'])
            user_qual_i_std = user_qual_i_std.reset_index()
            qual_num = pd.DataFrame(data={'user_id': [num for num in range(user_min, user_min+user_count)]})
            user_qual_i_mean_df = pd.merge(qual_num,user_qual_i_mean,on='user_id',how='left')
            user_qual_i_std_df = pd.merge(qual_num,user_qual_i_std,on='user_id',how='left')

            qual_ff = user_qual_i_std_df['std'] /(user_qual_i_mean_df['mean'] - qual_i_mean['mean'])
            qual_ff = qual_ff.fillna(0)
            qual_ff = qual_ff.values
            qual_ff = qual_ff.reshape((-1,1))

            if x == 0:
                qual_ff_all = qual_ff
            else:
                qual_ff_all = np.concatenate((qual_ff_all,qual_ff),axis=1)
        
    
    col = 'quality_1'
    q1_minus1_cnt = df[df[col] == -1 ].groupby('user_id').count()[col] 
    q1_minus1_cnt = q1_minus1_cnt.reset_index("user_id")
    q1_minus1_cnt_done = pd.merge(qual_num,q1_minus1_cnt,on='user_id',how='left')
    q1_minus1_cnt_done = q1_minus1_cnt_done.fillna(0)
    # q1_minus1_cnt_np = q1_minus1_cnt_done1.drop('user_id',axis=1).values
    # print(q1_minus1_cnt_np)

    ##quality_1에서 -1 비율
    qual_cnt = df.groupby('user_id').count()[col] 
    qual_cnt = qual_cnt.reset_index("user_id")
    qual_cnt.rename(columns = {col : col+'count'}, inplace = True)
    qual_cnt_done = pd.merge(qual_num,qual_cnt,on='user_id',how='left')

    q1_minus1_cnt_done[col+'_rate'] = q1_minus1_cnt_done[col] / qual_cnt_done[col+'count'] 
    q1_minus1_cnt_done[col+'_rate'] = q1_minus1_cnt_done[col+'_rate'].fillna(0)
    # q1_minus1_rate_np = q1_minus1_rate.drop('user_id',axis=1)
    qual_num = q1_minus1_cnt_done

    # print(qual_num)
    qual_num.drop('user_id',axis=1,inplace=True)
    qual_minus_val = qual_num.values

    qual_num = pd.DataFrame(data={'user_id': [num for num in range(user_min, user_min+user_count)]})
    #qual_num = pd.DataFrame(data={'user_id': [num for num in range(10000,25000)]})

    temp = df.groupby(['user_id','time']).count()

    quality_12= temp.reset_index()[temp.reset_index().fwver==12].user_id.value_counts()

    quality_12_df = quality_12.to_frame()
    quality_12_df = quality_12_df.reset_index()
    quality_12_df = quality_12_df.rename(columns = {'index' :'user_id','user_id':'counts'})
    quality_12_sum = quality_12_df.counts.sum()
    qual_12 = pd.merge(qual_num,quality_12_df,on='user_id',how='left')
    qual_12 = qual_12['counts']

    # 12/전체cnt
    qual_12_rate = qual_12 / quality_12_sum
    qual_12_rate= qual_12_rate.fillna(0).values.reshape(-1,1)

    quality_24= temp.reset_index()[temp.reset_index().fwver==24].user_id.value_counts()
    quality_24_df = quality_24.to_frame()
    quality_24_df = quality_24_df.reset_index()
    quality_24_df = quality_24_df.rename(columns = {'index' :'user_id','user_id':'counts'})
    quality_24_sum = quality_24_df.counts.sum()
    qual_24 = pd.merge(qual_num,quality_24_df,on='user_id',how='left')
    qual_24 = qual_24['counts']
    qual_24_count_np = qual_24.values

    # 12/전체cnt
    qual_24_rate = qual_24 / quality_24_sum
    qual_24_rate= qual_24_rate.fillna(0).values.reshape(-1,1)
    
    #24/12 비율
    qual_24_12_rate = qual_24 / qual_12
    qual_24_12_rate = qual_24_12_rate.fillna(0)
    qual_24_12_rate_np = qual_24_12_rate.values.reshape(-1,1)


    return np.concatenate((qual_val_all,qual_minus_val,qual_12_rate, qual_24_rate , qual_24_12_rate_np,qual_ff_all),axis=1)




def nun_err(df,ver):
    df_cp = df.copy()
    df_cp['errtype_errcode']= df_cp['errtype'].astype('str') + '_' + df_cp['errcode'].astype('str')
    nun_err = df_cp.groupby('user_id')['errtype','errcode','errtype_errcode'].nunique().reset_index()
    if ver =='train':
        nun_err.drop('user_id', axis=1, inplace=True)
        nun_err_val = nun_err.values
    else:
        nun_err.loc[-1] = [43262, 0, 0, 0]  # adding a row
        nun_err.index = nun_err.index + 1  # shifting index
        nun_err = nun_err.sort_values(by='user_id')  # sorting by index
        nun_err.drop('user_id', axis=1, inplace=True)
        nun_err_val = nun_err.values

    return nun_err_val




