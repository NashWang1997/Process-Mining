import pandas as pd
import datetime
import re
import random
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.preprocessing import sequence

def re_build_data(data_path, save_path):
    data_rebuilt = {}
    data_rebuilt['org:resource'] = []
    data_rebuilt['lifecycle:transition'] = []
    data_rebuilt['concept:name'] = []
    data_rebuilt['time:timestamp_short'] = []
    data_rebuilt['case:REG_DATE_short'] = []
    data_rebuilt['case:concept:name'] = []
    data_rebuilt['case:AMOUNT_REQ'] = []

    df = pd.read_csv(data_path)
    for index in range(df.shape[0]):
        data_rebuilt['org:resource'].append(df.iloc[index]['org:resource'])
        data_rebuilt['lifecycle:transition'].append(df.iloc[index]['lifecycle:transition'])
        data_rebuilt['concept:name'].append(df.iloc[index]['concept:name'])
        time_stamp_short = re.split('\.|\+', df.iloc[index]['time:timestamp'])
        data_rebuilt['time:timestamp_short'].append(time_stamp_short[0])
        reg_date_short = re.split('\.|\+', df.iloc[index]['case:REG_DATE'])
        data_rebuilt['case:REG_DATE_short'].append(reg_date_short[0])
        data_rebuilt['case:concept:name'].append(df.iloc[index]['case:concept:name'])
        data_rebuilt['case:AMOUNT_REQ'].append(df.iloc[index]['case:AMOUNT_REQ'])
    df_to_write = pd.DataFrame(data_rebuilt)
    df_to_write.to_csv(save_path, index=False)

def load_data(data_path, save_path):
    df = pd.read_csv(data_path)
    all_case_ids = set(df['case:concept:name'].values)
    num_trainset = len(all_case_ids) * 7 // 10
    random.seed(0)
    train_cids = random.sample(all_case_ids, num_trainset)
    train_cids_set = set(train_cids)
    test_cids_set = all_case_ids - train_cids_set
    print('train_data_size: ', len(train_cids_set), ', test_data_size: ', len(test_cids_set))

    activity_names = np.array(df['concept:name'].values)
    activity_names = np.reshape(activity_names, (activity_names.shape[0], 1))
    ohe_act = OneHotEncoder(sparse=False)
    ohe_act.fit(activity_names)

    weeks = np.array(list(range(0, 7)))
    weeks = np.reshape(weeks, (weeks.shape[0], 1))
    ohe_week = OneHotEncoder(sparse=False)
    ohe_week.fit(weeks)

    def generate_data(cid_set):
        data_set = []
        trace_length = 0
        for cid in cid_set:
            thisdf = df[df['case:concept:name']==cid]
            trace_length = max(trace_length, thisdf.shape[0])
            tmpdata = []

            start_time = datetime.datetime.strptime(thisdf.iloc[0]['time:timestamp_short'], '%Y-%m-%d %H:%M:%S')
            end_time = datetime.datetime.strptime(thisdf.iloc[-1]['time:timestamp_short'], '%Y-%m-%d %H:%M:%S')
            last_time = start_time

            for i in range(thisdf.shape[0]):
                row = [thisdf.iloc[i]['case:concept:name']]
                event_dt = datetime.datetime.strptime(thisdf.iloc[i]['time:timestamp_short'], '%Y-%m-%d %H:%M:%S')
                midnight_time = event_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                activity_name = np.array(thisdf.iloc[i]['concept:name'])
                activity_name = np.reshape(activity_name, (-1, 1))
                weekday = event_dt.weekday()
                weekday = np.reshape(np.array([weekday]), (-1, 1))

                row.extend(ohe_act.transform(activity_name).tolist()[0])
                row.append((event_dt - start_time).total_seconds()/3600/24)
                row.append((event_dt - last_time).total_seconds()/3600/24)
                last_time = event_dt
                row.append((event_dt - midnight_time).total_seconds()/3600/24)
                row.extend(ohe_week.transform(weekday).tolist()[0])
                row.append(thisdf.iloc[i]['case:AMOUNT_REQ'])

                row.append((end_time - event_dt).total_seconds()/3600/24)
                tmpdata.append(row)
                if i != thisdf.shape[0]-1:
                    data_set.append(tmpdata.copy())
        return data_set, trace_length

    train_data_set, max_train_trace_length = generate_data(train_cids_set)
    test_data_set, max_test_trace_length = generate_data(test_cids_set)

    min_value = [1e20] * (len(train_data_set[0][0]) - 1)
    max_value = [-1] * (len(train_data_set[0][0]) - 1)
    for element in train_data_set:
        for row in element:
            for i in range(len(row)-1):
                min_value[i] = min(min_value[i], row[i])
                max_value[i] = max(max_value[i], row[i])

    train_data_new = []
    for i in range(len(train_data_set)):
        seq = []
        for j in range(len(train_data_set[i])):
            row = [train_data_set[i][j][0]]
            for k in range(1, len(train_data_set[i][j])-1):
                row.append((train_data_set[i][j][k]-min_value[k]) / (max_value[k]-min_value[k]))
            row.append(train_data_set[i][j][-1])
            seq.append(row)
        train_data_new.append(seq)
    test_data_new = []
    for i in range(len(test_data_set)):
        seq = []
        for j in range(len(test_data_set[i])):
            row = [test_data_set[i][j][0]]
            for k in range(1, len(test_data_set[i][j])-1):
                row.append((test_data_set[i][j][k]-min_value[k]) / (max_value[k]-min_value[k]))
            row.append(test_data_set[i][j][-1])
            seq.append(row)
        test_data_new.append(seq)

    # max_length = max(max_train_trace_length, max_test_trace_length)
    train_data_set = sequence.pad_sequences(train_data_new, maxlen=10, dtype='float32')
    test_data_set = sequence.pad_sequences(test_data_new, maxlen=10, dtype='float32')
    np.save(save_path+'train_data_set_v1.npy', train_data_set)
    np.save(save_path+'test_data_set_v1.npy', test_data_set)
