import numpy as np
import pandas as pd
import os

def standardize(df):
    return (df - df.mean()) / df.std()

def fill_missing(df):
    return df.fillna(0)

def download_clean_data(folder_path, date, N):
    files = os.listdir(folder_path)

    datasets = []

    for file in files:
        if file == '.DS_Store':
            continue
        if int(file[15:23]) < date:
            continue
        df = pd.read_csv(folder_path + '/' + file,  encoding='utf-8')
        df['name'] = file
        datasets.append(df)

    datasets.sort(key=lambda x: x['name'][0])

    macro = pd.read_csv('Data/macro_data_amit_goyal.csv', encoding='utf-8')
    macro = macro[macro['yyyymm']>int(str(date)[:-2])]

    data = []
    ret = []

    for i,df in enumerate(datasets):
        
        df['mcap'] = df['SHROUT'] * df['prc']
        df.drop(['permno', 'DATE', 'Unnamed: 0', 'mve0', 'prc', 'SHROUT', 'sic2', 'name'], axis=1, inplace=True)
        df.dropna(thresh=60, axis=0, inplace=True)
        df = df[df['RET'] > -1]
        df = df.sort_values(by=['mcap'], ascending=False)
        df.drop(['mcap'], axis=1, inplace=True)
        df = df.head(N)
        ret.append(df['RET']-macro['Rfree'].values[i])
        df = df.drop(['RET'], axis=1)
        df = standardize(df)
        df = fill_missing(df)
        data.append(df)

    return data, ret