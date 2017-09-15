import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

if __name__ == '__main__':

    df = pd.read_csv('data/training.txt', delimiter=';', parse_dates=['time_interval_begin'], dtype={'link_ID': object})
    link_tops = pd.read_csv('../raw/gy_contest_link_top.txt', delimiter=';', dtype={'link_ID': object})
    link_tops.fillna(' ', inplace=True)
    in_links_df = pd.DataFrame(link_tops['in_links'].str.split('#').tolist(),
                               columns=['in_link1', 'in_link2', 'in_link3', 'in_link4'])
    out_links_df = pd.DataFrame(link_tops['out_links'].str.split('#').tolist(),
                                columns=['out_link1', 'out_link2', 'out_link3', 'out_link4'])
    link_tops = pd.merge(link_tops, in_links_df, left_index=True, right_index=True)
    link_tops = pd.merge(link_tops, out_links_df, left_index=True, right_index=True)
    link_tops = link_tops.drop(['in_links', 'out_links'], axis=1)
    link_tops = link_tops.fillna(np.nan)
    link_tops = link_tops.replace(r'\s+', np.nan, regex=True)

    df = pd.merge(df, link_tops, on='link_ID', how='left')

    def applyParallel(dfGrouped, func):
        retLst = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(func, check_pickle=False)(group) for name, group in dfGrouped)
        return pd.concat(retLst)

    def related_lagging(group):
        tmp = group[['link_ID', 'lagging1']].copy().set_index('link_ID')
        for index, row in group.iterrows():
            if str(row['in_link1']) != 'nan':
                group.loc[index, 'in_link1_lagging'] = tmp.loc[row['in_link1']]['lagging1']
            if str(row['in_link2']) != 'nan':
                group.loc[index, 'in_link2_lagging'] = tmp.loc[row['in_link2']]['lagging1']
            if str(row['in_link3']) != 'nan':
                group.loc[index, 'in_link3_lagging'] = tmp.loc[row['in_link3']]['lagging1']
            if str(row['in_link4']) != 'nan':
                group.loc[index, 'in_link4_lagging'] = tmp.loc[row['in_link4']]['lagging1']

            if str(row['out_link1']) != 'nan':
                group.loc[index, 'out_link1_lagging'] = tmp.loc[row['out_link1']]['lagging1']
            if str(row['out_link2']) != 'nan':
                group.loc[index, 'out_link2_lagging'] = tmp.loc[row['out_link2']]['lagging1']
            if str(row['out_link3']) != 'nan':
                group.loc[index, 'out_link3_lagging'] = tmp.loc[row['out_link3']]['lagging1']
            if str(row['out_link4']) != 'nan':
                group.loc[index, 'out_link4_lagging'] = tmp.loc[row['out_link4']]['lagging1']
        # print group.index.values[0]
        return group


    df = applyParallel(df.groupby(df['time_interval_begin']), related_lagging)
    # df = df.groupby('time_interval_begin').apply(related_lagging)

    df['in_link_mean'] = df[['in_link1_lagging', 'in_link2_lagging', 'in_link3_lagging', 'in_link4_lagging']].mean(
        axis=1)
    df['out_link_mean'] = df[['out_link1_lagging', 'out_link2_lagging', 'out_link3_lagging', 'out_link4_lagging']].mean(
        axis=1)
    df['in_link_mean'].fillna(3, inplace=True)
    df['out_link_mean'].fillna(3, inplace=True)
    df = df.drop(['in_link1', 'in_link2', 'in_link3', 'in_link4', 'out_link1', 'out_link2', 'out_link3',
                  'out_link4'], axis=1)
    df = df.drop(['in_link1_lagging', 'in_link2_lagging', 'in_link3_lagging', 'in_link4_lagging', 'out_link1_lagging',
                  'out_link2_lagging', 'out_link3_lagging', 'out_link4_lagging'], axis=1)
    df.to_csv('data/training1.txt', header=True, index=None, sep=';', mode='w')
