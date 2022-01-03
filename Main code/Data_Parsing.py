import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.impute import SimpleImputer

def parse_csv(filename, path='..\\Data', info_file='asset_details', imp_strategy=None, interp_method=None):
    '''
    - Read csv into dataframe
    - Add asset names
    - Remove timestamps that are not common for all coins
    - Sort by asset and timestamp
    - Impute missing values at minutes that are not common to all assets
    '''

    df_info = pd.read_csv(path + '\\' + info_file + '.csv')
    df = pd.read_csv(path + '\\' + filename + '.csv')
    timestamps = [t for t in pd.unique(df['timestamp']) if len(df[df['timestamp']==t])==len(pd.unique(df['Asset_ID']))]
    df = df[df['timestamp'].isin(timestamps)]

    df.sort_values(by=['Asset_ID', 'timestamp'], inplace=True)
    df = df.reset_index(drop=True)

    if imp_strategy != None:
        df = fill_missing_minutes(df=df, imp_strategy=imp_strategy)
    else:
        if interp_method != None:
            df = interpolate_missing_minutes(df=df, method=interp_method)

    df.insert(loc=2, column='Asset_Name', value=0)
    for (id, name) in zip(df_info['Asset_ID'], df_info['Asset_Name']):
        df['Asset_Name'] = np.where(df['Asset_ID'] == id, name, df['Asset_Name'])

    df['logR'] = np.nan
    for asset in pd.unique(df['Asset_Name']):
        df.loc[df['Asset_Name']==asset, 'logR'] = log_return(series=df[df['Asset_Name']==asset]['Close'], periods=15)
    df.loc[np.isnan(df['logR']), 'logR'] = -9999

    return df


def fill_missing_minutes(df, imp_strategy='median'):

    all_timestamps = range(df['timestamp'].min(), df['timestamp'].max()+60, 60)

    missing_timestamps = {asset: [t for t in all_timestamps if t not in df[df['Asset_ID']==asset]['timestamp'].to_numpy()]
                          for asset in pd.unique(df['Asset_ID'])}

    df_imputed = pd.DataFrame(data=[], columns=df.columns)

    for asset in pd.unique(df['Asset_ID']):

        df_asset = df[df['Asset_ID'] == asset]

        for t in missing_timestamps[asset]:
            new_row = {'timestamp': t}
            new_row['Asset_ID'] = asset
            new_row.update({col: np.nan for col in df_asset.columns if col not in ['timestamp', 'Asset_ID']})
            df_asset = df_asset.append(new_row, ignore_index=True)

        df_asset = df_asset.sort_values(by='timestamp').reset_index(drop=True)
        fill_nan = SimpleImputer(missing_values=np.nan, strategy=imp_strategy).fit(df_asset)

        df_imputed = df_imputed.append(pd.DataFrame(data=fill_nan.transform(df_asset), columns=df_imputed.columns), ignore_index=True)

    return df_imputed

def interpolate_missing_minutes(df, method='linear'):

    all_timestamps = range(df['timestamp'].min(), df['timestamp'].max()+60, 60)

    missing_timestamps = {asset: [t for t in all_timestamps if t not in df[df['Asset_ID']==asset]['timestamp'].to_numpy()]
                          for asset in pd.unique(df['Asset_ID'])}

    df_interpolated = pd.DataFrame(data=[], columns=df.columns)

    for asset in pd.unique(df['Asset_ID']):

        df_asset = df[df['Asset_ID'] == asset]

        for t in missing_timestamps[asset]:
            new_row = {'timestamp': t}
            new_row['Asset_ID'] = asset
            new_row.update({col: np.nan for col in df_asset.columns if col not in ['timestamp', 'Asset_ID']})
            df_asset = df_asset.append(new_row, ignore_index=True)

        df_asset = df_asset.sort_values(by='timestamp').reset_index(drop=True)

        df_interpolated = df_interpolated.append(df_asset.interpolate(method=method), ignore_index=True)

    return df_interpolated

def add_Market(df, path='..\\Data', info_file='asset_details'):
    '''
    - Add market as a new asset
    - Estimate market features as a weighted sum of asset features (weighted average)
    - Sort by asset and timestamp
    '''

    df_info = pd.read_csv(path + '\\' + info_file + '.csv')
    df_info['Weight'] /= df_info['Weight'].sum()

    df_market = pd.DataFrame(data=[], columns=df.columns)
    timestamps = [t for t in pd.unique(df['timestamp']) if len(df[df['timestamp']==t])==len(pd.unique(df['Asset_ID']))]
    df_market['timestamp'] = np.sort(timestamps)
    df_market['Asset_ID'] = df['Asset_ID'].max() + 1
    df_market['Asset_Name'] = 'Market'
    for col in df_market.columns:
        # if col in ['timestamp', 'Asset_ID', 'Asset_Name', 'Target', 'logR']:
        if col in ['timestamp', 'Asset_ID', 'Asset_Name', 'Target']:
            continue
        assets_col = pd.pivot(df.sort_values(by=['Asset_ID', 'timestamp'])[['Asset_ID', col]], columns='Asset_ID').to_numpy()
        df_market[col] = np.stack([weight * array[~np.isnan(array)]
                                   for (weight, array) in zip(df_info.sort_values(by='Asset_ID')['Weight'].to_numpy(),
                                                              assets_col.T)]).sum(axis=0).T
    df_market['Target'] = np.nan

    df = df.append(df_market).reset_index(drop=True)
    df.sort_values(by=['Asset_ID', 'timestamp'], inplace=True)
    df.loc[df['logR']==-9999, 'logR'] = np.nan
    return df

def add_Market_timeweight(df, weight_feat='Volume'):
    '''
    - Add market as a new asset
    - Weights are estimated actively over time based on 'weight feature'
    - Estimate market features as a weighted sum of asset features (weighted average)
    - Sort by asset and timestamp
    '''

    assets = [asset for asset in pd.unique(df['Asset_Name']) if asset != 'Market']
    df_weight = pd.DataFrame(data=np.c_[pd.unique(df['timestamp']),
                                        np.zeros((len(pd.unique(df['timestamp'])), len(assets)))],
                             columns=['timestamp']+assets)

    for t in pd.unique(df['timestamp']):
        df_dummy = df[df['timestamp']==t][['Asset_Name', weight_feat]]
        df_dummy[weight_feat] /= df_dummy[weight_feat].sum()
        for asset in assets:
            df_weight.loc[df_weight['timestamp']==t, asset] = df_dummy[df_dummy['Asset_Name']==asset][weight_feat].to_numpy()


    df_market = pd.DataFrame(data=[], columns=df.columns)
    timestamps = [t for t in pd.unique(df['timestamp']) if len(df[df['timestamp']==t])==len(pd.unique(df['Asset_ID']))]
    df_market['timestamp'] = np.sort(timestamps)
    df_market['Asset_ID'] = df['Asset_ID'].max().max() + 1
    df_market['Asset_Name'] = 'Time market'
    df.loc[np.isnan(df['logR']), 'logR'] = -9999
    for col in df_market.columns:
        # if col in ['timestamp', 'Asset_ID', 'Asset_Name', 'Target', 'logR']:
        if col in ['timestamp', 'Asset_ID', 'Asset_Name', 'Target']:
            continue
        assets_col = pd.pivot(df.sort_values(by=['Asset_ID', 'timestamp'])[['Asset_ID', col]], columns='Asset_ID').to_numpy()
        df_market[col] = np.stack([df_weight[asset] * array[~np.isnan(array)]
                                   for (asset, array) in zip(assets, assets_col.T)]).sum(axis=0).T
    df_market['Target'] = 0

    df = df.append(df_market).reset_index(drop=True)
    df.sort_values(by=['Asset_ID', 'timestamp'], inplace=True)
    df.loc[df['logR']==-9999, 'logR'] = np.nan
    return df

def get_corr_timeline(df, feature, lag=10):
    '''
    Estimate a feature's time line of cross-correlation between assets
    :param df: input dataframe
    :param feature: feature of interest
    :param lag: timeframe within the cross-correlation is estimated
    :return: timeline of cross-correlation for a feature
    '''
    df_corr = pd.DataFrame(data=[[np.nan for j in range(len(pd.unique(df['Asset_Name'])))]
                                 for i in range(len(pd.unique(df['timestamp'])))],
                           columns=pd.unique(df['Asset_Name']))
    for asset in pd.unique(df['Asset_Name']):
        df_corr[asset] = df[df['Asset_Name']==asset][feature].to_numpy()
    df_corr.reset_index(drop=True)
    corr_timeline = df_corr.groupby(df_corr.index // lag).corr()
    return corr_timeline


def log_return(series, periods=1):
    # Calculate the log(Return) as: R(t) = log(P(t+16)) - log(P(t+1))
    # return np.roll(np.log(series).diff(periods=periods), shift=-(periods+1))
    return np.append(arr=np.roll(np.log(series).diff(periods=periods), shift=-(periods+1))[:-periods],
                     values=np.ones(periods)*np.nan)


def asset_target(asset_logR, market_logR, roll_lag=3750):

    term1 = asset_logR['logR'].to_numpy() * market_logR['logR'].to_numpy()
    term2 = market_logR['logR'].to_numpy() ** 2

    df_roll = pd.DataFrame(data=np.c_[term1, term2], columns=['term1', 'term2'])

    beta = df_roll['term1'].rolling(window=roll_lag).mean() / df_roll['term2'].rolling(window=roll_lag).mean()

    target = asset_logR['logR'].to_numpy() - beta * market_logR['logR'].to_numpy()

    return target.to_numpy()


def normalize_data(df):
    '''
    Normalize data
    :param df: input dataframe
    :return: normalize data
    '''
    asset_mean = df.groupby('Asset_Name').mean()
    asset_std = df.groupby('Asset_Name').std()
    asset_min = df.groupby('Asset_Name').min()
    asset_max = df.groupby('Asset_Name').max()
    asset_spread = asset_max - asset_min
    df_norm, df_stand = copy.deepcopy(df), copy.deepcopy(df)
    for asset in pd.unique(df['Asset_Name']):
        for feature in df.columns:
            # if feature in ['Asset_ID', 'Asset_Name']:
            if feature in ['timestamp', 'Asset_ID', 'Asset_Name']:
                continue
            df_stand[feature] = np.where((df_stand['Asset_Name']==asset),
                                         (df_stand[feature] - asset_mean.loc[asset, feature]) / asset_std.loc[asset, feature],
                                          df_stand[feature])
            df_norm[feature] = np.where((df_norm['Asset_Name']==asset),
                                        (df_norm[feature] - asset_min.loc[asset, feature]) / asset_spread.loc[asset, feature],
                                         df_norm[feature])
    return df_stand, df_norm


def df_gradient(df, step=1):
    columns = [col for col in df.columns if col not in ['timestamp', 'Asset_ID', 'Asset_Name']]
    df_grad = pd.DataFrame(data=[], columns=df.columns)
    for asset in pd.unique(df['Asset_ID']):
        df_grad_asset = df[df['Asset_ID'] == asset].reset_index(drop=True)
        # Rate /second
        # df_grad_asset[columns] = df_grad_asset[columns].diff(periods=step).divide(df_grad_asset['timestamp'].diff(periods=step), axis='index')
        #Rate /min
        df_grad_asset[columns] = df_grad_asset[columns].diff(periods=step) / step
        df_grad = df_grad.append(df_grad_asset, ignore_index=True)
    return df_grad.reset_index(drop=True)

def df_rolling_mean(df, window):
    columns = [col for col in df.columns if col not in ['timestamp', 'Asset_ID', 'Asset_Name']]
    df_roll = pd.DataFrame(data=[], columns=df.columns)
    for asset in pd.unique(df['Asset_ID']):
        df_roll_asset = df[df['Asset_ID'] == asset].reset_index(drop=True)
        df_roll_asset[columns] = df_roll_asset[columns].rolling(window=window).mean()
        df_roll = df_roll.append(df_roll_asset, ignore_index=True)
    return df_roll.reset_index(drop=True)

def plot_autocorr(df, column, lags=None):
    '''
    Plots autocorrelation --> TO BE CHANGED
    :param df:
    :param column:
    :param lags:
    :return:nothing
    '''
    if lags == None:
        plot_acf(x=df[column].to_numpy())
    else:
        plot_acf(x=df[column].to_numpy(), lags=lags)


def plot_timeline(df, feature='Close', assets=None, highlight_market=True):
    '''
    Plot timeline of selected feature for selected assets
    :param df: input dataframe
    :param feature: feature to plot
    :param assets: assets to plot
    :param highlight_market: should the 'market' asset stand out?
    :return:nothing
    '''

    if assets == None:
        assets = list(pd.unique(df['Asset_Name']))
    sns.set(style='whitegrid', font_scale=1)
    fig = plt.figure()
    if highlight_market:
        if 'Market' in assets:
            assets.remove('Market')
        sns.lineplot(data=df[df['Asset_Name'].isin(assets)][['Asset_Name', 'timestamp', feature]],
                     x='timestamp',
                     y=feature,
                     hue='Asset_Name',
                     palette='colorblind')
        sns.lineplot(data=df[df['Asset_Name']=='Market'][['Asset_Name', 'timestamp', feature]],
                     x='timestamp',
                     y=feature,
                     hue='Asset_Name',
                     palette='bright', color='r', linewidth=2, zorder=11111)
    else:
        sns.lineplot(data=df[df['Asset_Name'].isin(assets)][['Asset_Name', 'timestamp', feature]],
                     x='timestamp',
                     y=feature,
                     hue='Asset_Name',
                     palette='colorblind')


def plot_corr_timeline(df, feature, assets=[], lag=10):
    '''
    Plot a feature's timeline of cross-correlation between assets
    :param df: input dataframe
    :param feature: feature of interest
    :param assets: assets to estimate and plot the cross-correlation
    :param lag: timeframe where the cross-correlation is estimated
    :return: nothing
    '''
    if not assets:
        assets = list(pd.unique(df['Asset_Name']))
    corr_timeline = get_corr_timeline(df=df[df['Asset_Name'].isin(assets)], feature=feature, lag=lag)
    sns.set(style='whitegrid', font_scale=1, palette='colorblind')
    fig = plt.figure()
    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets[i+1:]):
            plt.plot(corr_timeline.loc[:, asset1].loc[:, asset2], label=asset1+'-'+asset2)
    plt.legend()

#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    path = '..\\Data'

    # train_filename = 'train'
    train_filename = 'train_short'
    # train_filename = 'train_medium'


    df = parse_csv(path=path, filename=train_filename, imp_strategy=None)
    # df = parse_csv(path=path, filename=train_filename, imp_strategy='median')
    # df = parse_csv(path=path, filename=train_filename, imp_strategy='mean')
    # df = parse_csv(path=path, filename=train_filename, imp_strategy='most_frequent')
    # df = parse_csv(path=path, filename=train_filename, interp_method='linear')
    # df = parse_csv(path=path, filename=train_filename, interp_method='pad')


    df = add_Market(df=df, path=path)

    # df = add_Market_timeweight(df=df, weight_feat='Volume')

    df_grad = df_gradient(df=df, step=15)

    df_roll = df_rolling_mean(df=df, window=100)

    # df_stand, df_norm = normalize_data(df)

    plot_timeline(df=df, feature='logR')
    # plot_timeline(df=df_roll, feature='logR')
    plot_timeline(df=df_rolling_mean(df=df, window=10), feature='logR')
    plot_timeline(df=df_rolling_mean(df=df, window=100), feature='logR')
    plot_timeline(df=df_rolling_mean(df=df, window=500), feature='logR')

    # plot_timeline(df=df_grad, feature='Close')



    # plot_corr_timeline(df=df, feature='Close', assets=['Bitcoin', 'Ethereum'], lag=10)




    # df['rixardos'] = np.nan
    # for asset in pd.unique(df['Asset_Name']):
    #     df.loc[df['Asset_Name']==asset, 'rixardos'] = asset_target(asset_logR=df[df['Asset_Name']==asset][['logR']],
    #                                                                market_logR=df[df['Asset_Name']=='Market'][['logR']],
    #                                                                roll_lag=3750)
    # sns.pairplot(data=df[df['Asset_Name']!='Market'], x_vars='Target', y_vars='rixardos', hue='Asset_Name')
