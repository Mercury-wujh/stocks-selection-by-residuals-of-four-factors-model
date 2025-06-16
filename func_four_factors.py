import pandas as pd
import numpy as np
from tqdm import tqdm

#%% calculate factors
def calc_mkt(input_df, mon_end):
    df = input_df.copy()
    df['date'] = pd.to_datetime(df['date'])  # 需要转化成datetime格式才能计算gap,原df中的类型为object
    df['IPO_date'] = pd.to_datetime(df['IPO_date'])
    df['gap'] = ((df['date'] - df['IPO_date']).dt.days / 30)  # 计算间隔月份
    df['date'] = df['date'].dt.date
    df = df[df['gap'] >= 3]  # 筛选出上市满三个月的股票
    df['flag'] = df['date'].isin(mon_end).astype(int)  # 判断是否是月末
    df['weight'] = (
        (df['flag'] * df['circulating_market_cap']) / df.groupby('date')['circulating_market_cap'].transform('sum')
    )  # 计算月末日期的权重
    df['weight'] = df.replace(0, np.nan).groupby('code')['weight'].transform('ffill')  # 对非月末数据进行向前填充
    df['stock'] = df['return'] * df['weight']
    df['mkt'] = df.groupby('date')['stock'].transform('sum')
    return df


def calc_factor(df, mon_end, rk, type):
    """

    Parameters
    ----------
    df
    mon_end
    rk
    type

    Returns
    -------

    """
    if type == 'mom':
        temp = pd.DataFrame()
        for i in range(len(mon_end) - 1):
            block = df.loc[(df['date'] > mon_end.iloc[i]) & (df['date'] <= mon_end.iloc[i + 1]), :].copy()
            block['return'] = block['return'] + 1  # 收益率是乘性关系, 不能简单相加
            block['cum_return'] = block.groupby('code')['return'].cumprod()
            mon_end_data = block[block['date'].isin(mon_end)]
            temp = pd.concat([temp, mon_end_data], ignore_index=True)
    else:
        temp = df[df['date'].isin(mon_end)].copy()  # 取出月末的数据进行处理

    temp['rank'] = temp.groupby('date')[rk].rank(pct=True)  # 根据指定的指标进行排序
    # 对市值高低两组分别计算
    temp['low_sum'] = temp[temp['rank'] <= 1/3].groupby('date')['circulating_market_cap'].transform('sum')
    temp['high_sum'] = temp[temp['rank'] >= 2/3].groupby('date')['circulating_market_cap'].transform('sum')
    temp['low_weight'] = temp['circulating_market_cap'] / temp['low_sum']
    temp['high_weight'] = temp['circulating_market_cap'] / temp['high_sum']

    df = df.merge(temp[['code', 'date', 'low_weight', 'high_weight']], on=['date', 'code'], how='left')
    df = fill_mon(df, mon_end)
    df['low_return'] = df['low_weight'] * df['return']
    df['high_return'] = df['high_weight'] * df['return']
    df[type] = df.groupby('date')['low_return'].transform('sum') - df.groupby('date')['high_return'].transform('sum')

    return df


def fill_mon(df, mon_end):
    """
    fill nan with the data of the end of the month
    Parameters
    ----------
    df : pd.DataFrame
        merged with temp
    mon_end : ndarray
        the portfolio date / the end date of the month

    Returns
    -------
    df : pd.DataFrame
        df after filling nan by 'ffill'
    """
    for i in range(len(mon_end) - 1):
        mask = (df['date'] >= mon_end.iloc[i]) & (df['date'] < mon_end.iloc[i + 1])
        df.loc[mask, 'low_weight'] = df.loc[mask].groupby('code')['low_weight'].transform('ffill')
        df.loc[mask, 'high_weight'] = df.loc[mask].groupby('code')['high_weight'].transform('ffill')
    # 对最后一个月底之后的日期进行填充
    df.loc[df['date'] >= mon_end.iloc[-1], 'low_weight'] = df.loc[df['date'] >= mon_end.iloc[-1], :].groupby(
        'code')['low_weight'].transform('ffill')
    df.loc[df['date'] >= mon_end.iloc[-1], 'high_weight'] = df.loc[df['date'] >= mon_end.iloc[-1], :].groupby(
        'code')['high_weight'].transform('ffill')
    return df


#%% pre-process
def neutralize(factor_input, factor_style=None, factor_indus=None):
    '''
    NEUTRALIZE因子预处理：中性化

    Parameters
    ----------
    factor_input : (N,) numpy.ndarray
        单截面期因子向量.
    factor_style : (N,M) numpy.ndarray
        单截面期风格因子向量. The default is None.
    factor_indus : (N,) numpy.ndarray
        单截面期行业因子向量. The default is None.

    Returns
    -------
    factor_output : (N,) numpy.ndarray
        中性化后的单截面期因子向量.

    '''
    factor_input = np.reshape(factor_input, (factor_input.shape[0], -1))  # 转化成二维数组, 后续合并需要以矩阵的形式进行

    # 正确检查全0或全NaN的条件
    if np.all(factor_input == 0) or np.all(np.isnan(factor_input)):
        factor_output = factor_input.copy()
    else:
        # 保留原始的中性化因子输入，不覆盖它们！
        factor_output = None  # 或继续后续中性化处理

    # 读取风格因子（如对数市值因子）
    if factor_style is None:
        # shape中的0表示0列矩阵即无实际数据列,(100,0)的二维矩阵,防止后续拼接的时候报错
        factor_style = np.full((factor_input.shape[0], 0), np.nan)
    else:
        factor_style = np.reshape(factor_style, (factor_style.shape[0], -1))
    # 读取行业因子，将行业代码转换为哑变量矩阵
    if factor_indus is None:
        factor_indus_dummy = np.full((len(factor_input), 0), np.nan)
    else:
        factor_indus_dummy = pd.get_dummies(factor_indus).values
    # 拼接风格、行业因子
    factor_neutral = np.concatenate([factor_style, factor_indus_dummy], axis=1)

    # 判断是否存在中性化因子
    if factor_neutral.shape[1] > 0:
        # 如果存在中性化因子
        # 计算待中性化因子与风格、行业因子的相关系数
        tmp_concat = np.concatenate([factor_input, factor_neutral], axis=1)
        tmp_corr = pd.DataFrame(tmp_concat).corr().values[0, 1:]
        # 如果相关系数大于0.9，删除该风格或行业因子
        factor_neutral = factor_neutral[:, np.where(np.abs(tmp_corr) <= 0.9)[0]]
        # 待中性化因子为因变量，风格、行业因子为自变量，进行线性回归
        x = factor_neutral
        x = sm.add_constant(x)
        y = factor_input
        model = sm.OLS(y, x)
        results = model.fit()
        # 取残差为中性化后的因子
        factor_output = results.resid.copy()
    else:
        # 如果不存在中性化因子
        factor_output = factor_input.copy()
    return factor_output
#%% backtest
# IC test
def ic_test(
    df,
    close,
    pred_str,
    horizon=1,
    dropna=False
):
    """
    calc_ic.

    Parameters
    ----------
    df: pd.DataFrame
        initai data, contains 'date', 'code', pred_str
    close: pd.DataFrame
        label
    pred_str: str
        date_col

    Returns
    -------
    (pd.Series, pd.Series)
        ic and rank ic
    """
    label = close.pct_change(periods=horizon, axis=1).shift(periods=-horizon, axis=1)  # 未来horizon日的收益率作为label
    label = label.reset_index().rename(columns={'index': 'code'}).melt(
        id_vars='code', var_name='date', value_name='label')  # 宽数据转化成长数据
    label['date'] = pd.to_datetime(label['date']).dt.date
    df = df.merge(label, on=['code', 'date'], how='left')  # df中的数据是dropna之后的,将label与df进行匹配
    ic = df.groupby("date", group_keys=False)[[pred_str, 'label']].apply(
        lambda x: x[pred_str].corr(x['label'])
    )
    ric = df.groupby("date", group_keys=False)[[pred_str, 'label']].apply(
        lambda x: x[pred_str].corr(x['label'], method='spearman')
    )
    if dropna:
        return ic.dropna(), ric.dropna()
    else:
        return ic, ric


# layered backtesting
def layered_test(
    input_df: pd.DataFrame,
    close: pd.DataFrame,
    pred_str: str,
    quantile: float = 0.1,
    horizon: int = 1
):
    num_groups = int(1 / quantile)
    all_dates = input_df['date'].unique().tolist()  # 生成所有交易日
    all_tran_dates = all_dates[::horizon]  # 调仓日
    if all_tran_dates[-1] != all_dates[-1]:
        all_tran_dates.append(all_dates[-1])

    # 重构pred: 使两个调仓日之间的pred为首日pred,确保仓位在期间保持不变
    pred = input_df.pivot(index='code', columns='date', values=pred_str)
    pred_ = pred.copy()
    for i_date in range(1, len(all_tran_dates)):
        prev_trade_date = all_tran_dates[i_date - 1]
        prev_trade_date_ = all_dates[all_dates.index(prev_trade_date) + 1]
        curr_trade_date = all_tran_dates[i_date]
        num_days = all_dates.index(curr_trade_date) - all_dates.index(prev_trade_date_) + 1
        pred.loc[:, prev_trade_date_:curr_trade_date] = np.tile(
            pred_.loc[:, prev_trade_date].values, (num_days, 1)
        ).T
    pred = pred.reset_index().rename({'index': 'code'}).melt(
        id_vars='code', var_name='date').set_index(['date', 'code']).iloc[:, 0]  # 转化成长数据进行后续处理

    # 重构label
    all_ret = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for i_date in range(1, len(all_tran_dates)):
        prev_trade_date = all_tran_dates[i_date - 1]
        curr_trade_date = all_tran_dates[i_date]
        curr_price = close.loc[:, prev_trade_date:curr_trade_date]
        curr_ret = curr_price.iloc[:, 1:].div(curr_price.iloc[:, 0], axis=0) - 1  # 与每个交易区间的首日即调仓日的价格相比计算收益
        all_ret.loc[:, curr_ret.columns] = curr_ret.values
    all_ret = all_ret.reset_index().rename({'index': 'code'}).melt(
        id_vars='code', var_name='date').set_index(['date', 'code'])  # 转化成长数据进行后续处理
    label = all_ret.loc[pred.index, 'value']

    df = pd.DataFrame({'pred': pred, 'label': label})  # 将pred和label合并,进行后续分层计算每日收益
    group = df.groupby(level='date', group_keys=False)

    quantile_return = pd.DataFrame(
        columns=['layer%d' % i for i in range(1, num_groups + 1)],
        index=all_dates,
        dtype=float
    )  # 创建空df储存分层收益数据
    for i in range(num_groups - 1, -1, -1):
        lb, ub = i * quantile, (i + 1) * quantile
        curr_r = group.apply(
            lambda x: x.loc[(x.pred >= x.pred.quantile(lb)) & (x.pred < x.pred.quantile(ub)), 'label'].mean()
        )
        quantile_return.loc[:, 'layer%d' % (num_groups - i)] = curr_r.copy()

    quantile_return['avg'] = group.apply(lambda x: x.label.mean())  # 对按日期分组后的每组求label的均值

    # 计算相邻两日的收益率
    quantile_return_ = pd.DataFrame(0.0, index=quantile_return.index, columns=quantile_return.columns)
    for i_date in range(1, len(all_tran_dates)):
        prev_trade_date = all_tran_dates[i_date - 1]
        curr_trade_date = all_tran_dates[i_date]
        curr_ret = quantile_return.loc[prev_trade_date:curr_trade_date, :].copy()
        curr_ret.iloc[0, :] = 0
        curr_ret = (curr_ret + 1).pct_change()  # 将curr_ret的每个值加 1,然后计算百分比变化(收益率),近似于相对于前一日的收益率
        quantile_return_.loc[curr_ret.index[1:], :] = curr_ret.iloc[1:, :].values
    quantile_return_.iloc[0, :] = 0

    quantile_value_abs = (1 + quantile_return_).cumprod()  # 计算每个分层的累积收益率, 1+quantile_return_将收益率转换为收益倍数
    
    # 相对收益
    quantile_value_rel = quantile_value_abs.iloc[:, :-1].div(quantile_value_abs.iloc[:, -1], axis=0)

    # 分层超额
    excess_return = quantile_return_.iloc[:, :-1].sub(quantile_return_.iloc[:, -1], axis=0)
    excess_values = (excess_return + 1).cumprod()
    excess_layers = excess_values.dropna().iloc[-1, :]
    excess_layers = pd.DataFrame(excess_layers)
    excess_layers = excess_layers ** (252 / len(excess_return.index)) - 1

    return quantile_value_abs, quantile_value_rel, excess_layers


# 计算多空头收益
def calc_long_short_return(quantile_value_abs):
    quantile_return_abs = quantile_value_abs.pct_change()
    r_long = quantile_return_abs.iloc[:, 0]  # 第一列(layer1)为多头
    r_short = quantile_return_abs.iloc[:, -2]  # 倒数第二列(layer10)为空头
    r_avg = quantile_return_abs.iloc[:, -1]  # 倒数第一列为avg

    return r_long, r_short, (r_long - r_short) / 2, r_avg

