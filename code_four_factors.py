import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm

import sys
sys.path.append('D:/Codes/Dataset_Code/Code')  # 替换为func_four_factors上级路径
import func_four_factors as func

#%% read the initial data
data = pd.read_csv("D:/project/data.csv")  # the absolute path of your data
# date range
start_date = "20221231"
end_date = "20250531"

#%% 计算四因子
calc = data.iloc[:, 1:].copy()
calc['date'] = pd.to_datetime(calc['date'])  # 将date列转化成datetime格式
calc = calc[
    (calc['date'] >= pd.to_datetime(start_date)) & (calc['date'] <= pd.to_datetime(end_date))
].reset_index(drop=True)  # 按照设定的起始和终止时间筛选
mon_end = calc.groupby(calc['date'].dt.to_period('M'))['date'].max().reset_index(drop=True)  # 按照年月进行分组并取出每组最大值
calc['date'] = pd.to_datetime(calc['date']).dt.date  # 仅保留date
mon_end = mon_end.dt.date
if calc['date'].iloc[0] not in mon_end.values:  # 如果数据日期不是从月末开始，加入初始日期
    mon_end = pd.concat(
        [mon_end, pd.Series(calc['date'].iloc[0])]
    ).sort_values().reset_index(drop=True)

# 计算股票收益率
calc['return'] = calc.groupby('code')['close_adj'].pct_change()

# 构建市场因子
MKT = func.calc_mkt(calc, mon_end)
calc = calc.merge(MKT[['code', 'date', 'mkt']], on=['code', 'date'], how='left')

# 构建市值因子
SMB = func.calc_factor(
    calc[['code', 'date', 'circulating_market_cap', 'return']], mon_end, rk='circulating_market_cap', type='smb'
)
calc = calc.merge(SMB[['code', 'date', 'smb']], on=['code', 'date'], how='left')

# 构建估值因子
HML = func.calc_factor(calc, mon_end, rk='PB', type='hml')
calc = calc.merge(HML[['code', 'date', 'hml']], on=['code', 'date'], how='left')

# 构建动量因子
MOM = func.calc_factor(calc[['code', 'date', 'circulating_market_cap', 'return']], mon_end, rk='cum_return', type='mom')
calc = calc.merge(MOM[['code', 'date', 'mom']], on=['code', 'date'], how='left')

#%% 回归
reg_data = calc[['code', 'date', 'return', 'market_cap', 'indus_code', 'mkt', 'smb', 'hml', 'mom']]
reg_data = reg_data.set_index(['code', 'date'])
reg_data = reg_data.replace([np.inf, -np.inf], np.nan)
reg_data = reg_data.dropna()
X = reg_data[['mkt', 'smb', 'hml', 'mom']]
X = sm.add_constant(X)
Y = reg_data['return']
model = sm.OLS(Y, X)
results = model.fit()
reg_data['residual'] = results.resid  # 将残差项合并进原数据
print(results.summary())

#%% 因子预处理
def standardize(factor_input):
    '''
    STANDARDIZE因子预处理：标准化
    Parameters
    ----------
    factor_input : (N,) numpy.ndarray
        单截面期因子向量.

    Returns
    -------
    factor_output : (N,) numpy.ndarray
        标准化后的单截面期因子向量.
    '''
    if np.vstack((factor_input == 0, np.isnan(factor_input))).all():
        factor_output = factor_input
    else:
        factor_output = (factor_input - np.nanmean(factor_input)) / np.nanstd(factor_input)
    return factor_output


def neutralize(data_input, pred_str):
    '''
    NEUTRALIZE因子预处理：中性化

    Parameters
    ----------
    factor_input : (N,) numpy.ndarray
        单截面期因子向量.

    Returns
    -------
    factor_output : (N,) numpy.ndarray
        中性化后的单截面期因子向量.
    '''
    # 正确检查全0或全NaN的条件
    if np.all(data_input[pred_str] == 0) or np.all(np.isnan(data_input[pred_str])):
        factor_output = data_input[pred_str].copy()
    else:
        # 保留原始的中性化因子输入，不覆盖它们！
        data_copy = data_input.copy()  # 生成用于中性化的数据

    # 生成行业代码的哑变量(n×n的矩阵,n为行业数量)
    indus_dum = pd.get_dummies(data_copy['indus_code']).values
    # 合并风格因子(此处选择对数市值)和行业因子
    neu = np.concatenate([data_copy['log_mkt_cap'].values.reshape(-1, 1), indus_dum], axis=1)

    # 中性化处理
    # 计算待中性化因子与风格、行业因子的相关系数
    tmp_concat = np.concatenate([data_input[pred_str].values.reshape(-1, 1), neu], axis=1)
    tmp_corr = pd.DataFrame(tmp_concat).corr().values[0, 1:]
    # 如果相关系数大于0.9，删除该风格或行业因子
    neu = neu[:, np.where(np.abs(tmp_corr) <= 0.9)[0]]
    # 待中性化因子为因变量，风格、行业因子为自变量，进行线性回归
    x = neu
    x = sm.add_constant(x)
    y = data_input[pred_str]
    model = sm.OLS(y, x)
    results = model.fit()
    # 取残差为中性化后的因子
    factor_output = results.resid.copy()

    return factor_output


reg_data['residual'] = reg_data.groupby('date')['residual'].transform(
    lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
)  # 1%缩尾
reg_data['log_mkt_cap'] = np.log(reg_data['market_cap'])  # 市值对数化,用于市值中性化
factor = reg_data.groupby('date', group_keys=False).apply(
    lambda x: neutralize(x, pred_str='residual')
)  # 对截面因子进行中性化
factor = factor.groupby('date', group_keys=False).apply(standardize)  # 对截面因子进行标准化处理
factor = factor.reset_index().rename(columns={0: 'residual'})

#%% backtest
horizon = 1  # 调仓周期
close = calc.pivot(index='code', columns='date', values='close_adj')  # 将close列从长数据转化成宽数据,便于后续计算

# IC test
ic, ric = func.ic_test(factor, close, pred_str='residual')

# layered backtest
quantile_value_abs, quantile_value_rel, excess_layers = func.layered_test(
    factor, close, 'residual', 0.1, horizon
)
r_long, r_short, r_hedge, r_avg = func.calc_long_short_return(quantile_value_abs)

# bas=cktest results summary
# cumulative ic
evaluation_cum = pd.DataFrame(columns=['ic_cum', 'ric_cum'])
evaluation_cum['ic_cum'] = ic.iloc[::horizon].cumsum()
evaluation_cum['ric_cum'] = ric.iloc[::horizon].cumsum()
# indicators average
evaluation_avg = pd.Series(dtype=float)
evaluation_avg.loc['ic'] = np.nanmean(ic)
evaluation_avg.loc['rankic'] = np.nanmean(ric)
evaluation_avg.loc['icir'] = np.nanmean(ic) / np.nanstd(ic)
evaluation_avg.loc['rankicir'] = np.nanmean(ric) / np.nanstd(ric)
evaluation_avg.loc['top_return'] = 252 * r_long.mean()
evaluation_avg.loc['bottom_return'] = 252 * r_short.mean()
evaluation_avg.loc['hedge_return'] = 252 * r_hedge.mean()
evaluation_avg.loc['average_return'] = 252 * r_avg.mean()
evaluation_avg.loc['top_sharpe'] = np.sqrt(252) * r_long.mean() / r_long.std()
evaluation_avg.loc['bottom_sharpe'] = np.sqrt(252) * r_short.mean() / r_short.std()
evaluation_avg.loc['hedge_sharpe'] = np.sqrt(252) * (r_long - r_short).mean() / 2 / (r_long - r_short).std()

# save into excel
save_path = 'your_save_path'
writer = pd.ExcelWriter(save_path + 'evaluation_prediction.xlsx')
evaluation_avg.to_excel(writer, sheet_name='evaluation_avg')
evaluation_cum.to_excel(writer, sheet_name='evaluation_cum')
quantile_value_abs.to_excel(writer, sheet_name='quantile_value_abs')
quantile_value_rel.to_excel(writer, sheet_name='quantile_value_rel')
excess_layers.to_excel(writer, sheet_name='excess_layers')
writer.close()


