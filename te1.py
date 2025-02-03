import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tabulate import tabulate

plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号


def read_order(file):
    df = pd.read_csv(file, sep=' \s+', encoding='gb2312', engine='python')
    df_close = df.loc[df['类型'].str.contains('平')].dropna(axis=1)
    if len(df_close.columns) == 10:
        df_close = df_close.drop(columns=['序列'])
    df_close.columns = ['timestamp', 'product', 'order_type', 'close_price', 'share', 'pnl', 'pnl_ratio', 'property',
                        'max_drop']
    df_close.timestamp = pd.to_datetime(df_close.timestamp)
    #         # 强平导致并非所有close数据均存在open数据
    df_close['open_price'] = df_close.close_price.map(
        lambda x: float(x.split('/')[1] if len(x.split('/')) == 2 else x.split('/')[0]))
    df_close.close_price = df_close.close_price.map(lambda x: float(x.split('/')[0]))
    df['share'] = 1
    df_close['buy_sell'] = df_close.order_type.map(lambda x: 'S' if '空' in x else 'B')
    df_close['open_close'] = 'C'
    for col in ['pnl', 'property']:
        df_close[col] = df_close[col].map(
            lambda x: float(x.replace(',', '').replace('万', '')) * 10000 if '万' in x else float(x.replace(',', '')))
    df_close['trade_day'] = df_close.timestamp.dt.normalize()

    return df_close


def get_source_strategy(order_path, start_time, end_time):
    files = os.listdir(order_path)
    df_all = []
    for file in files:
        df = read_order(os.path.join(order_path, file))
        df = df.loc[(df.trade_day >= start_time) & (df.trade_day < end_time)]
        df['strategy'] = file.split('.')[0]
        df['trade_month'] = df.trade_day.dt.to_period('M')
        df_all.append(df)
    return pd.concat(df_all)


def get_sharpe_ratio(b):
    return b.mean() / b.std() * np.sqrt(250)


def return_stats(df, flag):
    day_return = df.groupby('trade_day').pnl.sum()
    dic = dict()
    # 总收益
    dic['total_return'] = day_return.sum()
    # 最大回撤
    dic['max_drawdown'] = np.max(np.maximum.accumulate(day_return.cumsum()) - day_return.cumsum())
    # 时间段
    dic['trade_day'] = list(day_return.index)
    # 回撤范围
    dic['range_drawdown'] = -1 * (np.maximum.accumulate(day_return.cumsum()) - day_return.cumsum()).values
    # 计算胜率
    dic['win_rate'] = df.loc[df.pnl > 0].shape[0] / df.shape[0]
    # 年化收益
    dic['annual_return'] = day_return.mean() * 250
    # 年化夏普比率
    dic['sharpe_ratio'] = get_sharpe_ratio(day_return)
    # 年化Calmar比率
    dic['calmar_ratio'] = dic['annual_return'] / dic['max_drawdown']
    # 盈亏比
    dic['profit_loss_ratio'] = df.loc[df.pnl > 0].pnl.mean() / df.loc[df.pnl < 0].pnl.abs().mean()
    # 交易次数
    dic['trade_count'] = df.shape[0]
    # 平均收益
    dic['average_return'] = df.loc[df.pnl > 0].pnl.mean()
    # 平均亏损
    dic['average_loss'] = df.loc[df.pnl < 0].pnl.mean()
    # 平均品种数量
    dic['average_product'] = df[['trade_month', 'product', 'strategy']].drop_duplicates().groupby(
        'trade_month').product.count().mean()
    # 实际交易品类
    dic['product_unique'] = df[['trade_month', 'product']].drop_duplicates().groupby(
        'trade_month').product.count().mean()
    # 类型
    dic['flag'] = flag
    return dic


# 新增函数：计算滚动夏普比率
def get_cumulative_sharpe(returns):
    cumulative_sharpe = []
    for i in range(1, len(returns) + 1):
        window = returns.iloc[:i]
        if window.std() == 0:
            sharpe = 0
        else:
            sharpe = (window.mean() / window.std()) * np.sqrt(250)
        cumulative_sharpe.append(sharpe)
    return pd.Series(cumulative_sharpe, index=returns.index)


def get_rolling_sharpe(returns, window=60):
    """计算滚动夏普比率"""

    def _calc_sharpe(x):
        if len(x) < 2 or x.std() == 0:
            return 0.0
        return (x.mean() / x.std()) * np.sqrt(250)

    return returns.rolling(window=window, min_periods=1).apply(_calc_sharpe, raw=False)


def get_yearly_stats(df):
    """计算年度统计指标"""
    yearly_stats = []

    # 按年份分组
    for year, df_year in df.groupby(df['trade_day'].dt.year):
        # 计算每日收益
        daily_return = df_year.groupby('trade_day').pnl.sum()
        if len(daily_return) == 0:
            continue

        # 年化收益率（总收益）
        # 假设本金为20w
        cash = 200000
        total_return = daily_return.sum()/cash

        # 年化夏普比率（考虑交易日天数）
        sharpe = (daily_return.mean() / daily_return.std()) * np.sqrt(252) if daily_return.std() != 0 else 0

        # 最大回撤（日内回撤）
        cumulative_returns = daily_return.cumsum()/cash
        max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

        yearly_stats.append({
            '年份': year,
            '年化收益率': total_return,
            '年化夏普比率': sharpe,
            '最大回撤率': max_drawdown
        })

    return pd.DataFrame(yearly_stats).set_index('年份')

def show_result(trader_dir, start_day, end_day, detail=False,rolling_window=120):
    df_source = get_source_strategy(trader_dir, start_day, end_day)
    all_stats_result = []
    df_plot = df_source.groupby(['strategy', 'trade_day']).pnl.sum().unstack(level=0).fillna(0)
    # ================= 新增年度统计 =================
    yearly_stats_df = get_yearly_stats(df_source)

    if detail:
        print("\n年度绩效统计：")
        display_df = yearly_stats_df.copy()

        # 格式化数字显示
        display_df['年化收益率'] = display_df['年化收益率'].map('{:,.2f}%'.format)
        display_df['年化夏普比率'] = display_df['年化夏普比率'].map('{:.2f}'.format)
        display_df['最大回撤率'] = display_df['最大回撤率'].map('{:,.2f}%'.format)

        # 打印表格
        print(tabulate(display_df.reset_index(), headers='keys', tablefmt='psql', numalign='right'))
        # 绘制累计收益
        df_plot.cumsum().plot(alpha=0.5, legend=True)
        split_date = '2023-01-01'
        colors = {'in_sample': '#1f77b4', 'out_sample': '#ff7f0e'}  # 蓝色和橘黄
        plt.title("PnL")
        plt.show()
        plt.figure(figsize=(12, 6))
        # 新增部分：计算并绘制累计夏普比率
        for strategy in df_plot.columns:
            cumulative_sharpe = get_cumulative_sharpe(df_plot[strategy])[120:]
            # 划分样本内外
            mask_in = cumulative_sharpe.index <= split_date
            mask_out = cumulative_sharpe.index > split_date
            # 计算每个策略的累计夏普比率
            plt.plot(cumulative_sharpe[mask_in].index,
                     cumulative_sharpe[mask_in],
                     color=colors['in_sample'],
                     linestyle='-',
                     linewidth=1.5,
                     alpha=0.7)

            plt.plot(cumulative_sharpe[mask_out].index,
                     cumulative_sharpe[mask_out],
                     color=colors['out_sample'],
                     linestyle='-',
                     linewidth=1.5,
                     alpha=0.7,
                     label=f'{strategy} (样本外)')

        for strategy in df_plot.columns:
            rolling_sharpe = get_rolling_sharpe(df_plot[strategy], rolling_window)[120:]
            mask_in_roll = rolling_sharpe.index <= split_date
            mask_out_roll = rolling_sharpe.index > split_date

            plt.plot(rolling_sharpe[mask_in_roll].index,
                     rolling_sharpe[mask_in_roll],
                     color=colors['in_sample'],
                     linestyle='--',
                     linewidth=1.2)

            plt.plot(rolling_sharpe[mask_out_roll].index,
                     rolling_sharpe[mask_out_roll],
                     color=colors['out_sample'],
                     linestyle='--',
                     linewidth=1.2,
                     label=f'{strategy} (滚动样本外)')
        plt.title(f'夏普比率对比（累计 vs {rolling_window}日滚动窗口）')
        plt.xlabel('日期')
        plt.ylabel('夏普比率')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    # 原有统计结果计算
    if detail:
        for strategy in df_source.strategy.unique():
            all_stats_result.append(return_stats(df_source.loc[df_source.strategy == strategy], strategy))
    all_stats_result.append(return_stats(df_source, 'total'))

    all_stats_result_df = pd.DataFrame(all_stats_result)
    return all_stats_result_df


source_order = r"./comb2"
start_day = '2017-01-01'
end_day = '2025-01-02'
result = show_result(source_order, start_day, end_day, detail=True)
print(result)
