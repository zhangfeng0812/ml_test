import pandas as pd
import glob
import streamlit as st
import altair as alt
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者你系统中的其他中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
def analyze_loss_events(daily_returns, threshold=-0.02):
    loss_events = daily_returns < threshold
    loss_frequency = loss_events.sum()

    if loss_frequency == 0:
        return 0, pd.Series()

    # 计算连续亏损分布
    streak_changes = (loss_events.astype(int).diff() == 1)
    streak_ids = streak_changes.cumsum()[loss_events]
    streak_lengths = streak_ids.value_counts().value_counts().sort_index()

    total_streaks = streak_lengths.sum()
    return loss_frequency, streak_lengths / total_streaks


def track_trading_curve(daily_returns, loss_threshold=-0.02,
                        profit_threshold=0.05, max_holding_days=10):
    shifted_returns = daily_returns.shift(1)
    position = False
    curve = []
    current_return = 0
    holding_days = 0

    for date, current_ret in daily_returns.items():
        prev_ret = shifted_returns.get(date, np.nan)

        # 离场条件判断
        if position:
            current_return += current_ret
            holding_days += 1

            if current_return >= profit_threshold or holding_days >= max_holding_days:
                position = False
                current_return = 0
                holding_days = 0

        # 入场条件判断
        if not position and not pd.isna(prev_ret):
            if prev_ret < loss_threshold:
                position = True
                current_return = current_ret  # 计入当日收益
                holding_days = 1

        curve.append(current_return if position else np.nan)

    return pd.Series(curve, index=daily_returns.index)


def plot_daily_returns(daily_returns, trading_curve):
    plt.figure(figsize=(14, 10))

    # 上方子图：原始策略的日收益曲线
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(daily_returns.index, daily_returns.cumsum(), label='Daily Returns', color='tab:blue')
    ax1.set_title("backtest")
    ax1.set_ylabel("accumulated profit")
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # 下方子图：交易盈利情况的折线图
    ax2 = plt.subplot(2, 1, 2)
    clean_curve = trading_curve.dropna()
    ax2.plot(clean_curve.index, clean_curve.cumsum(), label='Trading Returns', color='tab:red')
    ax2.set_title("trading curve")
    ax2.set_ylabel("accumulated profit")
    ax2.set_xlabel("Date")
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.tight_layout()
    st.pyplot(plt)


def calculate_trade_stats(trading_curve):
    clean_curve = trading_curve.dropna()
    changes = clean_curve.diff().fillna(0)

    entries = changes[changes > 0].index
    exits = changes[changes < 0].index

    # 对齐交易记录
    min_len = min(len(entries), len(exits))
    entries = entries[:min_len]
    exits = exits[:min_len]

    if min_len == 0:
        return 0, 0, 0

    returns = [trading_curve.loc[e] for e in exits]
    returns = pd.Series(returns)

    win_rate = returns.gt(0).mean()
    sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
    return sharpe, len(returns), win_rate
file = st.selectbox('选择对应的文件:',os.listdir("pickle/"), key='global_window')
LOSS_THRESHOLD= st.number_input("下跌阈值-- 1/100000", 0, 100, 1)/-100000
PROFIT_TARGET = st.number_input("上涨比率-- 1/10000", 0, 100, 1)/10000
MAX_HOLDING = st.number_input("最多持仓天数-- 1/10000", 0, 100, 1)
daily_returns = pickle.load(open('pickle/'+file, 'rb'))
if daily_returns is not None:
    # 亏损分析
    loss_count, loss_dist = analyze_loss_events(daily_returns, LOSS_THRESHOLD)
    st.write(f"大亏事件: {loss_count}次")
    st.write("连续亏损分布:\n", loss_dist)

    # 生成交易曲线
    trading_curve = track_trading_curve(
        daily_returns,
        loss_threshold=LOSS_THRESHOLD,
        profit_threshold=PROFIT_TARGET,
        max_holding_days=MAX_HOLDING
    )

    # 计算绩效指标
    sharpe, trades, win_rate = calculate_trade_stats(trading_curve)
    st.write(f"\n策略绩效:")
    st.write(f"夏普比率: {sharpe:.2f}")
    # st.write(f"交易次数: {trades}")
    st.write(f"胜率: {win_rate * 100:.1f}%")

    # 可视化结果
    plot_daily_returns(daily_returns, trading_curve)
