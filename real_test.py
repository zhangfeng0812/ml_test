import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="策略资产展示（最近五天）", layout="wide")
st.title("📈 策略资产对比分析（最近五个交易日）")

# 数据目录
DATA_ROOT = Path("./platform")

# 获取策略文件夹
strategies = [f.name for f in DATA_ROOT.iterdir() if f.is_dir() and not f.name.startswith(".")]

# 策略选择
selected_strategies = st.multiselect("选择策略（可多选）", strategies, default=strategies)

# 存储每个策略合并后的资产数据
strategy_total_assets = {}

# 每个策略单独合并品种资产
for strategy in selected_strategies:
    asset_path = DATA_ROOT / strategy / "asset"
    asset_files = sorted(asset_path.glob("*.log"))
    symbol_dfs = []

    for f in asset_files:
        df = pd.read_csv(f, header=None, names=["date", "asset"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        symbol_dfs.append(df)

    if symbol_dfs:
        # 合并所有品种
        strategy_df_sum = symbol_dfs[0].copy()
        for df in symbol_dfs[1:]:
            strategy_df_sum = strategy_df_sum.add(df, fill_value=0)

        # 只保留最近5个交易日
        strategy_df_sum = strategy_df_sum.tail(5)

        strategy_total_assets[strategy] = strategy_df_sum

# 标准化资产曲线图（更适合比较走势）
if strategy_total_assets:
    st.subheader("📈 标准化资产对比图（起点=1）")

    norm_df = pd.DataFrame()
    for strategy, df in strategy_total_assets.items():
        norm_series = df["asset"] / df["asset"].iloc[0]  # 标准化
        norm_df[strategy] = norm_series

    norm_df.index.name = "date"
    st.line_chart(norm_df)

# 可选：原始资产曲线（使用Plotly增强展示）
if strategy_total_assets:
    st.subheader("📉 原始资产折线图（Plotly）")

    raw_df = pd.DataFrame()
    for strategy, df in strategy_total_assets.items():
        raw_df[strategy] = df["asset"]
    raw_df["date"] = list(strategy_total_assets.values())[0].index
    raw_df = raw_df.set_index("date")

    fig = px.line(raw_df, x=raw_df.index, y=raw_df.columns, title="策略原始资产（最近五天）")
    fig.update_layout(xaxis_title="日期", yaxis_title="资产", legend_title="策略")
    fig.update_yaxes(tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)
st.markdown("---")
st.header("🔍 查询策略交易记录")

strategy_for_trades = st.selectbox("选择一个策略查看交易记录", strategies)

if strategy_for_trades:
    transaction_path = DATA_ROOT / strategy_for_trades / "transaction"
    transaction_files = sorted(transaction_path.glob("*.log"))
    all_trades = []

    for f in transaction_files:
        symbol = f.stem
        try:
            df = pd.read_csv(f, header=None, names=["time", "operation"], encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(f, header=None, names=["time", "operation"], encoding="gbk")  # 或 'latin1'
        df["symbol"] = symbol
        df["time"] = pd.to_datetime(df["time"])
        all_trades.append(df)
        print()

    if all_trades:
        trade_df = pd.concat(all_trades).sort_values("time").reset_index(drop=True)
        st.dataframe(trade_df, use_container_width=True)
    else:
        st.info("该策略没有找到任何交易记录文件。")
