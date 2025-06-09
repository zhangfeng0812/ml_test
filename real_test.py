import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="策略资产展示（最近五天）", layout="wide")
st.title("📈 策略资产展示（最近五个交易日）")

DATA_ROOT = Path("./platform")

# 获取所有策略文件夹
strategies = [f.name for f in DATA_ROOT.iterdir() if f.is_dir() and not f.name.startswith(".")]

# 选择策略
selected_strategies = st.multiselect("选择策略（可多选）", strategies, default=strategies)

# 用于汇总所有策略
strategy_total_assets = {}

for strategy in selected_strategies:
    st.header(f"📊 策略：{strategy}（总资产，最近五天）")

    asset_path = DATA_ROOT / strategy / "asset"
    asset_files = sorted(asset_path.glob("*.log"))
    symbol_dfs = []

    for f in asset_files:
        df = pd.read_csv(f, header=None, names=["date", "asset"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        symbol_dfs.append(df)

    # 合并所有品种
    if symbol_dfs:
        strategy_df_sum = symbol_dfs[0].copy()
        for df in symbol_dfs[1:]:
            strategy_df_sum = strategy_df_sum.add(df, fill_value=0)

        # 仅保留最后 5 个交易日
        strategy_df_sum = strategy_df_sum.tail(5)

        st.line_chart(strategy_df_sum)
        strategy_total_assets[strategy] = strategy_df_sum

# 所有策略的总资产展示
if strategy_total_assets:
    st.header("📈 所选策略资产合计（Total，最近五天）")
    total_df = pd.DataFrame()

    for strategy, df in strategy_total_assets.items():
        total_df[strategy] = df["asset"]


    total_df["Total"] = total_df.sum(axis=1)
    st.line_chart(total_df["Total"])
