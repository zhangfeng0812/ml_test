import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="策略资产与交易展示", layout="wide")
st.title("📈 策略资产展示（按策略汇总）")
st.title("张峰")

DATA_ROOT = Path("./platform")

# 获取所有策略文件夹
strategies = [f.name for f in DATA_ROOT.iterdir() if f.is_dir() and not f.name.startswith(".")]

# 选择策略
selected_strategies = st.multiselect("选择策略（可多选）", strategies, default=strategies)

# 所有策略的总资产数据汇总
strategy_total_assets = {}

for strategy in selected_strategies:
    st.header(f"📊 策略：{strategy}（总资产）")

    asset_path = DATA_ROOT / strategy / "asset"
    asset_files = sorted(asset_path.glob("*.log"))
    strategy_df_sum = pd.DataFrame()

    for f in asset_files:
        df = pd.read_csv(f, header=None, names=["date", "asset"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        if strategy_df_sum.empty:
            strategy_df_sum = df.copy()
        else:
            strategy_df_sum = strategy_df_sum.add(df, fill_value=0)

    if not strategy_df_sum.empty:
        st.line_chart(strategy_df_sum)
        strategy_total_assets[strategy] = strategy_df_sum

# 所有策略合计
if strategy_total_assets:
    st.header("📈 所选策略资产合计（Total）")
    total_df = pd.DataFrame()
    for k, df in strategy_total_assets.items():
        total_df[k] = df["asset"]
    total_df["Total"] = total_df.sum(axis=1)
    st.line_chart(total_df["Total"])
