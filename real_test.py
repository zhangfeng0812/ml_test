import streamlit as st
import pandas as pd
import os
from pathlib import Path

st.set_page_config(page_title="策略资产与交易展示", layout="wide")
st.title("📈 策略资产与交易展示")

DATA_ROOT = Path("./platform")

# 获取所有策略文件夹（假设不含 .py 文件）
strategies = [f.name for f in DATA_ROOT.iterdir() if f.is_dir() and not f.name.startswith(".")]

# 选择策略
selected_strategies = st.multiselect("选择策略（可多选）", strategies, default=strategies)

# 存储合并数据
all_assets = {}

# 展示每个策略的资产和交易
for strategy in selected_strategies:
    st.header(f"📊 策略：{strategy}")

    asset_path = DATA_ROOT / strategy / "asset"
    transaction_path = DATA_ROOT / strategy / "transaction"

    asset_files = sorted([f for f in asset_path.glob("*.log")])
    transaction_files = sorted([f for f in transaction_path.glob("*.log")])
    available_symbols = [f.stem for f in asset_files]

    selected_symbols = st.multiselect(f"选择策略 {strategy} 的品种", available_symbols, default=available_symbols,
                                      key=strategy)

    strategy_df_sum = pd.DataFrame()

    for symbol in selected_symbols:
        asset_file = asset_path / f"{symbol}.log"
        if asset_file.exists():
            df = pd.read_csv(asset_file, header=None, names=["date", "asset"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            st.subheader(f"📈 品种：{symbol}（资产变化）")
            st.line_chart(df)

            if strategy_df_sum.empty:
                strategy_df_sum = df.copy()
            else:
                strategy_df_sum = strategy_df_sum.add(df, fill_value=0)

            all_assets[f"{strategy}_{symbol}"] = df

        trans_file = transaction_path / f"{symbol}.log"
        if trans_file.exists():
            df_trans = pd.read_csv(trans_file, header=None, names=["time", "operation"])
            df_trans["time"] = pd.to_datetime(df_trans["time"])
            st.subheader(f"📃 品种：{symbol}（交易记录）")
            st.dataframe(df_trans)

# 全策略资产叠加图
if all_assets:
    st.header("📈 所选策略+品种的资产叠加")
    total_df = pd.DataFrame()
    for k, v in all_assets.items():
        total_df[k] = v["asset"]
    total_df["Total"] = total_df.sum(axis=1)
    st.line_chart(total_df["Total"])
