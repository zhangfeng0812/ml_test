import streamlit as st
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt

# 配置
st.set_page_config(page_title="策略可视化平台", layout="wide")

# 数据目录
DATA_ROOT = Path("./platform")

st.title("📈 策略资产与交易展示")

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

    # 获取所有品种文件名
    asset_files = sorted([f for f in asset_path.glob("*.csv")])
    transaction_files = sorted([f for f in transaction_path.glob("*.csv")])
    available_symbols = [f.stem for f in asset_files]

    selected_symbols = st.multiselect(f"选择策略 {strategy} 的品种", available_symbols, default=available_symbols,
                                      key=strategy)

    # 汇总资产绘图数据
    strategy_df_sum = pd.DataFrame()

    for symbol in selected_symbols:
        asset_file = asset_path / f"{symbol}.csv"
        if asset_file.exists():
            df = pd.read_csv(asset_file, header=None, names=["date", "asset"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df.sort_index()

            # 展示图表
            st.subheader(f"📈 品种：{symbol}（资产变化）")
            st.line_chart(df)

            # 存储叠加用
            if strategy_df_sum.empty:
                strategy_df_sum = df.copy()
            else:
                strategy_df_sum = strategy_df_sum.add(df, fill_value=0)

            # 存储给最终全局合并
            all_assets[f"{strategy}_{symbol}"] = df

        # 展示交易记录
        trans_file = transaction_path / f"{symbol}.csv"
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
