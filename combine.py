import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="策略收益展示", layout="wide")
st.title("📈 策略每日收益 & 累计收益展示（Plotly）")

# 文件路径
files = {
    "BOLL1": "./real/pkl/BOLL1.pkl",
    "Shock4": "./real/pkl/Shock4.pkl",
    "Shock5": "./real/pkl/Shock5.pkl",
    "l_2_10_10": "./real/pkl/L_2_10_10.pkl",
    "l_2_18_10": "./real/pkl/L_2_18_10.pkl",
}

data_dict = {}
for name, path in files.items():
    with open(path, "rb") as f:
        data = pickle.load(f)
        if isinstance(data, pd.Series):
            data = data.to_frame(name)
        elif isinstance(data, pd.DataFrame) and data.shape[1] == 1:
            data.columns = [name]
        data_dict[name] = data

# 合并策略每日收益率
df_daily_return = pd.concat(data_dict.values(), axis=1)

# 放大/缩小部分策略
df_daily_return["l_2_18_10"] *= 0.1
df_daily_return["Shock5"] *= 10

# 确保 index 为 datetime 类型
df_daily_return.index = pd.to_datetime(df_daily_return.index)

# 累计收益率
df_cum_return = df_daily_return.cumsum()

# 是否展示组合策略
show_combined = st.checkbox("➕ 显示策略汇总曲线（多策略组合）", value=True)
if show_combined:
    df_daily_return["Combined"] = df_daily_return.sum(axis=1)
    df_cum_return["Combined"] = df_daily_return["Combined"].cumsum()

# 展示类型选择
option = st.radio("选择展示类型", ("每日收益率", "累计收益率"))
df_plot = df_daily_return if option == "每日收益率" else df_cum_return

# 显示原始数据（可选）
if st.checkbox("📋 显示原始数据"):
    st.dataframe(df_plot.tail())

# Plotly 绘图
st.subheader("📊 策略收益曲线图（Plotly 交互图）")

fig = go.Figure()
for col in df_plot.columns:
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot[col],
        mode="lines",
        name=col,
        hovertemplate="时间: %{x}<br>收益: %{y:.5f}<extra>" + col + "</extra>"
    ))

fig.update_layout(
    title=f"{option}（可交互）",
    xaxis_title="时间",
    yaxis_title="收益",
    height=600,
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
