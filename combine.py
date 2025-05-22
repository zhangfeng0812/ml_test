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

# 加载所有策略数据
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

# 比例调整（根据你的需求）
df_daily_return["l_2_18_10"] *= 0.1
df_daily_return["Shock5"] *= 10

# 时间索引格式化
df_daily_return.index = pd.to_datetime(df_daily_return.index)

# 计算累计收益率
df_cum_return = df_daily_return.cumsum()

# 是否添加组合策略
show_combined = st.checkbox("➕ 显示策略汇总曲线（多策略组合）", value=True)
if show_combined:
    df_daily_return["Combined"] = df_daily_return.sum(axis=1)
    df_cum_return["Combined"] = df_daily_return["Combined"].cumsum()

# 选择每日 or 累计收益
option = st.radio("选择展示类型", ("累计收益率", "每日收益率"))
df_plot = df_daily_return if option == "每日收益率" else df_cum_return

# 日期范围过滤
st.subheader("📅 选择展示时间范围")
end_date = df_plot.index.max()
start_date = end_date - pd.DateOffset(months=2)

start, end = st.date_input(
    "请选择日期范围：",
    value=(start_date, end_date),
    min_value=df_plot.index.min(),
    max_value=df_plot.index.max()
)

df_plot = df_plot.loc[(df_plot.index >= pd.to_datetime(start)) & (df_plot.index <= pd.to_datetime(end))]

# 策略多选框
st.subheader("📌 选择要展示的策略")
strategies_available = list(df_plot.columns)
strategies_selected = st.multiselect(
    "请选择策略",
    options=strategies_available,
    default=strategies_available
)

# 根据策略选择过滤数据
df_plot_filtered = df_plot[strategies_selected]

# 显示原始数据
if st.checkbox("📋 显示原始数据"):
    st.dataframe(df_plot_filtered.tail())

# 绘制 Plotly 图
st.subheader("📊 策略收益曲线图（Plotly 交互图）")

fig = go.Figure()
for col in df_plot_filtered.columns:
    fig.add_trace(go.Scatter(
        x=df_plot_filtered.index,
        y=df_plot_filtered[col],
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
