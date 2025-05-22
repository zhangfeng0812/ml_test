import streamlit as st
import pandas as pd
import altair as alt
import pickle

st.set_page_config(page_title="策略收益展示", layout="wide")
st.title("📈 策略每日收益 & 累计收益展示（Altair）")

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

# 累计收益率
df_cum_return = df_daily_return.cumsum()

# 是否展示组合策略
show_combined = st.checkbox("➕ 显示策略汇总曲线（多策略组合）", value=True)
if show_combined:
    df_daily_return["Combined"] = df_daily_return.sum(axis=1)
    df_cum_return["Combined"] = df_daily_return["Combined"].cumsum()

# 选择展示类型
option = st.radio("选择展示类型", ("每日收益率", "累计收益率"))
df_plot = df_daily_return if option == "每日收益率" else df_cum_return

# 显示原始数据（可选）
if st.checkbox("📋 显示原始数据"):
    st.dataframe(df_plot.tail())

# 将数据转为 Altair 所需格式
df_long = df_plot.copy()
df_long["时间"] = df_long.index
df_long = df_long.reset_index(drop=True)
df_long = df_long.melt(id_vars="时间", var_name="策略", value_name="收益")

# Altair 绘图
st.subheader("📊 策略收益曲线图（Altair交互图）")
chart = alt.Chart(df_long).mark_line().encode(
    x=alt.X("时间:T", title="时间"),
    y=alt.Y("收益:Q", title="收益"),
    color="策略:N",
    tooltip=["时间:T", "策略:N", "收益:Q"]
).properties(
    width=1000,
    height=500,
    title=f"{option}"
).interactive()

st.altair_chart(chart, use_container_width=True)
