import streamlit as st
import pandas as pd
import altair as alt
import pickle
import matplotlib.pyplot as plt
st.set_page_config(page_title="策略收益展示", layout="wide")
st.title("📈 策略每日收益 & 累计收益展示（Altair）")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 文件路径
files = {
    "BOLL1": "./real/pkl/BOLL1.pkl",
    "Shock4": "./real/pkl/Shock4.pkl",
    "Shock5": "./real/pkl/Shock5.pkl",
    "l_2_10_10": "./real/pkl/L_2_10_10.pkl",
    "l_2_18_10": "./real/pkl/L_2_18_10.pkl",
}
# 加载数据
data_dict = {}
for name, path in files.items():
    with open(path, "rb") as f:
        data = pickle.load(f)
        if isinstance(data, pd.Series):
            data = data.to_frame(name)
        elif isinstance(data, pd.DataFrame) and data.shape[1] == 1:
            data.columns = [name]
        data_dict[name] = data

# 合并每日收益率
df_daily_return = pd.concat(data_dict.values(), axis=1)

# 设置索引为 datetime（如果不是）
if not pd.api.types.is_datetime64_any_dtype(df_daily_return.index):
    df_daily_return.index = pd.to_datetime(df_daily_return.index)

# 计算累计收益率
df_cum_return = df_daily_return.cumsum()

# 是否显示组合策略
show_combined = st.checkbox("➕ 显示策略汇总曲线（多策略组合）", value=True)

if show_combined:
    combined_daily = df_daily_return.sum(axis=1)
    df_daily_return["Combined"] = combined_daily
    df_cum_return["Combined"] = combined_daily.cumsum()

# 选择展示内容
option = st.radio("选择展示类型", ("每日收益率", "累计收益率"))

# 选取要展示的数据
df_plot = df_daily_return if option == "每日收益率" else df_cum_return

# 显示原始数据
if st.checkbox("📋 显示原始数据"):
    st.dataframe(df_plot.tail())

# 将数据转换为 Altair 支持的长格式
df_long = df_plot.reset_index().melt(id_vars=df_plot.index.name or "index",
                                     var_name="策略",
                                     value_name="收益")

# 将列名统一为 time/value 以兼容 Altair
df_long.rename(columns={df_plot.index.name or "index": "时间"}, inplace=True)

# Altair 绘图
st.subheader("📊 策略收益曲线图")

chart = alt.Chart(df_long).mark_line().encode(
    x="时间:T",
    y="收益:Q",
    color="策略:N",
    tooltip=["时间:T", "策略:N", "收益:Q"]
).properties(
    width=1000,
    height=500,
    title=f"{option} 曲线"
).interactive()

st.altair_chart(chart, use_container_width=True)