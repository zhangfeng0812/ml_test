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
data_dict = {}
for name, path in files.items():
    with open(path, "rb") as f:
        data = pickle.load(f)
        if isinstance(data, pd.Series):
            data = data.to_frame(name)
        elif isinstance(data, pd.DataFrame) and data.shape[1] == 1:
            data.columns = [name]
        data_dict[name] = data

# 合并所有策略每日收益率
df_daily_return = pd.concat(data_dict.values(), axis=1)
df_daily_return["l_2_18_10"]=df_daily_return["l_2_18_10"]*0.1
df_daily_return["Shock5"]=df_daily_return["Shock5"]*10
# 计算累计收益率（cumsum）
df_cum_return = df_daily_return.cumsum()

# 是否展示汇总曲线
show_combined = st.checkbox("➕ 显示策略汇总曲线（多策略组合）", value=True)

# 如果勾选了汇总按钮，计算汇总策略的收益
if show_combined:
    combined_daily = df_daily_return.sum(axis=1)
    df_daily_return["Combined"] = combined_daily
    df_cum_return["Combined"] = combined_daily.cumsum()

# 选择展示类型
option = st.radio("选择展示类型", ("每日收益率", "累计收益率"))

if st.checkbox("📋 显示原始数据"):
    if option == "每日收益率":
        st.dataframe(df_daily_return.tail())
    else:
        st.dataframe(df_cum_return.tail())

# 绘图
st.subheader("📊 策略收益曲线")
fig, ax = plt.subplots(figsize=(12, 6))

if option == "每日收益率":
    df_daily_return.plot(ax=ax)
    ax.set_title("每日收益率")
    ax.set_ylabel("收益率")
else:
    df_cum_return.plot(ax=ax)
    ax.set_title("累计收益率（Cumulative Return）")
    ax.set_ylabel("累计收益")

ax.set_xlabel("时间")
ax.grid(True)
st.pyplot(fig)