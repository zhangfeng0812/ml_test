import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# 设置页面标题
st.set_page_config(page_title="曲线展示", layout="wide")
st.title("📈 多策略曲线对比")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 文件路径
files = {
    "BOLL1": "./real/pkl/BOLL1.pkl",
    "Shock4": "./real/pkl/Shock4.pkl",
    "Shock5": "./real/pkl/Shock5.pkl",
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

# 合并所有策略收益率
df_daily_return = pd.concat(data_dict.values(), axis=1)

# 计算累计收益率（cumsum）
df_cum_return = df_daily_return.cumsum()

# 切换展示内容
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