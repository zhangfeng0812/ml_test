import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# 设置页面标题
st.set_page_config(page_title="曲线展示", layout="wide")
st.title("📈 多策略曲线对比")

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

# 合并数据以便统一绘图
df_all = pd.concat(data_dict.values(), axis=1)

# 显示原始数据（可选）
if st.checkbox("📋 显示原始数据"):
    st.dataframe(df_all.tail())

# 绘图
st.subheader("📊 策略曲线图")
fig, ax = plt.subplots(figsize=(12, 6))
df_all.plot(ax=ax)
ax.set_title("策略收益曲线")
ax.set_xlabel("时间")
ax.set_ylabel("收益")
ax.grid(True)
st.pyplot(fig)
