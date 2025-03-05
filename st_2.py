import pandas as pd
import glob
import os
import streamlit as st
import altair as alt
# 设置包含CSV文件的文件夹路径
folder_path = 'transaction/*.csv'  # 替换为你的实际路径

# 获取所有CSV文件列表
file_list = glob.glob(folder_path)
# 读取并合并CSV文件
df = pd.concat(
    (pd.read_csv(file, parse_dates=['exit_time']) for file in file_list),
    ignore_index=True
)

# 按exit_time排序
df = df.sort_values('exit_time').reset_index(drop=True)
total_win_rate = (df['pnl'] > 0).mean()
total = (
    df['pnl'].gt(0)          # 判断每笔交易是否盈利（True/False）
    .expanding()             # 创建从第一行到当前行的累积窗口
    .mean()                  # 计算累积窗口内胜率的均值（True=1, False=0）
)
# 计算最近100次交易的滚动胜率
st.write("策略全品种曲线")
selection = st.selectbox('选择滚动周期数:', options=[50,100,200,500,1000])
rolling = (df['pnl']
                              .gt(0)
                              .rolling(selection, min_periods=1)
                              .mean()
                             )
data = rolling.to_frame()
data.columns = ["rolling"]
data['x'] = data.index
data2 = total.to_frame()
data2.columns = ["total"]
data2['x'] = data2.index
lines = (
        alt.Chart(data, width=800, height=500)
        .mark_line(color='#ff7f0e')
        .encode(x="x", y="rolling")
    )
lines2 = (
        alt.Chart(data2, width=800, height=500)
        .mark_line(color='#1f77b4')
        .encode(x="x", y="total")
    )
st.altair_chart(lines+lines2)
st.write("策略分品种曲线")
fu = st.selectbox('选择对应的品种:', options=os.listdir('transaction/'))
selection2 = st.selectbox('选择滚动周期数:', options=[10,25,50,100,200])
df3 = pd.read_csv(os.path.join("transaction/",fu))
df3 = df3.sort_values('exit_time').reset_index(drop=True)
total2 = (
    df3['pnl'].gt(0)          # 判断每笔交易是否盈利（True/False）
    .expanding()             # 创建从第一行到当前行的累积窗口
    .mean()                  # 计算累积窗口内胜率的均值（True=1, False=0）
)
# 计算最近100次交易的滚动胜率
rolling2 = (df3['pnl']
                              .gt(0)
                              .rolling(selection2, min_periods=1)
                              .mean()
                             )
data3 = rolling2.to_frame()
data3.columns = ["rolling"]
data3['x'] = data3.index
data4 = total2.to_frame()
data4.columns = ["total"]
data4['x'] = data4.index
lines3 = (
        alt.Chart(data3, width=800, height=500)
        .mark_line(color='#ff7f0e')
        .encode(x="x", y="rolling")
    )
lines4 = (
        alt.Chart(data4, width=800, height=500)
        .mark_line(color='#1f77b4')
        .encode(x="x", y="total")
    )
st.altair_chart(lines3+lines4)