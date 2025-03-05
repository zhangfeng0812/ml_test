import pandas as pd
import glob
import streamlit as st
import altair as alt
# 设置包含CSV文件的文件夹路径
folder_path = 'transaction/*.csv'  # 替换为你的实际路径

# 获取所有CSV文件列表
file_list = glob.glob(folder_path)
print(file_list)
# 读取并合并CSV文件
df = pd.concat(
    (pd.read_csv(file, parse_dates=['exit_time']) for file in file_list),
    ignore_index=True
)

# 按exit_time排序
df = df.sort_values('exit_time').reset_index(drop=True)
total_win_rate = (df['pnl'] > 0).mean()

# 计算最近100次交易的滚动胜率
st.write("策略demo")
selection = st.selectbox('选择滚动周期数:', options=[50,100,200,500,1000])
rolling = (df['pnl']
                              .gt(0)
                              .rolling(selection, min_periods=1)
                              .mean()
                             )
data = rolling.to_frame()
data.columns = ["rolling"]
data['x'] = data.index
lines = (
        alt.Chart(data, width=800, height=500)
        .mark_line(color='#ff7f0e')
        .encode(x="x", y="rolling")
    )
st.altair_chart(lines)