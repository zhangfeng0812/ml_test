import pandas as pd
import glob
import streamlit as st
# 设置包含CSV文件的文件夹路径
folder_path = r'C:\AlleyOOP\10m\transaction\*.csv'  # 替换为你的实际路径

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

# 计算最近100次交易的滚动胜率
rolling_win_rate = (df['pnl']
                    .iloc[-100:]  # 取最后100条记录
                    .gt(0)        # 转换为布尔值（是否盈利）
                    .mean()       # 计算胜率
                   )

rolling = (df['pnl']
                              .gt(0)
                              .rolling(100, min_periods=1)
                              .mean()
                             )
st.write("test")