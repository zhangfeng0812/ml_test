import pandas as pd
import glob
import os
import streamlit as st
import altair as alt

# 设置响应式样式（保持你的原样式）
st.markdown("""
<style>
.vega-embed {
    width: 100% !important;
    height: auto !important;
}
canvas {
    touch-action: auto !important;
}
</style>
""", unsafe_allow_html=True)

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

# 新增：定义绘图函数避免重复代码

def create_line_chart(data, y_col, color):
    zoom = alt.selection_interval(
        bind='scales',
        on=["[mousedown, mouseup]", "[touchstart, touchend]"],
        translate=["[mousemove]", "[touchmove]"]
    )

    return alt.Chart(data).mark_line(color=color).encode(
        x='x:Q',
        y=alt.Y(f'{y_col}:Q', scale=alt.Scale(domain=[0, 1]))
    ).add_params(zoom)

# ----------------------------
# 第一部分：全品种曲线
# ----------------------------
st.write("策略全品种曲线")

selection = st.selectbox('选择滚动周期数:', options=[50, 100, 200, 500, 1000])
rolling = df['pnl'].gt(0).rolling(selection, min_periods=1).mean()
data = rolling.rename('rolling').reset_index().rename(columns={'index': 'x'})

total = df['pnl'].gt(0).expanding().mean()
data_total = total.rename('total').reset_index().rename(columns={'index': 'x'})

# 使用新绘图函数
chart_rolling = create_line_chart(data, 'rolling', '#ff7f0e')
chart_total = create_line_chart(data_total, 'total', '#1f77b4')

combined_chart = (chart_rolling + chart_total).resolve_scale(y='shared')
st.altair_chart(combined_chart, use_container_width=True)

# ----------------------------
# 第二部分：分品种曲线
# ----------------------------
st.write("策略分品种曲线")

fu = st.selectbox('选择对应的品种:', options=os.listdir('transaction/'))
selection2 = st.selectbox('选择滚动周期数:', options=[10, 25, 50, 100, 200])

# 数据加载和处理（保持你的原代码）
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

# 使用相同的绘图函数
chart_rolling2 = create_line_chart(data3, 'rolling', '#ff7f0e')
chart_total2 = create_line_chart(data4, 'total', '#1f77b4')

combined_chart2 = (chart_rolling2 + chart_total2).resolve_scale(y='shared')
st.altair_chart(combined_chart2, use_container_width=True)