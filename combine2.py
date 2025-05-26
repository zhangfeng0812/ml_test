import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 页面配置
st.set_page_config(page_title="策略收益展示", layout="wide")
st.title("📈 策略每日收益 & 累计收益展示（Plotly）")

# 策略文件路径
files = {
    "BOLL1": "./real/csv/BOLL1.csv",
    "BBI": "./real/csv/TouchBBI1.csv",
    "Shock4": "./real/csv/Shock4.csv",
    "Shock5": "./real/csv/Shock5.csv",
    "l_2_10_10": "./real/csv/L_2_10_10.csv",
    "l_2_18_10": "./real/csv/L_2_18_10.csv",
    # 可继续添加更多策略
}

# 加载数据并合并
merged_df = None

for name, path in files.items():
    df = pd.read_csv(path)
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df.set_index(df.columns[0], inplace=True)

    # 所有列求和，作为该策略收益
    df2 = pd.DataFrame(df.sum(axis=1), columns=[name])

    if merged_df is None:
        merged_df = df2
    else:
        merged_df = merged_df.join(df2, how='outer')

# 补全缺失值并计算总收益
merged_df = merged_df.ffill()
merged_df["Total"] = merged_df.sum(axis=1)

# 重置索引，准备绘图
merged_df.reset_index(inplace=True)
merged_df.rename(columns={merged_df.columns[0]: "Date"}, inplace=True)

# ------------------ Total 单独图 ------------------
st.subheader("🌟 策略组合 Total 每日收益")
fig_total = go.Figure()
fig_total.add_trace(go.Scatter(
    x=merged_df["Date"][60:],
    y=merged_df["Total"][60:],
    mode='lines',
    name="Total",
    line=dict(width=3, color="firebrick")
))
fig_total.update_layout(title="Total 策略每日收益曲线",
                        xaxis_title="日期", yaxis_title="收益",
                        hovermode="x unified", template="plotly_white")
st.plotly_chart(fig_total, use_container_width=True)

# ------------------ 所有策略每日收益图 ------------------
st.subheader("📊 所有策略每日收益")
fig_daily = go.Figure()
for col in merged_df.columns[1:-1]:
    fig_daily.add_trace(go.Scatter(
        x=merged_df["Date"], y=merged_df[col],
        mode='lines', name=col
    ))
fig_daily.update_layout(title="每日收益曲线（所有策略）",
                        xaxis_title="日期", yaxis_title="收益",
                        hovermode="x unified", template="plotly_white")
st.plotly_chart(fig_daily, use_container_width=True)


