import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="策略收益展示", layout="wide")
st.title("📈 策略每日收益 & 累计收益展示（Plotly）")

# 文件路径（CSV）
files = {
    "BOLL1": "./real/csv/BOLL1.csv",
    "BBI": "./real/csv/TouchBBI1.csv",
    "Shock4": "./real/csv/Shock4.csv",
    "Shock5": "./real/csv/Shock5.csv",
    "l_2_10_10": "./real/csv/L_2_10_10.csv",
    "l_2_18_10": "./real/csv/L_2_18_10.csv",
    # 你可以继续添加其他策略路径
}

# 加载 CSV 数据
data_dict = {}
for name, path in files.items():
    df = pd.read_csv(path)

    # 自动识别日期列并设为索引（假设第一列是日期）
    if "date" in df.columns or "日期" in df.columns:
        date_col = "date" if "date" in df.columns else "日期"
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    else:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)

    # 统一列名
    if df.shape[1] == 1:
        df.columns = [name]
    else:
        df = df.iloc[:, [0]]  # 只取第一列数据作为策略收益
        df.columns = [name]

    data_dict[name] = df

# 合并策略每日收益率
df_daily_return = pd.concat(data_dict.values(), axis=1)



# 时间索引格式化
df_daily_return.index = pd.to_datetime(df_daily_return.index)

# 计算累计收益率
df_cum_return = df_daily_return.cumsum()

# 是否添加组合策略
show_combined = st.checkbox("➕ 显示策略汇总曲线（多策略组合）", value=True)
if show_combined:
    df_daily_return["Combined"] = df_daily_return.sum(axis=1)
    df_cum_return["Combined"] = df_daily_return["Combined"]

# 选择每日 or 累计收益
option = st.radio("选择展示类型", ("累计收益率", "每日收益率"))
df_plot = df_daily_return if option == "每日收益率" else df_cum_return

# 日期范围过滤
st.subheader("📅 选择展示时间范围")



# 策略多选框
st.subheader("📌 选择要展示的策略")
strategies_available = list(df_plot.columns)
strategies_selected = st.multiselect(
    "请选择策略",
    options=strategies_available,
    default=strategies_available
)

# 根据策略选择过滤数据
df_plot_filtered = df_plot[strategies_selected]

# 显示原始数据
if st.checkbox("📋 显示原始数据"):
    st.dataframe(df_plot_filtered.tail())

# 绘图
st.subheader("📊 策略收益曲线图（Plotly 交互图）")

fig = go.Figure()
for col in df_plot_filtered.columns:
    fig.add_trace(go.Scatter(
        x=df_plot_filtered.index,
        y=df_plot_filtered[col],
        mode="lines",
        name=col,
        hovertemplate="时间: %{x}<br>收益: %{y:.5f}<extra>" + col + "</extra>"
    ))

fig.update_layout(
    title=f"{option}（可交互）",
    xaxis_title="时间",
    yaxis_title="收益",
    height=600,
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
