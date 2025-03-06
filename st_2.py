import pandas as pd
import glob
import streamlit as st
import altair as alt

# 设置全局样式
st.markdown("""
<style>
.vega-embed { width: 100% !important; height: auto !important; }
canvas { touch-action: auto !important; }
</style>
""", unsafe_allow_html=True)

def load_data(folder):
    """加载并合并交易数据"""
    return pd.concat(
        [pd.read_csv(f, parse_dates=['exit_time']) for f in glob.glob(folder)],
        ignore_index=True
    ).sort_values('exit_time').reset_index(drop=True)

def compute_metrics(df, window):
    """计算胜率指标"""
    return (
        df['pnlcomm'].gt(0).rolling(window, min_periods=1).mean().rename('rolling'),
        df['pnlcomm'].gt(0).expanding().mean().rename('total')
    )

def create_chart(data, colors=('#ff7f0e', '#1f77b4')):
    """创建双线图表"""
    base = alt.Chart(data.reset_index(), width=800, height=500)
    return (
        base.mark_line(color=colors[0]).encode(x='index', y='rolling') +
        base.mark_line(color=colors[1]).encode(x='index', y='total')
    ).interactive().add_selection(alt.selection_interval(bind='scales'))

# 全品种分析
st.header("全品种分析")
df = load_data('transaction/*.csv')
df2 = load_data('transaction2/*.csv')
window = st.selectbox('选择滚动周期数:', [50,100,200,500,1000], key='global_window')
st.write("不含手续费")
rolling, total = compute_metrics(df, window)
st.altair_chart(create_chart(pd.concat([rolling, total], axis=1)))
st.write("含手续费")
rolling2, total2 = compute_metrics(df2, window)
st.altair_chart(create_chart(pd.concat([rolling2, total2], axis=1)))
# 局部范围分析
col1, col2 = st.columns(2)
with col1: start = st.number_input("起始次数", 0, len(df)-1, 0)
with col2: end = st.number_input("结束次数", start+1, len(df), len(df))
st.write("不含手续费")
st.altair_chart(create_chart(pd.concat([rolling[start:end], total[start:end]], axis=1)))
st.write("含手续费")
st.altair_chart(create_chart(pd.concat([rolling2[start:end], total2[start:end]], axis=1)))

# 分品种分析
st.header("分品种分析")
symbol = st.selectbox('选择品种:', glob.glob('transaction/*.csv'))
window_symbol = st.selectbox('选择滚动周期数:', [10,25,50,100,200], key='symbol_window')

df_symbol = load_data(symbol)
rolling_s, total_s = compute_metrics(df_symbol, window_symbol)
st.altair_chart(create_chart(pd.concat([rolling_s, total_s], axis=1)))