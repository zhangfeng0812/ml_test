import streamlit
import streamlit as st
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import an_1
import pandas as pd
from datetime import datetime
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
ROOT = "comb11"

def get_portfolio(df,window_length=120):
    def calculate_risk_contribution(w, cov_matrix):
        """计算各资产的风险贡献"""
        port_vol = np.sqrt(w.T @ cov_matrix @ w)
        marginal_contrib = (cov_matrix @ w) / port_vol
        risk_contrib = w * marginal_contrib
        return risk_contrib / risk_contrib.sum()  # 归一化

    def risk_parity_objective(w, cov_matrix):
        """优化目标：风险贡献的方差最小化"""
        rc = calculate_risk_contribution(w, cov_matrix)
        target_rc = np.ones_like(w) / len(w)  # 目标风险贡献相等
        return np.sum((rc - target_rc) ** 2)

    def rolling_risk_parity(returns, window=120, rebalance_freq='M'):
        """修正后的滚动风险平价函数"""
        # 生成再平衡日期并过滤不存在于原始数据的日期
        reb_dates = returns.resample(rebalance_freq).last().index
        reb_dates = reb_dates[reb_dates.isin(returns.index)]

        weights = pd.DataFrame(index=reb_dates, columns=returns.columns)

        for i, date in enumerate(reb_dates):
            # 获取日期在原始数据中的位置
            date_pos = returns.index.get_loc(date)

            # 计算窗口起始位置，防止越界
            start_pos = max(0, date_pos - window)
            start_date = returns.index[start_pos]

            # 提取窗口数据（使用iloc避免KeyError）
            window_returns = returns.iloc[start_pos:date_pos]

            # 检查窗口数据长度
            if len(window_returns) < 10:  # 最小数据量要求
                weights.loc[date] = np.nan
                continue

            # 计算协方差矩阵
            from sklearn.covariance import LedoitWolf
            cov_matrix = LedoitWolf().fit(window_returns).covariance_
            # cov_matrix = window_returns.cov().values

            # 优化权重（逻辑保持不变）
            n_assets = len(returns.columns)
            initial_guess = np.ones(n_assets) / n_assets

            constraints = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'ineq', 'fun': lambda w: w}
            )

            result = minimize(
                risk_parity_objective,
                initial_guess,
                args=(cov_matrix,),
                method='SLSQP',
                constraints=constraints,
                tol=1e-6
            )

            if result.success:
                weights.loc[date] = result.x
            else:
                weights.loc[date] = np.nan

        # 前向填充权重
        weights = weights.resample('D').ffill().reindex(returns.index)
        return weights.dropna(how='all')

    rebalance_freq = 'M'  # 每月再平衡
    period = st.selectbox("select the specific period:",[30,60,90,120,240,500,1000])
    weights = rolling_risk_parity(df, window=period, rebalance_freq=rebalance_freq)
    portfolio_returns = (df* weights.shift(1)).sum(axis=1)  # 次日调仓

    # 可视化
    fig = plt.figure(figsize=(14, 8))

    # 绘制权重变化
    plt.subplot(2, 1, 1)
    for col in weights.columns:
        plt.plot(weights.index, weights[col], label=col)
    plt.title('Rolling Risk Parity Weights')
    plt.ylabel('Weight')
    plt.legend()

    # 绘制累计收益
    plt.subplot(2, 1, 2)
    cumulative_port = (1 + portfolio_returns).cumprod()
    cumulative_bm = (1 + df.mean(axis=1)).cumprod()  # 等权重作为基准
    plt.plot(cumulative_port, label='Risk Parity Portfolio')
    plt.plot(cumulative_bm, label='Equal Weight Benchmark', linestyle='--')
    plt.title('Cumulative Returns')
    plt.ylabel('Growth')
    plt.legend()

    plt.tight_layout()
    st.pyplot(fig)

    def calculate_metrics(returns):
        """计算绩效指标"""
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak

        return pd.Series({
            'Annualized Return': returns.mean() * 252,
            'Annualized Volatility': returns.std() * np.sqrt(252),
            'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(252),
            'Max Drawdown': drawdown.min(),
            'Calmar Ratio': returns.mean() * 252 / abs(drawdown.min())
        })

    port_metrics = calculate_metrics(portfolio_returns)
    bm_metrics = calculate_metrics(df.mean(axis=1))

    print("\n策略绩效对比:")
    st.table(pd.concat([port_metrics.rename('Risk Parity'),
               bm_metrics.rename('Equal Weight')], axis=1).T)


def return_stats(day_return, flag):
    def get_sharpe_ratio(b):
        return b.mean() / b.std() * np.sqrt(250)
    dic = dict()
    # 总收益
    dic['total_return'] = day_return.sum()
    # 最大回撤
    dic['max_drawdown'] = np.max(np.maximum.accumulate(day_return.cumsum()) - day_return.cumsum())
    # 时间段
    dic['trade_day'] = list(day_return.index)
    # 回撤范围
    dic['range_drawdown'] = -1*(np.maximum.accumulate(day_return.cumsum()) - day_return.cumsum()).values
    # 年化收益
    dic['annual_return'] = day_return.mean() * 250
    # 年化夏普比率
    dic['sharpe_ratio'] = get_sharpe_ratio(day_return)
    # 年化Calmar比率
    dic['calmar_ratio'] = dic['annual_return'] / dic['max_drawdown']
    dic['flag'] = flag
    return dic

def backtest_results():
    files = os.listdir(ROOT)
    lst = [f.split(".")[0] for f in files]
    return lst

def get_is_training_status(stras):
    score = 0
    dic = {"train_sharpe": round(a.train_table[a.train_table['flag'] == stras]['sharpe_ratio'], 2),
           "test_sharpe": round(a.test_table[a.test_table['flag'] == stras]['sharpe_ratio'], 2),
           "PASS":[],
           "FAIL":[]}
    # IS and OS sharpe
    if abs(float(dic["train_sharpe"]) - float(dic["test_sharpe"]))<0.5:
        dic["PASS"].append('<span style="color:green;"> > </span> The strategy was tested successfully both IS and OS.')
        score += 60
    else:
        dic["FAIL"].append('<span style="color:red;"> > </span> The strategy was tested **%s** both IS and OS is above cutoff of **0.5**.'%abs(round(float(dic["train_sharpe"]) - float(dic["test_sharpe"]),2)))
    # CALMAR
    calmar = float(round(a.train_table[a.train_table['flag'] == stras]['calmar_ratio'], 2))
    if calmar < 1:
        dic["FAIL"].append('<span style="color:red;"> > </span> The calmar of **%s** is below cutoff of **1.0**.'%calmar)
    elif calmar < 1.5:
        score += 60
        dic["PASS"].append('<span style="color:green;"> > </span>The calmar of **%s** is above cutoff of **1.0**.'%calmar)
    else:
        score += 80
        dic["PASS"].append('<span style="color:green;"> > </span> The calmar of strategy is spectacular.')
    # rolling-sharpe
    rolling_sharpe = round(float(a.rolling_train_sharpe.mean()[stras]),2)
    if rolling_sharpe > 1:
        dic["PASS"].append('<span style="color:green;"> > </span> The rolling-sharpe of **%s** is above cutoff of **1.0**.'%rolling_sharpe)
    else:
        dic["FAIL"].append(
            '<span style="color:red;"> > </span> The rolling-sharpe of **%s** is below cutoff of **1.0**.' % rolling_sharpe)
    # corr
    corr = a.corr.loc[stras]
    max_corr = corr[corr < corr.max()].max()
    if max_corr >0.3:
        dic["FAIL"].append('<span style="color:red;"> > </span> The correlation of strategy is too high.')
    else:
        score += 60
        dic["PASS"].append('<span style="color:green;"> > </span> The correlation of strategy is tested successfully.')
    # count
    train_count = int(a.train_table[a.train_table['flag'] == stras]['trade_count'])
    if train_count < 5000:
        dic["FAIL"].append('<span style="color:red;"> > </span> The number of trades is too low.')
    else:
        score += 60
        dic["PASS"].append('<span style="color:green;"> > </span> The number of trades is tested successfully.')
    # max drawdown duaration
    duration = int(a.train_table[a.train_table['flag'] == stras]['max_drawdown_duration'])
    if duration > 120:
        dic["FAIL"].append('<span style="color:red;"> > </span> The maximum drawdown duration of **%s** days is above cutoff of **120** days.'%duration)
    else:
        score += 60
        dic["PASS"].append('<span style="color:green;"> > </span> The duration of **%s** days is below cutoff of **120** days.'%duration)
    fu_per = float(round(a.is_fu_table[a.is_fu_table['flag'] == stras]["per"],2))
    if fu_per >= 0.8:
        score += 80
        dic["PASS"].append('<span style="color:green;"> > </span> The variety of full-future of **%s%%** is above cutoff of **80%%**.'% str(fu_per*100))
    if 0.6 < fu_per < 0.8:
        score += 60
        dic["FAIL"].append(
            '<span style="color:red;"> > </span> The variety of full-future of **%s%%** is below cutoff of **80%%**.' % str(fu_per*100))
    if fu_per <= 0.6:
        dic["FAIL"].append('<span style="color:red;"> > </span> The variety of full-future of **%s%%** is below cutoff of **80%%**.'% str(fu_per*100))
    score = int(score/6)
    dic["score"] = score
    return dic

def get_is_testing_status(stras):
    score = 0
    dic = {"train_sharpe": round(a.train_table[a.train_table['flag'] == stras]['sharpe_ratio'], 2),
           "test_sharpe": round(a.test_table[a.test_table['flag'] == stras]['sharpe_ratio'], 2),
           "PASS":[],
           "FAIL":[]}
    # IS and OS sharpe
    if abs(float(dic["train_sharpe"]) - float(dic["test_sharpe"]))<0.5:
        dic["PASS"].append('The strategy was tested successfully both IS and OS.')
        score += 60
    else:
        dic["FAIL"].append('The strategy was tested unsuccessfully both IS and OS.')
    # CALMAR
    calmar = float(round(a.test_table[a.test_table['flag'] == stras]['calmar_ratio'], 2))
    if calmar < 1:
        dic["FAIL"].append('The calmar of strategy was too low.')
    elif calmar < 1.5:
        score += 60
        dic["PASS"].append('The calmar of strategy was good.')
    else:
        score += 80
        dic["PASS"].append('The calmar of strategy was spectacular.')
    # corr
    corr = a.corr.loc[stras]
    max_corr = corr[corr < corr.max()].max()
    if max_corr >0.3:
        dic["FAIL"].append('The correlation of strategy was too high.')
    else:
        score += 60
        dic["PASS"].append('The correlation of strategy was tested successfully.')
    # count
    train_count = int(a.test_table[a.test_table['flag'] == stras]['trade_count'])
    if train_count < 2000:
        dic["FAIL"].append('The number of trades was too low.')
    else:
        score += 60
        dic["PASS"].append('The number of trades was tested successfully.')
    # max drawdown duaration
    duration = int(a.test_table[a.test_table['flag'] == stras]['max_drawdown_duration'])
    if duration > 120:
        dic["FAIL"].append('The maximum drawdown duration was too long.')
    else:
        score += 60
        dic["PASS"].append('The duration of drawdown was tested successfully.')
    fu_per = float(a.is_fu_table[a.is_fu_table['flag'] == stras]["per"])
    if fu_per > 80:
        score += 80
        dic["PASS"].append('The variety of full-future was tested successfully.')
    if fu_per > 60:
        score += 60
    else:
        dic["FAIL"].append('The variety of full-future was failed.')
    score = int(score/6)
    dic["score"] = score
    return dic

def get_painting(name,per,stage):
    if  per=='PnL':
        data1 = a.test_PnL.cumsum()[name].to_frame()
        data1.columns = ["test"]
        data2 = a.train_PnL.cumsum()[name].to_frame()
        data2.columns = ["train"]
    if per == 'Sharpe':
        data1 = a.test_sharpe[name].to_frame()
        data1.columns = ["test"]
        data2 = a.train_sharpe[name].to_frame()
        data2.columns = ["train"]
    if per=='RollingSharpe':
        data1 = a.rolling_test_sharpe[name].to_frame()
        data1.columns = ["test"]
        data2 = a.rolling_train_sharpe[name].to_frame()
        data2.columns = ["train"]
    data = data1.join(data2, how='left')
    data['trade_day'] = data.index

    lines = (
        alt.Chart(data, width=800, height=500)
        .mark_line(color='#ff7f0e')
        .encode(x="trade_day", y="test")
    )

    lines2 = (
        alt.Chart(data, width=800, height=500)
        .mark_line(color='#1f77b4')
        .encode(x="trade_day", y="train")
    )
    if stage:
        st.altair_chart(lines + lines2)
    else:
        st.altair_chart(lines2)

def get_assess(s):
    sharpe = float(s)
    if sharpe>2.5:
        return "Spectacular"
    if sharpe>2:
        return "Excellent"
    if sharpe>1.5:
        return "Good"
    if sharpe>1:
        return "Average"
    else:
        #return "Needs improvement"
        return "Failed"

def get_avg_data():
    """

    :return: is_data/test_data/json_data
    """
    with open('data/is_data.json', 'r') as json_file:
        d1 = json.load(json_file)
    with open('data/test_data.json', 'r') as json_file:
        d2 = json.load(json_file)
    with open('data/train_data.json', 'r') as json_file:
        d3 = json.load(json_file)
    return [d1,d2,d3]


@st.cache_data  #   添加缓存装饰
def load_data():
    return an_1.An(ROOT)

def page1():
    stras = st.selectbox('Choose the specific strategy:', options=backtest_results())
    if 'show_test_period' not in st.session_state:
        st.session_state.show_test_period = False

    # 定义切换状态的回调函数
    def toggle_test_period():
        st.session_state.show_test_period = not st.session_state.show_test_period

    # 创建按钮，动态显示文本
    bt1 = st.button(
        label="Hide test period" if st.session_state.show_test_period else "Show test period",
        on_click=toggle_test_period,
        key="toggle_button"
    )
    st.subheader('📈Chart')
    performance = st.selectbox('Performance', options= ["PnL","Sharpe","RollingSharpe"])
    # 打印对应的图片
    get_painting(stras, performance,st.session_state.show_test_period)
    st.subheader('📶IS Summary')
    st0, st1= st.columns(2)
    bt0,bt1,bt2,bt3 = st1.columns(4)
    bt0.write("Period")
    bt_train = bt1.button("Train")
    bt_test = bt2.button("Test")
    bt_is = bt3.button("IS")
    col0,col1, col2, col3, col4 = st.columns(5)

    if bt_train or (bt_train== False and bt_test == False and bt_is==False):
        sharpe = round(a.train_table[a.train_table['flag']==stras]['sharpe_ratio'],2)
        col0.metric("Aggregate Data",get_assess(sharpe))
        col1.metric("Sharpe",round(a.train_table[a.train_table['flag']==stras]['sharpe_ratio'],2),delta=round(float(a.train_table[a.train_table['flag']==stras]['sharpe_ratio'])-avg_data[2]["sharpe_ratio"],2),delta_color="inverse")
        col2.metric("Calmar",round(a.train_table[a.train_table['flag']==stras]['calmar_ratio'],2),delta=round(float(a.train_table[a.train_table['flag']==stras]['calmar_ratio'])-avg_data[2]["calmar_ratio"],2),delta_color="inverse")
        col3.metric("trade_count",a.train_table[a.train_table['flag']==stras]['trade_count'],delta=round(float(a.train_table[a.train_table['flag']==stras]['trade_count'])-avg_data[2]["trade_count"],0),delta_color="inverse")
        col4.metric("profit_loss_ratio",round(a.train_table[a.train_table['flag']==stras]['profit_loss_ratio'],2),delta=round(float(a.train_table[a.train_table['flag']==stras]['profit_loss_ratio'])-avg_data[2]["profit_loss_ratio"],2),delta_color="inverse")
    if bt_is:
        sharpe = round(a.is_table[a.is_table['flag'] == stras]['sharpe_ratio'], 2)
        col0.metric("Aggregate Data", get_assess(sharpe))
        col1.metric("Sharpe", round(a.is_table[a.is_table['flag'] == stras]['sharpe_ratio'], 2),delta=round(float(a.is_table[a.is_table['flag']==stras]['sharpe_ratio'])-avg_data[0]["sharpe_ratio"],2),delta_color="inverse")
        col2.metric("Calmar", round(a.is_table[a.is_table['flag'] == stras]['calmar_ratio'], 2),delta=round(float(a.is_table[a.is_table['flag']==stras]['calmar_ratio'])-avg_data[0]["calmar_ratio"],2),delta_color="inverse")
        col3.metric("trade_count", a.is_table[a.is_table['flag'] == stras]['trade_count'],delta=round(float(a.is_table[a.is_table['flag']==stras]['trade_count'])-avg_data[0]["trade_count"],0),delta_color="inverse")
        col4.metric("profit_loss_ratio", round(a.is_table[a.is_table['flag'] == stras]['profit_loss_ratio'], 2),delta=round(float(a.is_table[a.is_table['flag']==stras]['profit_loss_ratio'])-avg_data[0]["profit_loss_ratio"],2),delta_color="inverse")
    if bt_test:
        sharpe = round(a.test_table[a.test_table['flag'] == stras]['sharpe_ratio'], 2)
        col0.metric("Aggregate Data", get_assess(sharpe))
        col1.metric("Sharpe", round(a.test_table[a.test_table['flag'] == stras]['sharpe_ratio'], 2),delta=round(float(a.test_table[a.test_table['flag']==stras]['sharpe_ratio'])-avg_data[1]["sharpe_ratio"],2),delta_color="inverse")
        col2.metric("Calmar", round(a.test_table[a.test_table['flag'] == stras]['calmar_ratio'], 2),delta=round(float(a.test_table[a.test_table['flag']==stras]['calmar_ratio'])-avg_data[1]["calmar_ratio"],2),delta_color="inverse")
        col3.metric("trade_count", a.test_table[a.test_table['flag'] == stras]['trade_count'],delta=round(float(a.test_table[a.test_table['flag']==stras]['trade_count'])-avg_data[1]["trade_count"],0),delta_color="inverse")
        col4.metric("profit_loss_ratio", round(a.test_table[a.test_table['flag'] == stras]['profit_loss_ratio'], 2),delta=round(float(a.test_table[a.test_table['flag']==stras]['profit_loss_ratio'])-avg_data[1]["profit_loss_ratio"],2),delta_color="inverse")
    #st.write(a.test_table[a.test_table['flag']==stras][['win_rate','annual_return','sharpe_ratio','calmar_ratio','profit_loss_ratio','trade_count','average_return','average_loss']])
    if st.session_state.show_test_period:
        st.table(a.year_table[stras])
    else:
        st.table(a.year_table[stras][:-2])
    st.subheader('📊Correlation')
    s1, s2, s3= st.columns(3)
    corr = a.corr.loc[stras]
    max_corr = corr[corr < corr.max() ].max()
    min_corr = corr.min()
    s1.metric("Self Correlation","")
    s2.metric("Maximum",round(max_corr,2))
    s3.metric("Minimum",round(min_corr,2))
    st.subheader(' IS Testing Status')
    if bt_train or (bt_train== False and bt_test == False and bt_is==False):
        dic1 = get_is_training_status(stras)
    else:
        dic1 = get_is_testing_status(stras)
    st.metric("Score:",dic1["score"])
    exp = st.expander("%s PASS"%len(dic1["PASS"]))
    for k in dic1["PASS"]:
        exp.markdown(k, unsafe_allow_html=True)
    exp2 = st.expander("%s FAIL"%len(dic1["FAIL"]))
    for k in dic1["FAIL"]:
        exp2.markdown(k, unsafe_allow_html=True)
    exp3 = st.expander("PENDING")
    exp3.write("Parameter sensitivity check pending.")
def page2():
    options = st.multiselect("Choose the specific Strategies.",
                             backtest_results())
    if len(options)!=0:
        st.subheader("Correlation")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            a.corr[options].loc[options],
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            ax=ax
        )
        ax.set_title(f"Heatmap")
        st.pyplot(fig)
        st.subheader("Net Value Performance")
        st0, st1 = st.columns(2)
        year = st0.selectbox("Choose the specific years.",["all","2017","2018","2019","2020","2021","2022","2023","2024"])
        bt0, bt1, bt2, bt3 = st1.columns(4)
        bt0.write("Period")
        bt_train = bt1.button("Train")
        bt_test = bt2.button("Test")
        bt_is = bt3.button("IS")
        data = a.train_PnL
        if bt_test:
            data = a.test_PnL
        if year and year != "all":
            data = a.test_PnL[(a.test_PnL.index>str(year)) & (a.test_PnL.index<str(int(year)+1))]
        if bt_train:
            data = a.train_PnL
        df_lst = []
        temp_option = options.copy()
        for op in options:
            df1 = data.cumsum()[op].to_frame()
            df_lst.append(df1)
        options.append("sum")
        df = pd.concat(df_lst, axis=1, join="inner")
        df["sum"] = df.sum(axis=1)
        df['trade_day'] = df.index
        selection = alt.selection_point(
            fields=['column'],  # 绑定到折叠后的列名
            bind='legend'  # 确保图例可以交互
        )
        # 创建折线图
        chart = alt.Chart(df).transform_fold(
            fold=options,  # 需要折叠的列
            as_=['column', 'value']  # 定义新字段名称
        ).mark_line().encode(
            x='trade_day:T',  # 时间类型确保正确解析
            y='value:Q',
            color=alt.condition(
                selection,  # 根据选择器状态改变颜色
                alt.Color('column:N'),  # 隐藏默认图例（使用交互图例）
                alt.value('lightgray')
            ),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),  # 增加透明度区分
            tooltip=['trade_day:T', 'column:N', 'value:Q']
        ).add_params(
            selection  # 将选择器添加到图表
        ).properties(
            width=800,
            height=500,
            title="Strategy Net Value"
        ).interactive()
        zoom = alt.selection_interval(bind='scales', encodings=['x'])  # 仅允许x轴缩放
        chart = chart.add_params(zoom)
        chart
        st.subheader("Strategy Performance")
        out_df = []
        data2 = data[temp_option]
        data3 = data2.copy()
        data2["sum"] = data2.sum(axis=1)
        for col in data2.columns:
            out = return_stats(data2[col],col)
            out_df.append([out["flag"],out["total_return"],out["max_drawdown"],out["sharpe_ratio"],out["calmar_ratio"]])
        per = pd.DataFrame(out_df,columns=["Strategy","total_return","max_drawdown","sharpe_ratio","calmar_ratio"])
        st.table(per)
        total_drawdown = per["max_drawdown"].sum()-per["max_drawdown"].iloc[-1]
        # The basic line
        st1,st2,st3 = st.columns(3)
        st1.metric("Drawdown Withdraw (%)",round(100*(per["max_drawdown"].iloc[-1]-total_drawdown)/total_drawdown,2))
        st2.metric("Calmar Ratio",round(per["calmar_ratio"].iloc[-1],2),round(per["calmar_ratio"].iloc[-1]-max(per["calmar_ratio"].iloc[:-1]),2),delta_color="inverse")
        st3.metric("Sharpe Ratio",round(per["sharpe_ratio"].iloc[-1],2),round(per["sharpe_ratio"].iloc[-1]-max(per["sharpe_ratio"].iloc[:-1]),2),delta_color="inverse")
        st.subheader('Strategy Portfolio')
        get_portfolio(data3/5000000)
avg_data = get_avg_data()
title = 'Strategy Pool Analysis'
a = load_data()
st.title(title)
pg = st.navigation([
    st.Page(page1, title="Single Strategy Analysis", icon=":material/favorite:"),
    st.Page(page2, title="Multiple Strategy Analysis", icon=":material/favorite:"),
])
pg.run()


