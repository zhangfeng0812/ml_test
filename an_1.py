import pandas as pd
import numpy as np
import os
import json



class An:
    def __init__(self,root):
        self.root = root
        self.start_day = '2017-01-01'
        self.end_day = '2025-01-02'
        self.is_end_day =  '2022-01-01'
        self.semi_end_day =  '2024-01-01'
        self.train_sharpe = []
        self.test_sharpe = []
        self.rolling_train_sharpe = []
        self.rolling_test_sharpe = []
        self.train_PnL = []
        self.test_PnL = []
        self.year_table = []
        self.train_table = []
        self.test_table = []
        self.is_table = []
        self.is_fu_table = []
        self.corr = []
        self.data = self.show_result(root, self.start_day, self.end_day, detail=True)

    def read_order(self,file):
        df = pd.read_csv(file, sep=' \s+', encoding='gb2312', engine='python')
        df_close = df.loc[df['类型'].str.contains('平')].dropna(axis=1)
        if len(df_close.columns) == 10:
            df_close = df_close.drop(columns=['序列'])
        df_close.columns = ['timestamp', 'product', 'order_type', 'close_price', 'share', 'pnl', 'pnl_ratio',
                            'property',
                            'max_drop']
        df_close.timestamp = pd.to_datetime(df_close.timestamp)
        #         # 强平导致并非所有close数据均存在open数据
        df_close['open_price'] = df_close.close_price.map(
            lambda x: float(x.split('/')[1] if len(x.split('/')) == 2 else x.split('/')[0]))
        df_close.close_price = df_close.close_price.map(lambda x: float(x.split('/')[0]))
        df['share'] = 1
        df_close['buy_sell'] = df_close.order_type.map(lambda x: 'S' if '空' in x else 'B')
        df_close['open_close'] = 'C'
        for col in ['pnl', 'property']:
            df_close[col] = df_close[col].map(
                lambda x: float(x.replace(',', '').replace('万', '')) * 10000 if '万' in x else float(
                    x.replace(',', '')))
        df_close['trade_day'] = df_close.timestamp.dt.normalize()

        return df_close

    def get_source_strategy(self,order_path, start_time, end_time):
        files = os.listdir(order_path)
        df_all = []
        for file in files:
            df = self.read_order(os.path.join(order_path, file))
            df = df.loc[(df.trade_day >= start_time) & (df.trade_day < end_time)]
            df['strategy'] = file.split('.')[0]
            df['trade_month'] = df.trade_day.dt.to_period('M')
            df_all.append(df)
        return pd.concat(df_all)

    def get_sharpe_ratio(self,b):
        return b.mean() / b.std() * np.sqrt(250)

    def return_stats(self,df, flag):
        day_return = df.groupby('trade_day').pnl.sum()
        dic = dict()
        # 总收益
        dic['total_return'] = day_return.sum()
        # 最大回撤
        dic['max_drawdown'] = np.max(np.maximum.accumulate(day_return.cumsum()) - day_return.cumsum())
        # 时间段
        dic['trade_day'] = list(day_return.index)
        # 回撤范围
        dic['range_drawdown'] = -1 * (np.maximum.accumulate(day_return.cumsum()) - day_return.cumsum()).values
        # 计算胜率
        dic['win_rate'] = df.loc[df.pnl > 0].shape[0] / df.shape[0]
        # 年化收益
        dic['annual_return'] = day_return.mean() * 250
        # 年化夏普比率
        dic['sharpe_ratio'] = self.get_sharpe_ratio(day_return)
        # 年化Calmar比率
        dic['calmar_ratio'] = dic['annual_return'] / dic['max_drawdown']
        # 盈亏比
        dic['profit_loss_ratio'] = df.loc[df.pnl > 0].pnl.mean() / df.loc[df.pnl < 0].pnl.abs().mean()
        # 交易次数
        dic['trade_count'] = df.shape[0]
        # 平均收益
        dic['average_return'] = df.loc[df.pnl > 0].pnl.mean()
        # 平均亏损
        dic['average_loss'] = df.loc[df.pnl < 0].pnl.mean()
        # 平均品种数量
        dic['average_product'] = df[['trade_month', 'product', 'strategy']].drop_duplicates().groupby(
            'trade_month').product.count().mean()
        # 实际交易品类
        dic['product_unique'] = df[['trade_month', 'product']].drop_duplicates().groupby(
            'trade_month').product.count().mean()
        # 类型
        dic['flag'] = flag
        return dic

    def return_stats(self, df, flag):
        day_return = df.groupby('trade_day').pnl.sum()
        cumulative = day_return.cumsum()  # 计算累积收益

        dic = dict()
        dic['total_return'] = day_return.sum()
        cumulative_returns = day_return.cumsum()
        max_cumulative_returns = np.maximum.accumulate(cumulative_returns)
        drawdowns = max_cumulative_returns - cumulative_returns
        max_drawdown_end = np.argmax(drawdowns)
        max_drawdown_start = np.argmax(cumulative_returns[:max_drawdown_end])

        # 最大回撤天数
        max_drawdown_days = max_drawdown_end - max_drawdown_start
        dic['max_drawdown_duration'] = max_drawdown_days  # 新增字段
        # 计算最大回撤及其持续时间
        peak = cumulative.iloc[0]
        peak_idx = 0
        max_drawdown = 0
        max_duration = 0

        for i in range(1, len(cumulative)):
            current = cumulative.iloc[i]
            if current > peak:
                peak = current
                peak_idx = i
            else:
                drawdown = peak - current
                duration = i - peak_idx
                # 更新最大回撤和持续时间
                if (drawdown > max_drawdown) or (drawdown == max_drawdown and duration > max_duration):
                    max_drawdown = drawdown
                    max_duration = duration

        dic['max_drawdown'] = max_drawdown
        #dic['max_drawdown_duration'] = max_duration  # 新增字段

        # 其他指标（保持不变）
        dic['trade_day'] = list(day_return.index)
        dic['range_drawdown'] = -1 * (np.maximum.accumulate(cumulative) - cumulative).values
        dic['win_rate'] = df.loc[df.pnl > 0].shape[0] / df.shape[0]
        dic['annual_return'] = day_return.mean() * 250
        dic['sharpe_ratio'] = self.get_sharpe_ratio(day_return)
        dic['calmar_ratio'] = dic['annual_return'] / max_drawdown if max_drawdown != 0 else 0
        dic['profit_loss_ratio'] = df.loc[df.pnl > 0].pnl.mean() / df.loc[df.pnl < 0].pnl.abs().mean()
        dic['trade_count'] = df.shape[0]
        dic['average_return'] = df.loc[df.pnl > 0].pnl.mean()
        dic['average_loss'] = df.loc[df.pnl < 0].pnl.mean()
        dic['average_product'] = df[['trade_month', 'product', 'strategy']].drop_duplicates().groupby(
            'trade_month').product.count().mean()
        dic['product_unique'] = df[['trade_month', 'product']].drop_duplicates().groupby(
            'trade_month').product.count().mean()
        dic['flag'] = flag
        return dic

    def future_stats(self, df, flag):
        dic = dict()
        future_return = df.groupby('product').pnl.sum()
        dic["per"] = len(future_return[future_return.values>0])/len(future_return.values)
        dic['flag'] = flag
        return dic

    # 新增函数：计算滚动夏普比率
    def batch_cumulative_sharpe(self,returns_df, annualized_factor=np.sqrt(250)):
        """
        批量计算多列累积夏普比率（向量化优化版）

        参数：
            returns_df : DataFrame 收益数据（每列为一个策略/资产）
            annualized_factor : float 年化因子，默认日频数据√250

        返回：
            DataFrame 累积夏普比率序列（与输入同维度）
        """
        # 向量化计算累积均值和标准差
        expanding_mean = returns_df.expanding().mean()
        expanding_std = returns_df.expanding().std()

        # 处理零标准差（将夏普设为0）
        sharpe = np.divide(
            expanding_mean,
            expanding_std.replace(0, np.nan),  # 避免除零警告
            out=np.zeros_like(expanding_mean),
            where=(expanding_std != 0)
        ) * annualized_factor

        return sharpe

    def vectorized_rolling_sharpe(self,returns_df, window=120, min_periods=20, annualized_factor=np.sqrt(250)):
        """
        向量化滚动夏普比率计算 (支持多列并行)

        参数：
            returns_df : DataFrame 收益率数据 (每列为一个策略)
            window : int 滚动窗口大小
            min_periods : int 最小计算周期 (不足返回NaN)
            annualized_factor : float 年化因子

        返回：
            DataFrame 滚动夏普比率 (与输入同形状)
        """
        # 计算滚动指标
        rolling_mean = returns_df.rolling(window=window, min_periods=1).mean()
        rolling_std = returns_df.rolling(window=window, min_periods=1).std()

        # 向量化计算夏普比率
        sharpe = rolling_mean / rolling_std.replace(0, np.nan) * annualized_factor

        # 处理边界条件
        valid_periods = returns_df.rolling(window, min_periods=1).count() >= min_periods
        sharpe = sharpe.where(valid_periods, 0)  # 不满足最小周期设为0
        sharpe = sharpe.where(rolling_std != 0, 0)  # 零波动率设为0

        return sharpe

    def get_yearly_stats(self,df):
        """计算年度统计指标"""
        yearly_stats = []

        # 按年份分组
        for year, df_year in df.groupby(df['trade_day'].dt.year):
            # 计算每日收益
            daily_return = df_year.groupby('trade_day').pnl.sum()
            if len(daily_return) == 0:
                continue

            # 年化收益率（总收益）
            # 假设本金为20w
            cash = 200000
            total_return = daily_return.sum() / cash

            # 年化夏普比率（考虑交易日天数）
            sharpe = (daily_return.mean() / daily_return.std()) * np.sqrt(252) if daily_return.std() != 0 else 0

            # 最大回撤（日内回撤）
            cumulative_returns = daily_return.cumsum() / cash
            max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

            yearly_stats.append({
                '年份': year,
                '年化收益率': total_return,
                '年化夏普比率': sharpe,
                '最大回撤率': max_drawdown
            })

        return pd.DataFrame(yearly_stats).set_index('年份')

    def calculate_annual_metrics(self,returns_df, annual_trading_days=252):
        """
        按年计算每个策略的三项核心指标

        参数：
            returns_df : DataFrame 日度收益率数据（每列为一个策略）
            annual_trading_days : int 年化交易日数（默认252）

        返回：
            tuple: (年化收益率df, 夏普比率df, 最大回撤df)
        """
        # 转换为日期索引
        returns_df.index = pd.to_datetime(returns_df.index)

        # 按年分组
        years = returns_df.index.year.unique()

        # 预计算结果容器
        annual_returns = pd.DataFrame(index=years, columns=returns_df.columns)
        sharpe_ratios = pd.DataFrame(index=years, columns=returns_df.columns)
        max_drawdowns = pd.DataFrame(index=years, columns=returns_df.columns)

        # 向量化计算每个年份
        for year in years:
            # 获取当年数据
            mask = returns_df.index.year == year
            cash = 200000
            year_returns = returns_df[mask]

            # 跳过空年份
            if len(year_returns) == 0:
                continue

            # 计算年化收益率
            cum_ret = round(year_returns.sum() / cash,2)
            n_days = len(year_returns)
            annual_returns.loc[year] = cum_ret


            # 计算夏普比率（处理零波动）
            mean_ret = year_returns.mean()
            std_ret = year_returns.std()
            sharpe = (mean_ret / std_ret.replace(0, np.nan)) * np.sqrt(annual_trading_days)
            sharpe_ratios.loc[year] = sharpe.fillna(0)

            # 计算最大回撤（优化算法）
            def compute_drawdown(col):
                cumulative = col.cumsum()/cash
                peak = cumulative.cummax()
                drawdown = (peak - cumulative).max()
                return str(round(drawdown,2))+"%"

            max_drawdowns.loc[year] = year_returns.apply(compute_drawdown)
        dic={}
        for strategy in annual_returns.columns:
            df2=pd.concat([annual_returns[strategy],sharpe_ratios[strategy],max_drawdowns[strategy]],axis=1)
            df2.columns = ['annual_return','sharpe_ratio','max_drawdown']
            dic[strategy]=df2
        return dic
    def show_result(self,trader_dir, start_day, end_day, detail=False, rolling_window=120):
        df_source = self.get_source_strategy(trader_dir, start_day, end_day)
        all_stats_result = []
        df_plot = df_source.groupby(['strategy', 'trade_day']).pnl.sum().unstack(level=0).fillna(0)
        self.corr = df_plot.corr()
        # ================= 新增年度统计 =================
        #yearly_stats_df = self.get_yearly_stats(df_source)
        yearly_stats_df = self.calculate_annual_metrics(df_plot)
        self.year_table = yearly_stats_df
        if detail:
            split_date = '2023-01-01'
            colors = {'in_sample': '#1f77b4', 'out_sample': '#ff7f0e'}  # 蓝色和橘黄
            self.train_PnL = df_plot[df_plot.cumsum().index <= split_date]
            self.test_PnL = df_plot
            cumulative_sharpe = self.batch_cumulative_sharpe(df_plot)[120:]
            self.train_sharpe = cumulative_sharpe[cumulative_sharpe.index <= split_date]
            self.test_sharpe = cumulative_sharpe
            rolling_sharpe = self.vectorized_rolling_sharpe(df_plot, rolling_window)[120:]
            self.rolling_train_sharpe =rolling_sharpe[rolling_sharpe.index <= split_date]
            self.rolling_test_sharpe = rolling_sharpe

        # 原有统计结果计算
        is_lst= []
        test_lst = []
        is_fu_lst = []
        train_lst = []
        is_dic = {}
        is_fu_dic = {}
        test_dic = {}
        train_dic = {}
        if detail:
            for strategy in df_source.strategy.unique():
                all_stats_result.append(self.return_stats(df_source.loc[df_source.strategy == strategy], strategy))
                is_source = self.get_source_strategy(trader_dir, start_day, self.semi_end_day)
                is_lst.append(self.return_stats(is_source.loc[is_source.strategy == strategy], strategy))
                is_fu_lst.append(self.future_stats(is_source.loc[is_source.strategy == strategy], strategy))
                test_source = self.get_source_strategy(trader_dir, self.is_end_day, self.semi_end_day)
                test_lst.append(self.return_stats(test_source.loc[test_source.strategy == strategy], strategy))
                train_source = self.get_source_strategy(trader_dir, self.start_day, self.is_end_day)
                train_lst.append(self.return_stats(train_source.loc[train_source.strategy == strategy], strategy))
        all_stats_result.append(self.return_stats(df_source, 'total'))
        all_stats_result_df = pd.DataFrame(all_stats_result)
        try:
            os.makedirs(r"./data")
        except FileExistsError:
            pass
        self.is_fu_table = pd.DataFrame(is_fu_lst,index=df_source.strategy.unique())
        self.is_table = pd.DataFrame(is_lst)
        #self.test_table =all_stats_result_df
        self.test_table = pd.DataFrame(test_lst)
        self.train_table = pd.DataFrame(train_lst)
        # 计算IS，TEST
        for k in ['sharpe_ratio','calmar_ratio','profit_loss_ratio','trade_count']:
            is_dic[k] = self.is_table[k].sum()/self.is_table.shape[0]
            test_dic[k] = self.test_table[k].sum()/self.is_table.shape[0]
            train_dic[k] = self.train_table[k].sum()/self.is_table.shape[0]
        with open('./data/is_data.json', 'w', encoding="utf-8") as json_file:
            json.dump(is_dic, json_file, indent=4, ensure_ascii=False)
        with open('./data/test_data.json', 'w', encoding="utf-8") as json_file:
            json.dump(test_dic, json_file, indent=4, ensure_ascii=False)
        with open('./data/train_data.json', 'w', encoding="utf-8") as json_file:
            json.dump(train_dic, json_file, indent=4, ensure_ascii=False)
        return all_stats_result_df


#a = An(r"./comb")
