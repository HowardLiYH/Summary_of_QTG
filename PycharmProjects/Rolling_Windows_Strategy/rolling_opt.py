import warnings

warnings.filterwarnings('ignore')
import time
import akshare as ak
import numpy as np
import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import random
from myconfig import SqlConnManager
from ta.momentum import rsi
import scipy.stats


class RollingProcessor:
    @staticmethod
    def get_ETF_data(SecuCode):
        """

        :param SecuCode int/str

        :return ETF Data
        """
        sql = f"""SELECT * FROM [jydb].[dbo].SecuMain WHERE SecuCode = '{SecuCode}' """
        data = pd.read_sql(sql, SqlConnManager.conn_jy())
        InnerCode = data['InnerCode'][0]
        if len(data) > 1:
            for row in range(len(data)):
                InnerCode = data.iloc[row, :]['InnerCode']
                sql2 = f"""SELECT 
                        [InnerCode]
                      ,[InfoPublDate]
                      ,[InfoSource]
                      ,[EndDate]
                      ,[NV]
                      ,[UnitNV]
                      ,[AccumulatedUnitNV]
                    FROM [jydb].[dbo].[MF_NetValue]
                    where innercode = '{InnerCode}'  order by InfoPublDate ASC"""
                data2 = pd.read_sql(sql2, SqlConnManager.conn_jy())
                if len(data2) > 2:
                    return data2
        sql2 = f"""SELECT 
                    [InnerCode]
                  ,[InfoPublDate]
                  ,[InfoSource]
                  ,[EndDate]
                  ,[NV]
                  ,[UnitNV]
                  ,[AccumulatedUnitNV]
                FROM [jydb].[dbo].[MF_NetValue]
                where innercode = '{InnerCode}'  order by InfoPublDate ASC"""
        data2 = pd.read_sql(sql2, SqlConnManager.conn_jy())
        return data2

    @staticmethod
    def get_index_data(SecuCode):
        """

        :param SecuCode int/str

        :return ETF Data
        """
        sql = f"""select a.*
        from QT_IndexQuote a, SecuMain b
        where a.InnerCode = b.InnerCode and b.SecuCode = '{SecuCode}' """

        data = pd.read_sql(sql, SqlConnManager.conn_jy())
        return data

    @staticmethod
    # Align all the DFs in list to the size of the shortest one from CSV
    def get_data_euqal_period_CSV(list_of_codes):
        """
        :param list_of_names:

        :return: DataFrame
        """
        # parameter type: list
        etf_list = []
        for i in list_of_codes:
            file_path = os.path.join(r"C:\Users\ps\PycharmProjects\pythonProject", f"{str(i)}.csv")
            etf_data = pd.read_csv(file_path)
            etf_data = etf_data[['EndDate', 'AccumulatedUnitNV']].set_index('EndDate')
            etf_data = etf_data.rename(columns={'AccumulatedUnitNV': f"etf_{str(i)}"})
            etf_list.append(etf_data)

            def compare_index_difference(df1, df2):
                index1 = set(df1.index)
                index2 = set(df2.index)

                missing_indices_df1 = index2 - index1
                missing_indices_df2 = index1 - index2

                return missing_indices_df1, missing_indices_df2

        shortest_length = len(etf_list[0]) + 1
        shortest_index = -1

        # Find the shortest DataFrame
        list_of_index = []
        for i, df in enumerate(etf_list):
            list_of_index.append(i)
            if len(df) < shortest_length:
                shortest_length = len(df)
                shortest_index = i

        # Align all length based on the shortest DF
        for i, df in enumerate(etf_list):
            if i == shortest_index:
                continue
            etf_list[i] = df.drop(compare_index_difference(df, etf_list[shortest_index])[1])

        data = pd.concat([etf_list[0], etf_list[1]], axis=1)

        for index in list_of_index[2:]:
            data = pd.concat([data, etf_list[index]], axis=1)
        data = data.dropna().iloc[:]
        data.index = pd.to_datetime(data.index)

        # 计算每日涨跌幅和滚动收益率
        for column in data.columns:
            data[f'{column}_day_return'] = data[column].rolling(2).apply(lambda x: (x[-1] - x[0]) / x[0])

        return data

    @staticmethod
    # Align all the DFs in list to the size of the shortest one from SQL
    def get_data_euqal_period_SQL(list_of_codes):
        """
        :param list_of_names:

        :return: DataFrame
        """
        # parameter type: list
        etf_list = []
        for i in list_of_codes:
            etf_data = RollingProcessor.get_ETF_data(i)
            etf_data = etf_data[['EndDate', 'AccumulatedUnitNV']].set_index('EndDate')
            etf_data = etf_data.rename(columns={'AccumulatedUnitNV': f"etf_{str(i)}"})
            etf_list.append(etf_data)

            def compare_index_difference(df1, df2):
                index1 = set(df1.index)
                index2 = set(df2.index)

                missing_indices_df1 = index2 - index1
                missing_indices_df2 = index1 - index2

                return missing_indices_df1, missing_indices_df2

        shortest_length = len(etf_list[0]) + 1
        shortest_index = -1
        # Find the shortest DataFrame
        list_of_index = []
        for i, df in enumerate(etf_list):
            list_of_index.append(i)
            if len(df) < shortest_length:
                shortest_length = len(df)
                shortest_index = i

        # Align all length based on the shortest DF
        for i, df in enumerate(etf_list):
            if i == shortest_index:
                continue
            etf_list[i] = df.drop(compare_index_difference(df, etf_list[shortest_index])[1])

        data = pd.concat([etf_list[0], etf_list[1]], axis=1)

        for index in list_of_index[2:]:
            data = pd.concat([data, etf_list[index]], axis=1)
        data = data.dropna().iloc[:]
        data.index = pd.to_datetime(data.index)

        # 计算每日涨跌幅和滚动收益率
        for column in data.columns:
            data[f'{column}_day_return'] = data[column].rolling(2).apply(lambda x: (x[-1] - x[0]) / x[0])

        return data

    @staticmethod
    # Align all the DFs in list to the size of the shortest one
    def get_data_full_period(list_of_codes, list_of_types=None):
        """
        :param list_of_names:
        :param list_of_types:

        :return: DataFrame
        """

        # parameter type: list
        item_list = []
        for i in range(len(list_of_codes)):
            if list_of_types is None:
                print("Length of names and types does not match. Please input the list_of_types")
                etf_data = RollingProcessor.get_ETF_data(list_of_codes[i])
                etf_data = etf_data[['EndDate', 'AccumulatedUnitNV']].set_index('EndDate')
                etf_data = etf_data.rename(columns={'AccumulatedUnitNV': f"etf_{str(list_of_codes[i])}"})
                etf_data = etf_data[~etf_data.index.duplicated(keep='first')]
                item_list.append(etf_data)
            else:
                if list_of_types[i] == 'INDEX':
                    index_data = RollingProcessor.get_index_data(list_of_codes[i])
                    index_data = index_data[['TradingDay', 'ClosePrice']].set_index('TradingDay')
                    index_data = index_data.rename(columns={'ClosePrice': f"index_{str(list_of_codes[i])}"})
                    index_data = index_data[~etf_data.index.duplicated(keep='first')]
                    item_list.append(index_data)
                if list_of_types[i] == 'ETF':
                    etf_data = RollingProcessor.get_ETF_data(list_of_codes[i])
                    etf_data = etf_data[['EndDate', 'AccumulatedUnitNV']].set_index('EndDate')
                    etf_data = etf_data.rename(columns={'AccumulatedUnitNV': f"etf_{str(list_of_codes[i])}"})
                    etf_data = etf_data[~etf_data.index.duplicated(keep='first')]
                    item_list.append(etf_data)

        longest_length = len(item_list[0])
        longest_index = 0

        # Find the longest DataFrame
        list_of_index = []
        for i, df in enumerate(item_list):
            list_of_index.append(i)
            if len(df) > longest_length:
                longest_length = len(df)
                longest_index = i

        # Align all length based on the shortest DF
        data = item_list[longest_index]
        for i, df in enumerate(item_list):
            if i == longest_index:
                continue
            data = pd.concat([data, df], axis=1)
        data.index = pd.to_datetime(data.index)
        data = data

        # 计算每日涨跌幅和滚动收益率
        for column in data.columns:
            data[f'{column}_day_return'] = data[column].rolling(2).apply(lambda x: (x[-1] - x[0]) / x[0])
        data = data[1:]
        return data

    @staticmethod
    # Produce an optimized table with max returns from all windows in the window lists
    def rolling_optimization_table(window_list, data, target_num):
        """
        :param window_list:
        :param data:
        :param target_num:

        :return: data_table
        """
        # parameter type: list, DataFrame, int

        data_table = pd.DataFrame()

        for i in window_list:
            data_x = data.copy()
            signal_name = []

            for num, name in enumerate(data_x.columns[target_num:]):
                data_x[f'{name}_{str(i)}'] = data_x[data_x.columns[num]].rolling(window=i).apply(lambda x: x[-1] / x[0])
                signal_name.append(f'signal_{data_x.columns[num][4:]}')

            data_x = data_x.iloc[i:, :]
            for name in signal_name:
                data_x[name] = [0] * len(data_x)

            for row in range(len(data_x)):
                value = data_x.iloc[row][2 * target_num:3 * target_num].max()
                series = data_x.iloc[row]
                try:
                    position = series[2 * target_num:3 * target_num][series == value].idxmax()
                except:
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + target_num
                data_x.loc[data_x.index[row], series.index[new_position]] = 1

            for name in data_x.columns[-target_num:]:
                data_x[name] = data_x[name].shift(1)

            data_x = data_x[1:]

            data_x[f'signal_return_{str(i)}'] = [0] * len(data_x)
            arr = []
            X = target_num
            for row in range(len(data_x)):
                value = data_x.iloc[row][3 * X:4 * X].max()
                series = data_x.iloc[row]
                ## find the position of signal with 1
                position = series[3 * X:4 * X][series == value].idxmax()
                old_position = series.index.get_loc(position)
                new_position = old_position - 2 * X
                arr.append(series[new_position] + 1)

            data_x['arr'] = arr
            data_x[f'signal_return_{str(i)}'] = np.cumprod(arr, axis=0) * 100

            data_x[f'signal_return_{str(i)}'] = (100 / data_x[f'signal_return_{str(i)}'].values[0]) * data_x[
                f'signal_return_{str(i)}']
            data_table[f'signal_return_{str(i)}'] = data_x[f'signal_return_{str(i)}']
        return data_table

    @staticmethod
    # Produce a single table with name, day_return, window_return, signal_return
    def single_window_table(window_list, data, target_num):
        """
        :param window_list:
        :param data:
        :param target_num:

        :return: data_list
        """
        # parameter type: list, DataFrame, int

        data_list = []

        for i in window_list:
            data_x = data.copy()
            signal_name = []

            for num, name in enumerate(data_x.columns[target_num:]):
                data_x[f'{name}_{str(i)}'] = data_x[data_x.columns[num]].rolling(window=i).apply(lambda x: x[-1] / x[0])
                signal_name.append(f'signal_{data_x.columns[num][4:]}')

            data_x = data_x.iloc[i:, :]
            for name in signal_name:
                data_x[name] = [0] * len(data_x)

            for row in range(len(data_x)):
                value = data_x.iloc[row][2 * target_num:3 * target_num].max()
                series = data_x.iloc[row]
                try:
                    position = series[2 * target_num:3 * target_num][series == value].idxmax()
                except:
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + target_num
                data_x.loc[data_x.index[row], series.index[new_position]] = 1

            for name in data_x.columns[-target_num:]:
                data_x[name] = data_x[name].shift(1)

            data_x = data_x[1:]

            data_x[f'signal_return_{str(i)}'] = [0] * len(data_x)
            arr = []
            X = target_num
            for row in range(len(data_x)):
                value = data_x.iloc[row][3 * X:4 * X].max()
                series = data_x.iloc[row]
                ## find the position of signal with 1
                position = series[3 * X:4 * X][series == value].idxmax()
                old_position = series.index.get_loc(position)
                new_position = old_position - 2 * X
                arr.append(series[new_position] + 1)

            data_x['arr'] = arr
            data_x[f'signal_return_{str(i)}'] = np.cumprod(arr, axis=0) * 100

            data_x[f'signal_return_{str(i)}'] = (100 / data_x[f'signal_return_{str(i)}'].values[0]) * data_x[
                f'signal_return_{str(i)}']
        return data_x

    @staticmethod
    # Return the analysis of columns within the table
    def table_interpret(data_table):
        """
        :param data_table: DataFrame
        :return: data_table
        """
        # Calculate strategy performance metrics
        result = {}
        metrics = ["avg_return", "volatility", "sharpe", "max_drawdown", "win_rate"]
        for column in data_table.columns:
            result[column] = []
            for metric in metrics:
                metric_function = getattr(qs.stats, metric)
                r = metric_function(data_table[column])
                result[column].append(r)

        result = pd.DataFrame(result, index=metrics)

        print(result)

        # Plot performance curves
        _ = plt.figure(figsize=(15, 10))
        plt.xlabel('Time')
        plt.ylabel('Return')

        legend_list = []
        for name in data_table.columns:
            _ = plt.plot(np.array(data_table[name].index), data_table[name].values)
            legend_list.append(f'windows{name[14:]}_{result.loc["sharpe", name]}')
        _ = plt.legend(legend_list)

        ax = plt.gca()
        plt.title(f'Different Rolling Window size with their Sharpe Ratio on {len(data_table)} days since 2021-06',
                  fontname='Arial', fontsize=21)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(int(len(data_table) / 4)))

    @staticmethod
    # Return the Perfromance Table with list_of_codes and window_list
    def performance_test_equal_period(list_of_codes, window_list):
        """
        :param list_of_codes:['512480', '159865', '588080', '515790', '515080']
        :param window_list:[3,6,9,12,15,18,21,24,27]

        """
        # parameter type: list, list

        data = RollingProcessor.get_data_full_period(list_of_codes)
        target_num = len(list_of_codes)
        data_table = RollingProcessor.rolling_optimization_table(window_list, data, target_num)
        RollingProcessor.table_interpret(data_table)
        return None

    @staticmethod
    # Run loops with randomized combo to find optimal window size for each
    def trial_optimization(list_of_all_codes, window_list, attempts_time, sample_size):
        """
        :param list_of_all_codes:['512480', '159869', '512170', '512690', '512980',
                                '516950','159865', '588080', '515790', '515080']
        :param window_list:[3,6,9,12,15,18,21,24,27]
        :param attempts_time:100

        """
        # parameter type: list, list
        winner_codes = []
        bar_return = 100
        unique_codes = []
        for i in range(attempts_time):
            random_codes = random.sample(list_of_all_codes, sample_size)

            # Check if the new random_codes are already present in the set of unique codes
            n = 0
            while random_codes in unique_codes:
                random_codes = random.sample(list_of_all_codes, sample_size)
                n += 1
                if n > 20:
                    break

            # Add the new random_codes to the set of unique codes
            unique_codes.append(random_codes)

            data = RollingProcessor.get_data_full_period(random_codes)
            data_table = RollingProcessor.rolling_optimization_table(window_list, data, int(len(data.columns) / 2))
            last_row = data_table.iloc[-1]
            largest_value = last_row.max()
            largest_column = last_row.idxmax()
            #         print(f'Áttempt No.{i+1} on combination {random_codes} got {round(largest_value,3)} return at Windows={largest_column[14:]}')
            if largest_value > bar_return:
                print(f'HOORAY! The new bar is set at {round(largest_value, 3)}!!')
                bar_return = largest_value
                bar_column = largest_column
                winner_codes = random_codes
        print('***************************************************************************************')
        print('**********************************Result***********************************************')
        print(
            f'\n1)The optimized combination is {winner_codes}\n2)The max return from it is {bar_return} at  Windows={bar_column[14:]}')
        return None

    @staticmethod
    # Run loops with randomized combo to rank them by correlation from low to high
    def corr_test(list_of_all_codes, attempts_time, sample_size, display_num, window_list, method, comparison_target):
        """

        :param list_of_all_codes:['512480', '159869', '512170', '512690', '512980']
        :param attempts_time:1000
        :param sample_size:2
        :param display_num:100
        :param window_list:[3,6,9,12,15,18,21,24,27]
        :param method: 1. 'equal_peiord_csv'; 2. 'equal_peiord_sql'; 3. 'full_period'

        :return the top No. of combo with lowest correlation and their corresponding return, Sharpe, and Windows
        """
        # parameter type: list, list
        winner_corr = {}
        unique_codes = []

        for i in range(attempts_time):
            random_codes = random.sample(list_of_all_codes, sample_size)

            # Check if the new random_codes are already present in the set of unique codes
            n = 0
            while random_codes in unique_codes:

                random_codes = random.sample(list_of_all_codes, sample_size)
                n += 1
                if n > 10:
                    break

            # Add the new random_codes to the set of unique codes
            unique_codes.append(random_codes)

            data = RollingProcessor.get_data_full_period(random_codes)
            if method == 'equal_peiord_csv':
                data = RollingProcessor.get_data_euqal_period(random_codes)
            if method == 'equal_peiord_sql':
                data = RollingProcessor.get_data_euqal_period_new(random_codes)

            # Price Correlation
            data = data.iloc[1:, :].iloc[:, :sample_size]
            correlation = data[data.columns[0]].corr(data[data.columns[1]])

            # Return Correlation
            if comparison_target == 'Return':
                data = data.iloc[1:, :].iloc[:, sample_size:]
                correlation = data[data.columns[0]].corr(data[data.columns[1]])

            if len(winner_corr) > display_num - 1:
                if correlation > list(winner_corr.keys())[-1]:
                    continue
                if correlation is None:
                    continue
                # Remove the last key-value pair
                winner_corr.popitem()
            winner_corr[correlation] = random_codes
            # Sort the dictionary based on keys
            winner_corr = dict(sorted(winner_corr.items()))
        print('********************************************************************************')
        print(f'Here are the Top {len(winner_corr.keys())} Combination with the lowest correlation score')

        plot_corr = []
        plot_Sharpe = []
        plot_return = []
        for rank, correlation_score in enumerate(winner_corr.keys()):

            data = RollingProcessor.get_data_full_period(winner_corr[correlation_score])
            if method == 'equal_peiord_csv':
                data = RollingProcessor.get_data_euqal_period(winner_corr[correlation_score])
            if method == 'equal_peiord_sql':
                data = RollingProcessor.get_data_euqal_period_new(winner_corr[correlation_score])

            data_table = RollingProcessor.rolling_optimization_table(window_list, data, 2)
            last_row = data_table.iloc[-1]
            largest_value = last_row.max()
            largest_column = last_row.idxmax()
            plot_corr.append(round(correlation_score, 3))
            plot_return.append(round(largest_value, 3))

            # Calculate strategy performance metrics
            result = {}
            metrics = ["avg_return", "volatility", "sharpe", "max_drawdown", "win_rate"]

            for column in data_table.columns:
                result[column] = []
                for metric in metrics:
                    metric_function = getattr(qs.stats, metric)
                    r = metric_function(data_table[column])
                    result[column].append(r)

            result = pd.DataFrame(result, index=metrics)
            plot_Sharpe.append(round(result.loc["sharpe", largest_column], 3))

            print(
                f'No.{rank + 1}: {winner_corr[correlation_score]} with a score of {round(correlation_score, 3)} and its max return is {round(largest_value, 3)} with Sharpe {round(result.loc["sharpe", largest_column], 3)} at Windows={largest_column[14:]}')

        fig, axs = plt.subplots(1, 2, figsize=(15, 8))

        # Left subplot
        axs[0].scatter(plot_corr, plot_Sharpe)
        axs[0].set_title(f'Correlation vs. Sharpe on Windows={largest_column[14:]}')

        # Right subplot
        axs[1].scatter(plot_corr, plot_return)
        axs[1].set_title(f'Correlation vs. Return on Windows={largest_column[14:]}')

        plt.show()

        return None

    ##
    ##############
    # MA strategy
    @staticmethod
    def single_window_table_MA_Return(window_list, data, target_num):
        """
        :param window_list:
        :param data:
        :param target_num:

        :return: data_list
        """
        # parameter type: list, DataFrame, int

        data_list = []

        for i in window_list:
            data_x = data.copy()
            signal_name = []

            for num, name in enumerate(data_x.columns[target_num:]):
                data_x[f'{name}_{str(i)}'] = data_x[data_x.columns[num]].rolling(window=i)
                list_of_codes = ['510050', '510300', '159915', '159901', '588000']
                signal_name.append(f'signal_{data_x.columns[num][4:]}')

            data_x = data_x.iloc[i:, :]
            for name in signal_name:
                data_x[name] = [0] * len(data_x)

            for row in range(len(data_x)):
                value = data_x.iloc[row][2 * target_num:3 * target_num].max()
                series = data_x.iloc[row]
                try:
                    position = series[2 * target_num:3 * target_num][series == value].idxmax()
                except:
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + target_num
                data_x.loc[data_x.index[row], series.index[new_position]] = 1

            for name in data_x.columns[-target_num:]:
                data_x[name] = data_x[name].shift(1)

            data_x = data_x[1:]

            data_x[f'signal_return_{str(i)}'] = [0] * len(data_x)
            arr = []
            X = target_num
            for row in range(len(data_x)):
                value = data_x.iloc[row][3 * X:4 * X].max()
                series = data_x.iloc[row]
                ## find the position of signal with 1
                position = series[3 * X:4 * X][series == value].idxmax()
                old_position = series.index.get_loc(position)
                new_position = old_position - 2 * X
                arr.append(series[new_position] + 1)

            data_x['arr'] = arr
            data_x[f'signal_return_{str(i)}'] = np.cumprod(arr, axis=0) * 100

            data_x[f'signal_return_{str(i)}'] = (100 / data_x[f'signal_return_{str(i)}'].values[0]) * data_x[
                f'signal_return_{str(i)}']
        return data_x

    @staticmethod
    def rolling_optimization_table_MA(window_list, data, target_num):
        """
        :param window_list:
        :param data:
        :param target_num:

        :return: data_table
        """
        # parameter type: list, DataFrame, int

        data_table = pd.DataFrame()

        for i in window_list:
            data_x = data.copy()
            signal_name = []

            for num, name in enumerate(data_x.columns[target_num:]):
                data_x[f'{name}_{str(i)}'] = data_x[data_x.columns[num]].rolling(window=i).mean().pct_change()
                signal_name.append(f'signal_{data_x.columns[num][4:]}')

            data_x = data_x.iloc[i:, :]
            for name in signal_name:
                data_x[name] = [0] * len(data_x)

            for row in range(len(data_x)):
                value = data_x.iloc[row][2 * target_num:3 * target_num].max()
                series = data_x.iloc[row]
                try:
                    position = series[2 * target_num:3 * target_num][series == value].idxmax()
                except:
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + target_num
                data_x.loc[data_x.index[row], series.index[new_position]] = 1

            for name in data_x.columns[-target_num:]:
                data_x[name] = data_x[name].shift(1)

            data_x = data_x[1:]

            data_x[f'signal_return_{str(i)}'] = [0] * len(data_x)
            arr = []
            X = target_num
            for row in range(len(data_x)):
                value = data_x.iloc[row][3 * X:4 * X].max()
                series = data_x.iloc[row]
                ## find the position of signal with 1
                position = series[3 * X:4 * X][series == value].idxmax()
                old_position = series.index.get_loc(position)
                new_position = old_position - 2 * X
                arr.append(series[new_position] + 1)

            data_x['arr'] = arr
            data_x[f'signal_return_{str(i)}'] = np.cumprod(arr, axis=0) * 100

            data_x[f'signal_return_{str(i)}'] = (100 / data_x[f'signal_return_{str(i)}'].values[0]) * data_x[
                f'signal_return_{str(i)}']
            data_table[f'signal_return_{str(i)}'] = data_x[f'signal_return_{str(i)}']
        return data_table

    ##
    ###############################
    # Testing for conditions
    @staticmethod
    def test_better_return(list_of_all_codes, attempts_time, sample_size, display_num, window_list, method,
                           comparison_target):
        """

        :param list_of_all_codes:['512480', '159869', '512170', '512690', '512980']
        :param attempts_time:1000
        :param sample_size:2
        :param display_num:100
        :param window_list:[3,6,9,12,15,18,21,24,27]
        :param method: 1. 'equal_peiord_csv'; 2. 'equal_peiord_sql'; 3. 'full_period'

        :return the top No. of combo with lowest correlation and their corresponding return, Sharpe, and Windows
        """
        # parameter type: list, list
        winner_corr = {}
        unique_codes = []
        modified_R = 0
        original_R = 0
        for i in range(attempts_time):
            random_codes = random.sample(list_of_all_codes, sample_size)

            # Check if the new random_codes are already present in the set of unique codes
            n = 0
            while random_codes in unique_codes:

                random_codes = random.sample(list_of_all_codes, sample_size)
                n += 1
                if n > 10:
                    break

            # Add the new random_codes to the set of unique codes
            unique_codes.append(random_codes)

            data = RollingProcessor.get_data_full_period(random_codes)
            if method == 'equal_peiord_csv':
                data = RollingProcessor.get_data_euqal_period(random_codes)
            if method == 'equal_peiord_sql':
                data = RollingProcessor.get_data_euqal_period_new(random_codes)

            # Price Correlation
            data_P = data.iloc[1:, :].iloc[:, :sample_size]
            correlation = data_P[data_P.columns[0]].corr(data_P[data_P.columns[1]])

            # Return Correlation
            if comparison_target == 'Return':
                data_R = data.iloc[1:, :].iloc[:, sample_size:]
                correlation = data_R[data_R.columns[0]].corr(data_R[data_R.columns[1]])

            if len(winner_corr) > display_num - 1:
                if correlation > list(winner_corr.keys())[-1]:
                    continue
                if correlation is None:
                    continue
                # Remove the last key-value pair
                winner_corr.popitem()
            winner_corr[correlation] = random_codes
            # Sort the dictionary based on keys
            winner_corr = dict(sorted(winner_corr.items()))
        print('********************************************************************************')
        print(f'Here are the Top {len(winner_corr.keys())} Combination with the lowest correlation score')

        for rank, correlation_score in enumerate(winner_corr.keys()):

            data = RollingProcessor.get_data_full_period(winner_corr[correlation_score])
            if method == 'equal_peiord_csv':
                data = RollingProcessor.get_data_euqal_period(winner_corr[correlation_score])
            if method == 'equal_peiord_sql':
                data = RollingProcessor.get_data_euqal_period_new(winner_corr[correlation_score])

            data_table = RollingProcessor.rolling_optimization_table(window_list, data, 2)
            data_table_MA = RollingProcessor.rolling_optimization_table_MA(window_list, data, 2)

            last_row = data_table.iloc[-1]
            largest_value = last_row.max()
            largest_column = last_row.idxmax()

            last_row_MA = data_table_MA.iloc[-1]
            largest_value_MA = last_row_MA.max()
            largest_column_MA = last_row_MA.idxmax()

            # Calculate strategy performance metrics
            result = {}
            metrics = ["avg_return", "volatility", "sharpe", "max_drawdown", "win_rate"]

            for column in data_table.columns:
                result[column] = []
                for metric in metrics:
                    metric_function = getattr(qs.stats, metric)
                    r = metric_function(data_table[column])
                    result[column].append(r)

            result = pd.DataFrame(result, index=metrics)

            # Calculate strategy performance metrics on MA return Strat
            result_MA = {}
            metrics_MA = ["avg_return", "volatility", "sharpe", "max_drawdown", "win_rate"]

            for column_MA in data_table_MA.columns:
                result_MA[column_MA] = []
                for metric_MA in metrics_MA:
                    metric_function_MA = getattr(qs.stats, metric_MA)
                    r_MA = metric_function_MA(data_table_MA[column_MA])
                    result_MA[column_MA].append(r_MA)

            result_MA = pd.DataFrame(result_MA, index=metrics_MA)

            if largest_value_MA > largest_value:
                modified_R += 1
                print(
                    f'No.{rank + 1}: {winner_corr[correlation_score]} perform better on modified MA Strategy with a max return of {round(largest_value_MA, 3)} with Sharpe {round(result_MA.loc["sharpe", largest_column_MA], 3)} at Windows={largest_column_MA[14:]}')
                print(
                    f'     The original strategy Strategy has a max return {round(largest_value, 3)} with Sharpe {round(result.loc["sharpe", largest_column], 3)} at Windows={largest_column[14:]}')
                print(
                    '************************************************************************************************************')
            if largest_value > largest_value_MA:
                original_R += 1
                print(
                    f'No.{rank + 1}: {winner_corr[correlation_score]} perform better on the original Strategy with a max return {round(largest_value, 3)} with Sharpe {round(result.loc["sharpe", largest_column], 3)} at Windows={largest_column[14:]}')
                print(
                    f'     The modified MA strategy Strategy has a max return {round(largest_value_MA, 3)} with Sharpe {round(result_MA.loc["sharpe", largest_column_MA], 3)} at Windows={largest_column_MA[14:]}')
                print(
                    '************************************************************************************************************')

        print(
            '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(
            f'Given {len(winner_corr.keys())} combinations with lowest correlation score \n We found {modified_R} groups perform better on MA Strat and {original_R} perform better on the original Start')
        return None

    @staticmethod
    # Test on Variance Spread
    def test_stability(list_of_all_codes, attempts_time, sample_size, display_num, window_list, method,
                       comparison_target):
        """

        :param list_of_all_codes:['512480', '159869', '512170', '512690', '512980']
        :param attempts_time:1000
        :param sample_size:2
        :param display_num:100
        :param window_list:[3,6,9,12,15,18,21,24,27]
        :param method: 1. 'equal_peiord_csv'; 2. 'equal_peiord_sql'; 3. 'full_period'

        :return the top No. of combo with lowest correlation and their corresponding return, Sharpe, and Windows
        """
        # parameter type: list, list
        winner_corr = {}
        unique_codes = []
        modified_R = 0
        original_R = 0
        modified_R_v = []
        original_R_v = []
        for i in range(attempts_time):
            random_codes = random.sample(list_of_all_codes, sample_size)

            # Check if the new random_codes are already present in the set of unique codes
            n = 0
            while random_codes in unique_codes:

                random_codes = random.sample(list_of_all_codes, sample_size)
                n += 1
                if n > 10:
                    break

            # Add the new random_codes to the set of unique codes
            unique_codes.append(random_codes)

            data = RollingProcessor.get_data_full_period(random_codes)
            if method == 'equal_peiord_csv':
                data = RollingProcessor.get_data_euqal_period(random_codes)
            if method == 'equal_peiord_sql':
                data = RollingProcessor.get_data_euqal_period_new(random_codes)

            # Price Correlation
            data_P = data.iloc[1:, :].iloc[:, :sample_size]
            correlation = data_P[data_P.columns[0]].corr(data_P[data_P.columns[1]])

            # Return Correlation
            if comparison_target == 'Return':
                data_R = data.iloc[1:, :].iloc[:, sample_size:]
                correlation = data_R[data_R.columns[0]].corr(data_R[data_R.columns[1]])

            if len(winner_corr) > display_num - 1:
                if correlation > list(winner_corr.keys())[-1]:
                    continue
                if correlation is None:
                    continue
                # Remove the last key-value pair
                winner_corr.popitem()
            winner_corr[correlation] = random_codes
            # Sort the dictionary based on keys
            winner_corr = dict(sorted(winner_corr.items()))
        print('********************************************************************************')
        print(f'Here are the Top {len(winner_corr.keys())} Combination with the lowest correlation score')

        for rank, correlation_score in enumerate(winner_corr.keys()):

            data = RollingProcessor.get_data_full_period(winner_corr[correlation_score])
            if method == 'equal_peiord_csv':
                data = RollingProcessor.get_data_euqal_period(winner_corr[correlation_score])
            if method == 'equal_peiord_sql':
                data = RollingProcessor.get_data_euqal_period_new(winner_corr[correlation_score])

            data_table = RollingProcessor.rolling_optimization_table(window_list, data, 2)
            data_table_MA = RollingProcessor.rolling_optimization_table_MA(window_list, data, 2)

            last_row = data_table.iloc[-1]
            largest_value = last_row.max()
            largest_column = last_row.idxmax()
            #         sample_mean = np.mean(last_row)
            #         sample_var = 0
            #         for i in last_row:
            #             value = (i - sample_mean)**2
            #             sample_var += value
            #         sample_var = sample_var/(len(last_row)-1)

            last_row_MA = data_table_MA.iloc[-1]
            largest_value_MA = last_row_MA.max()
            largest_column_MA = last_row_MA.idxmax()
            #         sample_mean_MA = np.mean(last_row_MA)
            #         sample_var_MA = 0
            #         for i in last_row_MA:
            #             value = (i - sample_mean_MA)**2
            #             sample_var_MA += value
            #         sample_var_MA = sample_var_MA/(len(last_row_MA)-1)

            # Calculate strategy performance metrics
            result = {}
            metrics = ["avg_return", "volatility", "sharpe", "max_drawdown", "win_rate"]

            for column in data_table.columns:
                result[column] = []
                for metric in metrics:
                    metric_function = getattr(qs.stats, metric)
                    r = metric_function(data_table[column])
                    result[column].append(r)

            result = pd.DataFrame(result, index=metrics)
            old_var = result.iloc[2, :].var()
            print('________________________________________________________________')

            # Calculate strategy performance metrics on MA return Strat
            result_MA = {}
            metrics_MA = ["avg_return", "volatility", "sharpe", "max_drawdown", "win_rate"]

            for column_MA in data_table_MA.columns:
                result_MA[column_MA] = []
                for metric_MA in metrics_MA:
                    metric_function_MA = getattr(qs.stats, metric_MA)
                    r_MA = metric_function_MA(data_table_MA[column_MA])
                    result_MA[column_MA].append(r_MA)

            result_MA = pd.DataFrame(result_MA, index=metrics_MA)
            new_var = result_MA.iloc[2, :].var()

            if new_var < old_var:
                modified_R += 1
                difference = old_var - new_var
                modified_R_v.append(difference)
                print(f'No.{rank + 1}: {winner_corr[correlation_score]} with a lower variance on MA Strat')
                print(f'1) Old Strategy return has a variance of {old_var}\n{last_row}')
                print(f'2) MA Strategy return has a variance of {new_var}\n{last_row_MA}')
                print(
                    '************************************************************************************************************')

            if old_var < new_var:
                original_R += 1
                difference = new_var - old_var
                original_R_v.append(difference)
                print(f'No.{rank + 1}: {winner_corr[correlation_score]} with a lower variance on OLD Strat')
                print(f'1) Old Strategy return has a variance of {old_var}\n{last_row}')
                print(f'2) MA Strategy return has a variance of {new_var}\n{last_row_MA}')
                print(
                    '************************************************************************************************************')

        print(
            '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(
            f'Given {len(winner_corr.keys())} combinations with lowest correlation score \n We found {modified_R} groups shows a smaller variance on MA Strat with an average variance difference of {sum(modified_R_v) / len(modified_R_v)} and {original_R} groups shows a smaller variance on the original Start with an average variance difference of {sum(original_R_v) / len(original_R_v)}')
        return None

    @staticmethod
    # Test on Variance Spread
    def test_stability_ZONE(list_of_all_codes, attempts_time, sample_size, display_num, window_list, method,
                            comparison_target):
        """

        :param list_of_all_codes:['512480', '159869', '512170', '512690', '512980']
        :param attempts_time:1000
        :param sample_size:2
        :param display_num:100
        :param window_list:[3,6,9,12,15,18,21,24,27]
        :param method: 1. 'equal_peiord_csv'; 2. 'equal_peiord_sql'; 3. 'full_period'

        :return the top No. of combo with lowest correlation and their corresponding return, Sharpe, and Windows
        """
        # parameter type: list, list
        winner_corr = {}
        unique_codes = []
        modified_R = 0
        original_R = 0
        modified_R_v = []
        original_R_v = []
        for i in range(attempts_time):
            random_codes = random.sample(list_of_all_codes, sample_size)

            # Check if the new random_codes are already present in the set of unique codes
            n = 0
            while random_codes in unique_codes:

                random_codes = random.sample(list_of_all_codes, sample_size)
                n += 1
                if n > 10:
                    break

            # Add the new random_codes to the set of unique codes
            unique_codes.append(random_codes)

            data = RollingProcessor.get_data_full_period(random_codes)
            if method == 'equal_peiord_csv':
                data = RollingProcessor.get_data_euqal_period(random_codes)
            if method == 'equal_peiord_sql':
                data = RollingProcessor.get_data_euqal_period_new(random_codes)

            # Price Correlation
            data_P = data.iloc[1:, :].iloc[:, :sample_size]
            correlation = data_P[data_P.columns[0]].corr(data_P[data_P.columns[1]])

            # Return Correlation
            if comparison_target == 'Return':
                data_R = data.iloc[1:, :].iloc[:, sample_size:]
                correlation = data_R[data_R.columns[0]].corr(data_R[data_R.columns[1]])

            if len(winner_corr) > display_num - 1:
                if correlation > list(winner_corr.keys())[-1]:
                    continue
                if correlation is None:
                    continue
                if correlation > 0.5:
                    continue
                # Remove the last key-value pair
                winner_corr.popitem()
            winner_corr[correlation] = random_codes
            # Sort the dictionary based on keys
            winner_corr = dict(sorted(winner_corr.items()))
        print('********************************************************************************')
        print(f'Here are the Top {len(winner_corr.keys())} Combination with the lowest correlation score')

        for rank, correlation_score in enumerate(winner_corr.keys()):

            data = RollingProcessor.get_data_full_period(winner_corr[correlation_score])
            if method == 'equal_peiord_csv':
                data = RollingProcessor.get_data_euqal_period(winner_corr[correlation_score])
            if method == 'equal_peiord_sql':
                data = RollingProcessor.get_data_euqal_period_new(winner_corr[correlation_score])

            data_table = RollingProcessor.rolling_optimization_table(window_list, data, 2)
            data_table_MA = RollingProcessor.rolling_optimization_table_MA(window_list, data, 2)

            last_row = data_table.iloc[-1]
            largest_value = last_row.max()
            largest_column = last_row.idxmax()
            #         sample_mean = np.mean(last_row)
            #         sample_var = 0
            #         for i in last_row:
            #             value = (i - sample_mean)**2
            #             sample_var += value
            #         sample_var = sample_var/(len(last_row)-1)

            last_row_MA = data_table_MA.iloc[-1]
            largest_value_MA = last_row_MA.max()
            largest_column_MA = last_row_MA.idxmax()
            #         sample_mean_MA = np.mean(last_row_MA)
            #         sample_var_MA = 0
            #         for i in last_row_MA:
            #             value = (i - sample_mean_MA)**2
            #             sample_var_MA += value
            #         sample_var_MA = sample_var_MA/(len(last_row_MA)-1)

            # Calculate strategy performance metrics
            result = {}
            metrics = ["avg_return", "volatility", "sharpe", "max_drawdown", "win_rate"]

            for column in data_table.columns:
                result[column] = []
                for metric in metrics:
                    metric_function = getattr(qs.stats, metric)
                    r = metric_function(data_table[column])
                    result[column].append(r)

            result = pd.DataFrame(result, index=metrics)
            old_var = result.iloc[2, :].var()
            print('________________________________________________________________')

            # Calculate strategy performance metrics on MA return Strat
            result_MA = {}
            metrics_MA = ["avg_return", "volatility", "sharpe", "max_drawdown", "win_rate"]

            for column_MA in data_table_MA.columns:
                result_MA[column_MA] = []
                for metric_MA in metrics_MA:
                    metric_function_MA = getattr(qs.stats, metric_MA)
                    r_MA = metric_function_MA(data_table_MA[column_MA])
                    result_MA[column_MA].append(r_MA)

            result_MA = pd.DataFrame(result_MA, index=metrics_MA)
            new_var = result_MA.iloc[2, :].var()

            if max(result_MA.iloc[2, :].mean(), result.iloc[2, :].mean()) < 0.5:
                continue

            if new_var < old_var:
                modified_R += 1
                difference = old_var - new_var
                modified_R_v.append(difference)
                print(f'No.{rank + 1}: {winner_corr[correlation_score]} with a lower variance on MA Strat')
                print(f'1) Old Strategy return has a variance of {old_var}\n{last_row}')
                print(f'2) MA Strategy return has a variance of {new_var}\n{last_row_MA}')
                print(
                    '************************************************************************************************************')

            if old_var < new_var:
                original_R += 1
                difference = new_var - old_var
                original_R_v.append(difference)
                print(f'No.{rank + 1}: {winner_corr[correlation_score]} with a lower variance on OLD Strat')
                print(f'1) Old Strategy return has a variance of {old_var}\n{last_row}')
                print(f'2) MA Strategy return has a variance of {new_var}\n{last_row_MA}')
                print(
                    '************************************************************************************************************')

        print(
            '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(
            f'Given {len(winner_corr.keys())} combinations with lowest correlation score \n We found {modified_R} groups shows a smaller variance on MA Strat with an average variance difference of {sum(modified_R_v) / len(modified_R_v)} and {original_R} groups shows a smaller variance on the original Start with an average variance difference of {sum(original_R_v) / len(original_R_v)}')
        return None


    ##
    ##########################
    # RSI
    @staticmethod
    def apply_RSI(data_table, list_of_codes, window_list):
        ## Apply RSI
        for name in data_table.iloc[:, :len(list_of_codes)].columns:
            data_table[f'{name}_RSI'] = rsi(data_table[name], window=window_list[0])
        data_table = data_table.iloc[window_list[0]:, :]
        return data_table

    @staticmethod
    def RSI(window_list, data, target_num, list_of_codes):
        """
        :param window_list:
        :param data:
        :param target_num:

        :return: data_list
        """
        # parameter type: list, DataFrame, int

        data_list = []

        for i in window_list:
            data_x = data.copy()
            _data = data.copy()
            data_RSI = RollingProcessor.apply_RSI(_data, list_of_codes, window_list)

            signal_name = []
            for num, name in enumerate(data_x.columns[target_num:]):
                data_x[f'{name}_{str(i)}'] = data_x[data_x.columns[num]].rolling(window=i).apply(lambda x: x[-1] / x[0])
                signal_name.append(f'signal_{data_x.columns[num][4:]}')

            data_x = data_x.iloc[i:, :]
            for name in signal_name:
                data_x[name] = [0] * len(data_x)

            ############################################## Modified Zone for RSI ##################################################
            for row in range(len(data_x)):
                a = data_x.iloc[row][2 * target_num:3 * target_num]
                b = data_RSI.iloc[row][2 * target_num:3 * target_num]

                RSI_dict = {}
                for item1, item2 in zip(a, b):
                    RSI_dict[item1] = item2

                target = max(RSI_dict)

                if RSI_dict[target] >= 70:
                    if min(RSI_dict.values()) >= 70:
                        target = target
                    else:
                        target = next(
                            (key for key, value in sorted(RSI_dict.items(), key=lambda x: x[1], reverse=True) if
                             value < 70), None)
                series = data_x.iloc[row]
                value_index = series[series == target].index.tolist()
                try:
                    position = value_index[0]
                except:
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + target_num
                data_x.loc[data_x.index[row], series.index[new_position]] = 1

            #############################################################################################################

            for name in data_x.columns[-target_num:]:
                data_x[name] = data_x[name].shift(1)

            data_x = data_x[1:]

            data_x[f'signal_return_{str(i)}'] = [0] * len(data_x)
            arr = []
            X = target_num
            for row in range(len(data_x)):
                value = data_x.iloc[row][3 * X:4 * X].max()
                series = data_x.iloc[row]
                ## find the position of signal with 1
                position = series[3 * X:4 * X][series == value].idxmax()
                old_position = series.index.get_loc(position)
                new_position = old_position - 2 * X
                arr.append(series[new_position] + 1)

            data_x['arr'] = arr
            data_x[f'signal_return_{str(i)}'] = np.cumprod(arr, axis=0) * 100

            data_x[f'signal_return_{str(i)}'] = (100 / data_x[f'signal_return_{str(i)}'].values[0]) * data_x[
                f'signal_return_{str(i)}']
        return data_x

    @staticmethod
    def test_RSI_return(list_of_all_codes, attempts_time, sample_size, display_num, window_list, method,
                        comparison_target, target_num):
        """

        :param list_of_all_codes:['512480', '159869', '512170', '512690', '512980']
        :param attempts_time:1000
        :param sample_size:2
        :param display_num:100
        :param window_list:[3,6,9,12,15,18,21,24,27]
        :param method: 1. 'equal_peiord_csv'; 2. 'equal_peiord_sql'; 3. 'full_period'

        :return the top No. of combo with lowest correlation and their corresponding return, Sharpe, and Windows
        """
        # parameter type: list, list
        winner_corr = {}
        unique_codes = []
        modified_R = 0
        original_R = 0
        improvements = []
        difference = []
        for i in range(attempts_time):
            print(f'Attempt{i}')
            random_codes = random.sample(list_of_all_codes, sample_size)

            # Check if the new random_codes are already present in the set of unique codes
            n = 0
            while random_codes in unique_codes:

                random_codes = random.sample(list_of_all_codes, sample_size)
                n += 1
                if n > 10:
                    break

            # Add the new random_codes to the set of unique codes
            unique_codes.append(random_codes)

            data = RollingProcessor.get_data_full_period(random_codes)
            if method == 'equal_peiord_csv':
                data = RollingProcessor.get_data_euqal_period(random_codes)
            if method == 'equal_peiord_sql':
                data = RollingProcessor.get_data_euqal_period_new(random_codes)

            # Price Correlation
            data_P = data.iloc[1:, :].iloc[:, :sample_size]
            correlation = data_P[data_P.columns[0]].corr(data_P[data_P.columns[1]])

            # Return Correlation
            if comparison_target == 'Return':
                data_R = data.iloc[1:, :].iloc[:, sample_size:]
                correlation = data_R[data_R.columns[0]].corr(data_R[data_R.columns[1]])

            if len(winner_corr) > display_num - 1:
                if correlation > list(winner_corr.keys())[-1]:
                    continue
                if correlation is None:
                    continue
                # Remove the last key-value pair
                winner_corr.popitem()
            winner_corr[correlation] = random_codes
            # Sort the dictionary based on keys
            winner_corr = dict(sorted(winner_corr.items()))
        print('********************************************************************************')
        print(f'Here are the Top {len(winner_corr.keys())} Combination with the lowest correlation score')

        for rank, correlation_score in enumerate(winner_corr.keys()):

            data = RollingProcessor.get_data_full_period(winner_corr[correlation_score])
            if method == 'equal_peiord_csv':
                data = RollingProcessor.get_data_euqal_period(winner_corr[correlation_score])
            if method == 'equal_peiord_sql':
                data = RollingProcessor.get_data_euqal_period_new(winner_corr[correlation_score])

            #######################################################################################
            data_table = RollingProcessor.single_window_table(window_list, data, 2)
            data_table_RSI = rsi(window_list, data, 2)

            data_table_return = data_table.iloc[-1, 4 * target_num]
            data_table_RSI_return = data_table_RSI.iloc[-1, 4 * target_num]
            #######################################################################################

            #         # Calculate strategy performance metrics
            #         result = {}
            #         metrics = ["avg_return", "volatility", "sharpe", "max_drawdown", "win_rate"]

            #         for column in data_table.columns:
            #             result[column] = []
            #             for metric in metrics:
            #                 metric_function = getattr(qs.stats, metric)
            #                 r = metric_function(data_table[column])
            #                 result[column].append(r)

            #         result = pd.DataFrame(result, index=metrics)

            #         # Calculate strategy performance metrics on MA return Strat
            #         result_RSI = {}
            #         metrics_RSI = ["avg_return", "volatility", "sharpe", "max_drawdown", "win_rate"]

            #         for column_RSI in data_table_RSI.columns:
            #             result_RSI[column_RSI] = []
            #             for metric_RSI in metrics_RSI:
            #                 metric_function_RSI = getattr(qs.stats, metrics_RSI)
            #                 r_RSI = metric_function_MA(data_table_RSI[column_RSI])
            #                 result_RSI[column_RSI].append(r_RSI)

            #         result_RSI = pd.DataFrame(result_RSI, index=metric_RSI)

            if data_table_RSI_return > data_table_return:
                modified_R += 1
                improvements.append(data_table_RSI_return - data_table_return)
                print(
                    f'No.{rank + 1}: {winner_corr[correlation_score]} perform better on modified RSI return of {round(data_table_RSI_return, 3)} and improvement of {round(data_table_RSI_return - data_table_return, 3)} at Windows={window_list[0]}')
                print(
                    f'     The original strategy Strategy has a return {round(data_table_return, 3)} at Windows={window_list[0]}')
                print(
                    '************************************************************************************************************')
            if data_table_return > data_table_RSI_return:
                original_R += 1
                difference.append(data_table_return - data_table_RSI_return)
                print(
                    f'No.{rank + 1}: {winner_corr[correlation_score]} perform better on the original Strategy with a max return {round(data_table_return, 3)} and difference of {round(data_table_return - data_table_RSI_return, 3)}at Windows={window_list[0]}')
                print(
                    f'     The modified MA strategy Strategy has a return {round(data_table_RSI_return, 3)}  at Windows={window_list[0]}')
                print(
                    '************************************************************************************************************')

        print(
            '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(
            f'Given {len(winner_corr.keys())} combinations with lowest correlation score \n We found {modified_R} groups perform better on RSI Strat with an average improvement of {sum(improvements) / len(improvements)} and {original_R} perform better on the original Start with an average difference of {sum(difference) / len(difference)}')
        return None

