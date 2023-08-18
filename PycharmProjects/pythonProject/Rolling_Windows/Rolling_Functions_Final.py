import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
from ta.momentum import rsi
import pyodbc
import quantstats as qs
from math import isnan



# Initial Rolling Strategy
class Rolling:
    cnxn_jydb = pyodbc.connect('Driver={SQL Server};Server=192.168.0.144;Database=jydb;uid=zxing;pwd=zxing321')
    def __init__(self):
        pass

    def read_jydb(sql_query):
        """
        :param sql_query: str
        
        :return DataFrame
        """
        return pd.read_sql(sql_query, Rolling.cnxn_jydb)

    def get_ETF_data(SecuCode):
        """
        :param SecuCode int/str
        
        :return ETF Data
        """
        sql1 = f"""SELECT * FROM [jydb].[dbo].SecuMain WHERE SecuCode = '{SecuCode}' """
        data = Rolling.read_jydb(sql1)
        InnerCode = data['InnerCode'][0]
        if len(data)>1:
            for row in range(len(data)):
                InnerCode = data.iloc[row,:]['InnerCode']
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
                data2 = Rolling.read_jydb(sql2)
                if len(data2)> 2:
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
        data2 = Rolling.read_jydb(sql2)
        return data2

    def get_index_data(SecuCode):
        """
        :param SecuCode int/str
        
        :return Index Data
        """
        sql = f"""select a.*
        from QT_IndexQuote a, SecuMain b
        where a.InnerCode = b.InnerCode and b.SecuCode = '{SecuCode}' """
        
        data = Rolling.read_jydb(sql)
        return data

    def get_data(list_of_codes, period=None, list_of_types=None):
        """
        :param list_of_codes:
        :param period: ('full', 'equal')
        :param list_of_types: 1. 'ETF', 2. 'INDEX'

        :return: DataFrame
        """
        def compare_index_difference(df1, df2):
            index1 = set(df1.index)
            index2 = set(df2.index)

            missing_indices_df1 = index2 - index1
            missing_indices_df2 = index1 - index2

            return missing_indices_df1, missing_indices_df2
            
        # parameter type: list
        item_list = []
        for i in range(len(list_of_codes)):
            if list_of_types is None:
    #             print("Length of names and types does not match. Please input the list_of_types") 
                etf_data = Rolling.get_ETF_data(list_of_codes[i])
                etf_data = etf_data[['EndDate', 'AccumulatedUnitNV']].set_index('EndDate')
                etf_data = etf_data.rename(columns={'AccumulatedUnitNV': f"etf_{str(list_of_codes[i])}"})
                etf_data = etf_data[~etf_data.index.duplicated(keep='first')]
                item_list.append(etf_data)
            else:
                if list_of_types[i] == 'INDEX':
                    index_data = Rolling.get_index_data(list_of_codes[i])
                    index_data = index_data[['TradingDay', 'ClosePrice']].set_index('TradingDay')
                    index_data = index_data.rename(columns={'ClosePrice': f"index_{str(list_of_codes[i])}"})
                    index_data = index_data[~etf_data.index.duplicated(keep='first')]
                    item_list.append(index_data)
                if list_of_types[i] == 'ETF':
                    etf_data = Rolling.get_ETF_data(list_of_codes[i])
                    etf_data = etf_data[['EndDate', 'AccumulatedUnitNV']].set_index('EndDate')
                    etf_data = etf_data.rename(columns={'AccumulatedUnitNV': f"etf_{str(list_of_codes[i])}"})
                    etf_data = etf_data[~etf_data.index.duplicated(keep='first')]
                    item_list.append(etf_data)
            
            
        ## DataFrame Length Manipulation
        if period == "equal":
            print('Here at period == equal')
            shortest_length = len(item_list[0]) + 1
            shortest_index = -1
            # Find the shortest DataFrame
            list_of_index = []
            for i, df in enumerate(item_list):
                list_of_index.append(i)
                if len(df) < shortest_length:
                    shortest_length = len(df)
                    shortest_index = i
            # Align all length based on the shortest DF
            for i, df in enumerate(item_list):
                if i == shortest_index:
                    continue
                item_list[i] = df.drop(compare_index_difference(df, item_list[shortest_index])[1])

            data = pd.concat([item_list[0], item_list[1]], axis=1)

            for index in list_of_index[2:]:
                data = pd.concat([data, item_list[index]], axis=1)
            data = data.dropna().iloc[:]
            data.index = pd.to_datetime(data.index)

            # 计算每日涨跌幅和滚动收益率
            for column in data.columns:
                data[f'{column}_day_return'] = data[column].rolling(2).apply(lambda x:(x[-1]-x[0])/x[0])
            return data
    
        else:
            print('Here at starting from the longest period')
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
                data[f'{column}_day_return'] = data[column].rolling(2).apply(lambda x:(x[-1]-x[0])/x[0])
            data = data[1:]
            return data
    
    def rolling_optimization_table(data, window_list, target_num):
        """

        :param data: DataFrame from get_data
        :param window_list: [3,6,9,12,15,18,21,24,27]
        :param target_num: len(list_of_codes)

        :return: data_table
        """
        # parameter type: list, DataFrame, int
        
        data_table = pd.DataFrame()
        A = target_num
        for i in window_list:
            data_x = data.copy()
            signal_name = []
            
            for num, name in enumerate(data_x.columns[A:]):
                data_x[f'{name}_{str(i)}'] = data_x[data_x.columns[num]].rolling(window=i).apply(lambda x: x[-1]/x[0])
                signal_name.append(f'signal_{data_x.columns[num][4:]}')
            
            data_x = data_x.iloc[i:,:]
            for name in signal_name:
                data_x[name] = [0] * len(data_x)
            
            for row in range(len(data_x)):
                value = data_x.iloc[row][2*A:3*A].max()
                series = data_x.iloc[row]
                try:
                    position = series[2*A:3*A][series == value].idxmax()
                except:
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + A
                data_x.loc[data_x.index[row],series.index[new_position]] = 1
            
            for name in data_x.columns[-A:]:
                data_x[name] = data_x[name].shift(1)
            
            data_x = data_x[1:]
            
            data_x[f'signal_return_{str(i)}'] = [0] * len(data_x)
            arr = []

            for row in range(len(data_x)):
                value = data_x.iloc[row][3*A:4*A].max()
                series = data_x.iloc[row]
                ## find the position of signal with 1 
                position = series[3*A:4*A][series == value].idxmax()
                old_position = series.index.get_loc(position)
                new_position = old_position - 2*A
                arr.append(series[new_position] + 1)
                
            data_x['arr'] = arr
            data_x[f'signal_return_{str(i)}'] = np.cumprod(arr, axis=0)*100

            data_x[f'signal_return_{str(i)}'] = (100/data_x[f'signal_return_{str(i)}'].values[0]) * data_x[f'signal_return_{str(i)}'] 
            data_table[f'signal_return_{str(i)}'] = data_x[f'signal_return_{str(i)}']
        return data_table
    
    def single_window_table(data, window_list, target_num):
        """

        :param data: DataFrame from get_data
        :param window_list: [3,6,9,12,15,18,21,24,27]
        :param target_num: len(list_of_codes)

        :return: data_list
        """
        # parameter type: list, DataFrame, int
        
        data_list = []
        A = target_num
        for i in window_list:
            data_x = data.copy()
            signal_name = []
            
            for num, name in enumerate(data_x.columns[A:]):
                data_x[f'{name}_{str(i)}'] = data_x[data_x.columns[num]].rolling(window=i).apply(lambda x: x[-1]/x[0])
                signal_name.append(f'signal_{data_x.columns[num][4:]}')
            
            data_x = data_x.iloc[i:,:]
            for name in signal_name:
                data_x[name] = [0] * len(data_x)
            
            for row in range(len(data_x)):
                value = data_x.iloc[row][2*A:3*A].max()
                series = data_x.iloc[row]
                try:
                    position = series[2*A:3*A][series == value].idxmax()
                except:
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + A
                data_x.loc[data_x.index[row],series.index[new_position]] = 1
            
            for name in data_x.columns[-A:]:
                data_x[name] = data_x[name].shift(1)
            
            data_x = data_x[1:]
            
            data_x[f'signal_return_{str(i)}'] = [0] * len(data_x)
            arr = []
            
            for row in range(len(data_x)):
                value = data_x.iloc[row][3*A:4*A].max()
                series = data_x.iloc[row]
                ## find the position of signal with 1 
                position = series[3*A:4*A][series == value].idxmax()
                old_position = series.index.get_loc(position)
                new_position = old_position - 2*A
                arr.append(series[new_position] + 1)
                
            data_x['arr'] = arr
            data_x[f'signal_return_{str(i)}'] = np.cumprod(arr, axis=0)*100
            
    #         print(data_x)

            data_x[f'signal_return_{str(i)}'] = (100/data_x[f'signal_return_{str(i)}'].values[0]) * data_x[f'signal_return_{str(i)}'] 
        return data_x
    
    def signal_only(data, window_list, target_num):
        """

        :param data: DataFrame from get_data
        :param window_list: [3,6,9,12,15,18,21,24,27]
        :param target_num: len(list_of_codes)

        :return: data_list
        """
        # parameter type: list, DataFrame, int
        
        data_list = []
        A = target_num
        for i in window_list:
            data_x = data.copy()
            signal_name = []
            
            for num, name in enumerate(data_x.columns[A:]):
                data_x[f'{name}_{str(i)}'] = data_x[data_x.columns[num]].rolling(window=i).apply(lambda x: x[-1]/x[0])
                signal_name.append(f'signal_{data_x.columns[num][4:]}')
            
            data_x = data_x.iloc[i:,:]
            for name in signal_name:
                data_x[name] = [0] * len(data_x)
            
            for row in range(len(data_x)):
                value = data_x.iloc[row][2*A:3*A].max()
                series = data_x.iloc[row]
                try:
                    position = series[2*A:3*A][series == value].idxmax()
                except:
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + A
                data_x.loc[data_x.index[row],series.index[new_position]] = 1
            
        return data_x



# ROlling effective Test on return and sharpe
class Test(Rolling):

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
        plt.title(f'Different Rolling Window size with their Sharpe Ratio on {len(data_table)} days since {data_table.index[0]}', fontname='Arial', fontsize=21)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(int(len(data_table)/4)))
        plt.show()

    def get_Sharpe(data_table, column_name):
        """
        :param data_table: DataFrame
        :param column_name 
        :return: Specific Sharpe for the given column

        """
        result = {}
        metrics = ["avg_return", "volatility", "sharpe", "max_drawdown", "win_rate"]
        for column in data_table.columns:
            result[column] = []
            for metric in metrics:
                metric_function = getattr(qs.stats, metric)
                r = metric_function(data_table[column])
                result[column].append(r)

        result = pd.DataFrame(result, index=metrics)
        return result.loc[:, f"{column_name}"][2]

    def performance_test_equal_period(list_of_codes, window_list):
        """
        :param list_of_codes:['512480', '159865', '588080', '515790', '515080']
        :param window_list:[3,6,9,12,15,18,21,24,27]

        """
        # parameter type: list, list
        
        data = Rolling.get_data(list_of_codes)
        data_table  = Rolling.rolling_optimization_table(data, window_list, len(list_of_codes))
        Rolling.table_interpret(data_table)
        return None
    
    def trial_optimization(list_of_codes, window_list, attempts_time, sample_size):
        """
        :param list_of_all_codes:['512480', '159869', '512170', '512690', '512980', 
                                '516950','159865', '588080', '515790', '515080']
        :param attempts_time:100
        :param sample_size: 2

        """
        # parameter type: list, list
        winner_codes = []
        bar_return = 100
        unique_codes = []
        for i in range(attempts_time):
            random_codes = random.sample(list_of_codes, sample_size)
            
            # Check if the new random_codes are already present in the set of unique codes
            n = 0
            while random_codes in unique_codes:
                random_codes = random.sample(list_of_codes, sample_size)
                n += 1
                if n >20:
                    break
            
            # Add the new random_codes to the set of unique codes
            unique_codes.append(random_codes)
            data = Rolling.get_data(random_codes)
            data_table  = Rolling.rolling_optimization_table(data, window_list, int(len(data.columns)/2))
            last_row = data_table.iloc[-1]
            largest_value = last_row.max()
            largest_column = last_row.idxmax()
    #         print(f'Áttempt No.{i+1} on combination {random_codes} got {round(largest_value,3)} return at Windows={largest_column[14:]}')
            if largest_value > bar_return:
                print(f'HOORAY! The new bar is set at {round(largest_value,3)}!!')
                bar_return = largest_value
                bar_column = largest_column
                winner_codes = random_codes
        print('***************************************************************************************')
        print('**********************************Result***********************************************')
        print(f'\n1)The optimized combination is {winner_codes}\n2)The max return from it is {bar_return} at  Windows={bar_column[14:]}')
        return None
        
    def corr_test(list_of_codes, attempts_time, sample_size, display_num, comparison_target, window_list, method):
        """
        
        :param list_of_codes:['512480', '159869', '512170', '512690', '512980']
        :param attempts_time:1000
        :param sample_size:2
        :param display_num:100
        :param comparison_target: 1. 'Price'; 2. 'Return'
        :param window_list:[3,6,9,12,15,18,21,24,27]
        :param method: 1. 'equal'; 2. 'full_period'
        
        :return the top No. of combo with lowest correlation and their corresponding return, Sharpe, and Windows
        """
        # parameter type: list, list
        winner_corr = {}
        unique_codes = []
        
        for i in range(attempts_time):
            random_codes = random.sample(list_of_codes, sample_size)
            
            # Check if the new random_codes are already present in the set of unique codes
            n = 0
            while random_codes in unique_codes:

                random_codes = random.sample(list_of_codes, sample_size)
                n += 1
                if n >10:
                    break
            
            # Add the new random_codes to the set of unique codes
            unique_codes.append(random_codes)
            
            data = Rolling.get_data(random_codes)
            if method == 'equal':
                data = Rolling.get_data(random_codes, "equal")
                
            # Price Correlation
            data = data.iloc[1:,:].iloc[:,:sample_size]
            correlation = data[data.columns[0]].corr(data[data.columns[1]])
            
            # Return Correlation
            if comparison_target == 'Return':
                data = data.iloc[1:,:].iloc[:,sample_size:]
                correlation = data[data.columns[0]].corr(data[data.columns[1]])
                

            if len(winner_corr) > display_num - 1:
                if correlation > list(winner_corr.keys())[-1]:
                    continue
                if correlation is None:
                    continue
                # Remove the last key-value pair
                winner_corr.popitem()
            winner_corr[correlation]=random_codes
            # Sort the dictionary based on keys
            winner_corr = dict(sorted(winner_corr.items()))
        print('********************************************************************************')
        print(f'Here are the Top {len(winner_corr.keys())} Combination with the lowest correlation score')
        

        plot_corr = []
        plot_Sharpe = []
        plot_return = []
        for rank, correlation_score in enumerate(winner_corr.keys()):
            
            data = Rolling.get_data(winner_corr[correlation_score])
            if method == 'equal':
                data = Rolling.get_data(winner_corr[correlation_score], 'equal')
                
            data_table  = Rolling.rolling_optimization_table(data, window_list, 2)
            last_row = data_table.iloc[-1]
            largest_value = last_row.max()
            largest_column = last_row.idxmax()
            plot_corr.append(round(correlation_score,3))
            plot_return.append(round(largest_value,3))
            
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
            plot_Sharpe.append(round(result.loc["sharpe", largest_column],3))
            
            print(f'No.{rank+1}: {winner_corr[correlation_score]} with a score of {round(correlation_score,3)} and its max return is {round(largest_value,3)} with Sharpe {round(result.loc["sharpe", largest_column],3)} at Windows={largest_column[14:]}')
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 8))

        # Left subplot
        axs[0].scatter(plot_corr, plot_Sharpe)
        axs[0].set_title(f'Correlation vs. Sharpe on Windows={largest_column[14:]}')

        # Right subplot
        axs[1].scatter(plot_corr, plot_return)
        axs[1].set_title(f'Correlation vs. Return on Windows={largest_column[14:]}')

        plt.show()

        
        return None


# MA Return Strategy
class MA(Rolling):

    def single_window_table_MA_Return(window_list, data, target_num):
        """
        :param window_list: [21]
        :param data: DataFrame from get_data
        :param target_num: 2

        :return: data_list
        """
        # parameter type: list, DataFrame, int
        
        data_list = []

        for i in window_list:
            data_x = data.copy()
            signal_name = []
            print(data_x)
            for num, name in enumerate(data_x.columns[target_num:]):
                data_x[f'{name}_{str(i)}'] = data_x[data_x.columns[num]].rolling(window=i)
                signal_name.append(f'signal_{data_x.columns[num][4:]}')
            
            data_x = data_x.iloc[i:,:]
            for name in signal_name:
                data_x[name] = [0] * len(data_x)
            
            for row in range(len(data_x)):
                value = data_x.iloc[row][2*target_num:3*target_num].max()
                series = data_x.iloc[row]
                try:
                    position = series[2*target_num:3*target_num][series == value].idxmax()
                except:
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + target_num
                data_x.loc[data_x.index[row],series.index[new_position]] = 1
            
            for name in data_x.columns[-target_num:]:
                data_x[name] = data_x[name].shift(1)
            
            data_x = data_x[1:]
            
            data_x[f'signal_return_{str(i)}'] = [0] * len(data_x)
            arr = []
            X = target_num
            for row in range(len(data_x)):
                value = data_x.iloc[row][3*X:4*X].max()
                series = data_x.iloc[row]
                ## find the position of signal with 1 
                position = series[3*X:4*X][series == value].idxmax()
                old_position = series.index.get_loc(position)
                new_position = old_position - 2*X
                arr.append(series[new_position] + 1)
                
            data_x['arr'] = arr
            data_x[f'signal_return_{str(i)}'] = np.cumprod(arr, axis=0)*100

            data_x[f'signal_return_{str(i)}'] = (100/data_x[f'signal_return_{str(i)}'].values[0]) * data_x[f'signal_return_{str(i)}'] 
        return data_x

    def rolling_optimization_table_MA(data, window_list, target_num):
        """
        :param data: DataFrame from get_date
        :param target_num: 2

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
            
            data_x = data_x.iloc[i:,:]
            for name in signal_name:
                data_x[name] = [0] * len(data_x)
            
            for row in range(len(data_x)):
                value = data_x.iloc[row][2*target_num:3*target_num].max()
                series = data_x.iloc[row]
                try:
                    position = series[2*target_num:3*target_num][series == value].idxmax()
                except:
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + target_num
                data_x.loc[data_x.index[row],series.index[new_position]] = 1
            
            for name in data_x.columns[-target_num:]:
                data_x[name] = data_x[name].shift(1)
            
            data_x = data_x[1:]
            
            data_x[f'signal_return_{str(i)}'] = [0] * len(data_x)
            arr = []
            X = target_num
            for row in range(len(data_x)):
                value = data_x.iloc[row][3*X:4*X].max()
                series = data_x.iloc[row]
                ## find the position of signal with 1 
                try:
                    position = series[3*X:4*X][series == value].idxmax()
                except:
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position - 2*X
                arr.append(series[new_position] + 1)
                
            data_x['arr'] = arr
            data_x[f'signal_return_{str(i)}'] = np.cumprod(arr, axis=0)*100

            data_x[f'signal_return_{str(i)}'] = (100/data_x[f'signal_return_{str(i)}'].values[0]) * data_x[f'signal_return_{str(i)}'] 
            data_table[f'signal_return_{str(i)}'] = data_x[f'signal_return_{str(i)}']
        return data_table

# MA Test
class MA_Test(MA):

    def test_better_return(list_of_codes, attempts_time, sample_size, display_num, comparison_target, window_list, method):
        """
        
        :param list_of_codes:['512480', '159869', '512170', '512690', '512980']
        :param attempts_time:1000
        :param sample_size:2
        :param display_num:100
        :param comparison_target: DataFrame
        :param window_list:[3,6,9,12,15,18,21,24,27]
        :param method: 1. 'equal'; 3. 'full_period'
        
        :return the top No. of combo with lowest correlation and their corresponding return, Sharpe, and Windows
        """
        # parameter type: list, list
        winner_corr = {}
        unique_codes = []
        modified_R = 0
        original_R = 0
        for i in range(attempts_time):
            random_codes = random.sample(list_of_codes, sample_size)
            
            # Check if the new random_codes are already present in the set of unique codes
            n = 0
            while random_codes in unique_codes:

                random_codes = random.sample(list_of_codes, sample_size)
                n += 1
                if n >10:
                    break
            
            # Add the new random_codes to the set of unique codes
            unique_codes.append(random_codes)
            
            data = Rolling.get_data(random_codes)
            if method == 'equal':
                data = Rolling.get_data(random_codes, 'equal')
                
            # Price Correlation
            data_P = data.iloc[1:,:].iloc[:,:sample_size]
            correlation = data_P[data_P.columns[0]].corr(data_P[data_P.columns[1]])
            
            # Return Correlation
            if comparison_target == 'Return':
                data_R = data.iloc[1:,:].iloc[:,sample_size:]
                correlation = data_R[data_R.columns[0]].corr(data_R[data_R.columns[1]])
                

            if len(winner_corr) > display_num - 1:
                if correlation > list(winner_corr.keys())[-1]:
                    continue
                if correlation is None:
                    continue
                # Remove the last key-value pair
                winner_corr.popitem()
            winner_corr[correlation]=random_codes
            # Sort the dictionary based on keys
            winner_corr = dict(sorted(winner_corr.items()))
        print('********************************************************************************')
        print(f'Here are the Top {len(winner_corr.keys())} Combination with the lowest correlation score')
        

        for rank, correlation_score in enumerate(winner_corr.keys()):
            
            data = Rolling.get_data(winner_corr[correlation_score])
            if method == 'equal':
                data = Rolling.get_data(winner_corr[correlation_score])
    
            data_table  = Rolling.rolling_optimization_table(data, window_list, 2)
            data_table_MA  = Rolling.rolling_optimization_table_MA(data, window_list, 2)
            
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
                print(f'No.{rank+1}: {winner_corr[correlation_score]} perform better on modified MA Strategy with a max return of {round(largest_value_MA,3)} with Sharpe {round(result_MA.loc["sharpe", largest_column_MA],3)} at Windows={largest_column_MA[14:]}')
                print(f'     The original strategy Strategy has a max return {round(largest_value,3)} with Sharpe {round(result.loc["sharpe", largest_column],3)} at Windows={largest_column[14:]}')
                print('************************************************************************************************************')
            if largest_value > largest_value_MA:
                original_R += 1
                print(f'No.{rank+1}: {winner_corr[correlation_score]} perform better on the original Strategy with a max return {round(largest_value,3)} with Sharpe {round(result.loc["sharpe", largest_column],3)} at Windows={largest_column[14:]}')
                print(f'     The modified MA strategy Strategy has a max return {round(largest_value_MA,3)} with Sharpe {round(result_MA.loc["sharpe", largest_column_MA],3)} at Windows={largest_column_MA[14:]}')
                print('************************************************************************************************************')

                
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(f'Given {len(winner_corr.keys())} combinations with lowest correlation score \n We found {modified_R} groups perform better on MA Strat and {original_R} perform better on the original Start')
        return None
    
    def test_stability(list_of_codes, attempts_time, sample_size, display_num, window_list, comparison_target, method):
        """
        
        :param list_of_codes:['512480', '159869', '512170', '512690', '512980']
        :param attempts_time:1000
        :param sample_size:2
        :param display_num:100
        :param window_list:[3,6,9,12,15,18,21,24,27]
        :param comparison_target
        :param method: 1. 'equal'; 3. 'full_period'
        
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
            random_codes = random.sample(list_of_codes, sample_size)
            
            # Check if the new random_codes are already present in the set of unique codes
            n = 0
            while random_codes in unique_codes:

                random_codes = random.sample(list_of_codes, sample_size)
                n += 1
                if n >10:
                    break
            
            # Add the new random_codes to the set of unique codes
            unique_codes.append(random_codes)
            
            data = Rolling.get_data(random_codes)
            if method == 'equal':
                data = Rolling.get_data_euqal_period(random_codes, 'equal')
                
            # Price Correlation
            data_P = data.iloc[1:,:].iloc[:,:sample_size]
            correlation = data_P[data_P.columns[0]].corr(data_P[data_P.columns[1]])
            
            # Return Correlation
            if comparison_target == 'Return':
                data_R = data.iloc[1:,:].iloc[:,sample_size:]
                correlation = data_R[data_R.columns[0]].corr(data_R[data_R.columns[1]])
                

            if len(winner_corr) > display_num - 1:
                if correlation > list(winner_corr.keys())[-1]:
                    continue
                if correlation is None:
                    continue
                # Remove the last key-value pair
                winner_corr.popitem()
            winner_corr[correlation]=random_codes
            # Sort the dictionary based on keys
            winner_corr = dict(sorted(winner_corr.items()))
        print('********************************************************************************')
        print(f'Here are the Top {len(winner_corr.keys())} Combination with the lowest correlation score')
        

        for rank, correlation_score in enumerate(winner_corr.keys()):
            
            data = Rolling.get_data(winner_corr[correlation_score])
            if method == 'equal':
                data = Rolling.get_data_euqal_period(winner_corr[correlation_score], 'equal')

                
            data_table  = Rolling.rolling_optimization_table(data, window_list, 2)
            data_table_MA  = Rolling.rolling_optimization_table_MA(data, window_list, 2)
            
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
            old_var = result.iloc[2,:].var()
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
            new_var = result_MA.iloc[2,:].var()
            
            
            if new_var < old_var:
                modified_R += 1
                difference = old_var - new_var
                modified_R_v.append(difference)
                print(f'No.{rank+1}: {winner_corr[correlation_score]} with a lower variance on MA Strat')
                print(f'1) Old Strategy return has a variance of {old_var}\n{last_row}')
                print(f'2) MA Strategy return has a variance of {new_var}\n{last_row_MA}')
                print('************************************************************************************************************')
                
            

            if old_var < new_var:
                original_R += 1
                difference = new_var - old_var
                original_R_v.append(difference)
                print(f'No.{rank+1}: {winner_corr[correlation_score]} with a lower variance on OLD Strat')
                print(f'1) Old Strategy return has a variance of {old_var}\n{last_row}')
                print(f'2) MA Strategy return has a variance of {new_var}\n{last_row_MA}')
                print('************************************************************************************************************')
                
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(f'Given {len(winner_corr.keys())} combinations with lowest correlation score \n We found {modified_R} groups shows a smaller variance on MA Strat with an average variance difference of {sum(modified_R_v)/len(modified_R_v)} and {original_R} groups shows a smaller variance on the original Start with an average variance difference of {sum(original_R_v)/len(original_R_v)}')
        return None
    
    def test_stability_ZONE(list_of_codes, attempts_time, sample_size, display_num, comparison_target, window_list, method):
        """
        
        :param list_of_codes:['512480', '159869', '512170', '512690', '512980']
        :param attempts_time:1000
        :param sample_size:2
        :param display_num:100
        :param comparison_target: DataFrame 
        :param window_list:[3,6,9,12,15,18,21,24,27]
        :param method: 1. 'equal'; 3. 'full_period'
        
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
            random_codes = random.sample(list_of_codes, sample_size)
            
            # Check if the new random_codes are already present in the set of unique codes
            n = 0
            while random_codes in unique_codes:

                random_codes = random.sample(list_of_codes, sample_size)
                n += 1
                if n >10:
                    break
            
            # Add the new random_codes to the set of unique codes
            unique_codes.append(random_codes)
            
            data = Rolling.get_data(random_codes)
            if method == 'equal':
                data = Rolling.get_data(random_codes)

            # Price Correlation
            data_P = data.iloc[1:,:].iloc[:,:sample_size]
            correlation = data_P[data_P.columns[0]].corr(data_P[data_P.columns[1]])
            
            # Return Correlation
            if comparison_target == 'Return':
                data_R = data.iloc[1:,:].iloc[:,sample_size:]
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
            winner_corr[correlation]=random_codes
            # Sort the dictionary based on keys
            winner_corr = dict(sorted(winner_corr.items()))
        print('********************************************************************************')
        print(f'Here are the Top {len(winner_corr.keys())} Combination with the lowest correlation score')
        

        for rank, correlation_score in enumerate(winner_corr.keys()):
            
            data = Rolling.get_data(winner_corr[correlation_score])
            if method == 'equal':
                data = Rolling.get_data(winner_corr[correlation_score], 'equal')
                
            data_table  = Rolling.rolling_optimization_table(data, window_list, 2)
            data_table_MA  = Rolling.rolling_optimization_table_MA(data, window_list, 2)
            
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
            old_var = result.iloc[2,:].var()
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
            new_var = result_MA.iloc[2,:].var()
            
            if max(result_MA.iloc[2,:].mean(), result.iloc[2,:].mean()) < 0.5:
                continue
            
            
            if new_var < old_var:
                modified_R += 1
                difference = old_var - new_var
                modified_R_v.append(difference)
                print(f'No.{rank+1}: {winner_corr[correlation_score]} with a lower variance on MA Strat')
                print(f'1) Old Strategy return has a variance of {old_var}\n{last_row}')
                print(f'2) MA Strategy return has a variance of {new_var}\n{last_row_MA}')
                print('************************************************************************************************************')
                
            

            if old_var < new_var:
                original_R += 1
                difference = new_var - old_var
                original_R_v.append(difference)
                print(f'No.{rank+1}: {winner_corr[correlation_score]} with a lower variance on OLD Strat')
                print(f'1) Old Strategy return has a variance of {old_var}\n{last_row}')
                print(f'2) MA Strategy return has a variance of {new_var}\n{last_row_MA}')
                print('************************************************************************************************************')
                
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(f'Given {len(winner_corr.keys())} combinations with lowest correlation score \n We found {modified_R} groups shows a smaller variance on MA Strat with an average variance difference of {sum(modified_R_v)/len(modified_R_v)} and {original_R} groups shows a smaller variance on the original Start with an average variance difference of {sum(original_R_v)/len(original_R_v)}')
        return None


# RSI Strategy
class RSI_Strategy(Rolling):

    def apply_RSI(target_num, window_list, data_table):
        """
        :param list_of_codes: ['510050', '512890']
        :param window_list: 
        :param data_table: DataFrame

        :return: DataFrame
        """
        
        for name in data_table.iloc[:,:target_num].columns:
            data_table[f'{name}_RSI'] = rsi(data_table[name], window=window_list[0])
        data_table = data_table.iloc[window_list[0]:,:]
        return data_table
    
    def RSI(data, window_list, target_num):
        """
        :param data: DataFrame from get_data
        :param window_list: [21]
        :param target_num: 2 

        :return: data_list
        """
        # parameter type: list, DataFrame, int
        A = target_num
        for i in window_list:
            data_x = data.copy()
            _data = data.copy()
            data_RSI = RSI_Strategy.apply_RSI(target_num, window_list, _data)
            
            signal_name = []
            for num, name in enumerate(data_x.columns[A:]):
                data_x[f'{name}_{str(i)}'] = data_x[data_x.columns[num]].rolling(window=i).apply(lambda x: x[-1]/x[0])
                signal_name.append(f'signal_{data_x.columns[num][4:]}')
            
            data_x = data_x.iloc[i:,:]
            for name in signal_name:
                data_x[name] = [0] * len(data_x)
    
            ############################################## Modified Zone for RSI ##################################################
            for row in range(len(data_x)):
                a = data_x.iloc[row][2*A:3*A]
                b = data_RSI.iloc[row][2*A:3*A]

                RSI_dict = {}
                for item1, item2 in zip(a, b):
                    RSI_dict[item1] = item2

                target = max(RSI_dict)

                if RSI_dict[target] >= 70:
                    if min(RSI_dict.values()) >= 70:
                        target = target
                    else:
                        target = next((key for key, value in sorted(RSI_dict.items(), key=lambda x: x[1], reverse=True) if value < 70), None)
                series = data_x.iloc[row]
                value_index = series[series == target].index.tolist()
                try:
                    position = value_index[0]
                except:
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + A
                data_x.loc[data_x.index[row],series.index[new_position]] = 1
            
            #############################################################################################################
            
            for name in data_x.columns[-A:]:
                data_x[name] = data_x[name].shift(1)
            
            data_x = data_x[1:]
            
            data_x[f'signal_return_{str(i)}'] = [0] * len(data_x)
            arr = []
            
            for row in range(len(data_x)):
                value = data_x.iloc[row][3*A:4*A].max()
                series = data_x.iloc[row]
                ## find the position of signal with 1 
                position = series[3*A:4*A][series == value].idxmax()
                old_position = series.index.get_loc(position)
                new_position = old_position - 2*A
                arr.append(series[new_position] + 1)
                
            data_x['arr'] = arr
            data_x[f'signal_return_{str(i)}'] = np.cumprod(arr, axis=0)*100

            data_x[f'signal_return_{str(i)}'] = (100/data_x[f'signal_return_{str(i)}'].values[0]) * data_x[f'signal_return_{str(i)}'] 
        return data_x

# RSI Test
class RSI_Test(RSI_Strategy):

    def test_RSI_return(list_of_codes, attempts_time, sample_size, display_num, window_list, comparison_target, method, target_num):
        """
        
        :param list_of_codes:['512480', '159869', '512170', '512690', '512980']
        :param attempts_time:1000
        :param sample_size:2
        :param display_num:100
        :param window_list:[3,6,9,12,15,18,21,24,27]
        :param comparison_target: DataFrame
        :param method: 1. 'equal'; 2. 'full_period'
        :param target_num: 2
        
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
            random_codes = random.sample(list_of_codes, sample_size)
            
            # Check if the new random_codes are already present in the set of unique codes
            n = 0
            while random_codes in unique_codes:

                random_codes = random.sample(list_of_codes, sample_size)
                n += 1
                if n >10:
                    break
            
            # Add the new random_codes to the set of unique codes
            unique_codes.append(random_codes)
            
            data = Rolling.get_data(random_codes)
            if method == 'equal':
                data = Rolling.get_data(random_codes, 'equal')
                
            # Price Correlation
            data_P = data.iloc[1:,:].iloc[:,:sample_size]
            correlation = data_P[data_P.columns[0]].corr(data_P[data_P.columns[1]])
            
            # Return Correlation
            if comparison_target == 'Return':
                data_R = data.iloc[1:,:].iloc[:,sample_size:]
                correlation = data_R[data_R.columns[0]].corr(data_R[data_R.columns[1]])
                

            if len(winner_corr) > display_num - 1:
                if correlation > list(winner_corr.keys())[-1]:
                    continue
                if correlation is None:
                    continue
                # Remove the last key-value pair
                winner_corr.popitem()
            winner_corr[correlation]=random_codes
            # Sort the dictionary based on keys
            winner_corr = dict(sorted(winner_corr.items()))
        print('********************************************************************************')
        print(f'Here are the Top {len(winner_corr.keys())} Combination with the lowest correlation score')
        

        for rank, correlation_score in enumerate(winner_corr.keys()):
            
            data = Rolling.get_data(winner_corr[correlation_score])
            if method == 'equal':
                data = Rolling.get_data(winner_corr[correlation_score], 'equal')
                
            #######################################################################################   
            data_table  = Rolling.single_window_table(window_list, data, 2)
            data_table_RSI  = RSI_Strategy.RSI(window_list, data, 2)
            
            data_table_return = data_table.iloc[-1,4*target_num]
            data_table_RSI_return = data_table_RSI.iloc[-1,4*target_num]
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
                improvements.append(data_table_RSI_return-data_table_return)
                print(f'No.{rank+1}: {winner_corr[correlation_score]} perform better on modified RSI return of {round(data_table_RSI_return,3)} and improvement of {round(data_table_RSI_return-data_table_return,3)} at Windows={window_list[0]}')
                print(f'     The original strategy Strategy has a return {round(data_table_return,3)} at Windows={window_list[0]}')
                print('************************************************************************************************************')
            if data_table_return > data_table_RSI_return:
                original_R += 1
                difference.append(data_table_return-data_table_RSI_return)
                print(f'No.{rank+1}: {winner_corr[correlation_score]} perform better on the original Strategy with a max return {round(data_table_return,3)} and difference of {round(data_table_return-data_table_RSI_return,3)}at Windows={window_list[0]}')
                print(f'     The modified MA strategy Strategy has a return {round(data_table_RSI_return,3)}  at Windows={window_list[0]}')
                print('************************************************************************************************************')

                
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(f'Given {len(winner_corr.keys())} combinations with lowest correlation score \n We found {modified_R} groups perform better on RSI Strat with an average improvement of {sum(improvements)/len(improvements)} and {original_R} perform better on the original Start with an average difference of {sum(difference)/len(difference)}')
        return None

# Count Positions
class CountingPosition(RSI_Strategy):

    def position_change_cal(data_old):
        """
        :param data_old: DataFrame from get_data

        :return: DataFrame
        """
        data_new = data_old.copy()
        data_new['Position_change'] = len(data_new) * 0
        # Detect Position Change and Count it
        for row in range(len(data_new)):
            if sum(data_new.iloc[row, 6:8] == data_new.iloc[row-1, 6:8]) == 0:
                data_new.iloc[row,-1] = 1
        # Correct the first line
        data_new.iloc[0,-1] = 0
        print(f'The total position change has occured {data_new.iloc[:,-1].sum()} times.')
        return data_new.iloc[:,-1].sum()
    
    def position_change_return(data_new, target):
        """
        :param data_new: DataFrame from position_change_cal
        :param target: len(list_of_codes)

        :return: DataFrame
        """
        arr = data_new.iloc[:,target*4+1]
        data_new['Adjusted_arr'] = arr - data_new['Position_change']*arr*0.0025
        data_new['Adjusted_signal_return_21'] = np.cumprod(data_new['Adjusted_arr'], axis=0)*100
        data_new['Adjusted_signal_return_21'] = data_new['Adjusted_signal_return_21'] * 100 / data_new['Adjusted_signal_return_21'][0]
        print(f'The final return is {round(data_new.iloc[-1,-1], 3)}')
        return data_new
    
    def modified_RSI(data_new_RSI, target_num):
        """
        :param data_new_RSI: DataFrame from position_change_cal
        :param target_num: len(list_of_codes)

        :return: DataFrame
        """
        A = target_num
        i = 0 
        for row in range(len(data_new_RSI)):
            if data_new_RSI.iloc[row,-1] == 1:
                # Assuming `row` and `target_num` are defined
                start_index = A * 3
                end_index = A * 4

                # Create a boolean mask of values equal to 1 in the specified columns
                mask = (data_new_RSI.iloc[row-1, start_index:end_index] == 1)

                # Find the index of the first occurrence of value 1 using idxmax
                larger_one = mask.idxmax()
                larger_name = f'etf_{larger_one[7:]}_RSI'
                
                smaller_one = mask.idxmin()
                smaller_name = f'etf_{smaller_one[7:]}_RSI'
                
                if data_new_RSI.iloc[row-1][larger_name] > 70:
                    if data_new_RSI.iloc[row-1][smaller_name] < 70:
                        i += 1
                        print(f"This is the {i} time. One less Position Change Occured")
                        data_new_RSI.iloc[row-1]['Position_change'] = 0
                        data_new_RSI.iloc[row-1][larger_one] = 0
                        data_new_RSI.iloc[row-1][smaller_one] = 1
        if i == 0:
            print("No Position Change Occured. This function does not done any effect!")
        
        arr = []
        for row in range(len(data_new_RSI)):
            value = data_new_RSI.iloc[row][3*A:4*A].max()
            series = data_new_RSI.iloc[row]
            ## find the position of signal with 1 
            position = series[3*A:4*A][series == value].idxmax()
            old_position = series.index.get_loc(position)
            new_position = old_position - 2*A
            arr.append(series[new_position] + 1)
                
        data_new_RSI['arr'] = arr
        data_new_RSI['signal_return_21'] = np.cumprod(arr, axis=0)*100
        data_new_RSI['signal_return_21'] = (100/data_new_RSI['signal_return_21'].values[0]) * data_new_RSI['signal_return_21']
        
        return data_new_RSI

# Modified RSI Test
class Position_Test(CountingPosition):

    def modified_RSI_comparison(data_new_RSI, target_num):
        """
        :param data_new_RSI: DataFrame from position_change_cal
        :param target_num: len(list_of_codes)

        :return: DataFrame
        """
        i = 0 
        A = target_num
        for row in range(len(data_new_RSI)):
            if data_new_RSI.iloc[row,-1] == 1:
                # Assuming `row` and `target_num` are defined
                start_index = A * 3
                end_index = A * 4

                # Create a boolean mask of values equal to 1 in the specified columns
                mask = (data_new_RSI.iloc[row-1, start_index:end_index] == 1)

                # Find the index of the first occurrence of value 1 using idxmax
                larger_one = mask.idxmax()
                larger_name = f'etf_{larger_one[7:]}_RSI'
                
                smaller_one = mask.idxmin()
                smaller_name = f'etf_{smaller_one[7:]}_RSI'
                
                if data_new_RSI.iloc[row-1][larger_name] < data_new_RSI.iloc[row-1][smaller_name]:
                    i += 1
                    print(f"This is the {i} time. One less Position Change Occured")
                    data_new_RSI.iloc[row]['Position_change'] = 0
                    print(data_new_RSI.iloc[row][larger_one])
                    data_new_RSI.iloc[row][larger_one] = 1
                    print(data_new_RSI.iloc[row][larger_one])
                    data_new_RSI.iloc[row][smaller_one] = 0
        if i == 0:
            print("No Position Change Occured. This function does not done any effect!")
        
        arr = []
        for row in range(len(data_new_RSI)):
            value = data_new_RSI.iloc[row][3*A:4*A].max()
            series = data_new_RSI.iloc[row]
            ## find the position of signal with 1 
            position = series[3*A:4*A][series == value].idxmax()
            old_position = series.index.get_loc(position)
            new_position = old_position - 2*A
            arr.append(series[new_position] + 1)
                
        data_new_RSI['arr'] = arr
        data_new_RSI['signal_return_21'] = np.cumprod(arr, axis=0)*100
        data_new_RSI['signal_return_21'] = (100/data_new_RSI['signal_return_21'].values[0]) * data_new_RSI['signal_return_21']
        
        return data_new_RSI
    
# Option Strategy
class OPT(Rolling):
    def opt_strat_table(data, window_list, target_num):
        """

        :param data: DataFrame from get_data
        :param window_list: [3,6,9,12,15,18,21,24,27]
        :param target_num: len(list_of_codes)

        :return: data_list
        """
        # parameter type: list, DataFrame, int
        
        data_list = []
        A = target_num
        for i in window_list:
            data_x = data.copy()
            signal_name = []
            
            for num, name in enumerate(data_x.columns[A:]):
                ## Create the i day rolling window return
                data_x[f'{name}_{str(i)}'] = data_x[data_x.columns[num]].rolling(window=i).apply(lambda x: x[-1]/x[0])
                signal_name.append(f'signal_{data_x.columns[num][4:]}')
            
            data_x = data_x.iloc[i:,:]
            for name in signal_name:
                # Create Signal_ETF columns 
                data_x[name] = [0] * len(data_x)
            
            ## For Buy Long 
            for row in range(len(data_x)):
                # Locate the largest i day return for the given day (row)
                value = data_x.iloc[row][2*A:3*A].max()
                series = data_x.iloc[row]
                try:
                    position = series[2*A:3*A][series == value].idxmax()
                except:
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + A
                # Give the largest i day return 1 on its signal
                data_x.loc[data_x.index[row],series.index[new_position]] = 1
            
            ## For Sell Short
            for row in range(len(data_x)):
                # Locate the smallest i day return for the given day (row)
                value = data_x.iloc[row][2*A:3*A].min()
                series = data_x.iloc[row]
                try:
                    position = series[2*A:3*A][series == value].idxmin()
                except:
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + A
                # Give the largest i day return 1 on its signal
                data_x.loc[data_x.index[row],series.index[new_position]] = -1
            
            for name in data_x.columns[-A:]:
                data_x[name] = data_x[name].shift(1)
            
            data_x = data_x[1:]
            
            data_x[f'signal_return_{str(i)}'] = [0] * len(data_x)
            arr = []
            
            ## Create interim 'arr' for calculating 'signal_return'
            for row in range(len(data_x)):
                value_max = data_x.iloc[row][3*A:4*A].max()
                value_min = data_x.iloc[row][3*A:4*A].min()
                series = data_x.iloc[row]
                ## find the position of signal with 1 and -1
                position_max = series[3*A:4*A][series == value_max].idxmax()
                position_min = series[3*A:4*A][series == value_min].idxmin()
                old_position_max = series.index.get_loc(position_max)
                new_position_max = old_position_max - 2*A
                old_position_min = series.index.get_loc(position_min)
                new_position_min = old_position_min - 2*A
                arr.append(series[new_position_max] + 1 - series[new_position_min])
                
            data_x['arr'] = arr
            data_x[f'signal_return_{str(i)}'] = np.cumprod(arr, axis=0)*100
            
    #         print(data_x)

            data_x[f'signal_return_{str(i)}'] = (100/data_x[f'signal_return_{str(i)}'].values[0]) * data_x[f'signal_return_{str(i)}'] 
        return data_x

    def OPT_RSI(data, window_list, target_num, upper_bound, lower_bound):
        """
        :param data: DataFrame from get_data
        :param window_list: [21]
        :param target_num: 2 
        :param upper_bound: 70
        :param lower_bound: 40

        :return: data_list
        """
        # parameter type: list, DataFrame, int
        if upper_bound ==0 and lower_bound == 0:
            print("We are calculating conditions without RSI")
            return OPT.opt_strat_table(data, window_list, target_num)


        A = target_num
        for i in window_list:
            data_x = data.copy()
            _data = data.copy()
            data_RSI = RSI_Strategy.apply_RSI(target_num, window_list, _data)
            
            signal_name = []
            for num, name in enumerate(data_x.columns[A:]):
                data_x[f'{name}_{str(i)}'] = data_x[data_x.columns[num]].rolling(window=i).apply(lambda x: x[-1]/x[0])
                signal_name.append(f'signal_{data_x.columns[num][4:]}')
            
            data_x = data_x.iloc[i:,:]
            for name in signal_name:
                data_x[name] = [0] * len(data_x)
        
            # print(f"{data_x.iloc[:,-12:-6]}\n")
            # print(data_RSI.iloc[:,-6])
            ############################################## Modified Zone for RSI on Buy Long ##################################################
            ## For Buy Long
            for row in range(len(data_x)):
                a = data_x.iloc[row][2*A:3*A]
                b = data_RSI.iloc[row][2*A:3*A]
                # if row == 6:
                #     print(a)
                #     print(b)

                RSI_dict = {}
                for item1, item2 in zip(a, b):
                    RSI_dict[item1] = item2

                RSI_dict = {k: RSI_dict[k] for k in RSI_dict if not isnan(k)}
                # if row == 6:
                #     print(f'Here at row {row}, we have Buy Long RSI_dict as {RSI_dict}')
                target = max(RSI_dict)
                # if row == 6:
                #     print(f'The initial target is {target}')

                if RSI_dict[target] >= upper_bound:
                    if min(RSI_dict.values()) >= upper_bound:
                        # if row == 6:
                        #     print('@@@@@@@@@@We have reached Buy Long 1.0')
                        target = target
                    else:
                        # if row == 6:
                        #     print('@@@@@@@@@We have reached Buy Long 2.0')
                        #     print(f'The targets for RSI_dict when sorting RSI >70 are: {sorted(RSI_dict.items(), key=lambda x: x[1], reverse=True)}')
                        target = next((key for key, value in sorted(RSI_dict.items(), key=lambda x: x[1], reverse=True) if value < upper_bound), None)
                series = data_x.iloc[row]
                value_index = series[series == target].index.tolist()
                try:
                    position = value_index[0]
                except:
                    print(f'EXECPTION IS REACHED HERE AT ROW {row}')
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + A
                data_x.loc[data_x.index[row],series.index[new_position]] += 1
                # if row == 6:
                #     print(f'This is row {row}, the Buy Long RSI_dict is {RSI_dict} with target of {target}')
                #     print(f'The choice here is {series.index[new_position]}.\n')
                #     print(data_x.iloc[row,:][-6:])
                #     print('\n')
                # print(row, series.index[new_position], 1)
            
            ############################################# Modified Zone for RSI on Sell Short ####################################################
            ## For Sell Short
            for row in range(len(data_x)):
                a = data_x.iloc[row][2*A:3*A]
                b = data_RSI.iloc[row][2*A:3*A]

                RSI_dict = {}
                for item1, item2 in zip(a, b):
                    RSI_dict[item1] = item2

                RSI_dict = {k: RSI_dict[k] for k in RSI_dict if not isnan(k)}
                # if row == 6:
                #     print(f'Here at row {row}, we have Sell Short RSI_dict as {RSI_dict}')

                target = min(RSI_dict)
                # if row == 6:
                #     print(f'The initial target is {target}')

                if RSI_dict[target] <= lower_bound:
                    if max(RSI_dict.values()) <= lower_bound:
                        # if row == 6:
                        #     print('@@@@@@@@@@We have reached Sell Short 1.0')
                        target = target
                    else:
                        # if row == 6:
                        #     print('@@@@@@@@@@We have reached Sell Short 2.0')
                        target = next((key for key, value in sorted(RSI_dict.items(), key=lambda x: x[1], reverse=False) if value > lower_bound), None)
                        # if row == 6 :
                        #     print(f'The targets for RSI_dict when sorting RSI < 30 are: {sorted(RSI_dict.items(), key=lambda x: x[1], reverse=False)}\n')
                series = data_x.iloc[row]
                value_index = series[series == target].index.tolist()
                # if row == 6:
                #     print(f'Here is the value_index {value_index}\n')
                try:
                    position = value_index[0]
                except:
                    print(f'EXECPTION IS REACHED HERE AT ROW {row}')
                    continue
                # if row == 6:
                #     print('Here is reached')
                old_position = series.index.get_loc(position)
                new_position = old_position + A
                data_x.loc[data_x.index[row],series.index[new_position]] += -1
                # if row == 6:
                #     print(f'This is row {row}, the Sell Short RSI_dict is {RSI_dict} with target of {target}')
                #     print(f'The choice here is {series.index[new_position]}.\n')
                #     print(data_x.iloc[row,:][-6:])
                #     print('\n')
            # print(f'The length of data_x is {len(data_x)}\n')
            # print(data_x)
            ##########################################################################################################################
            
            for name in data_x.columns[-A:]:
                data_x[name] = data_x[name].shift(1)
            
            data_x = data_x[1:]
            

            # print('Here is the shifted part')
            # print(f'The length of data_x is {len(data_x)}\n')

            # print(data_x)


            data_x[f'signal_return_{str(i)}'] = [0] * len(data_x)
            arr = []
            

            ## Create interim 'arr' for calculating 'signal_return'
            for row in range(len(data_x)):
                value_max = data_x.iloc[row][3*A:4*A].max()
                value_min = data_x.iloc[row][3*A:4*A].min()
                series = data_x.iloc[row]
                ## find the position of signal with 1 and -1
                position_max = series[3*A:4*A][series == value_max].idxmax()
                position_min = series[3*A:4*A][series == value_min].idxmin()
                old_position_max = series.index.get_loc(position_max)
                new_position_max = old_position_max - 2*A
                old_position_min = series.index.get_loc(position_min)
                new_position_min = old_position_min - 2*A
                # if row == 6:
                #     print(f'For row {row}, we get arr as {series[new_position_max] + 1 - series[new_position_min]} with max return at {position_max} {series[new_position_max]} and min return at {position_min} {series[new_position_min]}')
                arr.append(series[new_position_max] + 1 - series[new_position_min])



            data_x['arr'] = arr
            data_x[f'signal_return_{str(i)}'] = np.cumprod(arr, axis=0)*100

            data_x[f'signal_return_{str(i)}'] = (100/data_x[f'signal_return_{str(i)}'].values[0]) * data_x[f'signal_return_{str(i)}'] 
        return data_x  

    def OPT_rolling_optimization_table(data, window_list, target_num):
        """

        :param data: DataFrame from get_data
        :param window_list: [3,6,9,12,15,18,21,24,27]
        :param target_num: len(list_of_codes)

        :return: data_table
        """
        # parameter type: list, DataFrame, int
        
        data_table = pd.DataFrame()
        A = target_num
        for i in window_list:
            data_x = data.copy()
            signal_name = []
            
            for num, name in enumerate(data_x.columns[A:]):
                data_x[f'{name}_{str(i)}'] = data_x[data_x.columns[num]].rolling(window=i).apply(lambda x: x[-1]/x[0])
                signal_name.append(f'signal_{data_x.columns[num][4:]}')
            
            data_x = data_x.iloc[i:,:]
            for name in signal_name:
                data_x[name] = [0] * len(data_x)
            
            for row in range(len(data_x)):
                value = data_x.iloc[row][2*A:3*A].max()
                series = data_x.iloc[row]
                try:
                    position = series[2*A:3*A][series == value].idxmax()
                except:
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + A
                data_x.loc[data_x.index[row],series.index[new_position]] = 1
            
            for name in data_x.columns[-A:]:
                data_x[name] = data_x[name].shift(1)
            
            data_x = data_x[1:]
            
            data_x[f'signal_return_{str(i)}'] = [0] * len(data_x)
            arr = []

            for row in range(len(data_x)):
                value = data_x.iloc[row][3*A:4*A].max()
                series = data_x.iloc[row]
                ## find the position of signal with 1 
                position = series[3*A:4*A][series == value].idxmax()
                old_position = series.index.get_loc(position)
                new_position = old_position - 2*A
                arr.append(series[new_position] + 1)
                
            data_x['arr'] = arr
            data_x[f'signal_return_{str(i)}'] = np.cumprod(arr, axis=0)*100

            data_x[f'signal_return_{str(i)}'] = (100/data_x[f'signal_return_{str(i)}'].values[0]) * data_x[f'signal_return_{str(i)}'] 
            data_table[f'signal_return_{str(i)}'] = data_x[f'signal_return_{str(i)}']
        return data_table
    
    def OPT_Position_Count(data, window_list, target_num, upper_bound, lower_bound):
        """
        :param data: DataFrame from get_data
        :param window_list: [21]
        :param target_num: 2 
        :param upper_bound: 70
        :param lower_bound: 40

        :return: data_list
        """
        # parameter type: list, DataFrame, int
        if upper_bound ==0 and lower_bound == 0:
            print("We are calculating conditions without RSI")
            return OPT.opt_strat_table(data, window_list, target_num)

        Long_last_chocie = ['Nothing']
        Long_Count = 0
        Short_last_chocie = ['Nothing']
        Short_Count = 0

        A = target_num
        for i in window_list:
            data_x = data.copy()
            _data = data.copy()
            data_RSI = RSI_Strategy.apply_RSI(target_num, window_list, _data)
            
            signal_name = []
            for num, name in enumerate(data_x.columns[A:]):
                data_x[f'{name}_{str(i)}'] = data_x[data_x.columns[num]].rolling(window=i).apply(lambda x: x[-1]/x[0])
                signal_name.append(f'signal_{data_x.columns[num][4:]}')
            
            data_x = data_x.iloc[i:,:]
            for name in signal_name:
                data_x[name] = [0] * len(data_x)
        
            # print(f"{data_x.iloc[:,-12:-6]}\n")
            # print(data_RSI.iloc[:,-6])
            ############################################## Modified Zone for RSI on Buy Long ##################################################
            ## For Buy Long
            for row in range(len(data_x)):
                a = data_x.iloc[row][2*A:3*A]
                b = data_RSI.iloc[row][2*A:3*A]
                # if row == 6:
                #     print(a)
                #     print(b)

                RSI_dict = {}
                for item1, item2 in zip(a, b):
                    RSI_dict[item1] = item2

                RSI_dict = {k: RSI_dict[k] for k in RSI_dict if not isnan(k)}
                # if row == 6:
                #     print(f'Here at row {row}, we have Buy Long RSI_dict as {RSI_dict}')
                target = max(RSI_dict)
                # if row == 6:
                #     print(f'The initial target is {target}')

                if RSI_dict[target] >= upper_bound:
                    if min(RSI_dict.values()) >= upper_bound:
                        # if row == 6:
                        #     print('@@@@@@@@@@We have reached Buy Long 1.0')
                        target = target
                    else:
                        # if row == 6:
                        #     print('@@@@@@@@@We have reached Buy Long 2.0')
                        #     print(f'The targets for RSI_dict when sorting RSI >70 are: {sorted(RSI_dict.items(), key=lambda x: x[1], reverse=True)}')
                        target = next((key for key, value in sorted(RSI_dict.items(), key=lambda x: x[1], reverse=True) if value < upper_bound), None)
                series = data_x.iloc[row]
                value_index = series[series == target].index.tolist()
                try:
                    position = value_index[0]
                except:
                    print(f'EXECPTION IS REACHED HERE AT ROW {row}')
                    continue
                old_position = series.index.get_loc(position)
                new_position = old_position + A
                data_x.loc[data_x.index[row],series.index[new_position]] += 1
                # if row == 6:
                #     # print(f'This is row {row}, the Buy Long RSI_dict is {RSI_dict} with target of {target}')
                #     print(f'The choice here is {series.index[new_position]}.\n')
                #     print(data_x.iloc[row,:][-6:])
                #     print('\n')
                if series.index[new_position] != Long_last_chocie[0]:
                    Long_last_chocie[0] = series.index[new_position]
                    Long_Count += 1
                # print(row, series.index[new_position], 1)
            
            ############################################# Modified Zone for RSI on Sell Short ####################################################
            ## For Sell Short
            for row in range(len(data_x)):
                a = data_x.iloc[row][2*A:3*A]
                b = data_RSI.iloc[row][2*A:3*A]

                RSI_dict = {}
                for item1, item2 in zip(a, b):
                    RSI_dict[item1] = item2

                RSI_dict = {k: RSI_dict[k] for k in RSI_dict if not isnan(k)}
                # if row == 6:
                #     print(f'Here at row {row}, we have Sell Short RSI_dict as {RSI_dict}')

                target = min(RSI_dict)
                # if row == 6:
                #     print(f'The initial target is {target}')

                if RSI_dict[target] <= lower_bound:
                    if max(RSI_dict.values()) <= lower_bound:
                        # if row == 6:
                        #     print('@@@@@@@@@@We have reached Sell Short 1.0')
                        target = target
                    else:
                        # if row == 6:
                        #     print('@@@@@@@@@@We have reached Sell Short 2.0')
                        target = next((key for key, value in sorted(RSI_dict.items(), key=lambda x: x[1], reverse=False) if value > lower_bound), None)
                        # if row == 6 :
                        #     print(f'The targets for RSI_dict when sorting RSI < 30 are: {sorted(RSI_dict.items(), key=lambda x: x[1], reverse=False)}\n')
                series = data_x.iloc[row]
                value_index = series[series == target].index.tolist()
                # if row == 6:
                #     print(f'Here is the value_index {value_index}\n')
                try:
                    position = value_index[0]
                except:
                    print(f'EXECPTION IS REACHED HERE AT ROW {row}')
                    continue
                # if row == 6:
                #     print('Here is reached')
                old_position = series.index.get_loc(position)
                new_position = old_position + A
                data_x.loc[data_x.index[row],series.index[new_position]] += -1
                # if row == 6:
                    # # print(f'This is row {row}, the Sell Short RSI_dict is {RSI_dict} with target of {target}')
                    # print(f'The choice here is {series.index[new_position]}.\n')
                    # print(data_x.iloc[row,:][-6:])
                    # print('\n')
                if series.index[new_position] != Short_last_chocie[0]:
                    Short_last_chocie[0] = series.index[new_position]
                    Short_Count += 1

            # print(f'The length of data_x is {len(data_x)}\n')
            # print(data_x)
            ##########################################################################################################################
            
            for name in data_x.columns[-A:]:
                data_x[name] = data_x[name].shift(1)
            
            data_x = data_x[1:]
            

            # print('Here is the shifted part')
            # print(f'The length of data_x is {len(data_x)}\n')

            # print(data_x)


            data_x[f'signal_return_{str(i)}'] = [0] * len(data_x)
            arr = []
            

            ## Create interim 'arr' for calculating 'signal_return'
            for row in range(len(data_x)):
                value_max = data_x.iloc[row][3*A:4*A].max()
                value_min = data_x.iloc[row][3*A:4*A].min()
                series = data_x.iloc[row]
                ## find the position of signal with 1 and -1
                position_max = series[3*A:4*A][series == value_max].idxmax()
                position_min = series[3*A:4*A][series == value_min].idxmin()
                old_position_max = series.index.get_loc(position_max)
                new_position_max = old_position_max - 2*A
                old_position_min = series.index.get_loc(position_min)
                new_position_min = old_position_min - 2*A
                # if row == 6:
                #     print(f'For row {row}, we get arr as {series[new_position_max] + 1 - series[new_position_min]} with max return at {position_max} {series[new_position_max]} and min return at {position_min} {series[new_position_min]}')
                arr.append(series[new_position_max] + 1 - series[new_position_min])



            data_x['arr'] = arr
            data_x[f'signal_return_{str(i)}'] = np.cumprod(arr, axis=0)*100

            data_x[f'signal_return_{str(i)}'] = (100/data_x[f'signal_return_{str(i)}'].values[0]) * data_x[f'signal_return_{str(i)}'] 
        return Long_Count, Short_Count  


    
    







# if __name__ == "__main__":
#     print("Initiliazing.............")
#     print('\n')
#     print('\n')
#     print('\n')


#     list_of_codes = ['510050', '510300', '159915', '159901', '588000', '510500']
#     window_list = [21]
#     data = Rolling.get_data(list_of_codes)
#     data_2016 = data.iloc[2666:,:]
#     Long_Count, Short_Count = OPT.OPT_Position_Count(data_2016, [12], len(list_of_codes), 60, 30)
#     print(Long_Count, Short_Count)


#     RSI_Table =  OPT.OPT_RSI(data_2016, [3], len(list_of_codes), 60, 20)
#     print(Test.get_Sharpe(RSI_Table, f'signal_return_3'))


#     No_RSI_Table = OPT.OPT_RSI(data_2016, [12], len(list_of_codes), 0, 0)
#     print(No_RSI_Table)
    ###################
    ### Basic Functions

    # Test 1: Get Basic Daily Return Data
    # Daily_Return = Rolling.get_data(['512480', '159869'])
    # print(Daily_Return)

    ## Test 2: Get the Best Window Performance from the selected ETF Combination
    # Roller = Rolling(['512480', '159865', '588080', '515790','515080'], [3,6,9,12,15,18,21,24,27], 'full_period')
    # Daily_Return = Rolling.get_data(['512480', '159865', '588080', '515790','515080'])
    # opt_data_table  = Rolling.rolling_optimization_table(Daily_Return, [3,6,9,12,15,18,21,24,27], 5)
    # print(opt_data_table)

    ## Test 3: Get the finalzied signal return based on the initial rolling window strategy, windows = 21
    # Daily_Return = Rolling.get_data(['512480', '159865', '588080', '515790','515080'])
    # data_table = Rolling.single_window_table(Daily_Return, [21], 5)
    # print(data_table)

    ## Test 3: Get Sharpe Results and Return Plots
    # Daily_Return = Rolling.get_data(['512480', '159865', '588080', '515790','515080'])
    # data_table  = Rolling.rolling_optimization_table(Daily_Return, [3,6,9,12,15,18,21,24,27], 5)
    # table_results = Test.table_interpret(data_table)

    ## Test 4: Get the optimized combination form the list of ETFs based on the highest return
    # test_instance = Test(['512480', '159865', '588080', '515790','515080'], [3,6,9,12,15,18,21,24,27])
    # Opt_result = Test.trial_optimization(['512480', '159865', '588080', '515790','515080'], [3,6,9,12,15,18,21,24,27], 300, 2)

    ## Test 5: Rank the randomized ETF combo based on their correlation (Either Daily Return or Price Return) along with Return overtim and Shape
    # Corr_result = Test.corr_test(['512480', '159865', '512170', '512690','512980', '516950', '588080', '515790', '515080', '159967'], 
    #                              1000, 2, 200, 'Price', [21], 'full_period')

    #################
    ### MA Functions:

    ## Test 1: Get a finalized signal return based on the MA strategy\
    # list_of_codes = ['512480', '159865', '512170', '512690','512980', '516950', '588080', '515790', '515080', '159967']
    # Daily_Return = Rolling.get_data(list_of_codes, 'equal')
    # data_table = MA.single_window_table_MA_Return([21], Daily_Return, len(list_of_codes))
    

    #################
    ### RSI Functions:

    ## Test 1: Add RSI on the initial DataFrame
    # list_of_codes = ['510050', '510300', '159915', '159901', '588000']
    # window_list = [21]
    # data = Rolling.get_data(list_of_codes)
    # _data = data.copy()
    # data_table = RSI_Strategy.apply_RSI(len(list_of_codes), window_list, _data)

    ## Test 2: Produce a new table guided by the RSI Strategy leaning toward RSI over rolling windows
    # Continued from Test 1 
    # RSI_Strategy.RSI(_data, [21], len(list_of_codes))

    #################

    ## Test 1: Show the performance 
    # list_of_codes = ['510050', '510300', '159915', '159901', '588000', '510500']
    # window_list = [21]
    # data = Rolling.get_data(list_of_codes)
    # data_2016 = data.iloc[2666:,:]
    # OPT_table = OPT.opt_strat_table(data_2016, window_list, len(list_of_codes))
    # print(OPT_table)

    # OG_table = Rolling.single_window_table(data_2016, window_list, len(list_of_codes))
    # print(OG_table)







    



