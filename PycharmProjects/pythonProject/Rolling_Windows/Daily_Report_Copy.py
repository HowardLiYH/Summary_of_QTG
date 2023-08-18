import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pyodbc
import logging
import datetime

cnxn_jydb = pyodbc.connect('Driver={SQL Server};Server=192.168.0.144;Database=jydb;uid=zxing;pwd=zxing321')

def read_jydb(sql_query):
    return pd.read_sql(sql_query, cnxn_jydb)

def get_ETF_data(SecuCode):
    """
    
    :param SecuCode int/str
    
    :return ETF Data
    """
    sql1 = f"""SELECT * FROM [jydb].[dbo].SecuMain WHERE SecuCode = '{SecuCode}' """
    data = read_jydb(sql1)
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
            data2 = read_jydb(sql2)
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
    data2 = read_jydb(sql2)
    return data2

def get_index_data(SecuCode):
    """
    
    :param SecuCode int/str
    
    :return ETF Data
    """
    sql = f"""select a.*
    from QT_IndexQuote a, SecuMain b
    where a.InnerCode = b.InnerCode and b.SecuCode = '{SecuCode}' """
    
    data = read_jydb(sql)
    return data

# Align all the DFs in list to the size of the shortest one 
def get_data(list_of_codes, period=None, list_of_types=None):
    """
    :param list_of_names:
    :param period: ('full', 'equal')
    :param list_of_types:

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
            etf_data = get_ETF_data(list_of_codes[i])
            etf_data = etf_data[['EndDate', 'AccumulatedUnitNV']].set_index('EndDate')
            etf_data = etf_data.rename(columns={'AccumulatedUnitNV': f"etf_{str(list_of_codes[i])}"})
            etf_data = etf_data[~etf_data.index.duplicated(keep='first')]
            item_list.append(etf_data)
        else:
            if list_of_types[i] == 'INDEX':
                index_data = get_index_data(list_of_codes[i])
                index_data = index_data[['TradingDay', 'ClosePrice']].set_index('TradingDay')
                index_data = index_data.rename(columns={'ClosePrice': f"index_{str(list_of_codes[i])}"})
                index_data = index_data[~etf_data.index.duplicated(keep='first')]
                item_list.append(index_data)
            if list_of_types[i] == 'ETF':
                etf_data = get_ETF_data(list_of_codes[i])
                etf_data = etf_data[['EndDate', 'AccumulatedUnitNV']].set_index('EndDate')
                etf_data = etf_data.rename(columns={'AccumulatedUnitNV': f"etf_{str(list_of_codes[i])}"})
                etf_data = etf_data[~etf_data.index.duplicated(keep='first')]
                item_list.append(etf_data)
        
        
    
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

################################# Convient W #########################################
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
            data_x[f'{name}_{str(i)}'] = data_x[data_x.columns[num]].rolling(window=i).apply(lambda x: x[-1]/x[0])
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
        
#         print(data_x)

        data_x[f'signal_return_{str(i)}'] = (100/data_x[f'signal_return_{str(i)}'].values[0]) * data_x[f'signal_return_{str(i)}'] 
    return data_x

if __name__ == '__main__':
    
    current_date = datetime.date.today()
    current_time = datetime.datetime.now().time()  
    formatted_date = current_date.strftime("%Y-%m-%d")
    formatted_time = current_time.strftime("%H:%M:%S") 
    try:
        list_of_codes = ['159967', '512890']
        window_list = [21]
        target_num = 2

        data = get_data(list_of_codes)
        data_old = single_window_table(window_list, data, target_num)

        data_old['etf_512890_return%'] = round(data_old['etf_512890']*(100/data_old.loc[:,"etf_512890"][0]),2)
        data_old['etf_159967_return%'] = round(data_old['etf_159967']*(100/data_old.loc[:,"etf_159967"][124]),2)

        data_old['512890_day_return%'] = round(data_old['etf_512890_day_return']*100 ,2)
        data_old['159967_day_return%'] = round(data_old['etf_159967_day_return']*100 ,2)
        data_old['159967_21day_return%'] = round(data_old['etf_159967_day_return_21']*100 - 100,2)
        data_old['512890_21day_return%'] = round(data_old['etf_512890_day_return_21']*100 - 100,2)

        data_old['512890 - 159967_21day_return%'] = round(100*((data_old['etf_512890_day_return_21'] - data_old['etf_159967_day_return_21'])/data_old['etf_512890_day_return_21']),2)

        data_old['Daily Strategy Gain%'] = round((data_old['arr'] - 1)*100,2)
        data_old['signal_return_21%'] = round(data_old['signal_return_21'],2)

        data_old = data_old.iloc[-22:,:]
        data_old.index = data_old.index.strftime('%m-%d')
        print(data_old.columns)

        # Save the DataFrame to a CSV file at the specified location
        data_old.to_csv(fr"/mnt/88disk/Howard/Daily_ETF_{current_date}.csv", index=True, date_format='%Y-%m-%d')

        # Configure logging to redirect output to a file
        logging.basicConfig(
            filename='report_log.txt',
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )

        # Redirect the standard output to the logger
        logger = logging.getLogger()
        stdout_handler = logging.StreamHandler()
        logger.addHandler(stdout_handler)

        # Example usage
        logging.info(f"The CSV has been successfully updated on {current_date } at time {current_time}")
        
    except Exception as e:
        logging.error("Error occurred: %s", str(e))
        


