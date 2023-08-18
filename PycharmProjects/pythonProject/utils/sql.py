import numpy as np
import pandas as pd
import pymssql
import pymysql
# from sqlalchemy import create_engine
from common.read_config import ReadConfig


class MySQLUtil:
    def __init__(self, section):
        self.RC = ReadConfig("../configs/pam_urls.ini")
        # 打开数据库连接
        self.connector = pymysql.connect(
            host=self.RC.get_str(section, "host"),
            port=self.RC.get_int(section, "port"),
            user=self.RC.get_str(section, "user"),
            password=self.RC.get_str(section, "password"),
            database=self.RC.get_str(section, "database"),  # 要连接的数据库
            # charset=self.RC.get_str("MYSQL", "charset")  # 设置数据库编码
        )
        self.Dic = {'SHFE': 'qtg_data.tbl_shfe_book_l2',
                    'INE': 'qtg_data.tbl_shfe_book_l2',
                    'DCE': 'qtg_data.tbl_dce_l2_book',
                    'CFFEX': 'qtg_data.tbl_future_book_l2',
                    'CZCE': 'qtg_data.tbl_czce_l2_book',
                    # --通联--
                    'CCFX': 'qtg_data.tbl_future_book_l2',  # 中国金融期货交易所
                    'XDCE': 'qtg_data.tbl_dce_l2_book',  # 大连商品交易所
                    'XSGE': 'qtg_data.tbl_shfe_book_l2',  # 上海期货交易所
                    'XZCE': 'qtg_data.tbl_czce_l2_book',  # 郑州商品交易所
                    'XSIE': 'qtg_data.tbl_shfe_book_l2',  # 上海国际能源交易中心
                    }
        # self.engine = create_engine(
        #     f"""mysql+pymysql://{self.RC.get_str(section, "user")}:{self.RC.get_str(section, "password")}@{self.RC.get_str(section, "host")}:{self.RC.get_int(section, "port")}/{self.RC.get_str(section, "database")}?charset=utf8""")

        # self.cur = self.connector.cursor()  # 创建一个游标对象
        # self.cursor = self.connector.cursor(pymysql.cursors.DictCursor)  # 设置为字典形式返回数据，默认为元组

    def df2mysql(self, df, table_name):
        # engine = create_engine('mysql+pymysql://gui:account4gui@192.168.0.133:3306/gui?charset=utf8')
        df.to_sql(name=table_name, con=self.engine, if_exists='append', index=False)

    def df_query_table(self, sql):
        return pd.read_sql(sql, con=self.connector)

    def close(self):
        # self.cursor.close()  # 关闭游标对象
        self.connector.close()  # 关闭数据库连接


class MySQLUtil_gui(MySQLUtil):

    def query_fund_products(self, fund_products_list):
        fund_products_str = ", ".join("'" + s.strip() + "'" for s in fund_products_list)
        sql = f"""SELECT sub_category,brief_name,wechat_userid as wechat_id
                    FROM gui.tbl_instance_info
                    where sub_category in ({fund_products_str})
                    and type!='Hedge'
                    and wechat_userid!='ChenYibo'
                    ;
                    """
        df = self.df_query_table(sql)
        return df
        # df_ = df.groupby('sub_category')['brief_name'].agg(lambda x: ','.join(x)).reset_index()
        #
        # fund_products_dict = df_.groupby('sub_category')['brief_name'].apply(list).to_dict()
        # brief_name_list = df_['brief_name'].tolist()
        # return fund_products_dict, brief_name_list

    def query_trade_record(self, today):
        sql = f"""SELECT instrument_id,side,cumulated_quantity as volume,price,brief_name,wechat_id
                    FROM gui.tbl_trade_record
                    where trading_day = {today}
                    and length(instrument_id)=10;"""
        df_ = self.df_query_table(sql)
        return df_

    def query_fund_position(self, today, fund):
        """
        查询非股票的持仓
        :param today:
        :return:
        """
        # fund_products_str = ", ".join("'" + s.strip() + "'" for s in fund_products_list)
        sql = f"""SELECT brief_name,instrument_id,net_posiition, total_buy,total_sell, sod_position,
                            yesterday_long,yesterday_short, today_long, today_short                    
                    FROM gui.tbl_position_record
                    where trading_day = {today}
                    and fund_name ='{fund}'
                    and position_name ="QTG"
                    and (instrument_id not like '%.SSE' and instrument_id not like '%.SZE');"""
        df_ = self.df_query_table(sql)
        return df_

    def query_leo_holding(self, today):
        """
        查询非股票的持仓
        :param today:
        :return:
        """
        # fund_products_str = ", ".join("'" + s.strip() + "'" for s in fund_products_list)
        sql = f"""SELECT brief_name,instrument_id,net_posiition, total_buy,total_sell, sod_position,
                            yesterday_long,yesterday_short, today_long, today_short                    
                    FROM gui.tbl_position_record
                    where trading_day ='{today}'
                    and position_name ="QTG"
                    and brief_name = 'DWSZ1'
                    and (instrument_id like '%.SSE' or instrument_id like '%.SZE');  """
        df_ = self.df_query_table(sql)
        return df_

    def query_fund_margin(self, today, fund):
        # fund_products_str = ", ".join("'" + s.strip() + "'" for s in fund_products_list)
        sql = f"""SELECT sum(margin_current) as margin_current  FROM gui.tbl_margin_status 
                    where trading_day = {today} 
                    and sub_category = '{fund}';"""
        df_ = self.df_query_table(sql)
        if not df_.empty:
            return df_.margin_current[0] * 1000
        else:
            return np.nan


class MySQLUtil_179(MySQLUtil):
    def get_symbol_last(self, today, symbol):
        sql = f"""SELECT last_price                  
                    FROM qtg_data.tbl_realtime_book 
                    where trading_day = {today}
                    and instrument_id = '{symbol}'
                    ORDER BY ID DESC LIMIT 1;"""
        sql2 = f"""SELECT last_price                  
                    FROM qtg_data.tbl_realtime_cffex_l2 
                    where trading_day = {today}
                    and instrument_id = '{symbol}'
                    ORDER BY ID DESC LIMIT 1;"""
        df_ = self.df_query_table(sql)
        if df_.empty:
            df_ = self.df_query_table(sql2)
        return df_

    def get_symbol_price(self, today, symbol_list):
        last_price_list = []
        for symbol in symbol_list:
            df_ = self.get_symbol_last(today, symbol)
            if not df_.empty:
                last_price = df_.last_price[0]
            else:
                last_price = np.nan
            last_price_list.append(last_price)
        df = pd.DataFrame({'instrument_id': symbol_list, 'last_price': last_price_list})
        df = df.dropna(axis=0, how='any')
        return df


class SQLUtil:
    def __init__(self, section):
        self.RC = ReadConfig("../configs/pam_urls.ini")
        # 打开数据库连接
        self.connector = pymssql.connect(
            host=self.RC.get_str(section, "host"),
            # port=self.RC.get_int(section, "port"),
            user=self.RC.get_str(section, "user"),
            password=self.RC.get_str(section, "password"),
            database=self.RC.get_str(section, "database"),  # 要连接的数据库
            # charset=self.RC.get_str("MYSQL", "charset")  # 设置数据库编码
        )
        # self.cur = self.connector.cursor()  # 创建一个游标对象
        # self.cursor = self.connector.cursor(pymysql.cursors.DictCursor)  # 设置为字典形式返回数据，默认为元组

    def df_query_table(self, sql):
        return pd.read_sql(sql, con=self.connector)

    def get_today_close(self, today):
        sql = f"""select CONCAT(
                            TICKER_SYMBOL,
                            '.',
                            CASE 
                                WHEN EXCHANGE_CD = 'XSHG' THEN 'SSE'
                                WHEN EXCHANGE_CD = 'XSHE' THEN 'SZE'
                                ELSE EXCHANGE_CD
                            END
                        ) AS instrument_id,
                        CLOSE_PRICE
                    from tonglian.dbo.mkt_equd_adj_af
                    where 
                    TRADE_DATE = '{today}'
                    order by TICKER_SYMBOL,TRADE_DATE asc;"""
        df = self.df_query_table(sql)
        return df

    def get_JY_symbol_CMValue(self, symbol):
        sql_Option = f"""SELECT 
                           [TradingCode] as instrument_id
                          ,[ContractSize] as CMValue
                          FROM [jydb].[dbo].[Opt_OptionContract]
                          where TradingCode = '{symbol}';"""

        sql_symbol = f"""SELECT 
                            ContractCode as instrument_id
                            ,CMValue
                            FROM [jydb].[dbo].[Fut_ContractMain]
                            where ContractCode = '{symbol}'"""
        df_ = self.df_query_table(sql_Option)
        if df_.empty:
            df_ = self.df_query_table(sql_symbol)
        return df_

    def get_JY_CMValue(self, symbol_list):
        CMValue_list = []
        for symbol in symbol_list:
            df_ = self.get_JY_symbol_CMValue(symbol)
            if not df_.empty:
                CMValue = df_.CMValue[0]
            else:
                CMValue = np.nan
            CMValue_list.append(CMValue)
        df = pd.DataFrame({'instrument_id': symbol_list, 'CMValue': CMValue_list})
        return df

    def get_pcr_info(self, symbol, start_date, end_date, feq):
        sql = f"""SELECT  *
                  FROM [tonglian].[dbo].[mkt_opt_stats]
                  where SECURITY_ID ='{symbol}'
                and  STATS_INTERVAL = {feq}
                and TRADE_DATE>='{start_date}' 
                and TRADE_DATE<='{end_date}';"""
        df = self.df_query_table(sql)
        return df

    def close(self):
        # self.cursor.close()  # 关闭游标对象
        self.connector.close()  # 关闭数据库连接


class SQLUtil_LEO(SQLUtil):
    def get_Account_Holding_Target(self, today):
        sql = f"""SELECT SecurityCode,SecurityPosition
                    FROM [Research].[dbo].[tbl_Account_Holding_Target] 
                    where AccountID = 'HY3_Acct_STK' 
                    and SecurityType = 'STK_A'
                    and Date = '{today}'; """
        df = self.df_query_table(sql)
        return df


if __name__ == '__main__':
    mysql_133 = MySQLUtil("MYSQL_133")
    mysql_155 = MySQLUtil("MYSQL_155")
    sql = SQLUtil("SQLServer")
    sql.JY_get_trading_time_dict('cu', '20230104')
