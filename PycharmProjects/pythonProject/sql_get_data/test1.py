import pandas as pd
from myconfig import SqlConnManager

if __name__ == '__main__':
    ula_code = '510050'


    sql = f"""
        SELECT a.TradingDate,b.ContractType, b.ExpirationDate, b.StrikePrice, a.ClosePrice, b.ContractCode
        From Opt_DailyQuote a,Opt_OptionContract b,Opt_ULAContract c
        where a.InnerCode = b.InnerCode and b.VarietyULAInnerCode = c.ULAInnerCode and c.ULACode = '{ula_code}'
        """

    data3 = pd.read_sql(sql, SqlConnManager.conn_jy())
    data3.to_csv(r"C:\Users\ps\PycharmProjects\pythonProject\data3.csv")

