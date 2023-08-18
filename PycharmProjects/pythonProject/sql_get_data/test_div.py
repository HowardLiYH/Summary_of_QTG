import pandas as pd
from myconfig import SqlConnManager

if __name__ == '__main__':
    ula_code = '510050'


    sql = f"""
        SELECT a.TradingDate,b.ContractType, b.ExpirationDate, a.ClosePrice,  b.ContractCode, e.ExgStrikePrice, e.AdjStrikePrice, e.ExgContractSize, e.AdjContractSize
        From Opt_DailyQuote a,Opt_OptionContract b,Opt_ULAContract c, Opt_DailyPreOpen d, Opt_Adjustment e
        where a.InnerCode = b.InnerCode and b.VarietyULAInnerCode = c.ULAInnerCode and c.ULACode = '{ula_code}' and d.InnerCode = a.InnerCode and e.Innercode = a.InnerCode
        """

    data4 = pd.read_sql(sql, SqlConnManager.conn_jy())
    # data4.to_csv(r"C:\Users\ps\PycharmProjects\pythonProject\data3.csv")
    data4.to_feather(r"C:\Users\ps\PycharmProjects\pythonProject\div.feather")

