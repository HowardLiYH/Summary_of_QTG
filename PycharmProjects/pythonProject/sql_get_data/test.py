import pandas as pd
from myconfig import SqlConnManager

if __name__ == '__main__':
    ula_code = '510050'


    sql = f"""
        SELECT a.*,b.ContractType
        From Opt_DailyQuote a,Opt_OptionContract b,Opt_ULAContract c
        where a.InnerCode = b.InnerCode and b.VarietyULAInnerCode = c.ULAInnerCode and c.ULACode = '{ula_code}'
        """

    data = pd.read_sql(sql, SqlConnManager.conn_jy())
    data.to_csv(r"C:\Users\ps\PycharmProjects\pythonProject\data.csv")
    g = data.groupby(['TradingDate','ContractType'])['OpenInterest'].sum()

    print(g)
