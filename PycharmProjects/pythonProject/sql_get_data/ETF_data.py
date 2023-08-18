import pandas as pd
from myconfig import SqlConnManager

if __name__ == '__main__':

        sql2 = f"""SELECT 
            [InnerCode]
          ,[InfoPublDate]
          ,[InfoSource]
          ,[EndDate]
          ,[NV]
          ,[UnitNV]
          ,[AccumulatedUnitNV]
          ,[DailyProfit]
          ,[LatestWeeklyYield]
          ,[NVDailyGrowthRate]
          ,[NVWeeklyGrowthRate]
          ,[DiscountRatio]
          ,[XGRQ]
          ,[JSID]
          ,[InvolvedDays]
        FROM [jydb].[dbo].[MF_NetValue]
        where innercode = '333579'  order by InfoPublDate ASC"""

        data2 = pd.read_sql(sql2, SqlConnManager.conn_jy())
        data2.to_csv(fr"C:\Users\ps\PycharmProjects\pythonProject\515790.csv")

