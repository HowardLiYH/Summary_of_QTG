from utils.sql import SQLUtil

if __name__ == '__main__':
    sql_tl = SQLUtil('SQL_tonglian')
    df = sql_tl.get_pcr_info('10000150', '20160101', '20230415', 4)
    sql_df = df
    sql_df.to_csv(r"C:\Users\ps\PycharmProjects\pythonProject\sql_df.csv")