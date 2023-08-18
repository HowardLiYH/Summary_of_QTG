import pandas as pd
from myconfig import SqlConnManager
from pathlib import Path
from meta import ETFOptContracts
from public_tools.public_instance import jy_coder,dtc


class DataQuerier:
    def __init__(self):
        self.conn = SqlConnManager.conn_jy()

    def query_etf_price(self,ula_code,start_date,end_date):
        sql = f"""
        SELECT SecuCode as contract,TradingDay,ClosePrice as "close",OpenPrice as "open",HighPrice as "high",
        LowPrice as "low",PrevClosePrice as "prev_close"
        FROM [jydb].[dbo].QT_FundsPerformanceHis a ,SecuMain b
        where b.SecuCode = '{ula_code}' and (TradingDay between '{start_date}' and '{end_date}') 
        and b.InnerCode = a.InnerCode and b.SecuCategory = 8
        order by TradingDay asc
        """
        res = pd.read_sql(sql,self.conn)
        res['str_date'] = res['TradingDay'].dt.strftime('%Y%m%d')
        return res

    def query_opt_price(self,ula_type,trading_code,ctrct_code):
        """
        查询期权合约生命周期的所有daily价格

        :param ula_type:
        :param trading_code:
        :param ctrct_code:
        :return:
        """
        if ula_type == 'etf':
            sql = f"""
            select b.ContractCode,a.TradingDate,a.OpenPrice as "open",
            a.HighPrice as "high",a.LowPrice as "low",a.ClosePrice as "close",a.SettlePrice as "settle"
            from Opt_OptionContract b
            left join (select * from Opt_DailyQuote) as a
            on b.InnerCode = a.InnerCode
            where b.ContractCode = '{ctrct_code}' and b.Exchange in ({str(jy_coder.get_opt_exg_list())[1:-1]}) 
            order by a.TradingDate
            """
        else:
            sql = f"""
            select b.TradingCode,a.TradingDate,a.OpenPrice as "open",
            a.HighPrice as "high",a.LowPrice as "low",a.ClosePrice as "close",a.SettlePrice as "settle"
            from Opt_OptionContract b
            left join (select * from Opt_DailyQuote) as a
            on b.InnerCode = a.InnerCode
            where b.TradingCode = '{trading_code}' and b.Exchange in ({str(jy_coder.get_opt_exg_list())[1:-1]}) 
            order by a.TradingDate
            """
        res = pd.read_sql(sql,self.conn)
        res['str_date'] = res['TradingDate'].dt.strftime('%Y%m%d')
        res.set_index('TradingDate',inplace=True)

        return res


if __name__ == '__main__':
    path = Path(r'\\192.168.0.88\Public\OptionDesk\DATA\database\1day_bar\etf_by_optmon')
    targ_etfs = ['510050','510300']
    dq = DataQuerier()

    end_date = '20230317'

    for ula_code in targ_etfs:
        ula_folder = path.joinpath(ula_code)
        if not ula_folder.exists():
            ula_folder.mkdir()
        start_date = dtc.query_opt_start_date(ula_code).strftime('%Y%m%d')
        etf_metas = ETFOptContracts(ula_code)
        all_etf_data = dq.query_etf_price(ula_code,start_date,end_date)
        all_etf_data.rename(columns={'TradingDay':'datetime'},inplace=True)
        all_etf_data.set_index('datetime',inplace=True)

        for d in dtc.get_trding_day_range(start_date,end_date):
            full_year_code = etf_metas.get_dueyymm_series(d)[0]
            yymm = full_year_code[2:]
            outfolder = ula_folder.joinpath(yymm)

            if not outfolder.exists():
                outfolder.mkdir()
                file_path = outfolder.joinpath(f'{ula_code}.pkl')
                _k_dict = etf_metas.t_table[d][full_year_code]
                # series_list_date = min([etf_metas.get_contract_meta(_k_dict[k][cp_type]).get_start_date() for k in _k_dict for cp_type in _k_dict[k]])
                last_main_due_day = dtc.get_ith_weekday(dtc.get_last_month(yymm),4,3)
                etf_data:pd.DataFrame = all_etf_data.loc[(all_etf_data['str_date'].between(last_main_due_day,etf_metas.get_dueday(full_year_code)) ),:]
                etf_data.to_pickle(str(file_path))
                print(yymm,'done')
            else:
                continue


