import datetime
from myconfig import SqlConnManager
import pandas as pd
import numpy as np
import math
from pathlib import Path


class RVCalculator:
    def __init__(self):
        self.conn_jy = SqlConnManager.conn_jy()
        self.root = Path(r'D:\code\Routine\MorningReportRVPart')

    def get_etf_data(self,ula_code,start_date,end_date):
        sql = f"""
        SELECT SecuCode,TradingDay,a.PrevClosePrice, a.OpenPrice, a.ClosePrice, a.HighPrice, a.LowPrice
        FROM [jydb].[dbo].[QT_FundsPerformanceHis] a ,SecuMain b
        where b.SecuCode = '{ula_code}' and (TradingDay between '{start_date}' and '{end_date}')
        and b.InnerCode = a.InnerCode and b.SecuCategory = 8
        order by TradingDay
        """
        ula_data = pd.read_sql(sql,self.conn_jy)
        # ula_data['str_date'] = ula_data['TradingDay'].apply(lambda x:x.strftime('%Y%m%d'))
        # ula_data.set_index('str_date',inplace=True)
        return ula_data

    def _add_price(self,targ_df,src_dict):
        targ_df['OpenPrice'] = [src_dict['o']]
        targ_df['HighPrice'] = [src_dict['h']]
        targ_df['LowPrice'] = [src_dict['l']]
        targ_df['ClosePrice'] = [src_dict['c']]

    def manually_add_tdy_price(self,ula_price:pd.DataFrame,tdy_price:dict,copy_col)->pd.DataFrame:
        tdy_df = pd.DataFrame(columns=ula_price.columns)
        tdy = pd.Timestamp(datetime.date.today())
        tdy_df['TradingDay'] = [tdy]
        for c in copy_col:
            tdy_df[c] = [ula_price.loc[len(ula_price)-1,c]]
        tdy_df['PrevClosePrice'] = [ula_price.loc[len(ula_price)-1,'ClosePrice']]
        self._add_price(tdy_df,tdy_price)
        return pd.concat([ula_price,tdy_df],ignore_index=True)

    def prep_etf_data(self,ula_price:pd.DataFrame,manually_add_tdy_price:dict = None):
        if manually_add_tdy_price:
            df = self.manually_add_tdy_price(ula_price,manually_add_tdy_price,copy_col=['SecuCode'])
        else:
            df = ula_price.copy(deep=True)
        df['DailyReturn'] = df['ClosePrice'] / df['PrevClosePrice']
        return df

    @staticmethod
    def cal_vol(df:pd.DataFrame,method,periods):
        tradingdays = 240
        if method == 'C2C':
            df['C2C_u'] = np.log(df['DailyReturn'])
            for prd in periods:
                df[f'C2C_{prd}'] = df['C2C_u'].rolling(periods[prd]).std() * math.sqrt(tradingdays)

        if method == 'RMS':
            df['SqrtRetrun'] = np.square(np.log(df['DailyReturn']))
            # print(df['SqrtRetrun'] )
            for prd in periods:
                df[f'RMS_{prd}'] = np.sqrt(df['SqrtRetrun'].rolling(periods[prd]).mean()) * math.sqrt(tradingdays)

        if method == 'EWMA':
            # EWMA: sigma_m^2 = sigma_n-1^2 * lambda + (u_n-1^2)*(1-lambda),lambda = (N-1)/(N+1) 由EWMA 中得span 代表
            df['EWMA_usqr'] = np.log(df['DailyReturn']) ** 2
            for prd in periods:
                df[f'EWMA_{prd}'] = np.sqrt(df['EWMA_usqr'].ewm(span=periods[prd], adjust=False).mean()) * math.sqrt(
                    tradingdays)

        if method == 'Parkinson':
            df['pks'] = (1.0 / (4.0 * math.log(2.0))) * ((df['HighPrice'] / df['LowPrice']).apply(np.log)) ** 2.0

            def f(v):
                return (tradingdays * v.mean()) ** 0.5

            for prd in periods:
                df[f'Parkinson_{prd}'] = df['pks'].rolling(periods[prd]).apply(func=f)

        if method == 'GarmanKlass':
            df['log_hl'] = 0.5 * ((df['HighPrice'] / df['LowPrice']).apply(np.log)) ** 2
            df['loc_cc'] = (2 * math.log(2) - 1) * ((df['ClosePrice'] / df['OpenPrice']).apply(np.log)) ** 2
            df['gk'] = df['log_hl'] - df['loc_cc']
            for prd in periods:
                df[f'GarmanKlass_{prd}'] = np.sqrt(df['gk'].rolling(periods[prd]).mean()) * math.sqrt(tradingdays)

        if method == 'RogersSatchell':
            df['norm_u'] = (df['HighPrice'] / df['OpenPrice']).apply(np.log)  # normalized high
            df['norm_d'] = (df['LowPrice'] / df['OpenPrice']).apply(np.log)  # normalized low
            df['norm_c'] = (df['ClosePrice'] / df['OpenPrice']).apply(np.log)  # normalized close
            df['rs'] = df['norm_u'] * (df['norm_u'] - df['norm_c']) + df['norm_d'] * (
                        df['norm_d'] - df['norm_c'])  # rogers and satchell
            for prd in periods:
                df[f'RogersSatchell_{prd}'] = np.sqrt(df['rs'].rolling(periods[prd]).sum() / periods[prd]) * math.sqrt(
                    tradingdays)

        if method == 'YangZhang':
            df['norm_o'] = (df['OpenPrice'] / df['PrevClosePrice']).apply(np.log)  # normalized open
            df['norm_u'] = (df['HighPrice'] / df['OpenPrice']).apply(np.log)  # normalized high
            df['norm_d'] = (df['LowPrice'] / df['OpenPrice']).apply(np.log)  # normalized low
            df['norm_c'] = (df['ClosePrice'] / df['OpenPrice']).apply(np.log)  # normalized close
            df['rs'] = df['norm_u'] * (df['norm_u'] - df['norm_c']) + df['norm_d'] * (
                        df['norm_d'] - df['norm_c'])  # rogers and satchell
            for prd in periods:
                k = 0.34 / (1.34 + (periods[prd] + 1) / (periods[prd] - 1))
                V_o = df['norm_o'].rolling(periods[prd]).var() * periods[prd] / (periods[prd] - 1)
                V_c = df['norm_c'].rolling(periods[prd]).var() * periods[prd] / (periods[prd] - 1)
                V_rs = df['rs'].rolling(periods[prd]).sum() / periods[prd]
                df[f'YangZhang_{prd}'] = (V_o + k * V_c + (1 - k) * V_rs).apply(np.sqrt) * math.sqrt(tradingdays)
        return df

    def main_cal(self,ula_code,methods,periods,start_date):
        """

        :param ula_code:
        :param methods:
        :param periods:
        :return:
        """
        res_dict = dict()
        tdy_date = datetime.date.today().strftime('%Y%m%d')
        raw_data = self.get_etf_data(ula_code,start_date,tdy_date)
        data = self.prep_etf_data(raw_data)
        for met in methods:
            res_dict[met] = self.cal_vol(data,met,periods)
        return res_dict


class FutRVCalculator(RVCalculator):
    def __init__(self,change_abnormal_prices:dict = None):
        """

        :param change_abnormal_prices:dict, IM:{IM2211: {'20221108':[colname,newvalue]}}. IM 出现了乌龙指，极大地拉偏了 vol。所以引入新机制：
        可以指定某些品种的特定交易日的数据改掉
        """
        super(FutRVCalculator, self).__init__()
        self.change_abnormal_prices = change_abnormal_prices

    def get_fut_data(self,ula_code,start_date,end_date,series_flag):
        """

        :param ula_code:
        :param start_date:
        :param end_date:
        :param series_flag: 月份序列标志 1 当月，2 次月
        :return:
        """
        sql = f"""
        select TradingDay,c.LastTradingDate,a.ContractCode,a.PrevClosePrice,a.OpenPrice,a.HighPrice,a.LowPrice,a.ClosePrice
        from Fut_TradingQuote a,Fut_FuturesContract b ,Fut_ContractMain c
        where b.TradingCode = '{ula_code}' and b.Exchange in (10,11,13,15,20) and a.OptionCode = b.ContractOption and a.ContractInnerCode = c.ContractInnerCode
        and a.SeriesFlag = {series_flag} and (a.TradingDay between '{start_date}' and '{end_date}')
        order by TradingDay
        """
        data = pd.read_sql(sql,self.conn_jy)
        data['str_date'] = data['TradingDay'].dt.strftime('%Y%m%d')
        # 有出现乌龙指，则做删除：
        if len(self.change_abnormal_prices)>0:
            if ula_code not in self.change_abnormal_prices:
                return data
            change_ctrcts = self.change_abnormal_prices[ula_code]
            # ctrct_data_part = data[data['ContractCode'].isin(change_ctrcts.keys())]
            for ctrct in change_ctrcts:
                change_dates_part = data[(data['ContractCode']==ctrct) & (data['str_date'].isin(change_ctrcts[ctrct].keys()))]
                for change_date in change_ctrcts[ctrct]:
                    change_infos = change_ctrcts[ctrct][change_date]
                    targ_idx = change_dates_part[change_dates_part['str_date'] == change_date].index
                    if len(targ_idx) > 1:
                        raise ValueError(f'{ctrct}交易日{change_date} 查到了多条数据')
                    for change_info in change_infos:
                        if change_info[0] not in data.columns:
                            raise ValueError(f'target change  column:{change_info[0]} NOT IN dataframe columns')
                        data.loc[targ_idx,change_info[0]] = change_info[1]
        return data

    def prep_fut_data(self,crrt_m_fut:pd.DataFrame,next_m_fut:pd.DataFrame,manually_add_tdy_price:dict = None):
        if manually_add_tdy_price:
            crrt_m_fut = self.manually_add_tdy_price(crrt_m_fut,manually_add_tdy_price[
                crrt_m_fut.loc[len(crrt_m_fut)-1,'ContractCode']],copy_col=['LastTradingDate','ContractCode'])
            next_m_fut = self.manually_add_tdy_price(next_m_fut, manually_add_tdy_price[
                next_m_fut.loc[len(next_m_fut) - 1, 'ContractCode']], copy_col=['LastTradingDate', 'ContractCode'])

        # crrt_m_fut['DailyReturn'] = crrt_m_fut['ClosePrice'] / crrt_m_fut['PrevClosePrice']
        # next_m_fut['DailyReturn'] = next_m_fut['ClosePrice'] / next_m_fut['PrevClosePrice']
        crrt_m_fut['str_date'] = crrt_m_fut['TradingDay'].apply(lambda x:x.strftime('%Y%m%d'))
        crrt_m_fut['str_exp_date'] = crrt_m_fut['LastTradingDate'].apply(lambda x: x.strftime('%Y%m%d'))
        roll_part = crrt_m_fut[crrt_m_fut['str_date'] == crrt_m_fut['str_exp_date']].index
        targ_cols = ['TradingDay','PrevClosePrice','OpenPrice','HighPrice','LowPrice','ClosePrice']
        df = crrt_m_fut.loc[:,targ_cols]
        df.loc[roll_part,targ_cols] = next_m_fut.loc[roll_part,targ_cols]
        df['DailyReturn'] = df['ClosePrice'] / df['PrevClosePrice']
        return df

    def main_cal(self,ula_code,methods,periods,start_date):
        """

        :param ula_code:
        :param methods:
        :param periods:
        :return:
        """
        res_dict = dict()
        tdy_date = datetime.date.today().strftime('%Y%m%d')
        crrt_m = self.get_fut_data(ula_code,start_date,tdy_date,1)
        next_m = self.get_fut_data(ula_code, start_date, tdy_date, 2)
        data = self.prep_fut_data(crrt_m,next_m)
        for met in methods:
            res_dict[met] = self.cal_vol(data,met,periods)
        return res_dict


class VolOutputMnger:
    """ Output Manager: 将输出保存在txt与截屏中
    """
    @staticmethod
    def cal_hist_vol(targ_ula_list,methods,periods,vol_calculator:RVCalculator,start_date):
        res_dict = dict()
        for ula_code in targ_ula_list:

            res = vol_calculator.main_cal(ula_code,methods,periods,start_date)
            for met in methods:
                temp_res = res[met]
                temp_res['str_date'] = temp_res['TradingDay'].dt.strftime('%Y%m%d')
                temp_res = temp_res.set_index('str_date')
                res[met] = temp_res[[f'{met}_rv1W', f'{met}_rv1M']].apply(lambda x:round(x,4)).reset_index()
            res_dict[ula_code] = res
        return res_dict


def main_vol():
    etf_vol = RVCalculator()
    methods = ['C2C', 'RMS', 'YangZhang', 'EWMA']
    pers = {'rv1W': 5, 'rv2W': 10, 'rv1M': 21, 'rv2M': 41, 'rv3M': 62, 'rv4M': 83,
            'rv5M': 104, 'rv6M': 124}
    # IM 乌龙指，将开盘价换成指数开盘
    fut_vol = FutRVCalculator(change_abnormal_prices={'IM':{'IM2211': {'20221108': [['OpenPrice', 6707.01],
                                                                                    ['LowPrice',6647.44]]
                                                                       }}})
    targ_etf = ['510050','510300','510500']
    targ_fut = ['IH','IF','IC','IM']
    vom=VolOutputMnger()
    res_etf = vom.cal_hist_vol(targ_etf, methods, pers, etf_vol,'20200110')
    res_fut = vom.cal_hist_vol(targ_fut, methods, pers, fut_vol,'20200110')


if __name__ == '__main__':
    main_vol()