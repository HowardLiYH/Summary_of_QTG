from typing import Union

import pandas as pd
from myconfig import SqlConnManager, OnlineDiskManager
from meta import ETFOptContracts, IdxOptContracts, CommOptContracts, FutContracts

from opt_lab.calculations.RV_prev import RVCalculator, FutRVCalculator
from opt_lab.configs.deriv_info import idx_to_related_fut
from opt_lab.configs.public_instance import eu_opt_cal, am_opt_cal, dtc, jy_coder
from optcal import EuOptCalculator, AmOptCalculator


class CalETFATMVol:
    def __init__(self):
        self.conn_jy = SqlConnManager.conn_jy()
        self._loaded_metas = dict()
        self._meta_constructor = {'etf': ETFOptContracts, 'idx': IdxOptContracts, 'comm': CommOptContracts}
        self._calculator = {'etf': eu_opt_cal, 'idx': eu_opt_cal, 'comm': am_opt_cal}
        self._tc = dtc
        self._root = OnlineDiskManager.get_research_data_path().joinpath('OptLab/atm_vol')

        # import rv calculator
        self.etf_vol = RVCalculator()
        self._rv_methods = ['C2C', 'RMS', 'YangZhang', 'EWMA']
        self._rv_pers = {'rv1w': 5, 'rv4w': 21, }
        # IM 乌龙指，将开盘价换成指数开盘
        self.fut_vol = FutRVCalculator(change_abnormal_prices={'IM': {'IM2211': {'20221108': [['OpenPrice', 6707.01],
                                                                                              ['LowPrice', 6647.44]]
                                                                                 }}})

    def query_ula_spot_price(self, ula_code, ula_type, start_date, end_date):
        if ula_type == 'idx':
            table_name = '[jydb].[dbo].[QT_IndexQuote]'
            jy_type = 4
        elif ula_type == 'etf':
            table_name = '[jydb].[dbo].QT_FundsPerformanceHis'
            jy_type = 8
        else:
            raise ValueError('spot price only applies to index opt and etf opt')

        sql = f"""
        SELECT SecuCode,TradingDay,ClosePrice as cls_prc,PrevClosePrice as prev_cls
        FROM {table_name} a ,SecuMain b -- 谨慎使用定制表，因为会缺失历史数据
        where b.SecuCode = '{ula_code}' and (TradingDay between '{start_date}' and '{end_date}') 
        and b.InnerCode = a.InnerCode and b.SecuCategory = {jy_type}
        order by TradingDay asc   
        """
        ula_price_df = pd.read_sql(sql, self.conn_jy)
        return ula_price_df

    def _get_opt_metas(self, ula_code, ula_type) -> Union[ETFOptContracts, IdxOptContracts, CommOptContracts]:
        if ula_code not in self._loaded_metas:
            if ula_type == 'etf':
                self._loaded_metas[ula_code] = self._meta_constructor[ula_type](ula_code)
            elif ula_type == 'idx':
                self._loaded_metas[ula_code] = self._meta_constructor[ula_type](ula_code, FutContracts(
                    idx_to_related_fut[ula_code]))
            elif ula_type == 'comm':
                self._loaded_metas[ula_code] = self._meta_constructor[ula_type](ula_code, FutContracts(ula_code))
            else:
                raise ValueError(f'wrong type:{ula_type}')
        return self._loaded_metas[ula_code]

    def _choose_main_opt_series(self, targ_date, opt_metas: CommOptContracts):
        """
        对于etf 期权来讲，是否是主力，由到期日判断。里到期日两天以上，取当月，否则取次月。
        对于其他两个有ula fut 的来讲，主力与 ula fut 一致

        :param targ_date:
        :param opt_metas:
        :return:
        """
        if opt_metas.opt_type == 'etf_opt':
            crrt_main_ttm = self._tc.get_days_diff(opt_metas.get_dueday_series(targ_date)[0], targ_date)
            if crrt_main_ttm >= 2:
                return opt_metas.get_dueyymm_series(targ_date)[0]
            else:
                return opt_metas.get_dueyymm_series(targ_date)[1]
        else:
            crrt_main_fut = opt_metas.get_main_ula(targ_date)
            crrt_opt_full = crrt_main_fut.full_code[len(crrt_main_fut.prod):]
            if crrt_opt_full not in opt_metas.exp_dates:
                return opt_metas.ula_fut_meta.get_next_main(crrt_main_fut).full_year_code

            crrt_main_ttm = self._tc.get_days_diff(opt_metas.get_dueday(crrt_opt_full), targ_date)
            if crrt_main_ttm >= 2:
                return crrt_main_fut.full_year_code
            else:
                return opt_metas.ula_fut_meta.get_next_main(crrt_main_fut).full_year_code

    def query_opt_price(self, ula_type, trading_code, ctrct_code, targ_date):
        """
        查询期权合约收盘价。如果没有收盘价则取结算价

        :param ula_type:
        :param trading_code:
        :param ctrct_code:
        :param targ_date:
        :return:
        """
        if ula_type == 'etf':
            sql = f"""
            select b.ContractCode,b.ListingDate,a.OpenPrice,a.HighPrice,a.LowPrice,a.ClosePrice,a.SettlePrice
            from Opt_OptionContract b
            left join (select * from Opt_DailyQuote as a where a.TradingDate = '{targ_date}') as a
            on b.InnerCode = a.InnerCode
            where b.ContractCode = '{ctrct_code}' and b.Exchange in ({str(jy_coder.get_opt_exg_list())[1:-1]}) 
            """
        else:
            sql = f"""
            select b.TradingCode,b.ListingDate,a.OpenPrice,a.HighPrice,a.LowPrice,a.ClosePrice,a.SettlePrice
            from Opt_OptionContract b
            left join (select * from Opt_DailyQuote as a where a.TradingDate = '{targ_date}') as a
            on b.InnerCode = a.InnerCode
            where b.TradingCode = '{trading_code}' and b.Exchange in ({str(jy_coder.get_opt_exg_list())[1:-1]}) 
            """
        res = pd.read_sql(sql, self.conn_jy)
        assert len(res) == 1, '查询到 0 个 或 多个合约。'
        return res.loc[0, 'ClosePrice'] if res.loc[0, 'ClosePrice'] else res.loc[0, 'SettlePrice']

    def cal_atm_vol(self, opt_ula_code, ula_type, start_date, end_date):
        """

        :param opt_ula_code:
        :param ula_type: idx or etf or comm
        :param start_date:
        :param end_date:
        :return:
        """
        ula_price = self.query_ula_spot_price(opt_ula_code, ula_type, start_date, end_date)
        opt_metas = self._get_opt_metas(opt_ula_code, ula_type)
        start_date, end_date = dtc.adj_date_range(start_date, end_date)
        targ_dates = self._tc.get_trding_day_range(start_date, end_date)
        res_df = pd.DataFrame()
        res_df['date'] = targ_dates
        res_df['ula_close'] = ula_price['cls_prc']
        res_df.set_index('date', inplace=True)
        atm_vol = dict()
        ctrct_series = dict()
        ttm_info = dict()
        for targ_date in targ_dates:
            # main_series = self._choose_main_opt_series(targ_date,opt_metas)
            all_series = opt_metas.get_dueyymm_series(targ_date)
            for i in range(len(all_series)):
                if i not in ctrct_series:
                    ctrct_series[i] = []
                if i not in atm_vol:
                    atm_vol[i] = []
                if i not in ttm_info:
                    ttm_info[i] = []

                this_series = all_series[i]
                combo = opt_metas.get_atm_combo(this_series, targ_date, ula_price=res_df.loc[targ_date, 'ula_close'])
                cal: EuOptCalculator = self._calculator[ula_type]
                vol_res = dict()
                for cp_type in ['C', 'P']:
                    opt_meta = combo[cp_type]
                    opt_prc = self.query_opt_price(ula_type, opt_meta.trading_code, opt_meta.contract_code, targ_date)
                    opt_vol = cal.get_imp_vol(cp_type, ud_prc=res_df.loc[targ_date, 'ula_close'],
                                              k=float(opt_meta.get_strike(targ_date)),
                                              ttm=self._tc.get_days_diff(targ_date, opt_meta.get_exp_date()) / 245,
                                              r=0.02, div=0., option_prc=opt_prc)

                    vol_res[cp_type] = opt_vol

                atm_vol[i].append((vol_res['C'] + vol_res['P']) / 2)
                ctrct_series[i].append(this_series)
                ttm_info[i].append(self._tc.get_days_diff(targ_date, opt_metas.get_dueday(this_series)))
        # res_df['atm_vol'] = atm_vol
        # res_df['main_series'] = ctrct_series

        for vol_col in atm_vol:
            res_df[f'atm_vol_{vol_col + 1}m'] = atm_vol[vol_col]
        for ctrct_col in ctrct_series:
            res_df[f'ctrct_{ctrct_col + 1}m'] = ctrct_series[ctrct_col]
        for ttm_col in ttm_info:
            res_df[f'ttm_{ttm_col + 1}m'] = ttm_info[ttm_col]

        rv_start_date = self._tc.get_last_nth_trd_day(start_date, 30)  # ula默认取到30个交易日之前
        res_df = self._cal_rv(res_df, ula_type, opt_ula_code, rv_start_date)

        res_df.reset_index().to_feather(self._root.joinpath(f'{opt_ula_code}.ftr'))
        print(opt_ula_code, 'done')
        return res_df

    def _cal_rv(self, imp_vol_res: pd.DataFrame, ula_type, ula_code, start_date):
        vol_calculator = self.etf_vol if ula_type == 'etf' else self.fut_vol
        ula_code = idx_to_related_fut[ula_code] if ula_type == 'idx' else ula_code
        res = vol_calculator.main_cal(ula_code, self._rv_methods, self._rv_pers, start_date)
        for met in self._rv_methods:
            temp_res = res[met]
            temp_res['str_date'] = temp_res['TradingDay'].dt.strftime('%Y%m%d')
            temp_res = temp_res[['str_date', f'{met}_rv1w', f'{met}_rv4w']].set_index('str_date')

            imp_vol_res = imp_vol_res.merge(temp_res, how='left', left_index=True, right_index=True)
        return imp_vol_res


if __name__ == '__main__':
    cav = CalETFATMVol()
    # xo_res = cav.cal_atm_vol('000300','idx',dtc.query_opt_start_date('000300').strftime('%Y%m%d'),'20230317')
    etf_res = cav.cal_atm_vol('510050', 'etf', dtc.query_opt_start_date('510050').strftime('%Y%m%d'), '20230317')
    print()
