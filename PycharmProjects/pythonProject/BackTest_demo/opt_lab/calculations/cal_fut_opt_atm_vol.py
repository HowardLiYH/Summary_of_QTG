from typing import Union

import pandas as pd
from myconfig import SqlConnManager, OnlineDiskManager
from meta import ETFOptContracts, IdxOptContracts, CommOptContracts, FutContracts

from opt_lab.calculations.RV_prev import RVCalculator, FutRVCalculator
from opt_lab.calculations.cal_atm_vol import CalETFATMVol
from opt_lab.configs.deriv_info import idx_to_related_fut
from opt_lab.configs.public_instance import eu_opt_cal, am_opt_cal, dtc, jy_coder
from optcal import EuOptCalculator, AmOptCalculator


class CalFutOptATMVol(CalETFATMVol):
    def _query_one_fut_close(self, ctrct_code, targ_date):
        sql = f"""
        select b.ContractCode,a.ClosePrice
        from Fut_ContractMain b
        left join (select * from Fut_TradingQuote a where a.TradingDay = '{targ_date}') a
        on a.ContractInnerCode = b.ContractInnerCode
        where b.ContractCode = '{ctrct_code}' and b.ExchangeCode in ({str(jy_coder.get_fut_exg_list())[1:-1]})
        """
        return pd.read_sql(sql, self.conn_jy).loc[0, 'ClosePrice']

    def query_fut_price(self, fut_metas: FutContracts, start_date, end_date):
        res = pd.DataFrame()
        res['date'] = list(self._tc.get_trding_day_range(start_date, end_date))
        fut_cls = []
        for d in res['date']:
            _fut_cls = [self._query_one_fut_close(c.contract_code, d) for c in fut_metas.get_contract_series(d)]
            fut_cls.append(_fut_cls)

        res[[f'fut_close_{i + 1}m' for i in range(len(fut_cls[0]))]] = fut_cls
        return res

    def cal_atm_vol(self, opt_ula_code, ula_type, start_date, end_date):
        """

        :param opt_ula_code:
        :param ula_type: idx or etf or comm
        :param start_date:
        :param end_date:
        :return:
        """
        opt_metas = self._get_opt_metas(opt_ula_code, ula_type)
        start_date, end_date = dtc.adj_date_range(start_date, end_date)
        # targ_dates = self._tc.get_trding_day_range(start_date,end_date)
        res_df = self.query_fut_price(opt_metas.ula_fut_meta, start_date, end_date)

        res_df.set_index('date', inplace=True, drop=False)
        atm_vol = dict()
        ctrct_series = dict()
        ttm_info = dict()
        for targ_date in res_df['date']:
            # main_series = self._choose_main_opt_series(targ_date,opt_metas)
            all_series = opt_metas.get_dueyymm_series(targ_date)
            fut_series = {c.full_year_code for c in opt_metas.ula_fut_meta.get_contract_series(targ_date)}
            all_series = [s for s in all_series if s in fut_series]
            if len(all_series) < len(fut_series):
                all_series = [all_series[0]] * (len(fut_series) - len((all_series))) + all_series
            for i in range(len(all_series)):
                if i not in ctrct_series:
                    ctrct_series[i] = []
                if i not in atm_vol:
                    atm_vol[i] = []
                if i not in ttm_info:
                    ttm_info[i] = []

                this_series = all_series[i]
                ula_price = res_df.loc[targ_date, f'fut_close_{i + 1}m']
                combo = opt_metas.get_atm_combo(this_series, targ_date, ula_price)
                cal = self._calculator[ula_type]
                vol_res = dict()
                for cp_type in ['C', 'P']:
                    opt_meta = combo[cp_type]
                    opt_prc = self.query_opt_price(ula_type, opt_meta.trading_code, opt_meta.contract_code, targ_date)
                    opt_vol = cal.get_imp_vol(cp_type, ud_prc=ula_price, k=float(opt_meta.get_strike(targ_date)),
                                              ttm=self._tc.get_days_diff(targ_date, opt_meta.get_exp_date()) / 245,
                                              r=0.02, div=0., option_prc=opt_prc)
                    vol_res[cp_type] = opt_vol

                atm_vol[i].append((vol_res['C'] + vol_res['P']) / 2)
                ctrct_series[i].append(this_series)
                ttm_info[i].append(self._tc.get_days_diff(targ_date, opt_metas.get_dueday(this_series)))
                print(targ_date, this_series, 'done')
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

        res_df.reset_index(drop=True).to_feather(self._root.joinpath(f'{opt_ula_code}.ftr'))
        print(opt_ula_code, 'done')
        return res_df


if __name__ == '__main__':
    cav = CalFutOptATMVol()
    xo_res = cav.cal_atm_vol('000852', 'idx', dtc.query_opt_start_date('000852').strftime('%Y%m%d'), '20230317')
    # xo_res = cav.cal_atm_vol('000300','idx',dtc.query_opt_start_date('000300').strftime('%Y%m%d'),'20191230')
