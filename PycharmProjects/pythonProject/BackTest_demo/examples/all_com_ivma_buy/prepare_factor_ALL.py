from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from public_tools.MyTT import ATR, TR, IDATR, IDTR, PKS, O2CPKS, C2CPKS, RSI
import pandas as pd


class FactorPreparer:
    def _load_pickle_data(self, factor_root):
        _factors = []
        for data_folder in factor_root.iterdir():
            for file in data_folder.iterdir():
                df = pd.read_pickle(file)
                # df['term'] = str(data_folder)[-4:]
                _factors.append(df)
        factor_df = pd.concat(_factors)
        return factor_df

    def calc_rv(self, factor_df, ret_rate_col, calc_window, targ_rv_name, adj=False):
        if not adj:
            if targ_rv_name in ['C2C', 'C2O', 'intraday']:
                return factor_df[ret_rate_col].rolling(calc_window).std() * sqrt(250)
            elif targ_rv_name == 'ATR':
                return ATR(factor_df['close'], factor_df['high'], factor_df['low'], calc_window)
            elif targ_rv_name == 'IDATR':
                return IDATR(factor_df['open'], factor_df['close'], calc_window)
            elif targ_rv_name == 'PKS':
                return PKS(factor_df['high'], factor_df['low'], calc_window)
            elif targ_rv_name == 'O2CPKS':
                return O2CPKS(factor_df['open'], factor_df['close'], calc_window)
            elif targ_rv_name == 'C2CPKS':
                return C2CPKS(factor_df['close'], calc_window)
            else:
                raise ValueError(fr"unknown target rv: {targ_rv_name}")
        else:
            if targ_rv_name in ['C2C', 'C2O', 'intraday']:
                return factor_df[ret_rate_col].rolling(calc_window).std() * sqrt(250)
            elif targ_rv_name == 'ATR':
                return (TR(factor_df['close_adj'], factor_df['prev_close_adj'], factor_df['high_adj'],
                           factor_df['low_adj']) / factor_df['_adj_factor']).rolling(calc_window).mean()
            elif targ_rv_name == 'IDATR':
                return (IDTR(factor_df['open_adj'], factor_df['close_adj'], factor_df['prev_close_adj']) / factor_df[
                    '_adj_factor']).rolling(calc_window).mean()
            elif targ_rv_name == 'PKS':
                return PKS(factor_df['high_adj'], factor_df['low_adj'], calc_window)
            elif targ_rv_name == 'O2CPKS':
                return O2CPKS(factor_df['open_adj'], factor_df['close_adj'], calc_window)
            elif targ_rv_name == 'C2CPKS':
                return C2CPKS(factor_df['close_adj'], factor_df['prev_close_adj'], calc_window)
            else:
                raise ValueError(fr"unknown target rv: {targ_rv_name}")

    def process_factor(self, kwargs):
        _factor_root = Path(kwargs['factor_root'])

        if 'adj' not in kwargs or not kwargs['adj']:
            factor_df = self._load_pickle_data(_factor_root)
        else:
            factor_df = pd.read_feather(_factor_root)
            factor_df.index = pd.to_datetime(factor_df['str_date'])

        iv_col = kwargs['iv_col']

        ivma_fast = kwargs['ivma_fast']
        ivma_slow = kwargs['ivma_slow']
        RIS_p = kwargs['rsi_p']

        # ---- 全品种聚合的信号 ------------
        if kwargs['sig'] == 'all':
            all_df = pd.read_pickle(rf'\\192.168.0.88\Public\OptionDesk\DATA\ResearchData\OptLab\atm_vol\all_atm_vol.pkl')
            all_df['ivma_fast'] = all_df['all_atm'].rolling(ivma_fast, min_periods=ivma_fast).mean()
            all_df['ivma_slow'] = all_df['all_atm'].rolling(ivma_slow, min_periods=ivma_slow).mean()
            all_df['all_atm_RSI'] = RSI(all_df['all_atm'], RIS_p)


            all_df['entrysignal'] = False
            # shortvol_cond = (all_df['all_atm'] < all_df['ivma_fast']) & (all_df['ivma_fast'] < all_df['ivma_slow'])
            # shortvol_cond = (all_df['all_atm'] < all_df['ivma_fast'])

            cond = (all_df['all_atm'] > all_df['ivma_fast']) & (all_df['ivma_fast'] > all_df['ivma_slow'])

            # cond = (all_df['all_atm'] > all_df['ivma_fast']) & (all_df['ivma_fast'] > all_df['ivma_slow']) & \
            #        (all_df['ivma_slow'] > all_df['all_atm'].rolling(30, min_periods=30).mean())

            # cond = (all_df['all_atm'] > all_df['ivma_fast']) & (all_df['ivma_fast'] > all_df['ivma_slow']) & \
            #        (all_df['all_atm_RSI'] > 55)
            # cond = (all_df['ivma_fast'] > all_df['ivma_slow'])

            if iv_col == 'atm_vol_1m':
                factor_df.loc[factor_df['ttm_1m'] <= 10, 'atm_vol_1m'] = factor_df.loc[factor_df['ttm_1m'] <= 10, 'atm_vol_2m']
                factor_df.ffill(inplace=True)

            factor_df['my_RSI'] = RSI(factor_df[iv_col], RIS_p)
            factor_df['my_ivma_fast'] = factor_df[iv_col].rolling(ivma_fast, min_periods=1).mean()
            factor_df['my_ivma_slow'] = factor_df[iv_col].rolling(ivma_slow, min_periods=1).mean()

            # cond = cond & (factor_df['my_ivma_fast'] > factor_df['my_ivma_slow']) & \
            #        (factor_df['my_RSI'] > 0)

            # cond = cond & (factor_df['my_ivma_fast'] > factor_df['my_ivma_slow']) & \
            #        (factor_df['my_ivma_slow'] > factor_df[iv_col].rolling(30, min_periods=1).mean())

            # cond = cond & (factor_df[iv_col] > factor_df['my_ivma_fast']) & (factor_df['my_ivma_fast'] > factor_df['my_ivma_slow'])

            cond = cond & (factor_df['my_ivma_fast'] > factor_df['my_ivma_slow'])

            all_df.loc[cond, 'entrysignal'] = True

            factor_df = pd.concat([factor_df, all_df['entrysignal']], axis=1).dropna()
        # ------------------------------

        # ---- 自己品种的信号 ------------
        if kwargs['sig'] == 'own':
            if iv_col == 'atm_vol_1m':
                factor_df.loc[factor_df['ttm_1m'] <= 10, 'atm_vol_1m'] = factor_df.loc[factor_df['ttm_1m'] <= 10, 'atm_vol_2m']
                factor_df.ffill(inplace=True)
            factor_df['my_RSI'] = RSI(factor_df[iv_col], RIS_p)
            factor_df['my_ivma_fast'] = factor_df[iv_col].rolling(ivma_fast, min_periods=1).mean()
            factor_df['my_ivma_slow'] = factor_df[iv_col].rolling(ivma_slow, min_periods=1).mean()

            # cond = (factor_df['my_ivma_fast'] > factor_df['my_ivma_slow']) & \
            #        (factor_df['my_RSI'] > 55)
            factor_df['entrysignal'] = False

            # cond = (factor_df[iv_col] > factor_df['my_ivma_fast']) & (factor_df['my_ivma_fast'] > factor_df['my_ivma_slow']) & \
            #        (factor_df['my_RSI'] > 55)

            # cond = (factor_df[iv_col] > factor_df['my_ivma_fast']) & (factor_df['my_ivma_fast'] > factor_df['my_ivma_slow'])
            cond = (factor_df['my_ivma_fast'] > factor_df['my_ivma_slow'])

            factor_df.loc[cond, 'entrysignal'] = True
        # ------------------------------
        # factor_df.dropna(inplace=True)
        if kwargs['begindate']:
            begindate = pd.Timestamp(kwargs['begindate'])
            factor_df = factor_df[factor_df.index >= begindate]
        if kwargs['enddate']:
            enddate = pd.Timestamp(kwargs['enddate'])
            factor_df = factor_df[factor_df.index <= enddate]
        return factor_df
