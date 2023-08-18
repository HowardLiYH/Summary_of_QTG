from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from public_tools.MyTT import ATR, TR, IDATR, IDTR, PKS, O2CPKS, C2CPKS
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

        hv_short_win = kwargs['hv_short_win']
        hv_long_win = kwargs['hv_long_win']

        ma_fast = kwargs['ma_fast']
        ma_slow = kwargs['ma_slow']

        rv_col = kwargs['rv_col']
        hv_name = f'{rv_col}_HV'

        if 'adj' not in kwargs or not kwargs['adj']:
            factor_df['C2C_ret'] = factor_df['close'] / factor_df['prev_close'] - 1

            factor_df['C2O_ret'] = factor_df['open'] / factor_df['prev_close'] - 1
            factor_df['intraday_ret'] = factor_df['close'] / factor_df['open'] - 1

            factor_df[f'{hv_name}_short'] = self.calc_rv(factor_df, ret_rate_col=f'{rv_col}_ret',
                                                         calc_window=hv_short_win, targ_rv_name=rv_col)
            factor_df[f'{hv_name}_long'] = self.calc_rv(factor_df, ret_rate_col=f'{rv_col}_ret',
                                                        calc_window=hv_long_win, targ_rv_name=rv_col)

        else:
            factor_df['C2C_ret'] = factor_df['close_adj'] / factor_df['prev_close_adj'] - 1

            factor_df['C2O_ret'] = factor_df['open_adj'] / factor_df['prev_close_adj'] - 1
            factor_df['intraday_ret'] = factor_df['close_adj'] / factor_df['open_adj'] - 1

            factor_df[f'{hv_name}_short'] = self.calc_rv(factor_df, ret_rate_col=f'{rv_col}_ret',
                                                         calc_window=hv_short_win, targ_rv_name=rv_col, adj=True)
            factor_df[f'{hv_name}_long'] = self.calc_rv(factor_df, ret_rate_col=f'{rv_col}_ret',
                                                        calc_window=hv_long_win, targ_rv_name=rv_col, adj=True)

        # 由于已经确定了两种交易mode（见 readme），所以只需要算特定的列，并且约定 hv slow 只与 ma slow 搭配；hv fast 只与ma fast 搭配
        factor_df[f'{hv_name}_short.ma_fast'] = factor_df[f'{hv_name}_short'].rolling(ma_fast).mean()
        factor_df[f'{hv_name}_long.ma_slow'] = factor_df[f'{hv_name}_long'].rolling(ma_slow).mean()

        factor_df['shortvol'] = False

        shortvol_cond = factor_df[f'{hv_name}_short.ma_fast'] < factor_df[f'{hv_name}_long.ma_slow']

        # factor_df.loc[factor_df['ttm_1m'] <= 3, 'atm_vol_1m'] = None
        # factor_df.ffill(inplace=True)

        factor_df.loc[shortvol_cond, 'shortvol'] = True

        factor_df.dropna(subset=[i for i in factor_df.columns if 'ma_slow' in i], inplace=True)
        if kwargs['begindate']:
            begindate = pd.Timestamp(kwargs['begindate'])
            factor_df = factor_df[factor_df.index >= begindate]
        if kwargs['enddate']:
            enddate = pd.Timestamp(kwargs['enddate'])
            factor_df = factor_df[factor_df.index <= enddate]

        # factor_df.loc[factor_df['ttm_1m'] <= 3, 'atm_vol_1m'] = None
        # factor_df.ffill(inplace=True)
        #
        # factor_df['atm_vol_1m'].plot(label=f'atm_vol_1m_{kwargs["ula"]}', figsize=(15, 9))
        # factor_df[f'{hv_name}_long'].plot(label=f'RV_{hv_name}win{kwargs["hv_long_win"]}_{kwargs["ula"]}', figsize=(15, 9))
        # (factor_df['atm_vol_1m'] - factor_df[f'{hv_name}_long']).plot(label=f'risk_premium_{kwargs["ula"]}', figsize=(15, 9))
        # plt.axhline(0)
        # plt.title(f"{kwargs['ula']}")
        # plt.legend()
        # plt.savefig(f"plots/IV_vs_RV/IV1m_{rv_col}_{kwargs['ula']}.png")
        # plt.close()

        return factor_df
