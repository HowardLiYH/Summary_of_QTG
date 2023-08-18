from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from data_manager import DataManager
from data_manager.data_processors import BuySellPriceDefiner
from strategies.all_com_ivma_buy.mylongvol import *


def test_singleYYMM(df, YYMM, data_manager: DataManager, rollpos, kwargs):
    _delay_p = kwargs['delay_p']
    _optmeta = kwargs['optmeta']
    _Strat = kwargs['strategy']
    _fillthennext = kwargs['fillthennext']
    _rollterm_day = kwargs['rollterm_day']
    _delta_th = kwargs['delta_th']
    _Klvl = kwargs['Klvl']
    _commodity = kwargs['commodity']

    dtc = kwargs['dtc']

    # if _Strat == Strat2 or _Strat == Strat5:
    strat = _Strat(df, data_manager, YYMM, _delay_p, _optmeta, _rollterm_day,
                   _delta_th, _Klvl, dtc, rollpos=rollpos, fillthennext=_fillthennext, commodity=_commodity)  # todo delete kwargs
    # else:
    #     raise ValueError('For now, use Strategy 2 or 5 only')
    # if _Strat == Strat3:
    #     _rollinK_day = kwargs['rollinK_day']
    #     strat = _Strat(df, data_manager, YYMM, _delay_p, _optmeta, _rollterm_day,
    #                    _delta_th, _Klvl, _rollinK_day, dtc, rollpos=rollpos, fillthennext=_fillthennext)

    return strat.run()


class myprocessor(BuySellPriceDefiner):
    def process(self, data) -> pd.DataFrame:
        df = self.use_price(data, price_col='close')
        df['next_open'] = df['open'].shift(-1)
        return df


class TempTools:
    @staticmethod
    def draw_heatmap(df, output_name, out_folder: Path, vmin=None, vmax=None):
        plt.figure(figsize=(15, 9))
        plt.title(output_name)
        sns.heatmap(data=df, cmap=sns.diverging_palette(220, 10, sep=1, n=10), annot=True, fmt=".2f", vmin=vmin,
                    vmax=vmax)
        plt.savefig(out_folder.joinpath(f'{output_name}.png'))
        plt.close()
