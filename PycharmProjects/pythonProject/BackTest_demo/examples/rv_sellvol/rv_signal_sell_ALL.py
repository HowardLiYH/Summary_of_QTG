import json
from myconfig import OnlineDiskManager
import pandas as pd
from meta import FutContracts, ETFOptContracts, IdxOptContracts, CommOptContracts
from timecal import DateTimeCalculator
import datetime
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import seaborn as sns
from data_manager import data_processors
from data_manager.data_processors import BuySellPriceDefiner
# from ..data_manager.data_processors import BuySellPriceDefiner
import os
from data_manager.manager import DataManager
from scipy.stats import skew, kurtosis
# from qtg_option_desk_lib.qtg_tools.MyTT.MyTT import RSI

from temp_tools import TempTools
from myconfig import OnlineDiskManager
from temp_tools import test_singleYYMM, myprocessor
from myshortvol import Strat2, Strat5
from prepare_factor_ALL import FactorPreparer


def test_allYYMM(kwargs):
    _fp = FactorPreparer()

    # ----- user define -----
    optmeta = kwargs['optmeta']
    futmeta = kwargs['futmeta']
    factor_df = _fp.process_factor(kwargs)
    # -----------------------

    rollpos = None
    trade_pnl_list = []
    pos_time = []
    pnl_list = []
    pnl_update = pd.DataFrame()
    pnl_values = pd.DataFrame()
    day_pnl = pd.DataFrame()

    # if 'term' not in factor_df.columns:
    #     factor_df['term'] = factor_df['opt_1m']

    # return 0, 0, 0

    for YYMM, YYMMfactor_df in factor_df.groupby('ctrct_1m'):

        dm = DataManager(data_processor=myprocessor())

        # ----- add ETF data to instrument_dict -----
        if kwargs['etf_root']:
            _etf_root = pathlib.Path(rf"{kwargs['etf_root']}\{YYMM}")
            for files in _etf_root.iterdir():
                dm.add_instrument(files.stem, files, 'index',
                                  add_info_dict={'mult': kwargs['etfmult'], 'slipfix': kwargs['etf_slipfix']})

        # ----- add idx data to instrument_dict -----
        if kwargs['idx_root']:
            _idx_root = pathlib.Path(rf"{kwargs['idx_root']}\{YYMM}")
            for contractfile in _idx_root.iterdir():
                dm.add_instrument(contractfile.stem, contractfile, 'index_opt',
                                  {'mult': kwargs['futmult'], 'slipfix': kwargs['fut_slipfix']})

        # ----- add option data to instrument_dict -----
        if kwargs['opt_root']:
            _opt_root = pathlib.Path(rf"{kwargs['opt_root']}\{YYMM}")
            for contractfile in _opt_root.iterdir():
                this_meta = optmeta.get_contract_meta(contractfile.stem)
                dm.add_instrument(contractfile.stem, contractfile, 'index_opt',
                                  {'mult': kwargs['optmult'], 'slipfix': kwargs['opt_slipfix']}, data_meta=this_meta)

        # ----- add fut data to instrument_dict -----
        if kwargs['fut_root']:
            _fut_root = pathlib.Path(rf"{kwargs['fut_root']}\{YYMM}")
            for contractfile in _fut_root.iterdir():
                this_meta = futmeta.get_contract_meta(contractfile.stem)
                dm.add_instrument(contractfile.stem, contractfile, 'index_opt',
                                  {'mult': kwargs['futmult'], 'slipfix': kwargs['fut_slipfix']}, data_meta=this_meta)

        # ---------------------------------------------

        pnl, rollpos, trades = test_singleYYMM(YYMMfactor_df, YYMM, dm, rollpos, kwargs)
        # trades_recorder.append([p for ctrct in trades for p in trades[ctrct] if p.closed])
        for dataname, positions in trades.items():
            for pos in positions:
                if pos.closed:
                    postime = (pos.closetime - pos.entrytime).total_seconds() / 3600
                    pos_time.append(postime)
                    trade_pnl_list.append(pos.size * (pos.closeprice - pos.entryprice) * pos.mult)
                # else:
                #     trade_pnl_list.append(pos.size * (pos.term_endprice - pos.entryprice) * pos.mult)

        pnl_list.append(pnl / kwargs['initial_cash'])
        if len(pnl_list) == 1:
            pnl_update = pnl_list[0]
            print()
        if len(pnl_list) >= 2:
            pnl_list[-1]['pnl'] = pnl_list[-2].iloc[-1]['pnl'] + pnl_list[-1]['pnl']
            pnl_update = pd.concat([pnl_update, pnl_list[-1]])
            print()

    pnl_values['values'] = pnl_update['pnl'] + 1
    day_pnl['values'] = pnl_values['values'].resample('1D').last().dropna()

    ### Calmar ###
    day_pnl['peak'] = day_pnl['values'].expanding(min_periods=1).max()
    day_pnl['drawdown'] = day_pnl['values'] / day_pnl['peak'] - 1
    day_pnl['nocompound_dd'] = day_pnl['values'] - day_pnl['peak']
    MDD = -day_pnl['drawdown'].min()
    # MDD = abs(day_pnl['nocompound_dd']).max()
    # Calmar = ((day_pnl['values'][-1] / day_pnl['values'][0]) ** (250 / len(day_pnl['values'])) - 1) / MDD
    AnnRet = (day_pnl['values'][-1] - 1) / (len(day_pnl['values']) / 240)
    Calmar = AnnRet / MDD

    #### win ratio
    if len(trade_pnl_list):
        num_positive = len([i for i in trade_pnl_list if i > 0])
        win_ratio = num_positive / len(trade_pnl_list)
        avg_win = np.mean([i for i in trade_pnl_list if i > 0])
        avg_loss = np.mean([i for i in trade_pnl_list if i < 0])
        pl_ratio = avg_win / (abs(avg_loss))
        trade_freq = len(trade_pnl_list) / (len(day_pnl['values']) / 240)
        pos_time_std = np.std(pos_time)
        pos_time_avg = sum(pos_time) / len(pos_time) if len(pos_time) else np.inf
    else:
        win_ratio = 0
        pl_ratio = 0
        trade_freq = 0
        pos_time_std = 0
        pos_time_avg = 0
    #### Max Drawdown #####
    day_pnl['peak'] = day_pnl['values'].expanding(min_periods=1).max()
    day_pnl['drawdown'] = day_pnl['values'] / day_pnl['peak'] - 1
    day_pnl['nocompound_dd'] = day_pnl['values'] - day_pnl['peak']
    day_MDD = -day_pnl['values'].diff().min()

    #### Sharpe Ratio #####
    day_pnl['day_return'] = day_pnl['values'].diff()
    day_pnl.loc[day_pnl.index[0], 'day_return'] = day_pnl.iloc[0]['values'] - 1
    vol = day_pnl['day_return'].std() / math.sqrt(240)
    Sharpe = (day_pnl['day_return'].mean() - 1.01 ** (1 / 240) + 1) / vol if vol else 0

    if 'ula' in kwargs:
        print(f"Product:             {kwargs['ula']}")
    print(f"Backtesting Period:  {day_pnl.index[0].date()} to {day_pnl.index[-1].date()}")
    print(f"End Value:           {day_pnl['values'][-1]}")
    print(f"Ann Return:          {AnnRet}")
    print(f"MDD abs:             {-day_pnl['nocompound_dd'].min()}")
    print(f"MDD compound:        {MDD}")
    print(f"Day MDD:             {day_MDD}")
    print(f"Sharpe Ratio:        {Sharpe}")
    print(f"Calmar Ratio:        {Calmar}")
    print(f"day Calmar Ratio:    {AnnRet / day_MDD}")
    print(f'Win Ratio:           {win_ratio}')
    print(f'Profit/Loss Ratio:   {pl_ratio}')
    print(f'trade_freq:          {trade_freq} trades/year')
    print(f'pos_time_avg:        {pos_time_avg} hours')
    print(f'pos_time_std:        {pos_time_std} hours')

    return day_pnl, {'AnnRet': AnnRet, 'Sharpe': Sharpe, 'Calmar': Calmar, 'trade_freq': trade_freq,
                     'pos_time_std': pos_time_std}, factor_df
    # return factor_df.index[0]


def main(ula, fut, ula_type):
    # ula = '510050'
    # ula = '510300'
    ##########################################
    strat = Strat2 if ula_type != 'commodity' else Strat5
    # ula = '510300'  # 510050, 510300, 510500, 000300, 000852
    # fut = 'IF'  # IH, IF, IC, IM
    # ula_type = 'ETF'  # ETF, index

    # begindate = '20170101'
    # enddate = '20211210'
    # begindate = '20150604'
    # begindate = '20200101'
    # begindate = '20220302'
    begindate = None
    enddate = '20230317'

    # opt_slipfix = 1  # for index, 滑点=1个指数点
    # opt_slipfix = 2 / 10000  # 单张合约交易成本1.6元， 单日垮式垮式开平仓包含滑点共8元
    rollterm_day = 1

    if ula_type == 'ETF':
        opt_slipfix = 3 / 10000
    elif ula_type == 'index':
        opt_slipfix = 1
    else:
        df = pd.read_csv(r"\\192.168.0.88\Public\OptionDesk\DATA\ResearchData\Opt_secumain\slippage.csv", index_col='prod')
        opt_slipfix = df.loc[ula, 'slipage'] / 2
        # opt_slipfix = 0
    Klvl = 1  # 指数改成3
    if ula_type == 'ETF':
        Klvl = 1
        # Klvl = 99
    elif ula_type == 'index':
        Klvl = 3
        # Klvl = 99
    elif ula_type == 'commodity':
        Klvl = 3
        # if ula == 'AU' or ula == 'SR':
        #     Klvl = 2
        # Klvl = 99
        rollterm_day = 10


    ##########################################

    factor_root = ''
    opt_root = ''
    initial_price = 0
    inital_cash = 0

    optmeta = None
    futmeta = None

    optmult = None
    futmult = None
    etfmult = None
    idxmult = None

    fut_root = r''
    etf_root = r''
    idx_root = r''

    fut_slipfix = 0
    etf_slipfix = 0
    idx_slipfix = 0

    if ula_type == 'ETF':
        optmeta = ETFOptContracts(ula)
        factor_root = OnlineDiskManager.get_mkt_db_path().joinpath(rf'1day_bar\etf_by_optmon\{ula}')
        opt_root = OnlineDiskManager.get_mkt_db_path().joinpath(rf'1day_bar\opt_by_optmon\{ula}')

    if ula_type == 'index':
        futmeta = FutContracts(fut)
        optmeta = IdxOptContracts(ula, futmeta)
        factor_root = OnlineDiskManager.get_mkt_db_path().joinpath(rf'1day_bar\idx_by_futmon\{ula}')
        opt_root = OnlineDiskManager.get_mkt_db_path().joinpath(rf'1day_bar\opt_by_futmon\{ula}')

    if ula_type == 'commodity':
        futmeta = FutContracts(fut)
        optmeta = CommOptContracts(ula, futmeta)
        factor_root = OnlineDiskManager.get_path_map(
            r"\\192.168.0.88\Public\OptionDesk\DATA\ResearchData\OptLab\atm_vol").joinpath(rf'{ula}.ftr')
        opt_root = OnlineDiskManager.get_mkt_db_path().joinpath(rf'1day_bar\opt_by_commoptmon\{ula}')
        fut_root = OnlineDiskManager.get_mkt_db_path().joinpath(rf'1day_bar\fut_by_commoptmon\{ula}')

    _factor_root = pathlib.Path(factor_root)

    adj = False
    if ula_type != 'commodity':
        start_data = pd.read_pickle(min([month for month in _factor_root.iterdir()]).joinpath(f'{ula}.pkl'))
        adj = False
    else:
        start_data = pd.read_feather(factor_root)
        adj = True

    begindate = begindate if begindate else start_data['str_date'][0]

    initial_price = start_data['open'].values[0]

    _opt_root = pathlib.Path(opt_root)
    _pick_one_meta = optmeta.get_contract_meta(next(iter(optmeta.metas.keys())))
    optmult = _pick_one_meta.get_contract_size(_pick_one_meta.get_start_date())

    initial_cash = initial_price * optmult

    print(f'prod = {ula}, begindate = {begindate}, initial_cash = {initial_cash}, opt_slipfix = {opt_slipfix}')

    dtc = DateTimeCalculator()

    # ret_dict = defaultdict(dict)

    delay_p = 0
    fillthennext = False
    IVterm = 1
    # delta_th = 0.4
    # res_dict = defaultdict(dict)

    # delta_th_list = [0.2, 0.3, 0.4, 0.5, 0.6]
    delta_th_list = [0.4]
    # delta_th_list = [0.2]
    # delta_th_list = [0.2]
    # delta_th_list = [0.6]
    #
    # hv_targ_cols = ['C2C', 'intraday', 'C2O', 'ATR', 'IDATR']
    # hv_targ_cols = ['C2C', 'intraday', 'C2O', 'ATR', 'IDATR', 'PKS', 'O2CPKS', 'C2CPKS']
    # hv_targ_cols = ['C2C', 'intraday', 'C2O', 'ATR', 'IDATR', 'PKS', 'O2CPKS', 'C2CPKS']  # C2CPKS在ETF上垃圾
    # hv_targ_cols = ['C2C', 'intraday', 'ATR', 'C2O', 'PKS']
    # hv_targ_cols = ['C2C', 'PKS']
    # hv_targ_cols = ['C2C', 'ATR']
    # hv_targ_cols = ['C2C', 'ATR', 'intraday', 'C2O']
    hv_targ_cols = ['C2C']
    # hv_targ_cols = ['IDATR']
    # hv_targ_cols = ['PKS', 'O2CPKS', 'C2CPKS']
    #
    # hv_fast_windows = [1, 2, 5, 10, 20, 30, 40, 50, 60, 90, 120, 150, 180, 210, 250]
    # hv_slow_windows = [5, 10, 20, 30, 40, 50, 60, 90, 120, 150, 180, 210, 250]

    # hv_fast_windows = [1, 2, 5, 10]
    # hv_slow_windows = [20, 30, 40, 50]

    # hv_fast_windows = [10]
    # hv_slow_windows = [20]
    # ma_fast_windows = [30]
    # ma_slow_windows = [30]

    ma_fast_windows = [5]
    ma_slow_windows = [5]
    hv_fast_windows = [5]
    hv_slow_windows = [40]
    #
    # ma_fast_windows = [5]
    # ma_slow_windows = [5]
    # hv_fast_windows = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # hv_slow_windows = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    #

    # ma_fast_windows = [1]
    # ma_slow_windows = [40]
    # hv_fast_windows = [30]
    # hv_slow_windows = [30]

    # ma_fast_windows = [1]
    # ma_slow_windows = [40, 45]
    # hv_fast_windows = [25, 30, 35]
    # hv_slow_windows = [25, 30, 35]

    #
    # hv_fast_windows = [120]
    # hv_slow_windows = [120]
    # ma_fast_windows = [1]
    # ma_slow_windows = [20]

    # ma_fixed_window = [1, 2, 3, 4, 5, 10, 15, 20, 30]  # same fast vs slow ma window for mode 2
    # ma_fixed_window = [1]  # same fast vs slow ma window for mode 2

    # ------ test ------------
    # delta_th_list = [0.4, 0.5]
    # hv_targ_cols = ['C2C', 'ATR']
    # hv_fast_windows = [60, 90]
    # ma_slow_windows = [30, 40]

    # res_root = OnlineDiskManager.get_research_data_path().joinpath('strategies/sell_opts/rv_sell_opts/trading_mode_2')
    # if not res_root.exists():
    #     res_root.mkdir()

    # for Klvl in [0, 1, 2, 3, 4, 5, 6]:
    # for Klvl in [2, 4]:
    # for Klvl in [1]:
    # plt.figure(figsize=(15, 9))
    for a in [0]:
        targ_d = delta_th_list[0]
    # for targ_d in delta_th_list:
        # _output_root = res_root.joinpath(f'{ula}/delta_th{targ_d}')
        # _output_root.mkdir(exist_ok=True, parents=True)
        for ma_fast in ma_fast_windows:
            for ma_slow in ma_slow_windows:
                # output_root = res_root.joinpath(f'{_output_root}/ma{ma_win}')
                # output_root.mkdir(exist_ok=True, parents=True)
                for rv_col in hv_targ_cols:
                    # AnnRet_dict = defaultdict(dict)
                    # Sharpe_dict = defaultdict(dict)
                    # Calmar_dict = defaultdict(dict)
                    # TradFrq_dict = defaultdict(dict)
                    # PosTime_dict = defaultdict(dict)
                    for hv_win_fast in hv_fast_windows:
                        for hv_win_slow in hv_slow_windows:

                            if hv_win_fast >= hv_win_slow:
                            # if ma_fast != ma_slow:
                                continue

                            # if hv_win_fast == 1 and 'ATR' not in rv_col:
                            #     continue

                            _kwargs = {
                                'strategy': strat,
                                'ula': ula,
                                # my strategy params
                                'Klvl': Klvl,  # 'nearK', 'otm1', 'itm1'
                                'rv_col': rv_col,
                                'hv_short_win': hv_win_fast,
                                'hv_long_win': hv_win_slow,
                                'ma_fast': ma_fast,
                                'ma_slow': ma_slow,
                                'adj': adj,

                                'rollterm_day': rollterm_day,
                                'delta_th': targ_d,
                                # 'riskprem_pctl': 0.8,
                                'rollinK_day': 10,
                                # 'HVmethod': HVmethod,

                                # engine setup
                                'delay_p': delay_p,
                                'fillthennext': fillthennext,
                                'optmeta': optmeta,
                                'futmeta': futmeta,

                                'dtc': dtc,
                                'opt_slipfix': opt_slipfix,
                                'etf_slipfix': etf_slipfix,
                                'fut_slipfix': fut_slipfix,
                                'initial_cash': initial_cash,
                                'begindate': begindate,
                                'enddate': enddate,

                                # data root
                                'factor_root': factor_root,
                                'opt_root': opt_root,
                                'fut_root': fut_root,
                                'etf_root': etf_root,
                                'idx_root': idx_root,

                                # mult data
                                'optmult': optmult,
                                'futmult': futmult,
                                'etfmult': etfmult,
                                'idxmult': idxmult
                            }

                            ret, res, factor_df = test_allYYMM(_kwargs)
                            # max_begindate = max(max_begindate, test_allYYMM(_kwargs))

                            # AnnRet_dict[f'hv_short{hv_win_fast}'][f'hv_long{hv_win_slow}'] = res['AnnRet']
                            # Sharpe_dict[f'hv_short{hv_win_fast}'][f'hv_long{hv_win_slow}'] = res['Sharpe']
                            # Calmar_dict[f'hv_short{hv_win_fast}'][f'hv_long{hv_win_slow}'] = res['Calmar']
                            # TradFrq_dict[f'hv_short{hv_win_fast}'][f'hv_long{hv_win_slow}'] = res['trade_freq']
                            # PosTime_dict[f'hv_short{hv_win_fast}'][f'hv_long{hv_win_slow}'] = res['pos_time_std']

                            # if res['Sharpe'] > 0.1:
                            #     ret['values'].plot(figsize=(15, 9), label=f'SMA_{ma_fast}_{ma_slow}_HV_{hv_win_fast}_{hv_win_slow}_{ula}')
                            # ret['values'].plot(figsize=(15, 9), label=f'SMA_{ma_fast}_{ma_slow}_HV_{hv_win_fast}_{hv_win_slow}_{ula}')

                            ret['values'].plot(figsize=(15, 9),
                                               label=f'SMA_{ma_fast}_{ma_slow}_HV_{hv_win_fast}_{hv_win_slow}_{ula}_{rv_col}_Klvl{Klvl}')
                            # ret['values'].plot(figsize=(15, 9),
                            #                    label=f'{ula}_no_signal_sell_delta_th{targ_d}_Klvl{Klvl}')
                            plt.legend(prop={'size': 12}, loc='upper left')
                            plt.title(f'{ula}')
                            # plt.savefig(f'plots/all_params/{ula}_SMA_{ma_fast}_{ma_slow}_HV_{hv_win_fast}_{hv_win_slow}_{rv_col}_Klvl{Klvl}.png')
                            # plt.savefig(f'plots/all_params/{ula}_no_signal_sell_delta_th{targ_d}_Klvl{Klvl}.png')
                            # plt.show()
                            # plt.close()
                            # # ret['values'].plot()
                            # sns.lineplot(data=ret['values'],
                            #              label=f'SMA_{ma_fast}_{ma_slow}_HV_{hv_win_fast}_{hv_win_slow}_{ula}_{rv_col}')
                            # print()

                            # ret['values'].plot(figsize=(15, 9), label=f'no_signal_sell_{ula}')
                            # ret['values'].plot(figsize=(15, 9), label=f'no_signal_sell_Klvl{Klvl}_{ula}')

                            # ret.to_pickle(f'res/all_params/{ula}_SMA_{ma_fast}_{ma_slow}_HV_{hv_win_fast}_{hv_win_slow}_{rv_col}_Klvl{Klvl}.pkl')
                            # ret.to_pickle(f'res/all_params/{ula}_no_signal_sell_delta_th{targ_d}_Klvl{Klvl}.pkl')

                            # if HV_SMA_fast_p != HV_SMA_slow_p:
                            #     ret['values'].plot(label=f'SMA_{HV_SMA_fast_p}_{HV_SMA_slow_p}_HV{HVwindow}_Klvl{Klvl}_delta_th{delta_th}_rollday{rollterm_day}')
                            # else:
                            #     ret['values'].plot(label=f'SMA_{HV_SMA_slow_p}_HV_{HVwindow}_{HVwindow2}_Klvl{Klvl}_delta_th{delta_th}_rollday{rollterm_day}')

                            # sns.lineplot(ret['values'],label=f'hv_calc_{hv_calc_win}_hv_rolling_{ma_slow_win}_Klvl{Klvl}_delta_th{targ_d}_rollday{rollterm_day}',)

    plt.legend(prop={'size': 12}, loc='upper left')
    plt.title(f'{ula}')
    # plt.savefig(f'plots/SMA_5_5_HV_5_40/SMA_{ula}')
    plt.show()
    plt.close()
            # # # #
            # df_AnnRet = pd.DataFrame(AnnRet_dict)
            # df_Sharpe = pd.DataFrame(Sharpe_dict)
            # df_Calmar = pd.DataFrame(Calmar_dict)
            # df_TradFrq = pd.DataFrame(TradFrq_dict)
            # df_PosTime = pd.DataFrame(PosTime_dict)
            #
            # TempTools.draw_heatmap(df_Sharpe, f'Sharpe_{rv_col}_ma{ma_win}', out_folder=output_root)
            # TempTools.draw_heatmap(df_Calmar, f'Calmar_{rv_col}_ma{ma_win}', out_folder=output_root)
            # TempTools.draw_heatmap(df_TradFrq, f'TradFrq_{rv_col}_ma{ma_win}', out_folder=output_root)
            # TempTools.draw_heatmap(df_PosTime, f'PosTime_{rv_col}_ma{ma_win}', out_folder=output_root)
            # TempTools.draw_heatmap(df_AnnRet, f'return_{rv_col}_ma{ma_win}', out_folder=output_root)


if __name__ == '__main__':

    product_list = [
        'CU',
        'AL',  # 有色
        'AU',
        # 'AG',  # 历史太短,不够预热
        # 'RB',  # 历史太短,不够预热
        'I',  # 黑色
        'RU',
        'TA',
        'MA', 'SC',  # 化工
        'C', 'M', 'P',
        'CF',
        'SR',
        'OI'  # 农产品
    ]
    #
    # product_list = [
    #     'CU', 'AL',  # 有色
    #     'AU',
    #     # 'AG',  # 历史太短,不够预热
    #     # 'RB',  # 历史太短,不够预热
    #     # 'I',  # 黑色
    #     'RU',
    #     # 'TA',
    #     # 'MA',
    #     'SC',
    #     # 'C', 'M', 'P', 'CF',
    #     # 'SR',
    #     # 'OI'  # 农产品
    # ]

    # product_list = ['OI', 'P', 'CF', 'SR']  # 套利
    # product_list = ['CU', 'AL']  # 套利
    #
    main('510050', 'IH', 'ETF')
    # main('510300', 'IF', 'ETF')
    # main('000300', 'IF', 'index')
    # main('510500', 'IC', 'ETF')
    # main('000852', 'IM', 'index')
    # for prod in product_list:
    #     main(prod, prod, 'commodity')

    # main('I', 'I', 'commodity')
    # main('SR', 'SR', 'commodity')  # 白糖可以无脑卖,Klvl=2， delta_th = 0.2最佳
    # main('AU', 'AU', 'commodity')  # 黄金无脑卖也凑合，但是有点奇怪,Klvl=0， delta_th = 0.2最佳

    # main('M', 'M', 'commodity')
    # main('AU', 'AU', 'commodity')  # sma_5_5_HV_5_40 用不同的RV_col 均值为正
    # main('CU', 'CU', 'commodity')  # sma_5_5_HV_5_40 用不同的RV_col 均值为正
    # main('AL', 'AL', 'commodity')  # sma_5_5_HV_5_40 用不同的RV_col 均值为正
    # main('RU', 'RU', 'commodity')  # sma_5_5_HV_5_40 用不同的RV_col 均值为正
    # main('SC', 'SC', 'commodity')  # sma_5_5_HV_5_40 用不同的RV_col 均值为正
    # main('OI', 'OI', 'commodity')  # sma_5_5_HV_5_40 用不同的RV_col 均值为正

    # main('AU', 'AU', 'commodity')
    # main('SR', 'SR', 'commodity')
    # main('SC', 'SC', 'commodity')

    # plt.legend(prop={'size': 7}, loc='upper left')
    # plt.grid()
    # # plt.title(f'all commodities')
    # # plt.title(f'all ETFs & indices')
    # # plt.savefig(f'plots/SMA{ma_win}_{ula}')
    # plt.show()
    # plt.close()

    # main('000852','IM','index')
    # main('510050', 'IH', 'ETF')
    # main('510300', 'IF', 'ETF')
    # main('510500', 'IC', 'ETF')

    # for prod in product_list:
    #     main(prod, prod, 'commodity')
    #     # main('I', 'I', 'commodity')
    #     # break
    # plt.legend(prop={'size': 7}, loc='upper left')
    # plt.title(f'all etfs')
    # # plt.savefig(f'plots/SMA{ma_win}_{ula}')
    # plt.show()
    # plt.close()
