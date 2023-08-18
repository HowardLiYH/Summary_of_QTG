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
from data_manager.data_processors import BuySellPriceDefiner
import os
from data_manager import DataManager
from scipy.stats import skew, kurtosis
# from qtg_option_desk_lib.qtg_tools.MyTT.MyTT import RSI

from strategies.all_com_rv_sell.temp_tools import TempTools
from myconfig import OnlineDiskManager
from strategies.all_com_ivma_buy.temp_tools import test_singleYYMM, myprocessor
from strategies.all_com_ivma_buy.mylongvol import *
from strategies.all_com_ivma_buy.prepare_factor_ALL import FactorPreparer


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

    if 'term' not in factor_df.columns:
        factor_df['term'] = factor_df['opt_1m']
    if 'ctrct_1m' in factor_df.columns:
        factor_df['term'] = factor_df['ctrct_1m']

    # return 0, 0, 0

    for YYMM, YYMMfactor_df in factor_df.groupby('term'):

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
    strat = BuyOnEntrySig
    # ula = '510300'  # 510050, 510300, 510500, 000300, 000852
    # fut = 'IF'  # IH, IF, IC, IM
    # ula_type = 'ETF'  # ETF, index

    # begindate = '20160104'
    # enddate = '20211210'
    # begindate = '20150604'
    # begindate = '20220101'
    begindate = None
    # enddate = '20230317'
    enddate = None

    # opt_slipfix = 1  # for index, 滑点=1个指数点
    # opt_slipfix = 2 / 10000  # 单张合约交易成本1.6元， 单日垮式垮式开平仓包含滑点共8元

    if ula_type == 'ETF':
        opt_slipfix = 3 / 10000
    elif ula_type == 'index':
        opt_slipfix = 1
    else:
        df = pd.read_csv(r"\\192.168.0.88\Public\OptionDesk\DATA\ResearchData\Opt_secumain\slippage.csv",
                         index_col='prod')
        opt_slipfix = df.loc[ula, 'slipage'] / 3
        # opt_slipfix = 0

    rollterm_day = 1
    iv_col = 'atm_vol_2m'

    commodity = False
    # Klvl = 1  # 指数改成3
    if ula_type == 'ETF':
        Klvl = 1
    elif ula_type == 'index':
        Klvl = 3
    elif ula_type == 'commodity':
        Klvl = 3
        # if ula == 'AU':
        #     Klvl = 2
        rollterm_day = 10
        iv_col = 'atm_vol_1m'
        commodity = True

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

    # pctl_th_list = np.arange(0.52, 0.7, 0.02)
    # pctl_th_list = np.arange(0.52, 0.7, 0.02)
    pctl_th_list = [0.52, 0.54, 0.56]
    # pctl_th_list = [0.56]
    pctl_window = 250
    # iv_ma_p_list = range(5, 16)
    iv_ma_p_list = [11, 12, 13]
    # iv_ma_p_list = [8]
    targ_d = 0.2
    sig = 'all'  # 'all', 'own'

    # ivma_fast_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # ivma_fast_list = [2, 3, 4, 5]
    # ivma_slow_list = [5, 10, 15, 20, 25, 30]
    ivma_fast_list = [5]
    ivma_slow_list = [10]
    # for Klvl in [0, 1, 2, 3, 4, 5, 6]:
    for a in [0]:
        for ivma_fast in ivma_fast_list:
            for ivma_slow in ivma_slow_list:

                if ivma_fast >= ivma_slow:
                    continue

                _kwargs = {
                    'strategy': strat,
                    'ula': ula,
                    # my strategy params

                    'iv_col': iv_col,
                    'ivma_fast': ivma_fast,
                    'ivma_slow': ivma_slow,
                    'rsi_p': 30,
                    'commodity': commodity,
                    "sig": sig,  # 'all', 'own'

                    'Klvl': Klvl,
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
                # ret['values'].plot(figsize=(15, 9), label=f'SMA_{ma_fast}_{ma_slow}_HV_{hv_win_fast}_{hv_win_slow}_{ula}_{rv_col}')
                # ret['values'].plot(figsize=(15, 9), label=f'no_signal_sell_{ula}')

                ret['values'].plot(figsize=(15, 9), label=f'{ula}_ma_1_{ivma_fast}_{ivma_slow}_Klvl{Klvl}_delta{targ_d}_{rollterm_day}dayroll_{sig}sig')
                #
                # plt.legend(prop={'size': 12}, loc='upper left')
                # plt.title(f'{ula}')
                # plt.savefig(f'plots/all_params/{ula}_pctl{pctl_th}_ma{iv_ma_p}_Klvl{Klvl}.png')
                # # plt.show()
                # plt.close()

                # ret.to_pickle(f'res/all_params/{ula}_pctl{pctl_th}_ma{iv_ma_p}_Klvl{Klvl}.pkl')
                # if HV_SMA_fast_p != HV_SMA_slow_p:
                #     ret['values'].plot(label=f'SMA_{HV_SMA_fast_p}_{HV_SMA_slow_p}_HV{HVwindow}_Klvl{Klvl}_delta_th{delta_th}_rollday{rollterm_day}')
                # else:
                #     ret['values'].plot(label=f'SMA_{HV_SMA_slow_p}_HV_{HVwindow}_{HVwindow2}_Klvl{Klvl}_delta_th{delta_th}_rollday{rollterm_day}')

                # sns.lineplot(ret['values'],label=f'hv_calc_{hv_calc_win}_hv_rolling_{ma_slow_win}_Klvl{Klvl}_delta_th{targ_d}_rollday{rollterm_day}',)

    # plt.legend(prop={'size': 12}, loc='upper left')
    # plt.title(f'{ula}')
    # # plt.savefig(f'plots/IV_shorvol_{ula}.png')
    # plt.show()
    # plt.close()
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

    return ret


if __name__ == '__main__':

    product_list = [
        'CU',
        'AL',  # 有色
        'AU',
        'AG',  # 历史太短,不够预热
        'RB',  # 历史太短,不够预热
        'I',  # 黑色
        'RU',
        'TA',
        'MA', 'SC',  # 化工
        # 'C',
        # 'M',
        'P', 'CF',
        # 'SR',
        'OI'  # 农产品
    ]
    #
    # product_list = [
    # #     'CU',
    # #     'AL',  # 有色
    # #     'AU',
    # #     'AG',  # 历史太短,不够预热
    # #     'RB',  # 历史太短,不够预热
    #     'I',  # 黑色
    #     'RU',
    # #     'TA',
    #     'MA', 'SC',  # 化工
    # #     'C',
    # #     'M',
    #     'P', 'CF',
    # #     'SR',
    # #     'OI'  # 农产品
    # ]

    product_list = [
        # 'CU', 'AL',  # 有色
        'AU',
        'AG',  # 历史太短,不够预热
        'RB',  # 历史太短,不够预热
        'I',  # 黑色
        'RU',
        'TA',
        'MA',
        'SC',
        # 'C',
        # 'M',
        'P', 'CF',
        # 'SR',
        'OI'  # 农产品
    ]

    # product_list = ['OI', 'P', 'CF', 'SR']  # 套利
    # product_list = ['CU', 'AL']  # 套利
    # df = pd.DataFrame()
    # ret = main('510050', 'IH', 'ETF')
    # ret.rename(columns={'values': f'{510050}_values'}, inplace=True)
    # df = pd.concat([df, ret[f'{510050}_values']], axis=1)
    # ret = main('510300', 'IF', 'ETF')
    # ret.rename(columns={'values': f'{510300}_values'}, inplace=True)
    # df = pd.concat([df, ret[f'{510300}_values']], axis=1)
    # # # main('000300', 'IF', 'index')
    # ret = main('510500', 'IC', 'ETF')
    # ret.rename(columns={'values': f'{510500}_values'}, inplace=True)
    # df = pd.concat([df, ret[f'{510500}_values']], axis=1)
    # ret = main('000852', 'IM', 'index')
    # ret.rename(columns={'values': f'000852_values'}, inplace=True)
    # df = pd.concat([df, ret[f'000852_values']], axis=1)

    # for prod in product_list:
    #     main(prod, prod, 'commodity')

    #
    df = pd.DataFrame()
    for prod in product_list:
        ret = main(prod, prod, 'commodity')
        ret.rename(columns={'values': f'{prod}_values'}, inplace=True)
        df = pd.concat([df, ret[f'{prod}_values']], axis=1)

    plt.legend(prop={'size': 7}, loc='upper left')
    # plt.title(f'all etfs')
    # plt.savefig(f'plots/SMA{ma_win}_{ula}')
    plt.show()
    plt.close()

    df['all_pnl'] = df.mean(axis=1)
    df['all_pnl'].plot()
    plt.title(product_list)
    plt.show()

    # # main('SR', 'SR', 'commodity')
    # main('M', 'M', 'commodity')
    # main('I', 'I', 'commodity')
    # # main('AU', 'AU', 'commodity')
    # main('CU', 'CU', 'commodity')
    # main('AL', 'AL', 'commodity')
    # main('RU', 'RU', 'commodity')
    # main('OI', 'OI', 'commodity')
    # main('AU', 'AU', 'commodity')
    # main('SR', 'SR', 'commodity')
    # main('SC', 'SC', 'commodity')

    # plt.legend(prop={'size': 12}, loc='upper left')
    # plt.grid()
    # plt.title(f'all commodities')
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
    # # plt.title(f'all etfs')
    # # plt.savefig(f'plots/SMA{ma_win}_{ula}')
    # plt.show()
    # plt.close()
