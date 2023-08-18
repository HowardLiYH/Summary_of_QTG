########################################################################################################
# The file for getting more features, which include:
#
############## Direct Information ####################
# open
# high
# low
# close
# volume
# turnover
#
############## Require Calculation ###################
# return[ log(close/prev_close) ]
# ma5 ma20 ma60
# vol_ma5 vol_ma20 vol_ma60
# Open Interest Put Call Ratio
# Volume PCR


########################################################################################################


from pathlib import Path
import pandas as pd
import numpy as np
from meta import ETFOptContracts
from timecal import DateTimeCalculator
import math


def load_data(full_year_code, contract_code):
    root = Path(r"\\192.168.0.88\Public\OptionDesk\DATA\database\1day_bar\opt_by_optmon\510050")
    targ_month = root.joinpath(full_year_code[2:])
    # Specific contract code inside the folder
    file = targ_month.joinpath(f'{contract_code}.pkl')
    return pd.read_pickle(str(file))


def load_ula_price(full_year_code):
    root = Path(r"\\192.168.0.88\Public\OptionDesk\DATA\database\1day_bar\etf_by_optmon\510050")
    targ_month = root.joinpath(full_year_code[2:])
    # Specific contract code inside the folder
    contract_code = '510050'
    file = targ_month.joinpath(f'{contract_code}.pkl')
    return pd.read_pickle(str(file))


def ula_ma(df_tick: pd.DataFrame, Ns: [5,20,60]):
    """成交量加权均价
    :param df_tick:
    :param N:
    :return:
    """
    for N in Ns:
        df_tick[f"ula_ma{N}"] = (df_tick["close"]).rolling(
            window=N,
            min_periods=1,
        ).sum() / N
    return df_tick


def ula_vol_ma(df_tick: pd.DataFrame, Ns: [5,20,60]):
    """成交量加权均价
    :param df_tick:
    :param N:
    :return:
    """
    for N in Ns:
        df_tick[f"ula_vol_ma{N}"] = (df_tick["volume"]).rolling(
            window=N,
            min_periods=1,
        ).sum() / N
    return df_tick


if __name__ == '__main__':
    # Setting Up
    error_date = []
    dtc = DateTimeCalculator()
    etf_metas = ETFOptContracts('510050')
    database_df = pd.DataFrame()
    call_df = pd.DataFrame()
    put_df = pd.DataFrame()
    # Start looping in the files
    for full_year_code in etf_metas.series_info_by_ctrct:
        start_d, end_d = etf_metas.get_effday(full_year_code), etf_metas.get_dueday(full_year_code)
        for targ_date in dtc.get_trding_day_range(start_d, end_d):
            print(f"{start_d}, {end_d}")
            # Add a stop preventing over-looping into empty files
            if targ_date[:-4] == '2015':
                continue
            if targ_date[:-4] != '20161229':
                continue

            contracts = etf_metas.get_series_ctrcts(full_year_code)

            check_list = []

            for contract in contracts:
                new_data = load_data(full_year_code, contract.contract_code)
                check_list.append(new_data)
            if end_d >= '20230524':
                break
            if pd.to_datetime(targ_date) not in load_ula_price(full_year_code).index[:-4]:
                continue
            database_df = pd.concat([database_df, load_ula_price(full_year_code)], axis=0)
            # three_exp_dates = etf_metas.get_dueday_series(targ_date)[:3]
            # first_exp_dates = etf_metas.get_dueday_series(targ_date)[:1]
            all_exp_dates = etf_metas.get_dueday_series(targ_date)
            for due_date in all_exp_dates:
                # print(f"Initialize concating {targ_date} from {full_year_code} with due {due_date}_____________________________________")
                k_cp_list = etf_metas.get_strikes(full_year_code, targ_date)
                # print(f'k_0 is {etf_metas.get_strikes(full_year_code, targ_date,0)}')
                # print(f'k_1 is {etf_metas.get_strikes(full_year_code, targ_date,1)}')
                # print(f'k_cp is {etf_metas.get_strikes(full_year_code, targ_date)}')
                targ_c_contract = [etf_metas.get_contract_by_strike(targ_date, full_year_code, 'C', k) for k in
                                   k_cp_list]
                targ_p_contract = [etf_metas.get_contract_by_strike(targ_date, full_year_code, 'P', k) for k in
                                   k_cp_list]

                ################################################################################################
                # # get all the 14 K strike price of the specific date given the input full year code and target date
                # all_info = load_ula_price(full_year_code)
                # close = load_ula_price(full_year_code).loc[pd.to_datetime(targ_date)]['close']
                # k_list = etf_metas.get_otm_strikes(full_year_code, targ_date, close, 8, 0)
                # k_cp_list = k_list['P'] + k_list['C']
                # # A list contains 14 specific contract info with all the call contracts based on each strike in k_list and the given the input full year code and target date
                # # Including ITM, ATM, OTM
                # targ_c_contract = [etf_metas.get_contract_by_strike(targ_date, full_year_code, 'C', k, 0) for k in
                #                    k_list['C']]
                # targ_p_contract = [etf_metas.get_contract_by_strike(targ_date, full_year_code, 'P', k, 0) for k in
                #                    k_list['P']]
                #####################################################################################################

                for c in targ_c_contract:
                    # if c.contract_code != '100750':
                    #     continue
                    try:
                        new_c_data = load_data(full_year_code, c.contract_code)
                    except:
                        error_date.append(targ_date)
                        print(f"On Call Contract. No alter_time = 1 info on {targ_date} with due date {due_date}. The error contact code is {c.contract_code}")
                    # try:
                    #     new_c_data = load_data(full_year_code, c.contract_code)
                    #     print(f"This is call load_data operation status with {c.contract_code}")
                    # except:
                    #     print(f'@@@@@@@@@@@@@@@@@@@@@@@@@The issue with call is {c.contract_code} @@@@@@@@@@@@@@@@@@@@@@@@@@')
                call_df = pd.concat([call_df, new_c_data], axis=0)
                # call_df = pd.concat([call_df, load_data(full_year_code, c.contract_code)], axis=0)


                for p in targ_p_contract:
                    try:
                        new_p_data = load_data(full_year_code, p.contract_code)

                    except:
                        print(f"On Put Contract. No alter_time = 1 info on {targ_date} with due date {due_date}. The error contact code is {c.contract_code}")
                put_df = pd.concat([put_df, new_p_data], axis=0)
    database_df = database_df[~database_df.index.duplicated(keep='first')]
    # For three expiration dates
    # call_df = pd.DataFrame(call_df.groupby(call_df.index).sum())
    # put_df = pd.DataFrame(put_df.groupby(put_df.index).sum())
    # garch_df = database_df[['open', 'high', 'low', 'close','volume', 'turnover', 'prev_close']]
    # garch_df['price_return'] = np.log(garch_df['close']/garch_df['prev_close'])
    # garch_df = ula_ma(garch_df, [5,20,60])
    # garch_df = ula_vol_ma(garch_df, [5, 20, 60])
    # garch_df['op_volume'] = call_df['volume'] + put_df['volume']
    # garch_df['op_turnover'] = call_df['turnover'] + put_df['turnover']
    # garch_df['oi_pcr'] = put_df['openinterest'] / call_df['openinterest']
    # garch_df['vol_pcr'] = put_df['volume'] / call_df['volume']
    # garch_df.to_csv(r"C:\Users\ps\PycharmProjects\pythonProject\garch_df.csv")
    # call_df.to_csv(r"C:\Users\ps\PycharmProjects\pythonProject\call_df.csv")
    # put_df.to_csv(r"C:\Users\ps\PycharmProjects\pythonProject\put_df.csv")

    # For the first expiration date
    first_call_df = call_df
    first_put_df = put_df
    first_garch_df = database_df[['open', 'high', 'low', 'close','volume', 'turnover', 'prev_close']]
    first_garch_df['price_return'] = np.log(first_garch_df['close']/first_garch_df['prev_close'])
    first_garch_df = ula_ma(first_garch_df, [5,20,60])
    first_garch_df = ula_vol_ma(first_garch_df, [5, 20, 60])
    # first_garch_df['op_volume'] = call_df['volume'] + put_df['volume']
    # first_garch_df['op_turnover'] = call_df['turnover'] + put_df['turnover']
    # first_garch_df['oi_pcr'] = put_df['openinterest'] / call_df['openinterest']
    # first_garch_df['vol_pcr'] = put_df['volume'] / call_df['volume']
    first_garch_df.to_csv(r"C:\Users\ps\PycharmProjects\pythonProject\first_garch_df.csv")
    call_df.to_csv(r"C:\Users\ps\PycharmProjects\pythonProject\check1_call_df3.csv")
    put_df.to_csv(r"C:\Users\ps\PycharmProjects\pythonProject\check1_put_df3.csv")

    error = pd.DataFrame()
    error['error_date'] = error_date
    error.to_csv(r"C:\Users\ps\PycharmProjects\pythonProject\error.csv")

    print(database_df)
