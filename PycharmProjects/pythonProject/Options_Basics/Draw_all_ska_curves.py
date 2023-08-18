from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from meta import ETFOptContracts
from optcal import EuOptCalculator
from timecal import DateTimeCalculator
import scipy.interpolate


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


def plot_inter_all(x, y, z, e):
    f = scipy.interpolate.interp1d(x, y)
    xnew = np.arange(min(x), max(x), 0.025)
    ynew = f(xnew)
    plt.plot(x, y, ',', color=str(z))
    plt.plot(xnew, ynew, '-', color=str(z), label=f'Due_{e + 1}')


def sort_categorical_val(a):
    skew_25[a].append(delta_25_skew)
    skew_10[a].append(delta_10_skew)
    Kur_25[a].append(kurtosis_25)
    Kur_10[a].append(kurtosis_10)
    A[a].append(vol_50_cp_interp)


if __name__ == '__main__':
    dtc = DateTimeCalculator()
    # Set Up
    etf_metas = ETFOptContracts('510050')
    date_index = 0
    skew_25 = {0: [], 1: [], 2: []}
    skew_10 = {0: [], 1: [], 2: []}
    Kur_25 = {0: [], 1: [], 2: []}
    Kur_10 = {0: [], 1: [], 2: []}
    A = {0: [], 1: [], 2: []}

    for full_year_code in etf_metas.series_info_by_ctrct:
        date_index = 0
        # if full_year_code != '201804':
        #     continue
        if full_year_code[:4] == '2015' or full_year_code == '201601':
            continue
        ###############################################
        start_d, end_d = etf_metas.get_effday(full_year_code), etf_metas.get_dueday(full_year_code)
        print(f'Yo! We are now at {full_year_code}')
        for targ_date in dtc.get_trding_day_range(start_d, end_d):
            # Without 20200203 & 20180209
            if targ_date[:4] == '2015' or targ_date == '20200203' or targ_date == '20180209':
                continue
            if end_d >= '20230524':
                break
            if pd.to_datetime(targ_date) not in load_ula_price(full_year_code).index[:-4]:
                continue
            four_exp_dates = etf_metas.get_dueday_series(targ_date)
            three_exp_dates = four_exp_dates[:3]
            for due_date in three_exp_dates:
                due = due_date[:-2]
                k_cp_list = etf_metas.get_strikes(due, targ_date, 0)
                targ_c_contract = [etf_metas.get_contract_by_strike(targ_date, due, 'C', k, 0) for k in
                                   k_cp_list]
                targ_p_contract = [etf_metas.get_contract_by_strike(targ_date, due, 'P', k, 0) for k in
                                   k_cp_list]
                datas_c = [load_data(full_year_code, c.contract_code) for c in targ_c_contract]
                datas_p = [load_data(full_year_code, c.contract_code) for c in targ_p_contract]

                close_list_c = [d.loc[pd.Timestamp(targ_date), 'close'] for d in datas_c]
                close_list_p = [d.loc[pd.Timestamp(targ_date), 'close'] for d in datas_p]

                # Calculating ula price with ATM Forward
                diff_cp_list = np.array(close_list_c) - np.array(close_list_p)
                mini_index = np.nanargmin(abs(diff_cp_list))
                k_atm, c_atm, p_atm = float(k_cp_list[mini_index]), close_list_c[mini_index], close_list_p[mini_index]
                ula_price = round((c_atm - p_atm + float(k_atm)), 4)

                ula_all = load_ula_price(full_year_code)
                supposed_ula = ula_all.loc[pd.to_datetime(targ_date), 'close']

                ###################################################################################################
                # Calculate Call VOL
                eu_opt_cal = EuOptCalculator()
                vol_c_list = []
                for i in range(len(close_list_c)):
                    opt_meta = targ_c_contract[i]
                    k = k_cp_list[i]
                    vol_c = eu_opt_cal.get_imp_vol(opt_meta.cp_type, ula_price, float(k),
                                                   ttm=dtc.get_days_diff(targ_date, opt_meta.get_exp_date()) / 252, r=0,
                                                   div=0,
                                                   option_prc=close_list_c[i])
                    vol_c_list.append(vol_c)

                # Calculate Put VOL
                vol_p_list = []
                for i in range(len(close_list_p)):
                    opt_meta = targ_p_contract[i]
                    k = k_cp_list[i]

                    # calculate the iv through get_imp_vol
                    vol_p = eu_opt_cal.get_imp_vol(opt_meta.cp_type, ula_price, float(k),
                                                   ttm=dtc.get_days_diff(targ_date, opt_meta.get_exp_date()) / 252, r=0,
                                                   div=0,
                                                   option_prc=close_list_p[i])
                    vol_p_list.append(vol_p)

                #######################################################################################################
                # Calculate Call Delta
                delta_c_list = []
                for i in range(len(close_list_c)):
                    opt_meta = targ_c_contract[i]
                    k = k_cp_list[i]
                    vol = vol_c_list[i]
                    delta = eu_opt_cal.get_greeks(opt_meta.cp_type, ula_price, float(k),
                                                  ttm=dtc.get_days_diff(targ_date, opt_meta.get_exp_date()) / 252, r=0,
                                                  div=0,
                                                  vol=vol)[0]
                    delta = 1 - delta
                    delta_c_list.append(delta)

                # Calculate Put Delta
                delta_p_list = []
                for i in range(len(close_list_c)):
                    opt_meta = targ_p_contract[i]
                    k = k_cp_list[i]
                    vol = vol_p_list[i]
                    delta = eu_opt_cal.get_greeks(opt_meta.cp_type, ula_price, float(k),
                                                  ttm=dtc.get_days_diff(targ_date, opt_meta.get_exp_date()) / 252,
                                                  r=0,
                                                  div=0,
                                                  vol=vol)[0]
                    delta = abs(delta)
                    delta_p_list.append(delta)
                ########################################################################################################
                # Calculate delta_25_skew and delta_10_skew
                # try:
                #     seperator_c = np.where(np.array(delta_c_list) > 0.5)[0][0]
                # except:
                #     print(f'Dropping {targ_date}__________________________')
                #     continue

                try:
                    seperator_c = np.where(np.array(delta_c_list) > 0.5)[0][0]
                except:
                    _targ_cal_list = np.where(np.array(delta_c_list) > 0.5)
                    print(f'ERROR on {targ_date} with duedate {due_date} with all four exp dates as {four_exp_dates}')
                    # print(f'length of the delta_c_list over 0.5 is {len(_targ_cal_list)}, with ttm as {dtc.get_days_diff(targ_date, opt_meta.get_exp_date())}')
                    print(f'ulaprice is {ula_price} and the supposed closing ula is {supposed_ula}.')
                    print(f'Close_list_c is {close_list_c}')
                    print(f'Close_list_p is {close_list_p}')
                    print(f'Difference between them is {diff_cp_list}')
                    print(f'The full K List is {k_cp_list} and the closest K is {k_atm} with corresponding Call Atm at {c_atm} and Put ATM at {p_atm}')
                    # print(f'The full OTM_CP_Delta List is {delta_cp_list}')
                    print('_____________________________________________________________________________________________')
                    print('_____________________________________________________________________________________________')
                    continue

                delta_cp_list = delta_p_list[:seperator_c] + delta_c_list[seperator_c:]
                vol_cp_list = vol_p_list[:seperator_c] + vol_c_list[seperator_c:]

                y_cp_interp = scipy.interpolate.interp1d(delta_cp_list, vol_cp_list)

                f = scipy.interpolate.interp1d(delta_cp_list, vol_cp_list, fill_value='extrapolate')
                try:
                    vol_50_cp_interp = y_cp_interp(0.5)
                except:
                    vol_50_cp_interp = f(0.5)
                if max(delta_cp_list) < 0.75:
                    vol_25_ccp_interp = f(0.75)
                else:
                    vol_25_ccp_interp = y_cp_interp(0.75)
                if max(delta_cp_list) < 0.6:
                    vol_10_ccp_interp = f(0.6)
                else:
                    if min(delta_cp_list) > 0.6:
                        vol_10_pcp_interp = f(0.6)
                    else:
                        vol_10_ccp_interp = y_cp_interp(0.6)
                if min(delta_cp_list) > 0.25:
                    vol_25_pcp_interp = f(0.25)
                else:
                    vol_25_pcp_interp = y_cp_interp(0.25)
                if min(delta_cp_list) > 0.4:
                    vol_10_pcp_interp = f(0.4)
                else:
                    vol_10_pcp_interp = y_cp_interp(0.4)

                delta_25_skew = vol_25_pcp_interp / vol_25_ccp_interp
                delta_10_skew = vol_10_pcp_interp / vol_10_ccp_interp
                kurtosis_25 = (vol_25_pcp_interp + vol_25_ccp_interp) / (2 * vol_50_cp_interp)
                if kurtosis_25 > 1.175:
                    print('___________________________________________________________________________________________________')
                    print(f'AHHHHHH HA! This {targ_date} with due date {due_date} has a bizzare kurtosis_25 of {kurtosis_25}')
                    print('___________________________________________________________________________________________________')
                kurtosis_10 = (vol_10_pcp_interp + vol_10_ccp_interp) / (2 * vol_50_cp_interp)
                if kurtosis_10 < 0.85:
                    print('___________________________________________________________________________________________________')
                    print(f'AHHHHHH HA! This {targ_date} with due date {due_date} has a bizzare kurtosis_10 of {kurtosis_10}')
                    print('___________________________________________________________________________________________________')

                sort_categorical_val(four_exp_dates.index(due_date))
            date_index += 1
            #########################################################################################################
    color_plate = ['pink', 'c', 'sandybrown', 'y', 'mediumseagreen']
    four_features = [skew_25, skew_10, Kur_25, Kur_10, A]
    Title = ['Skew_25D', 'Skew_10D', 'Kurtosis_25D', 'Kurtosis_10D', "ATMVOL"]

    for feature in four_features:
        index = four_features.index(feature)
        # plt.subplot(2, 2, index + 1)
        fig, ax = plt.subplots()
        for count in range(len(feature)):
            print(f'{Title[index]}_{count+1} has length: {len(feature[count])}')
            plot_inter_all([i for i in range(1, len(feature[count]) + 1)], feature[count], color_plate[count], count)
        plt.title(f'{Title[index]} vs. Time')

        # res = pd.DataFrame()
        # s, e = dtc.adj_date_range('20151224', '20220320')
        # res['str_date'] = dtc.get_trding_day_range(s, e)
        # res['values'] = range(len(res))
        # choose_these_dates = res['str_date'].values[::180]
        # plt.xticks(res.index[::30], choose_these_dates, rotation=45)

        ax.set_xticklabels(['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'])
        plt.legend()
        print('________________________________')
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.savefig(f'features_{Title[index]}.png')
        plt.clf()
