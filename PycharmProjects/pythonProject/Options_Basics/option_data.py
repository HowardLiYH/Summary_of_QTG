#######################################################################################################################
# Author: Howard Li
# The below functions take into account all the option data (C, P, K) from 201602 to 202303.

# Option Data for calculating the below indicators
# 1. Skew calculated by 25 Delta
# 2. Skew calculated by 10 Delta
# 3. Kurtosis calculated by 25 Delta
# 4. Kurtosis calculated by 10 Delta
# 5. At-the-money implied volatility

# The data here has excluded the last three trading days data before the expiry date.
# The data here has excluded the outliers date encountering circuit breakers on '20180209' and '20200203'.
# The data here is attempted to create the initial benchmark model for predicting the IV return on the second day.
#######################################################################################################################


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


def list_to_df(five_features, Title):
    feature_df = pd.DataFrame()
    for feature in five_features:
        for sequence in feature:
            index = sequence + 1
            column_name = f'{Title[five_features.index(feature)]}_due_{index}'
            feature_df[column_name] = feature[sequence]
    return feature_df


if __name__ == '__main__':
    # Setting Up
    count  = 0
    dtc = DateTimeCalculator()
    etf_metas = ETFOptContracts('510050')
    targ_date_list = []
    skew_25 = {0: [], 1: [], 2: [], 3: []}
    skew_10 = {0: [], 1: [], 2: [], 3: []}
    Kur_25 = {0: [], 1: [], 2: [], 3: []}
    Kur_10 = {0: [], 1: [], 2: [], 3: []}
    A = {0: [], 1: [], 2: [], 3: []}

    # Start looping in the files
    for full_year_code in etf_metas.series_info_by_ctrct:

        # Make sure the data start from 201602
        # if full_year_code != '201804':
        #     continue
        if full_year_code[:4] == '2015':
            continue
        start_d, end_d = etf_metas.get_effday(full_year_code), etf_metas.get_dueday(full_year_code)
        print(f'Yo! We are now at {full_year_code}')
        for targ_date in dtc.get_trding_day_range(start_d, end_d):
            #
            # print(f'The target date is {targ_date}')

            # # Take out 20200203 & 20180209
            # if targ_date[:4] == '2015' or targ_date == '20200203' or targ_date == '20180209':
            #     continue

            # Add a stop preventing over-looping into empty files
            if end_d >= '20230524':
                break

            # Make sure all targ_date is within the trading date and take out the last three days
            # if pd.to_datetime(targ_date) not in load_ula_price(full_year_code).index[:-4]:
            #     continue

            info = load_ula_price(full_year_code)

            if pd.to_datetime(targ_date) not in load_ula_price(full_year_code).index[:-4]:
                continue

            targ_date_list.append(targ_date)
            # Take out the last three trading days before the expiry date
            # three_exp_dates = etf_metas.get_dueday_series(targ_date)[:3]
            all_exp_dates = etf_metas.get_dueday_series(targ_date)
            for due_date in all_exp_dates:
                due = due_date[:-2]
                k_cp_list = etf_metas.get_strikes(due, targ_date, 0)
                k_cp_list1 = etf_metas.get_strikes(due, targ_date, 1)
                k_all = etf_metas.get_strikes(due, targ_date)

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

                test = eu_opt_cal.get_imp_vol('C', 2.6044, 2.5, 20, 0, 0, 0.0834)
                test2 = eu_opt_cal.get_imp_vol('P', 2.6044, 2.5, 20, 0, 0, 0.063243)


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
                    ttm = dtc.get_days_diff(targ_date, opt_meta.get_exp_date())
                    delta = eu_opt_cal.get_greeks(opt_meta.cp_type, ula_price, float(k),
                                                  ttm=dtc.get_days_diff(targ_date, opt_meta.get_exp_date()) / 252,
                                                  r=0,
                                                  div=0,
                                                  vol=vol)[0]
                    delta = abs(delta)
                    delta_p_list.append(delta)
                ########################################################################################################
                # Get OTM Call and Put's Delta and IV
                try:
                    seperator_c = np.where(np.array(delta_c_list) > 0.5)[0][0]
                    delta_cp_list = delta_p_list[:seperator_c] + delta_c_list[seperator_c:]
                    vol_cp_list = vol_p_list[:seperator_c] + vol_c_list[seperator_c:]

                    # Interpolation set up
                    y_cp_interp = scipy.interpolate.interp1d(delta_cp_list, vol_cp_list)

                    # Extrapolation set up
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
                            vol_10_ccp_interp = f(0.6)
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

                    # Calculate the four indicators
                    delta_25_skew = vol_25_pcp_interp / vol_25_ccp_interp
                    delta_10_skew = vol_10_pcp_interp / vol_10_ccp_interp
                    kurtosis_25 = (vol_25_pcp_interp + vol_25_ccp_interp) / (2 * vol_50_cp_interp)
                    kurtosis_10 = (vol_10_pcp_interp + vol_10_ccp_interp) / (2 * vol_50_cp_interp)


                # if no OTM Call
                except IndexError:
                    count += 1
                    print(
                        f'Line 183 has an IndexError{count} and the delta_c_list is {delta_c_list} on {targ_date} with due {due_date}')
                    print(
                        f'the delta_p_list is {delta_p_list} on {targ_date} with due {due_date}')
                    print(
                        f'The value {None} has been added to the given targets\n')

                    delta_25_skew = None
                    delta_10_skew = None
                    kurtosis_25 = None
                    kurtosis_10 = None
                    vol_50_cp_interp = None



                sort_categorical_val(all_exp_dates.index(due_date))
            #########################################################################################################

    # Plotting the data
    color_plate = ['pink', 'c', 'sandybrown', 'y', 'mediumseagreen']
    five_features = [skew_25, skew_10, Kur_25, Kur_10, A]
    Title = ['Skew_25D', 'Skew_10D', 'Kurtosis_25D', 'Kurtosis_10D', "ATMVOL"]

    # for feature in five_features:
    #     index = five_features.index(feature)
    #     fig, ax = plt.subplots()
    #     for count in range(len(feature)):
    #         print(f'{Title[index]}_{count + 1} has length: {len(feature[count])}')
    #         plot_inter_all([i for i in range(1, len(feature[count]) + 1)], feature[count], color_plate[count], count)
    #     plt.title(f'{Title[index]} vs. Time')
    #
    #     ax.set_xticklabels(['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'])
    #     plt.legend()
    #     fig = plt.gcf()
    #     fig.set_size_inches(18.5, 10.5)
    #
    #     # Save the plot into local with corresponding feature names
    #     plt.savefig(f'features_OG_{Title[index]}.png')
    #     plt.clf()

    feature_df = list_to_df(five_features, Title)
    feature_df['target_date'] = targ_date_list
    feature_df = feature_df.set_index('target_date')
    feature_df['next_ATMVOL_due_1'] = feature_df['ATMVOL_due_1'].shift(-1)
    feature_df['IV_return_due_1'] = feature_df['next_ATMVOL_due_1'] / feature_df['ATMVOL_due_1']
    feature_df.to_csv(r"C:\Users\ps\PycharmProjects\pythonProject\no_last4_feature_df.csv")
