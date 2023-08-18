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


if __name__ == '__main__':
    dtc = DateTimeCalculator()
    i = 0
    # Open the folder named '510050'
    etf_metas = ETFOptContracts('510050')
    ############################## For Test
    df = pd.DataFrame(columns=['contract_code', 'trading_code', 'Delta', 'K', 'IV', 'CP_type', 't', 'r'])
    ###############################
    for full_year_code in etf_metas.series_info_by_ctrct:
        ############### For Test
        if full_year_code != '201804':
            continue
        #     ##################
        start_d, end_d = etf_metas.get_effday(full_year_code), etf_metas.get_dueday(full_year_code)
        for targ_date in dtc.get_trding_day_range(start_d, end_d):
            ula_all = load_ula_price(full_year_code)
            targ_index = pd.to_datetime(targ_date)
            if targ_index not in ula_all.index:
                continue
            # ula_price = ula_all.loc[targ_index, 'close']
            # ula_price = 2.7293
            # if targ_date != '20191121':
            #     continue
            # get all the 14 K strike price of the specific date given the input full year code and target date
            # k_ula_list = etf_metas.get_otm_strikes(full_year_code, targ_date, ula_price, 8, 0)
            k_cp_list = etf_metas.get_strikes(full_year_code, targ_date, 0)

            # A list contains 14 specific contract info with all the call contracts based on each strike in k_list and the given the input full year code and target date
            # Including ITM, ATM, OTM
            targ_c_contract = [etf_metas.get_contract_by_strike(targ_date, full_year_code, 'C', k, 0) for k in
                               k_cp_list]
            targ_p_contract = [etf_metas.get_contract_by_strike(targ_date, full_year_code, 'P', k, 0) for k in
                               k_cp_list]

            # Load all the contract from the list with 14 contracts and get the contract code to create a list of DataFrames
            # The full year params here is used to set the path
            datas_c = [load_data(full_year_code, c.contract_code) for c in targ_c_contract]
            datas_p = [load_data(full_year_code, c.contract_code) for c in targ_p_contract]

            # Initializing
            eu_opt_cal = EuOptCalculator()

            # Access the list of DataFrame and get the closing price through each DataFrame and  compile them into a list in form of float
            close_list_c = [d.loc[pd.Timestamp(targ_date), 'close'] for d in datas_c]
            close_list_p = [d.loc[pd.Timestamp(targ_date), 'close'] for d in datas_p]

            # Calculating ula price with ATM Forward
            diff_cp_list = np.array(close_list_c) - np.array(close_list_p)
            mini_index = np.argmin(abs(diff_cp_list))
            k_atm, c_atm, p_atm = float(k_cp_list[mini_index]), close_list_c[mini_index], close_list_p[mini_index]
            ula_price = round((c_atm - p_atm + float(k_atm)), 4)

            # Use call and its corresponding exp to calculate all the call vol
            vol_c_list = []
            for i in range(len(close_list_c)):
                opt_meta = targ_c_contract[i]
                k = k_cp_list[i]
                ttm=dtc.get_days_diff(targ_date, opt_meta.get_exp_date())
                due =  opt_meta.get_exp_date()
                # calculate the iv through get_imp_vol
                vol_c = eu_opt_cal.get_imp_vol(opt_meta.cp_type, ula_price, float(k),
                                               ttm=dtc.get_days_diff(targ_date, opt_meta.get_exp_date()) / 252, r=0,
                                               div=0,
                                               option_prc=close_list_c[i])
                vol_c_list.append(vol_c)

            # Use put and its corresponding exp to calculate all the put vol
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
            # Plot the Call VOL vs. K
            plt.clf()
            plt.subplot(1, 2, 1)
            k_plot_list = [float(i) for i in k_cp_list]
            # Linear Interpolation
            y_c_interp = scipy.interpolate.interp1d(k_plot_list, vol_c_list)
            k_val = np.linspace(min(k_plot_list), max(k_plot_list), 50)
            vol_c_interp = y_c_interp(k_val)

            plt.plot(k_plot_list, vol_c_list, '^', color='r')
            plt.plot(k_val, vol_c_interp, '-.', color='r', label='Call_imp_vol')

            # Call Annotate
            mask = np.array([(ele > ula_price) for ele in k_plot_list])
            annotate_list = np.array(k_plot_list)[mask]
            for i in range(3):
                annotate_index = k_plot_list.index(annotate_list[i])
                vol_annotate = vol_c_list[annotate_index]
                plt.annotate(text=f'{round(100*vol_annotate,2)}%', xy=(annotate_list[i], vol_annotate),
                             xytext=(annotate_list[i]+i*0.05, vol_annotate-0.05), arrowprops=dict(arrowstyle='<|-|>', connectionstyle='angle', color='red'))



            # Plot the Put VOL vs. K
            # Linear Interpolation
            y_p_interp = scipy.interpolate.interp1d(k_plot_list, vol_p_list)
            vol_p_interp = y_p_interp(k_val)

            plt.plot(k_plot_list, vol_p_list, 'v', color='b')
            plt.plot(k_val, vol_p_interp, '-.', color='b', label='Put_imp_vol')

            plt.axvline(x=ula_price, color='cyan', linestyle='dashdot', label='ATM_forward_price')
            plt.legend()
            plt.xlim(2.3, 3.3)
            plt.ylim(0.13, 0.46)

            plt.title(f'Implied volatility {targ_date} with ula {ula_price} at 14:59:00 vs strike')

            # Put Annotate
            mask = np.array([(ele < ula_price) for ele in k_plot_list])
            annotate_list = np.array(k_plot_list)[mask]
            for i in range(1,4):
                annotate_index = k_plot_list.index(annotate_list[-i])
                vol_annotate = vol_p_list[annotate_index]
                plt.annotate(text=f'{round(100 * vol_annotate, 2)}%', xy=(annotate_list[-i], vol_annotate),
                             xytext=(annotate_list[-i]-i*0.05, vol_annotate - 0.05),
                             arrowprops=dict(arrowstyle='<|-|>', connectionstyle='angle', color='blue'))
            ########################################################################################################

            # Plot the Call VOL vs. Delta
            plt.subplot(1, 2, 2)
            delta_c_list = []
            for i in range(len(close_list_c)):
                opt_meta = targ_c_contract[i]
                k = k_cp_list[i]
                vol = vol_c_list[i]
                delta = eu_opt_cal.get_greeks(opt_meta.cp_type, ula_price, float(k),
                                              ttm=dtc.get_days_diff(targ_date, opt_meta.get_exp_date()) / 252, r=0,
                                              div=0,
                                              vol=vol)[0]
                delta = 1- delta
                delta_c_list.append(delta)

            # Linear Interpolation
            y_cd_interp = scipy.interpolate.interp1d(delta_c_list, vol_c_list)
            delta_val = np.linspace(min(delta_c_list), max(delta_c_list), 50)
            vol_interp = y_cd_interp(delta_val)

            plt.plot(delta_c_list, vol_c_list, '^', color='r')
            plt.plot(delta_val, vol_interp, '-.', color='r', label='Call_imp_vol')

            # Plot for Put Delta and Vol
            delta_p_list = []
            for i in range(len(close_list_c)):
                opt_meta = targ_p_contract[i]
                k = k_cp_list[i]
                vol = vol_p_list[i]
                delta = eu_opt_cal.get_greeks(opt_meta.cp_type, ula_price, float(k),
                                              ttm=dtc.get_days_diff(targ_date, opt_meta.get_exp_date()) / 252, r=0,
                                              div=0,
                                              vol=vol)[0]
                delta = abs(delta)
                delta_p_list.append(delta)

            # Linear Interpolation
            y_interp = scipy.interpolate.interp1d(delta_p_list, vol_p_list)
            delta_val = np.linspace(min(delta_p_list), max(delta_p_list), 50)
            vol_interp = y_interp(delta_val)

            plt.plot(delta_p_list, vol_p_list, 'v', color='b')
            plt.plot(delta_val, vol_interp, '-.', color='b', label='Put_imp_vol')

            plt.title(f'Implied volatility {targ_date} with ula {ula_price} at 14:59:00 vs delta')
            plt.axvline(x=0.5, color='cyan', linestyle='dashdot', label='0.5 delta')
            plt.axvline(x=0.25, color='purple', linestyle='--', label='0.25 delta put')
            plt.axvline(x=0.75, color='orange', linestyle='--', label='0.25 delta call')
            plt.legend()
            plt.xlim(0.0, 1.0)
            plt.ylim(0.13, 0.46)

            # Call Annotate
            mask = np.array([(ele > 0.5) for ele in delta_c_list])
            annotate_list = np.array(delta_c_list)[mask]
            for i in range(3):
                annotate_index = delta_c_list.index(annotate_list[i])
                vol_annotate = vol_c_list[annotate_index]
                plt.annotate(text=f'{round(100 * vol_annotate, 2)}%', xy=(annotate_list[i], vol_annotate),
                             xytext=(annotate_list[i] + i * 0.05, vol_annotate - 0.05),
                             arrowprops=dict(arrowstyle='<|-|>', connectionstyle='angle', color='red'))

            # Put Annotate
            mask = np.array([(ele < 0.5) for ele in delta_c_list])
            annotate_list = np.array(delta_c_list)[mask]
            for i in range(1, 4):
                annotate_index = delta_c_list.index(annotate_list[-i])
                vol_annotate = vol_p_list[annotate_index]
                plt.annotate(text=f'{round(100 * vol_annotate, 2)}%', xy=(annotate_list[-i], vol_annotate),
                             xytext=(annotate_list[-i] - i * 0.05, vol_annotate - 0.05),
                             arrowprops=dict(arrowstyle='<|-|>', connectionstyle='angle',
                                             color='blue'))

            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)
            plt.savefig(f'{targ_date}_K_delta.png')
            ##########################################################################################################

            #         for i in range(len(close_list)):
            #             opt_meta = targ_contract[i]
            #             k = k_cp_list[i]
            #             vol = vol_list[i]
            #             delta = delta_list[i]
            #             new_row = pd.DataFrame(
            #                 {'contract_code': opt_meta.contract_code, 'trading_code': opt_meta.trading_code, 'Delta': delta, 'K': k, 'IV': vol,
            #                  'CP_type': opt_meta.cp_type, 't':dtc.get_days_diff(targ_date, opt_meta.get_exp_date()) / 252, 'r':0}, index=[1])
            #             df.loc[len(df)] = new_row.iloc[0]
            #
            # print(df)