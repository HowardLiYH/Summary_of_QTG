import pandas as pd
import numpy as np
from pathlib import Path
from timecal import DateTimeCalculator
from matplotlib import pyplot as plt
import math
from optcal.tool_funcs import ToolFuncs
import scipy.interpolate
from meta import ETFOptContracts
import warnings
warnings.filterwarnings('ignore')
import datetime


def load_ula_price(full_year_code):
    root = Path(r"\\192.168.0.88\Public\OptionDesk\DATA\database\1day_bar\etf_by_optmon\510050")
    targ_month = root.joinpath(full_year_code[2:])
    # Specific contract code inside the folder
    contract_code = '510050'
    file = targ_month.joinpath(f'{contract_code}.pkl')
    return pd.read_pickle(str(file))


def df_drop_duplicates(date_range, num, df):
    name = pd.DataFrame()
    type_df = df[df['ContractType'] == num]
    for i in date_range:
        medium = type_df[type_df['TradingDate'] == str(i)]
        modified = medium[~medium.ContractCode.duplicated(keep='first')].drop(['ContractType', ], axis=1)
        new_type = modified.groupby(['TradingDate', 'ExpirationDate']).sum().drop(['ContractCode', ], axis=1)
        name = pd.concat([name, new_type], axis=0)
    return name


def get_imp_vol(cp_type, ud_prc, k, ttm, r, div, option_prc):
    # parameter type: str, float, float, float, float, float, float
    # returns implied volatility, type list of floats

    # parameters:
    max_vol_adj = 0.1

    # 档位
    if ud_prc <= 0:
        return 1e-4  # 负ud价格不能计算 vol。直接返回最小值。
    if cp_type == 'P':
        if k - ud_prc >= option_prc:
            return 1e-4  # 负的时间价值
    elif cp_type == 'C':
        if ud_prc - k >= option_prc:
            return 1e-4

    sqrt_t = math.sqrt(ttm)
    exp_dt = math.exp(div * ttm)
    exp_mns_dt = 1 / exp_dt  # math.exp(-div * ttm)
    exp_mns_rt = math.exp(-r * ttm)
    # take a guess: impliedvol = 0.25
    imp_vol = 0.25
    max_iterations = 30
    for i in range(0, max_iterations):
        up = math.log(ud_prc / k) + ttm * (r - div + imp_vol * imp_vol / 2)
        down = imp_vol * sqrt_t
        d1 = up / down
        d2 = d1 - down
        if cp_type == 'C':
            Nd1 = ToolFuncs.cdf(d1)
            Nd2 = ToolFuncs.cdf(d2)
            bs_price = ud_prc * exp_mns_dt * Nd1 - k * exp_mns_rt * Nd2  # corrected by Chenhao on 10.12: 加负号
        elif cp_type == 'P':
            N_d1 = ToolFuncs.cdf(-d1)
            N_d2 = ToolFuncs.cdf(-d2)
            bs_price = k * exp_mns_rt * N_d2 - ud_prc * exp_mns_dt * N_d1  # corrected by Chenhao on 10.12: 加负号
        else:
            raise ValueError(f'wrong cp_type: {cp_type}')

        if abs(bs_price - option_prc) <= 0.0001 or imp_vol < 1e-8:
            return imp_vol

        ed1 = ToolFuncs.ndf(d1)
        vega = ud_prc * exp_dt * sqrt_t * ed1
        if vega < 1e-12:
            vega = 1e-12
        delta_vol = (option_prc - bs_price) / vega
        delta_vol = min(delta_vol, max_vol_adj)
        delta_vol = max(delta_vol, -max_vol_adj)
        imp_vol = max(delta_vol + imp_vol, imp_vol / 1000)
    return imp_vol


def get_greeks(cp_type, ud_prc, k, ttm, r, div, vol):
    """

    :param cp_type:
    :param ud_prc:
    :param k:
    :param ttm:
    :param r:
    :param vol:
    :param div:
    :return: delta, gamma, vega, theta, rho, vanna, vomma
    """
    # parameter type: str, float, float, float, float, float, float
    # returns BSGreeks, type list of float

    sqrt_t = math.sqrt(ttm)
    up = math.log(ud_prc / k) + ttm * (r - div + vol * vol / 2)
    down = vol * math.sqrt(ttm)
    d1 = up / down
    d2 = d1 - down
    # expDT = math.exp( div * t ) # corrected on 2021-07-29 by XING
    exp_mns_dt = math.exp(-div * ttm)
    exp_mns_rt = math.exp(-r * ttm)

    # delta
    Nd1 = ToolFuncs.cdf(d1)
    if cp_type == 'C':
        delta = exp_mns_dt * Nd1
    elif cp_type == 'P':
        delta = exp_mns_dt * (Nd1 - 1)
    else:
        raise ValueError('Wrong cp_type')

    # gamma
    ed1 = ToolFuncs.ndf(d1)
    gamma = exp_mns_dt / ud_prc / vol / sqrt_t * ed1

    Nd2 = ToolFuncs.cdf(d2)
    N_d2 = ToolFuncs.cdf(-d2)

    # rho
    if cp_type == 'C':
        rho = k * ttm * exp_mns_rt * Nd2  # / 100
    elif cp_type == 'P':
        rho = -k * ttm * exp_mns_rt * N_d2  # / 100
    else:
        raise ValueError('Wrong cp_type')

    a = ud_prc * vol * exp_mns_dt / (2 * sqrt_t) * ed1
    N_d1 = ToolFuncs.cdf(-d1)
    # theta
    # annualTradingDays = 245
    if cp_type == 'C':
        b = r * k * exp_mns_rt * Nd2
        c = div * ud_prc * exp_mns_dt * Nd1  # corrected on 2021-07-29 by XING
        theta = (-a - b + c)  # / annualTradingDays
    elif cp_type == 'P':
        b = r * k * exp_mns_rt * N_d2
        c = div * ud_prc * exp_mns_dt * N_d1  # corrected on 2021-07-29 by XING
        theta = (-a + b - c)  # / annualTradingDays
    else:
        raise ValueError('Wrong cp_type')

    # vega
    vega = ud_prc * exp_mns_dt * sqrt_t * ed1  # / 100  # corrected on 2021-07-29 by XING

    # reference for above : https://www.macroption.com/black-scholes-formula/

    return [delta, gamma, vega, theta, rho]


def indicator_generation(delta_c_list, delta_p_list, c_iv_list, p_iv_list, seperator_c):
    # Interpolate and Calculate ATM_VOL, Skew_10, Skew_25, Kurtosis_10, and Kurtosis_25
    delta_cp_list = delta_p_list[:seperator_c] + delta_c_list[seperator_c:]
    iv_cp_list = p_iv_list[:seperator_c] + c_iv_list[seperator_c:]

    # Interpolation set up
    y_cp_interp = scipy.interpolate.interp1d(delta_cp_list, iv_cp_list)

    # Extrapolation set up
    f = scipy.interpolate.interp1d(delta_cp_list, iv_cp_list, fill_value='extrapolate')

    delta_range = [0.5, 0.75, 0.6, 0.25, 0.4]
    keys = ['vol_50', 'vol_75', 'vol_60', 'vol_25', 'vol_40']
    interp_range = {key: None for key in keys}

    for a, b in enumerate(delta_range):
        try:
            interp_range[keys[a]] = f(b)
        except:
            interp_range[keys[a]] = y_cp_interp(b)

    # Calculate the four indicators
    delta_25_skew = interp_range['vol_25'] / interp_range['vol_75']
    delta_10_skew = interp_range['vol_40'] / interp_range['vol_60']
    kurtosis_25 = (interp_range['vol_25'] + interp_range['vol_75']) / (2 * interp_range['vol_50'])
    kurtosis_10 = (interp_range['vol_40'] + interp_range['vol_60']) / (2 * interp_range['vol_50'])
    atm_vol = interp_range['vol_50']

    return delta_25_skew, delta_10_skew, kurtosis_25, kurtosis_10, atm_vol


def plot_inter_all(x, y, z, e):
    f = scipy.interpolate.interp1d(x, y)
    xnew = np.arange(min(x), max(x), 0.025)
    ynew = f(xnew)
    plt.plot(x, y, ',', color=str(z))
    plt.plot(xnew, ynew, '-', color=str(z), label=f'Due_{e + 1}')


def list_to_df(five_features, Title):
    feature_df = pd.DataFrame()
    for feature in five_features:
        for sequence in feature:
            index = sequence + 1
            column_name = f'{Title[five_features.index(feature)]}_due_{index}'
            feature_df[column_name] = feature[sequence]
    return feature_df

def create_prices(old, correspond_index):
    date_range = correspond_index
    dtc = DateTimeCalculator()
    old.TradingDate = pd.to_datetime(old.TradingDate)
    old = old[old.TradingDate.isin(date_range)]

    count = 0

    Skew_25 = {0: [], 1: [], 2: [], 3: []}
    Skew_10 = {0: [], 1: [], 2: [], 3: []}
    Kur_25 = {0: [], 1: [], 2: [], 3: []}
    Kur_10 = {0: [], 1: [], 2: [], 3: []}
    A = {0: [], 1: [], 2: [], 3: []}

    for date in date_range:

        # # Take out the last four days before expirations
        # full_year_code = date.replace('-','')[:-2]
        # last_four = load_ula_price(full_year_code).index[-4:]
        # if date in last_four:
        #     continue

        # # todo for debugg to fix delta
        # if date != '2019-05-29':
        #     continue

        call_price = old[old.ContractType == 2]
        put_price = old[old.ContractType == 3]

        # Call and K
        c_middleman = call_price[call_price.TradingDate == date]
        c_middleman.drop_duplicates(subset='ContractCode', inplace=True)
        c_middleman = c_middleman.reset_index().drop(['index', ], axis=1)

        # Update the StrikePrice
        for row in range(len(c_middleman)):
            calculator = etf_metas.get_contract_meta(str(c_middleman.loc[row]['ContractCode']))

            timestamp = c_middleman.loc[row]['TradingDate']
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")[:-9].replace('-', '')

            correct_strike = calculator.get_strike(timestamp_str)
            c_middleman.at[row, 'StrikePrice'] = float(correct_strike)

        k_list = c_middleman.StrikePrice
        c_list = c_middleman.ClosePrice
        exp_list = c_middleman.ExpirationDate
        contract_list = c_middleman.ContractCode

        # Put
        p_middleman = put_price[put_price.TradingDate == date]
        p_middleman.drop_duplicates(subset='ContractCode', inplace=True)
        p_middleman = p_middleman.reset_index().drop(['index', ], axis=1)
        p_list = p_middleman.ClosePrice

        # DateFrame with K, C, and P
        functional_df = pd.DataFrame()
        functional_df = pd.concat([functional_df, k_list], axis=1)
        functional_df = pd.concat([functional_df, c_list], axis=1)
        functional_df.rename(columns={'ClosePrice': 'C_ClosePrice'}, inplace=True)
        functional_df = pd.concat([functional_df, p_list], axis=1)
        functional_df.rename(columns={'ClosePrice': 'P_ClosePrice'}, inplace=True)
        functional_df = pd.concat([functional_df, exp_list], axis=1)
        functional_df = pd.concat([functional_df, contract_list], axis=1)

        functional_df = functional_df.sort_values('StrikePrice', ascending=True)

        # Calculate ATMVOL, Skew, Kurtosis for each trading date with its corrsponding due date
        list_of_expiries = np.sort(c_middleman.ExpirationDate.unique())

        for num in range(len(list_of_expiries)):
            new_functional_df = functional_df[functional_df['ExpirationDate'] == list_of_expiries[num]]
            new_functional_df = new_functional_df.reset_index().drop(['index', ], axis=1)

            # Calculating ula price with ATM Forward
            diff_cp_list = new_functional_df['C_ClosePrice'] - new_functional_df['P_ClosePrice']
            min_row = new_functional_df.loc[np.nanargmin(abs(diff_cp_list))]
            k_atm, c_atm, p_atm = min_row.StrikePrice, min_row.C_ClosePrice, min_row.P_ClosePrice
            ula_price = c_atm - p_atm + k_atm

            # Calculate IV row by row
            c_iv_list = []
            p_iv_list = []

            cur_date = date.replace('-', '')
            exp_date = list_of_expiries[num].strip('-').replace('-', '')

            ttm = dtc.get_days_diff(cur_date, exp_date) / 252
            ud_prc = ula_price
            r = 0
            div = 0
            c_type = 'C'
            p_type = 'P'

            # todo: Later try to modify into form of DataFrame and see if it works
            for row in range(len(new_functional_df)):
                k = new_functional_df['StrikePrice'][row]
                c_option_prc = new_functional_df['C_ClosePrice'][row]
                p_option_prc = new_functional_df['P_ClosePrice'][row]

                c_iv = get_imp_vol(c_type, ud_prc, k, ttm, r, div, c_option_prc)
                p_iv = get_imp_vol(p_type, ud_prc, k, ttm, r, div, p_option_prc)

                c_iv_list.append(c_iv)
                p_iv_list.append(p_iv)

            new_functional_df['c_iv'] = c_iv_list
            new_functional_df['p_iv'] = p_iv_list

            # Calculate Delta row by row
            delta_c_list = []
            delta_p_list = []

            for row in range(len(new_functional_df)):
                k = new_functional_df['StrikePrice'][row]
                c_iv = new_functional_df['c_iv'][row]
                p_iv = new_functional_df['p_iv'][row]

                delta_c = get_greeks(c_type, ud_prc, k, ttm, r, div, c_iv)[0]
                delta_p = get_greeks(p_type, ud_prc, k, ttm, r, div, p_iv)[0]

                delta_c_list.append(1 - delta_c)
                delta_p_list.append(abs(delta_p))

            delta_c_list = sorted(delta_c_list)
            delta_p_list = sorted(delta_p_list)

            # Calculate OTM Call and Put's Delta and IV
            try:
                seperator_c = np.where(np.array(delta_c_list) > 0.5)[0][0]
                delta_25_skew, delta_10_skew, kurtosis_25, kurtosis_10, atm_vol = \
                    indicator_generation(delta_c_list, delta_p_list, c_iv_list, p_iv_list, seperator_c)
                if atm_vol > 1.0:
                    print(f'Today is {date} with {num}th due date {list_of_expiries[num]} with ttm as {ttm * 252}')

            except IndexError:
                count += 1
                # print(f'Today is {date} with {num}th due date {list_of_expiries[num]}')
                # print(f'IndexError{count} and ttm is {ttm}: \n')
                # print(f'the delta_c_list is {delta_c_list} \n')
                # print(f'the delta_p_list is {delta_p_list}\n')
                # print(f'____________________________________________________________________')
                delta_25_skew = None
                delta_10_skew = None
                kurtosis_25 = None
                kurtosis_10 = None
                atm_vol = None

            Skew_25[num].append(delta_25_skew)
            Skew_10[num].append(delta_10_skew)
            Kur_25[num].append(kurtosis_25)
            Kur_10[num].append(kurtosis_10)
            A[num].append(atm_vol)

    return Skew_25, Skew_10, Kur_25, Kur_10, A


if __name__ == '__main__':
    etf_metas = ETFOptContracts('510050')

    # Raw SQL Date
    k = pd.read_csv(r"C:\Users\ps\PycharmProjects\pythonProject\data3.csv")
    k = k.drop(['Unnamed: 0', ], axis=1)
    k.TradingDate = pd.to_datetime(k.TradingDate)

    features = pd.read_csv(r"C:\Users\ps\PycharmProjects\pythonProject\Features_Finalization.csv")
    # features.tradingday = pd.to_datetime(features.tradingday)
    features.tradingday = features.tradingday

    # test = ['2018-04-28']
    # test = ['2018-03-29', '2018-03-30', '2018-04-02', '2018-04-03', '2018-04-04', '2018-04-09', '2018-04-10',
    #         '2018-04-11', '2018-04-12', '2018-04-13', '2018-04-16', '2018-04-17', '2018-04-18', '2018-04-19',
    #         '2018-04-20', '2018-04-23', '2018-04-24', '2018-04-25']
    # test = pd.to_datetime(test)

    # create_prices(k, test)

    Skew_25, Skew_10, Kur_25, Kur_10, A = create_prices(k, features.tradingday)

    five_features = [Skew_25, Skew_10, Kur_25, Kur_10, A]
    Title = ['Skew_25D', 'Skew_10D', 'Kurtosis_25D', 'Kurtosis_10D', "ATMVOL"]

    feature_df = list_to_df(five_features, Title)
    feature_df['next_ATMVOL_due_1'] = feature_df['ATMVOL_due_1'].shift(-1)
    feature_df['IV_return_due_1'] = feature_df['next_ATMVOL_due_1'] / feature_df['ATMVOL_due_1']
    feature_df.to_csv(r"C:\Users\ps\PycharmProjects\pythonProject\final_feature_df.csv")

    # print(f'Skew_25 with length {len(Skew_25[0])} is {Skew_25} \n')
    # print(f'Skew_10 with length {len(Skew_10[0])} is {Skew_10} \n')
    # print(f'Kur_25 with length {len(Kur_25[0])} is {Kur_25} \n')
    # print(f'Kur_10 with length {len(Kur_10[0])} is {Kur_10} \n')
    # print(f'A is with length {len(A[0])} {A} \n')
    #
    # # Plotting the data
    # color_plate = ['pink', 'c', 'sandybrown', 'y', 'mediumseagreen']
    # five_features = [Skew_25, Skew_10, Kur_25, Kur_10, A]
    # Title = ['Skew_25D', 'Skew_10D', 'Kurtosis_25D', 'Kurtosis_10D', "ATMVOL"]
    #
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
    #     plt.savefig(f'div_features_NoLast4_{Title[index]}.png')
    #     plt.clf()
