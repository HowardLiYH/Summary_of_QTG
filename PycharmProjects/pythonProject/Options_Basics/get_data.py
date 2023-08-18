from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from meta import ETFOptContracts
from optcal import EuOptCalculator
from timecal import DateTimeCalculator



def load_data(full_year_code,contract_code):
    root = Path(r"\\192.168.0.88\Public\OptionDesk\DATA\database\1day_bar\opt_by_optmon\510050")
    targ_month = root.joinpath(full_year_code[2:])
    # Specific contract code inside the folder
    file = targ_month.joinpath(f'{contract_code}.pkl')
    return pd.read_pickle(str(file))

def load_targ_date(full_year_code,contract_code):
    root = Path(r"\\192.168.0.88\Public\OptionDesk\DATA\database\1day_bar\etf_by_optmon\510050")
    targ_month = root.joinpath(full_year_code[2:])
    # Specific contract code inside the folder
    contract_code = '510050'
    file = targ_month.joinpath(f'{contract_code}.pkl')
    return pd.read_pickle(str(file))

if __name__ == '__main__':
    full_year_code = '202301'
    targ_date = '20230119'
    ula_price = float(2.73)

    dtc = DateTimeCalculator()

    # Open the folder named '510050'
    etf_metas = ETFOptContracts('510050')

    # calculate vol curve of 202301 on 20230119

    # Get the four corresponding expiration dates in form of yyymm given the input target date
    series_info = etf_metas.get_dueyymm_series(targ_date)

    # get all the 14 K strike price of the specific date given the input full year code and target date
    k_list = etf_metas.get_otm_strikes(full_year_code, targ_date, ula_price, 8, 0)
    k_cp_list = k_list['P'] + k_list['C']
    # A list contains 14 specific contract info with all the call contracts based on each strike in k_list and the given the input full year code and target date
    # Including ITM, ATM, OTM
    targ_c_contract = [etf_metas.get_contract_by_strike(targ_date, full_year_code, 'C', k, 0) for k in k_list['C']]
    targ_p_contract = [etf_metas.get_contract_by_strike(targ_date, full_year_code, 'P', k, 0) for k in k_list['P']]
    targ_contract = targ_p_contract + targ_c_contract

    # Load all the contract from the list with 14 contracts and get the contract code to create a list of DataFrames
    # The full year params here is used to set the path
    datas = [load_data(full_year_code, c.contract_code) for c in targ_contract]

    # Initializing
    eu_opt_cal = EuOptCalculator()

    # Access the list of DataFrame and get the closing price through each DataFrame and  compile them into a list in form of float
    close_list = [d.loc[pd.Timestamp(targ_date), 'close'] for d in datas]

    # Initializing a list for volatility
    vol_list = []

    # Since total len of close_list is 14, i goes from 0 to 13
    for i in range(len(close_list)):
        opt_meta = targ_contract[i]
        k = k_cp_list[i]

        # calculate the iv through get_imp_vol
        ################# Look into more on get_imp_vol
        vol = eu_opt_cal.get_imp_vol(opt_meta.cp_type, 2.829, float(k),
                                     ttm=dtc.get_days_diff(targ_date, opt_meta.get_exp_date()) / 252, r=0, div=0,
                                     option_prc=close_list[i])
        vol_list.append(vol)

    plt.scatter(k_cp_list, vol_list)
    plt.show()


