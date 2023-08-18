import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
from ta.momentum import rsi
import pyodbc
import quantstats as qs
from Rolling_Functions_Final import * 







if __name__ == "__main__":

    window_list = ['510050', '510300', '510500', '159915', '159901', '588000']

    Daily_Return = Rolling.get_data(window_list)
    data_table = Rolling.single_window_table(Daily_Return, [21], len(window_list))
    print(data_table)

    list_of_codes = ['510050', '510300', '159915', '159901', '588000', '510500']
    window_list = [21]
    data = Rolling.get_data(list_of_codes)
    data_2016 = data.iloc[2666:,:]
    OPT_table = OPT.opt_strat_table(data_2016, window_list, len(list_of_codes))
    print(OPT_table)

    OG_table = Rolling.single_window_table(data_2016, window_list, len(list_of_codes))
    print(OG_table)