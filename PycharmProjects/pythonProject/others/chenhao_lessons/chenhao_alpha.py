import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from timecal import DateTimeCalculator

dtc = DateTimeCalculator()

res = pd.DataFrame()
s,e =dtc.adj_date_range('20151224','20220320')
res['str_date'] = dtc.get_trding_day_range(s,e)
res['values'] = range(len(res))
res['int_index'] = range(len(res))

#
# plt.plot(res['int_index'],res['values'])
# choose_these_dates = res['str_date'].values[::30]
#
# plt.xticks(res.index[::30], choose_these_dates, rotation=45)
# plt.show()

# plt.plot(res['str_date'],res['values'])
# plt.show()
#
# plt.plot(res['int_index'],res['values'])
# plt.show()
#
# plt.plot(pd.to_datetime(res['str_date']),res['values'])
# plt.show()
