import matplotlib.pyplot as plt
import pandas as pd
from timecal import DateTimeCalculator
import numpy as np


if __name__ == '__main__':

    dtc = DateTimeCalculator()

    np.random.seed(6)

    df = pd.DataFrame()
    df['str_date'] = dtc.get_trding_day_range('20220104','20220121')
    df['value'] = np.random.randint(0,100,size=len(df['str_date']))
    df['int_index'] = range(len(df['str_date']))




    # # # fig = plt.gcf()
    # # # fig.set_size_inches(18.5, 10.5)
    # # plt.title('str_date as x axis')
    # # plt.plot(df['str_date'].values,df['value'].values)
    # # plt.show()
    # #
    # # fig = plt.gcf()
    # # fig.set_size_inches(18.5, 10.5)
    # plt.title('int index as x axis')
    # plt.plot(df['int_index'].values,df['value'].values)
    # plt.show()
    #
    # # plan A
    # # fig.set_size_inches(18.5, 10.5)
    # plt.title('dt_date as x axis')
    # plt.plot(pd.to_datetime(df['str_date']),df['value'].values)
    # plt.show()
    # # pros and cons: pros 方便且以年为单位时，看不出,cons: non trade day

    # plan B:
    plt.title('dt_date as x axis')
    plt.plot(df['int_index'], df['value'].values)
    df.loc[df.index[::5],'str_date']
    plt.show()



