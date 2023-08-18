import datetime
import multiprocessing
from multiprocessing import cpu_count,Queue,Manager
from pathlib import Path
import pandas as pd
from timecal import DateTimeCalculator
from meta import FutContracts


class MultiDataLoader:
    def __init__(self,use_cores:int = 5):
        """

        :param use_cores: how many cores to use to load data
        """
        self.num_procs = use_cores if use_cores < cpu_count() else cpu_count() - 1
        self.pool = multiprocessing.Pool(processes=self.num_procs)

    @staticmethod
    def _load_data_for_one_proce(data_tuple):
        data_name = data_tuple[0]
        data_path = data_tuple[1]
        data = pd.read_feather(data_path)
        return data_name,data

    def load_data(self,data_queue:Queue,one_day_path_dict:dict,multi_proce = True):
        """

        :param one_day_path_dict: contains paths to load. Usually ONLY contains !!ONE DAY!!'s data. Dict: data_name -> data path
        :param data_queue: data queue to convey data from loader to bt
        :param multi_proce: 是否使用 multiprocess。如果为False 则直接使用for 循环读取（相当于单核读取）
        :return:
        """
        if multi_proce:
            _datas = self.pool.map(self._load_data_for_one_proce,[(data_name,one_day_path_dict[data_name]) for data_name in one_day_path_dict])
            datas = {d[0]: d[1] for d in _datas}
        else:
            datas = {data_name:pd.read_feather(one_day_path_dict[data_name]) for data_name in one_day_path_dict}

        data_queue.put_nowait(datas)



if __name__ == '__main__':
    dl = MultiDataLoader(use_cores=4)
    data_q = Manager().Queue()
    dl_single = MultiDataLoader(1)

    dtc = DateTimeCalculator()
    start_date = '20220104'
    end_date = '20221230'

    ih_metas = FutContracts('IH')
    if_metas = FutContracts('IF')
    root = Path(r'\\192.168.0.88\Public\OptionDesk\DATA\database\1min_bar\fut')
    # signle load IF
    t0 = datetime.datetime.now()
    for date in dtc.get_trding_day_range(start_date, end_date):
        ctrct_list = [c.simp_code for c in if_metas.get_contract_series(date)]
        task_list = {c_code: root.joinpath(f'IF/{c_code}/{date}.ftr') for c_code in ctrct_list}
        dl.load_data(data_q, task_list, multi_proce=False)
        print(f'IF {date} done')
    t1 = datetime.datetime.now()
    print(t1 - t0)

    # multi load IH
    t0 = datetime.datetime.now()
    for date in dtc.get_trding_day_range(start_date,end_date):
        ctrct_list = [c.simp_code for c in ih_metas.get_contract_series(date)]
        task_list = {c_code: root.joinpath(f'IH/{c_code}/{date}.ftr') for c_code in ctrct_list}
        dl.load_data(data_q,task_list,multi_proce=True)
        print(f'IH {date} done')
    t1 = datetime.datetime.now()
    print(t1 - t0)




