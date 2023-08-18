"""
将 instrument dict 变成类，实现懒加载
"""
from pathlib import Path
from public_tools.public_instance import etf_marg_cal, idx_marg_cal, comm_marg_cal, fut_marg_cal
import pandas as pd
from data_manager.data_processors import Processor


class DataManager:
    def __init__(self, data_processor: Processor = None):
        """

        :param data_processor: 如果需要对读取的数据做计算或者其他操作，则输入processor
        """
        self._loaded = dict()  # key: dataname,value:df
        # instrument_dict[data_name] = {'data': ins_data, 'type': opt_type, 'mult': mult, 'margin': marg}
        self.ins_info = dict()  # key: data name, value: meta or dict. 对于 deriv 是meta，对于 etf 是 dict
        self._margin_cals = {
            'etf': None, 'etf_opt': etf_marg_cal, 'comm_opt': comm_marg_cal, 'index_opt': idx_marg_cal,
            'fut': fut_marg_cal,
            'index': None
        }
        self._ins_data_path = dict()  # key:data, value :path
        self._data_processor = data_processor

    def add_instrument(self, data_name, data_path: Path, data_type: str, add_info_dict: dict, data_meta=None):
        """


        :param data_name:
        :param data_type: fut, etf, index_opt, comm_opt, etf_opt
        :param add_info_dict: 其他需要添加的信息. 尽量在 add info dict 多添加会用到的信息，而不要使用meta。因为 info dict 的信息调用最方便
        :return:
        """

        self.ins_info[data_name] = {'meta': data_meta, 'type': data_type, 'margin': self._margin_cals[data_type]}
        self.ins_info[data_name].update(add_info_dict)
        self._ins_data_path[data_name] = data_path

    def _load_data(self, data_path: Path) -> pd.DataFrame:
        file_type = data_path.suffix
        if file_type == '.csv':
            return pd.read_csv(data_path)
        elif file_type == '.pkl':
            return pd.read_pickle(data_path)
        elif file_type == '.ftr':
            return pd.read_feather(data_path)
        else:
            raise ValueError(f'Unknown data type: {file_type}')

    def get_data(self, data_name, do_copy=False):
        """
        得到数据。

        :param do_copy: if true 则 返回 dataframe 的 copy，否则返回 指针。返回指针时可能导致：应用端修改dataframe，同时数据端dataframe也改变
        :return:
        """
        if data_name not in self.ins_info:
            raise KeyError('需要先调用 add instument 加入合约')

        if data_name not in self._loaded:
            data = self._load_data(self._ins_data_path[data_name])
            if self._data_processor:
                data = self._data_processor.process(data)
            self._loaded[data_name] = data

        return self._loaded[data_name] if not do_copy else self._loaded[data_name].copy(deep=True)

    def update_data(self, data_name, new_data: pd.DataFrame):
        """
        在外部处理之后，有些 dataframe 需要更新到 loaded 中 方便重复使用。

        :return:
        """
        self._loaded[data_name] = new_data

    def get_item(self, data_name, item_name):
        if data_name not in self.ins_info:
            raise KeyError(f'未添加数据{data_name}')
        if item_name not in self.ins_info[data_name]:
            raise KeyError(f'未定义数据{data_name}的{item_name}')
        return self.ins_info[data_name][item_name]

    def get_type(self, data_name):
        return self.get_item(data_name, 'type')

    def get_meta(self, data_name):
        return self.get_item(data_name, 'meta')

    def get_mult(self, data_name):
        """
        对于期权来讲是 init mult
        :param data_name:
        :return:
        """
        return self.get_item(data_name, 'mult')

    def get_margin_calculator(self, data_name):
        return self.get_item(data_name, 'margin')

