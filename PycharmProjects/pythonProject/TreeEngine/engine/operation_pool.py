import pandas as pd


from TreeEngine.engine.operations import register_pool,Operation


class OperationPool:
    def __init__(self):
        self.operations = {op.op_name:op for op in register_pool}

    def get_op(self,op_name) -> Operation:
        if op_name not in self.operations:
            raise KeyError(f'operation {op_name} not registered yet. ')
        return self.operations[op_name]


if __name__ == '__main__':
    op_pool = OperationPool()

    # test_data = pd.read_feather(r'D:\code\ResearchCode\TreeEngine\experiments\test_data.ftr')
    test_data = pd.read_feather(r"C:\Users\ps\PycharmProjects\TreeEngine\test.feather")

    # 生成数据
    tasks = {
        'last_close':
            {
                'EMA': [[5],[10],[15],[20]],
             },
        'vwap':
            {
                'EMA':[[5],[10],[15],[20]],
            },
        'volume':
            {
                'EMA':[[5],[10],[15],[20]],
            }
    }

    for data_name in tasks:
        for op_name in tasks[data_name]:
            _op = op_pool.get_op(op_name)
            for targ_param in tasks[data_name][op_name]:
                _op.cal(test_data, data_name, targ_param)



    # 计算AR
    calculation_tasks = {
        'AR':['last_close','vwap'],
        'DCUpper':['last_close','vwap','volume'],
        'DCLower':['last_close','vwap','volume','vwap.EMA_5'],
        'ATR':['last']
    }
