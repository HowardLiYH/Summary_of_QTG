import re
import os
import numpy as np
import pandas as pd
from TreeEngine.tools import re_tool, common_calculations, CommonCals, RETool
import pyodbc


# Operation: 针对一个时间序列（包含其OHLC）进行一个操作，返回一列。


class Operation:
    def __init__(self):
        self.op_name = None
        self.param_num: int = 0
        self.param_range = []
        self._init()
        # self.loader = FactorLoader()

    def _init(self):
        pass

    def __str__(self):
        return self.op_name

    def add_op(self, data_name, op_param_list):
        """
        attach operation name to data name

        :return:
        """
        return f'{data_name}.{self.op_name}' + '_'.join([''] + [str(p) for p in op_param_list])

    def _special_check(self,param_list):
        """
        某些operation 除了 常规的 param list check 之外，还需要特殊的check。此时重写此私有函数即可。

        :param param_list:
        :return:
        """
        return True

    def _check_param_list(self, param_list):
        """
        检查输入的 param list 是否符合 init 中定义的长度与范围

        :param param_list:
        :return:
        """
        if len(param_list) != self.param_num:
            raise ValueError('输入的param的个数与 self.param_num 不符')
        for i in range(len(param_list)):
            p = param_list[i]
            if not (self.param_range[i][0] <= p <= self.param_range[i][1]):
                raise ValueError(f'param list 中的第{i}个参数，不在范围{self.param_range[i][:2]}内')
        self._special_check(param_list)

    def cal(self,  data,data_name, param_list, check_param=True, skip_exists = True):
        """

        :param data_name:
        :param data:
        :param param_list: 该operation的参数，按顺序放入。
        :param check_param: 是否对输入的 param list 进行检查：1. 检查长度，2. 检查每个参数的范围。
        :param skip_exists: 为 True ，则判断 res name 是否在 data 中。若在则跳过。避免重复计算。为False 则重新计算。
        :return:
        """
        if check_param:
            self._check_param_list(param_list)
        res_name = self.add_op(data_name, param_list)
        if skip_exists and res_name in data.columns:
            return res_name,data[res_name]
        ## Rename the column, replace all infinities with nan, and forward fill the nan with the previous value
        ## Going down to _cal belonged to each method
        res_data = self._cal(data,data_name, param_list).rename(res_name).replace(np.inf,np.nan).replace(-np.inf,np.nan).ffill()
        # if attach_res:
        #     data.loc[:,res_name] = res_data
        return res_name, res_data

    def _cal(self, data,data_name, param_list) -> pd.Series:
        return data[data_name]


class TR(Operation):

    def _init(self):
        self.op_name = 'TR'
        self.param_num: int = 0
        self.param_range = []

    def _cal(self, data,data_name, param_list):
        return CommonCals.true_range(data,data_name)


class DCUpper(Operation):
    def _init(self):
        self.op_name = 'DCUpper'
        self.param_num: int = 1
        self.param_range = [(0, np.inf, int), ]


    def _cal(self, data,data_name, param_list):
        res_data = data[data_name].rolling(param_list[0]).max()
        return res_data


class DCLower(Operation):
    def _init(self):
        self.op_name = 'DCLower'
        self.param_num: int = 1
        self.param_range = [(0, np.inf, int), ]

    def _cal(self, data,data_name, param_list):
        res_data = data[data_name].rolling(param_list[0]).min()
        return res_data


class CoarseDCUpper(Operation):
    def _init(self):
        self.op_name = 'CoarseDCUpper'
        self.param_num: int = 2
        self.param_range = [(0,np.inf,int),(0,np.inf,int),]
        self._ema = EMA()

    def _cal(self, data,data_name, param_list):
        n,m = param_list[0],param_list[1]
        if 'high' not in data.columns:
            raise ValueError('Do Not Find Column named  high in the input DataFrame')
        if data_name != 'high':
            raise ValueError('Sorry, the input data_name must be high')
        _,high_ema = self._ema.cal(data, data_name, [n])
        res_data = high_ema.rolling(m).max()
        return res_data


class CoarseDCLower(Operation):
    def _init(self):
        self.op_name = 'CoarseDCLower'
        self.param_num: int = 2
        self.param_range = [(0,np.inf,int),(0,np.inf,int),]
        self._ema = EMA()

    def _cal(self, data,data_name, param_list):
        n,m = param_list[0],param_list[1]
        if 'low' not in data.columns:
            raise ValueError('Do Not Find Column named  high in the input DataFrame')
        if data_name != 'low':
            raise ValueError('Sorry, the input data_name must be low')
        _,low_ema = self._ema.cal(data, data_name, [n])
        res_data = low_ema.rolling(m).min()
        return res_data

class ATR(Operation):
    def _init(self):
        self.op_name = 'ATR'
        self.param_num: int = 1
        self.param_range = [(0, np.inf, int), ]
        self._tr_op = TR()

    def _cal(self, data,data_name, param_list):
        n = param_list[0]
        _,true_range = self._tr_op.cal(data,data_name, [], check_param=True, skip_exists=True)
        tr_ma = true_range.rolling(n).mean()
        return tr_ma


class EMA(Operation):
    def _init(self):
        self.op_name = 'EMA'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        return data[data_name].ewm(span=param_list[0], adjust=False,min_periods=param_list[0]).mean()


class SMA(Operation):
    def _init(self):
        self.op_name = 'SMA'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        return data[data_name].rolling(param_list[0],min_periods=param_list[0]).mean()


# Area ratio
class AR(Operation):
    def _init(self):
        self.op_name = 'AR'
        self.param_num: int = 3 # 参数是对应价格的两个SMA均线的window size 和look back window
        self.param_range = [(0, np.inf, int),(0, np.inf, int),(0, np.inf, int), ]

    def _special_check(self,param_list):
        """
        对于 ATR，我们要求：param list[0] 是 fast n，param list[1]是 slow n。fast n 一定要严格小于 slow n

        :param param_list:
        :return:
        """
        if not (param_list[0] < param_list[1]):
            raise ValueError('对于ATR，param[0],fast n, 必须小于 param[1], slow n.')

    def _cal(self, data,data_name, param_list,ma_name = 'EMA'):
        # fast_ma_col = f'{data_name}.{ma_name}_{param_list[0]}'
        # slow_ma_col = f'{data_name}.{ma_name}_{param_list[1]}'

        slow_ma_col, slow_ma = EMA().cal(data, data_name, [param_list[0]])
        data[slow_ma_col] = slow_ma
        fast_ma_col, fast_ma = EMA().cal(data, data_name, [param_list[1]])
        data[fast_ma_col] = fast_ma


        ma_slow = data[slow_ma_col]
        ma_fast = data[fast_ma_col]
        look_back = param_list[2]
        area = abs(ma_slow - ma_fast).rolling(look_back, min_periods=0).sum()
        _temp_df = pd.DataFrame(index = data.index)
        _temp_df.loc[ma_slow < ma_fast, 'MA_check'] = 1
        _temp_df.loc[ma_slow > ma_fast, 'MA_check'] = -1
        _temp_df['MA_cross'] = 0
        _temp_df.loc[_temp_df['MA_check'] != _temp_df['MA_check'].shift(1), 'MA_cross'] = 1
        _temp_df.loc[_temp_df.index[0], 'MA_cross'] = 0
        # _temp_df['X_n'] = _temp_df['MA_cross'].rolling(look_back, min_periods=0).sum()  # cross point numbers
        res = area * (np.maximum(ma_slow,ma_fast) / np.minimum(ma_slow,ma_fast)) ** 2 / look_back / data[data_name]
        return res


# 假设ratio 只能比 同类数据 的 不同周期
class Ratio(Operation):
    def _init(self):
        self.op_name = 'Ratio'

    def _check_param_list(self, param_list):
        """
        Ratio 的param list 完全由输入的 data 决定。如果输入的 param list 不对，就直接找不到column。不用另行检查

        :param param_list:
        :return:
        """
        return True

    def _cal(self, data,data_name, param_list):
        # re.match(r'[\dA-Za-z]+(?:_[\dA-Za-z]+)+?((?:_\d+)*)', 'asdfasdf_3af666f_5_10_10').groups()
        orig_param_str = RETool.get_last_op_params(data_name)
        if len(orig_param_str) == 0:
            raise ValueError('目前Ratio只能作用于同类数据同类操作的不同参数。因此必须有 param string')
        new_param_str = '_'.join([''] + [str(p) for p in param_list])
        deno_name = data_name.rstrip(orig_param_str) + new_param_str
        return data[data_name] / data[deno_name]


class Diff(Operation):
    def _init(self):
        self.op_name = 'Diff'

    def _check_param_list(self, param_list):
        """
        Diff 的param list 完全由输入的 data 决定。如果输入的 param list 不对，就直接找不到column。不用另行检查

        :param param_list:
        :return:
        """
        return True

    def _cal(self, data,data_name, param_list):
        orig_param_str = RETool.get_last_op_params(data_name)
        if len(orig_param_str) == 0:
            raise ValueError('目前Diff只能作用于同类数据同类操作的不同参数。因此必须有 param string')
        new_param_str = '_'.join([''] + [str(p) for p in param_list])
        deno_name = data_name.rstrip(orig_param_str) + new_param_str
        return data[data_name] - data[deno_name]


class RSI(Operation):

    def _init(self):
        self.op_name = 'RSI'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        dif = data[data_name] - data[data_name].shift(1)
        n = param_list[0]
        res = pd.Series(data=0,index=dif.index)
        deno = CommonCals.sma(np.abs(dif),n)
        res[deno>1e-4] = CommonCals.sma(np.maximum(dif, 0),n) / deno * 100  # 这样处理避免分母为0
        return res


class PDI(Operation):
    def _init(self):
        self.op_name = 'PDI'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]
        self._tr_op = TR()

    def _cal(self, data,data_name, param_list):
        _,true_range = self._tr_op.cal(data, data_name, [], check_param=True, skip_exists=True)
        high = data[data_name+'_high']
        low = data[data_name+'_low']
        hd = high - high.shift(1)
        ld = low.shift(1) - low
        dmp = pd.Series(np.where((hd > 0) & (hd > ld), hd, 0)).rolling(param_list[0]).sum()
        pdi = dmp * 100 / true_range
        return pdi


class MDI(Operation):
    def _init(self):
        self.op_name = 'MDI'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]
        self._tr_op = TR()

    def _cal(self, data,data_name, param_list):
        _,true_range = self._tr_op.cal(data,data_name,  [], check_param=True, skip_exists=True)
        high = data[data_name+'_high']
        low = data[data_name+'_low']
        ld = low.shift(1) - low
        hd = high - high.shift(1)
        dmm = pd.Series(np.where((ld > 0) & (ld > hd), ld, 0)).rolling(param_list[0]).sum()
        mdi = dmm * 100 / true_range
        return mdi


class ADX(Operation):
    def _init(self):
        self.op_name = 'ADX'
        self.param_num: int = 2
        self.param_range = [(0,np.inf,int),(0,np.inf,int),]
        self._pdi = PDI()
        self._mdi = MDI()

    def _cal(self, data,data_name, param_list):
        n,m = param_list[0],param_list[1]
        _,mdi = self._mdi.cal(data, data_name, [n])
        _,pdi = self._pdi.cal(data, data_name, [n])
        adx = (np.abs(mdi - pdi) / (pdi + mdi) * 100).rolling(m).mean()
        return adx


class ADXR(Operation):
    def _init(self):
        self.op_name = 'ADXR'
        self.param_num: int = 2
        self.param_range = [(0,np.inf,int),(0,np.inf,int),]
        self._adx = ADX()

    def _cal(self, data,data_name, param_list):
        n,m = param_list[0],param_list[1]
        _,adx = self._adx.cal(data, data_name,param_list)
        adxr = (adx + adx.shift(m)) / 2
        return adxr



############   CTA Features #################
class T_Mom(Operation):
    def _init(self):
        self.op_name = 'T_Mom'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        current_price = data[data_name]
        past_price = data[data_name].shift(param_list[0]+ 1)
        raw_momentum_factor = current_price / past_price - 1
        return raw_momentum_factor


##----------------------- P.25 趋势动量因子 ------------------------------##
class T_RSM(Operation):
    def _init(self):
        self.op_name = 'T_RSM'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        # Todo: Test the function with accurate targets
        #  Here we dont have the data for CONTANGO or BACKWARDATION, so we used close price and prev_close to represent the ideology instead

        current_CONTANGO =data[data_name[0]]
        prev_CONTANGO = data[data_name[1]]
        sign = np.sign(current_CONTANGO - prev_CONTANGO)
        avg_sign = sign.rolling(param_list[0]).mean()

        return avg_sign

class T_RSI(Operation):

    def _init(self):
        self.op_name = 'T_RSI'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        dif = data[data_name] - data[data_name].shift(1)
        n = param_list[0]
        res = pd.Series(data=0,index=dif.index)
        deno = CommonCals.sma(np.abs(dif),n)
        res[deno>1e-4] = CommonCals.sma(np.maximum(dif, 0),n) / deno * 100  # 这样处理避免分母为0
        return res

class T_MAratio(Operation):

    def _init(self):
        self.op_name = 'T_MAratio'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    # def _cal(self, data,data_name, param_list):
    #     current_price = data[data_name]
    #     past_ma_price = CommonCals.sma(data[data_name],param_list[0]+1)
    #     result_MAratio = current_price / past_ma_price
    #     return result_MAratio

    def _cal(self, data,data_name, param_list):
        current_price = data[data_name]
        past_ma_price = data[data_name].shift(param_list[0]+1).rolling(param_list[0]+1).mean()
        result_MAratio = current_price / past_ma_price
        return result_MAratio


class T_MAcross(Operation):
    def _init(self):
        self.op_name = 'T_MAcross'
        self.param_num: int = 2 # 参数是对应价格的两个SMA均线的window size 和look back window
        self.param_range = [(0, np.inf, int),(0, np.inf, int)]

    def _special_check(self,param_list):
        """
        对于 T_MAcross，我们要求：param list[0] 是 fast n，param list[1]是 slow n。fast n 一定要严格小于 slow n

        :param param_list:
        :return:
        """
        if not (param_list[0] < param_list[1]):
            raise ValueError('对于T_MAcross，param[0],fast n, 必须小于 param[1], slow n.')

    def _cal(self, data,data_name, param_list):
        data['fast_ma'] = CommonCals.sma(data[data_name], param_list[0]+1)
        data['slow_ma'] = CommonCals.sma(data[data_name], param_list[1]+1)

        res = (data['fast_ma'] - data['slow_ma'])/data['slow_ma']
        return res


class T_Overnight(Operation):

    def _init(self):
        self.op_name = 'T_Overnight'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        for target in ('open', 'prev_close'):
            if target not in data.columns:
                raise ValueError('Input DataFrame missing required open, prev_close columns')
        open_data = data['open']
        prev_close_data = data['prev_close']
        res = ((open_data - prev_close_data)/prev_close_data).rolling(param_list[0]+1).mean()
        return res


class T_intraday(Operation):

    def _init(self):
        self.op_name = 'T_intraday'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        for target in ('open', 'high', 'low', 'close'):
            if target not in data.columns:
                raise ValueError('Input DataFrame missing required open, close, high, low columns')
        open_data = data['open']
        close_data = data['close']
        res = ((close_data - open_data)/close_data).rolling(param_list[0]+1).mean()
        return res


class T_CumStep(Operation):

    def _init(self):
        self.op_name = 'T_CumStep'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        if 'open' and 'close' and 'high' and 'low' not in data.columns:
            raise ValueError('Do not find column named open, close, high, and low in the input DataFrame')
        sign = np.sign(data['close'] - data['open'])
        res = (((2*(data['high']-data['low'])*sign)-(data['close'] - data['open']))/data['close']).rolling(param_list[0]+1).mean()
        return res


class T_STDS(Operation):

    def _init(self):
        self.op_name = 'T_STDS'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        for target in ('open', 'high', 'low', 'close'):
            if target not in data.columns:
                raise ValueError('Input DataFrame missing required open, close, high, low columns')
        sign = np.sign(data['close'] - data['open'])

        ## 开始计算 GK， GK的公式在波动率因子中，Page. 33 和 34
        h = np.log(data['high']) - np.log(data['open'])
        l = np.log(data['low']) - np.log(data['open'])
        c = np.log(data['close']) - np.log(data['open'])
        rowlling_sum = np.sqrt((0.5*(h-l)**2-(2*np.log(2)-1)*c**2).rolling(param_list[0]+1).sum())
        GK = rowlling_sum*np.sqrt(252/param_list[0])
        res = sign * GK * (1/param_list[0])
        return res


class T_Rank(Operation):

    def _init(self):
        self.op_name = 'T_Rank'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    ## The DF here is stacked
    def _cal(self, data,data_name, param_list):
        N = len(data_name)
        df = data[data_name]
        df = df.pct_change()
        ranked_df = df.rank(axis=1, method='max', ascending=False)
        # print(ranked_df)
        for name in data_name:
            ranked_df[name] = (ranked_df[name] - (N+1)/2)/np.sqrt((N+1)*(N-1)/12)
            ranked_df[name] = ranked_df[name].rolling(param_list[0]).sum()*(1/param_list[0])
        return ranked_df.stack()



        #
        # close_data = data['close']
        # prev_close_data = data['close'].shift(1)
        # pct_change = (close_data-prev_close_data)/prev_close_data
        # mean_return = pct_change.mean()
        # std_return = pct_change.std()
        # normalized_returns = (pct_change - mean_return) / std_return
        # res = normalized_returns.rolling(window=param_list[0]).mean()
        # return normalized_returns


##----------------------- P.26 波动率因子 ------------------------------##
class V_STD(Operation):

    def _init(self):
        self.op_name = 'V_STD'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        close_data = data[data_name]
        prev_close_data = data[data_name].shift(1)
        day_return = (close_data-prev_close_data)/prev_close_data
        return_mean = day_return.rolling(window=param_list[0]+1).mean()
        culmulative_return = ((day_return - return_mean)**2).rolling(window=param_list[0]+1).sum()*(1/param_list[0])*np.sqrt(252)
        return culmulative_return



class Vol_EMA(Operation):

    def _init(self):
        pass

    def _cal(self, data,data_name, param_list):
        pass



class V_RS(Operation):

    def _init(self):
        self.op_name = 'V_RS'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        for target in ('open', 'high', 'low', 'close'):
            if target not in data.columns:
                raise ValueError('Input DataFrame missing required open, close, high, low columns')
        open_data = data['open']
        close_data = data['close']
        high_data = data['high']
        low_data = data['low']
        h = np.log(high_data) - np.log(open_data)
        l = np.log(low_data) - np.log(open_data)
        c = np.log(close_data) - np.log(open_data)
        rowlling_sum = np.sqrt(abs((h*h*(h-c)-l*(l-c)).rolling(param_list[0]+1).sum()))
        RS = rowlling_sum*np.sqrt(252/param_list[0])
        return RS



class V_GK(Operation):

    def _init(self):
        self.op_name = 'V_GK'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        for target in ('open', 'high', 'low', 'close'):
            if target not in data.columns:
                raise ValueError('Input DataFrame missing required open, close, high, low columns')
        open_data = data['open']
        close_data = data['close']
        high_data = data['high']
        low_data = data['low']
        h = np.log(high_data) - np.log(open_data)
        l = np.log(low_data) - np.log(open_data)
        c = np.log(close_data) - np.log(open_data)
        rowlling_sum = np.sqrt(abs(0.5*(h-l)**2-(2*np.log(2)-1)*c**2).rolling(param_list[0]+1).sum())
        GK = rowlling_sum*np.sqrt(252/param_list[0])
        return GK



class V_PK(Operation):

    def _init(self):
        self.op_name = 'V_PK'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        for target in ('open', 'high', 'low', 'close'):
            if target not in data.columns:
                raise ValueError('Input DataFrame missing required open, close, high, low columns')
        open_data = data['open']
        close_data = data['close']
        high_data = data['high']
        low_data = data['low']
        h = np.log(high_data) - np.log(open_data)
        l = np.log(low_data) - np.log(open_data)
        c = np.log(close_data) - np.log(open_data)
        rowlling_sum = np.sqrt(abs((h-l)**2).rolling(param_list[0]+1).sum())
        PK = rowlling_sum*np.sqrt(252/(4*np.log(2)*param_list[0]))
        return PK



##----------------------- P.28 趋势因子 require time, further clarification needed ------------------------------##
class Mom_RSI(Operation):

    def _init(self):
        self.op_name = 'Mom_RSI'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        for target in ('open', 'high', 'low', 'close'):
            if target not in data.columns:
                raise ValueError('Input DataFrame missing required open, close, high, low columns')

        data['up'] = data.apply(lambda row: row['close'] if row['close'] > row['prev_close'] else -10000,
                                          axis=1)
        data['up'] = data['up'].rolling(window=5).apply(lambda x: x[x!=-10000].mean())

        data['down'] = data.apply(lambda row: row['close'] if row['close'] < row['prev_close'] else -10000,
                                          axis=1)
        data['down'] = data['down'].rolling(window=5).apply(lambda x: x[x!=-10000].mean())


        RSI = 100 - 100/(1+ data['up']/data['down'])
        return RSI


class Mom_STOCHRSI(Operation):
    ## This function requires two inputs which are High time and Low time
    ## After confirming the exact info with High time and Low time, we just simply apply the RSI function
    def _init(self):
        self.op_name = 'Mom_STOCHRSI'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
       pass


class Mom_ULTOSC(Operation):

    def _init(self):
        self.op_name = 'Mom_ULTOSC'
        self.param_num: int = 1
        self.param_range = [(0,np.inf,int),]

    def _cal(self, data,data_name, param_list):
        for target in ('open', 'high', 'low', 'close'):
            if target not in data.columns:
                raise ValueError('Input DataFrame missing required open, close, high, low columns')

        BP = data['close'] - min(data['prev_close'], data['close'])
        TR = max(data['prev_close'], data['high']) - min(data['prev_close'], data['low'])
        # UO =



        return RSI


register_pool = [TR(), DCUpper(), DCLower(), CoarseDCUpper(), CoarseDCLower(),
                 ATR(), EMA(), SMA(), AR(), Ratio(), Diff(), RSI(), PDI(), MDI(), ADX(), ADXR(),
                 T_Mom()]


def act_download(answer, col_name, code):
    if answer == 'True':
        df = pd.DataFrame()
        df[f'{name}'] = col_d
        df = df.dropna().sort_index()
        col_name = col_name.split('.')[-1]
        #
        #
        # print(col_name)
        # print('--------------------\n')
        # print(df)
        #
        #
        # # # Specify the directory path and file name for saving the Feather file
        directory_path = fr"\\192.168.0.88\Public\OptionDesk\DATA\ResearchData\FactorModel\day\{code}\factors"
        file_name = f"{col_name}.pkl"
        # fr"/mnt/88disk/Howard/Daily_ETF_{current_date}.csv"
        # Create the directory if it doesn't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Save the DataFrame to a Feather file
        df.to_pickle(os.path.join(directory_path, file_name))
        print(f'Successfully Operated on {col_name}\n')
    elif answer == 'Two':
        df = col_df.copy()
        df = df.dropna().sort_index()
        col_name = col_name.split('.')[-1]
        #
        #
        # print(col_name)
        # print('--------------------\n')
        # print(df)
        #
        #
        # # # Specify the directory path and file name for saving the Feather file
        directory_path = fr"\\192.168.0.88\Public\OptionDesk\DATA\ResearchData\FactorModel\day\{code}\factors"
        file_name = f"{col_name}.pkl"
        # fr"/mnt/88disk/Howard/Daily_ETF_{current_date}.csv"
        # Create the directory if it doesn't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Save the DataFrame to a Feather file
        df.to_pickle(os.path.join(directory_path, file_name))
        print(f'Successfully Operated on {col_name}\n')
    else:
        print(f'Answer is not True. There is an error occured on {col_name}!\n')
# def test_get_col_name(data_name,op:Operation,param_list):
#     return op.add_op(data_name, param_list)


if __name__ == '__main__':
    for i in [5,10,15,20,40,60,120,240]:
    # for i in [5]:
        print(f'|---------------------------This is the round operation for i = {i}---------------------------------|\n')
        name = '588000'
        # test_df = pd.read_feather(r"C:\Users\ps\PycharmProjects\TreeEngine\price_data.feather")
        # test_df = pd.read_pickle(r"C:\Users\ps\PycharmProjects\TreeEngine\000300.pkl")
        # test_df = pd.read_pickle(fr"C:\Users\ps\PycharmProjects\TreeEngine\{name}.pkl")
        # test_df = pd.read_feather(r"C:\Users\ps\PycharmProjects\TreeEngine\test.feather")

        test_df = pd.read_pickle(fr"\\192.168.0.88\Public\OptionDesk\DATA\ResearchData\FactorModel\day\{name}\{name}.pkl")

        test_df_two = pd.read_pickle(r"C:\Users\ps\PycharmProjects\TreeEngine\commodity_trimmed.pkl")
        # test_df_two = pd.read_pickle(r"C:\Users\ps\PycharmProjects\TreeEngine\two_ETFs.pkl")
        # test_df = pd.read_pickle(r"\\192.168.0.88\Public\OptionDesk\DATA\ResearchData\FactorModel\day\000852\factors\T_Overnight_5.pkl")
        # print(test_df)
        # print(test_df.columns)
        answer = 'True'
        number = i
        test_df['prev_close_adj'] = test_df['close_adj'].shift(1)
        test_df = test_df.iloc[1:,:]
        test_df['open'] = test_df['open_adj']
        test_df['high'] = test_df['high_adj']
        test_df['low'] = test_df['low_adj']
        test_df['close'] = test_df['close_adj']
        test_df['prev_close'] = test_df['prev_close_adj']
        test_df = test_df[['open', 'high', 'low', 'close', 'prev_close']]




        ##----------------------- P.25 趋势动量因子 ------------------------------##
        # # T_Mom
        test_T_Mom = T_Mom()
        col_name, col_d= test_T_Mom.cal(test_df, 'close',[number])
        act_download(answer, col_name, name)
        #
        #
        # # T_RSM
        test_T_RSM = T_RSM()
        col_name, col_d= test_T_RSM.cal(test_df, ['close','prev_close'],[number])
        act_download(answer, col_name, name)
        #
        # #
        # # # T_RSI
        test_T_RSI = T_RSI()
        col_name, col_d = test_T_RSI.cal(test_df, 'close', [number])
        act_download(answer, col_name, name)
        #
        #
        # # T_MAratio
        test_T_MAratio = T_MAratio()
        col_name, col_d= test_T_MAratio.cal(test_df, 'close',[number])
        act_download(answer, col_name, name)
        #
        #
        # # T_MAcross
        test_T_MAcross = T_MAcross()
        col_name, col_d= test_T_MAcross.cal(test_df, 'close',[number, number+5])
        act_download(answer, col_name, name)
        #
        # T_Overnight
        test_T_Overnight = T_Overnight()
        col_name, col_d= test_T_Overnight.cal(test_df, f'{name}',[number])
        act_download(answer, col_name, name)

        # T_intraday
        test_T_intraday = T_intraday()
        col_name, col_d= test_T_intraday.cal(test_df, f'{name}',[number])
        act_download(answer, col_name, name)

        # T_CumStep
        test_T_CumStep = T_CumStep()
        col_name, col_d= test_T_CumStep.cal(test_df, f'{name}',[number])
        act_download(answer, col_name, name)

        # T_STDS
        test_T_STDS = T_STDS()
        col_name, col_d= test_T_STDS.cal(test_df, f'{name}',[number])
        act_download(answer, col_name, name)

        # # T_Rank
        # # DF here is stacked, if need to unstack, make sure to execute after getting the col_d
        # test_T_Rank = T_Rank()
        # col_name, col_d= test_T_Rank.cal(test_df_two, ['AU', 'CU', 'I', 'SC'],[number])
        # # col_name, col_d = test_T_Rank.cal(test_df_two, ['000300', '000852'],[number])
        # col_df = col_d.unstack()
        # col_df = col_df.loc[:,[name]]
        # print(col_df)
        # act_download('Two', col_name, name)

        #----------------------- P.26 波动率因子 ------------------------------##
        # # V_STD
        test_V_STD = V_STD()
        col_name, col_d= test_V_STD.cal(test_df, 'close',[number])
        act_download(answer, col_name, name)


        # # V_RS
        test_V_RS = V_RS()
        col_name, col_d= test_V_RS.cal(test_df, f'{name}',[number])
        act_download(answer, col_name, name)


        # # V_GK
        test_V_GK = V_GK()
        col_name, col_d= test_V_GK.cal(test_df, f'{name}',[number])
        act_download(answer, col_name, name)

        #
        # V_PK
        test_V_PK = V_PK()
        col_name, col_d= test_V_PK.cal(test_df, f'{name}',[number])
        act_download(answer, col_name, name)

        ##----------------------- P.28 趋势因子 ------------------------------##
        # # Mom_RSI
        test_Mom_RSI = Mom_RSI()
        col_name, col_d= test_Mom_RSI.cal(test_df, f'{name}',[number])
        act_download(answer, col_name, name)
















        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #









    ########################### old features ######################
    # ## TR
    # Can input any column name since it will solely check if you have close, high, and low and manipulate these columns
    # test_tr = TR()
    # col_name, col_d = test_tr.cal(test_df, 'etf_510050', [])

    # DCUpper
    # test_DCUpper = DCUpper()
    # col_name, col_d = test_DCUpper.cal(test_df, 'close', [5])

    # DCLower
    # test_DCLower = DCLower()
    # col_name, col_d = test_DCLower.cal(test_df, 'close', [5])

    # # CoarseDCUpper
    # test_CoarseDCUpper = CoarseDCUpper()
    # col_name, col_d = test_CoarseDCUpper.cal(test_df, 'high', [5,10])

    # CoarseDCLower
    # test_CoarseDCLower = CoarseDCLower()
    # col_name, col_d = test_CoarseDCLower.cal(test_df, 'low', [5,10])

    # ATR
    # test_ATR = ATR()
    # col_name, col_d = test_ATR.cal(test_df, 'close', [5])

    ## EMA
    # test_ema = EMA()
    # col_name, col_d= test_ema.cal(test_df, 'close',[5])


    ## SMA
    # test_sma = SMA()
    # col_name, col_d= test_sma.cal(test_df, 'close',[5])

    # # AR
    # test_ar = AR()
    # col_name, col_d = test_ar.cal(data=test_df, data_name='close', param_list=[5, 10, 3])

    ## Ratio()
    ## TBD

    ## Diff()
    ## TBD


    # # RSI
    # test_RSI = RSI()
    # col_name, col_d = test_RSI.cal(data=test_df, data_name='close', param_list=[5])

    ## PDI(), MDI(), ADX(), ADXR()
    ## Not Doing









    #
    #
    #
    # print(col_name)
    # print('--------------------\n')
    # print(col_d)

#     test_ema.cal('last_close',test_df,[10])
#     col_name,col_d = test_ar.cal(data_name='last_close', data=test_df, param_list = [5,10,3], check_param=True)
#
#     test_ratio = Ratio()
#     print(test_get_col_name('last_cose.EMA_5',test_ratio,[10]))
#     test_ratio.cal(test_get_col_name('last_close',test_ema,[5]),test_df,[10])

