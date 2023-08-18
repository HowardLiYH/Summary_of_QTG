import numpy as np


class CommonCals:
    @staticmethod
    def sma(data,win):
        return data.rolling(window=win,min_periods=win).mean()

    @staticmethod
    def true_range(data_src, data_name):
        if 'close' and 'high' and 'low' not in data_src.columns:
            raise ValueError('Do Not Find Column named close or high or low in the input DataFrame')
        close = data_src['close']
        high = data_src['high']
        low = data_src['low']

        last_close = close.shift(1)
        true_range = np.maximum(np.maximum((high - low), np.abs(last_close - high)), np.abs(last_close - low))
        return true_range

