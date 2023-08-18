"""
定义 生成 buy sell 的 函数。
注意，为了方便传参，所有的函数都需要用 kwargs 作为输入
"""
import pandas as pd


class Processor:
    def process(self, data) -> pd.DataFrame:
        raise NotImplementedError


class BuySellPriceDefiner(Processor):
    def process(self, data) -> pd.DataFrame:
        return self.use_price(data, price_col='close')

    @staticmethod
    def use_price(data_src: pd.DataFrame, price_col: str):
        data_src['buyprice'] = data_src[price_col]
        data_src['sellprice'] = -data_src[price_col]
        return data_src


class TradeOpenProcessor(Processor):
    def process(self, data) -> pd.DataFrame:
        # can do something here to process quotes for individual instruments
        return self.use_price(data, price_col='open')

    @staticmethod
    def use_price(data_src: pd.DataFrame, price_col: str):
        data_src['buyprice'] = data_src[price_col]
        data_src['sellprice'] = -data_src[price_col]
        return data_src


class TradeCloseProcessor(Processor):
    def process(self, data) -> pd.DataFrame:
        # can do something here to process quotes for individual instruments
        return self.use_price(data, price_col='close')

    @staticmethod
    def use_price(data_src: pd.DataFrame, price_col: str):
        data_src['buyprice'] = data_src[price_col]
        data_src['sellprice'] = -data_src[price_col]
        return data_src
