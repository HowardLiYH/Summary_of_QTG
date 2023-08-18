"""
@ Description   : 
@ Authour       : Jackson Xing
@ Date          : 2023/3/20 19:59
"""
from collections import defaultdict
from copy import copy
import pandas as pd
import warnings
from meta.optmeta import ETFOptContractMeta
from data_manager.manager import DataManager
from optengine.position import Position


class MultiDataBTEngine:

    def __init__(self, factordata, data_manager: DataManager, optmetas=None, delay_p=None, rollpos=None,
                 fillthennext=False):
        """
        :param factordata: main dataframe consisting of factors and a main datetime index, with which the engine is run
        :param delay_p: delay period of an execution after a trade signal is sent
        :param rollpos: if None, the engine initializes with zero pos for all instruments, else, initialize with rollpos
        :param fillthennext: controls the order of trades and strategy.next()
                            defaults to False, i.e. fills orders after next() is called.
                            Works if delay_p != 0. Otherwise, orders will be filled during next()


        NOTE:
            delay_p:
            - can be an integer or a dict of {dataname: delay_p,}
              if an integer is applied, the same delay_p is applied to all instruments

            instrument_dict:
            - dataname:-> str, name you use to call a particular instrument
            - 'data': must have ['close'], indexed with datetime object, NON-OPTIONAL.

                ['buyprice', 'sellprice'] are optional,
                If not included, must pass price to buy() sell() and close_all()

                MUST BE AWARE:
                    if included in dataframe, 'sellprice' MUST have NEGATIVE values!
                    if prices are passed to buy() sell() and close_all(), they MUST must be positive

                ['slipfix', 'slippct'] are optional. if not included, the instrument dict can have
                either 'slipfix' or 'slippct' key-value pair under dataname's value.
                If no slippage info is provided, 0 slippage is assumed

                index should ideally be the same as factordata.index. If not, you must know what you're doing!

            - 'mult':
                multiplier applied to each unit of size of instrument, NON-OPTIONAL.

            - 'type':
                etf_opt, index_opt, comm_opt, index_fut, comm_fut etc., NON-OPTIONAL.
                Used to call margin.get_margin() with different arguments if needed

            - 'margin':
                class that can calculate margin used, OPTIONAL.

            optmetas:
                when 'opt' is in instrument type, optmetas MUST BE PROVIDED

        NOTE:
            when working under delay mode,
            MUST MAKE SURE that your previous order is filled, through filled(dataname) or unfilled(dataname)
            i.e. unfilled_dict[dataname] is False before sending new orders

        NOTE:
            always assume locked positions, so stats for each trade can be available.
            this means that when using 'close_all', you would close out all locked positions, meaning more trading cost

            if you wish to get pnl from of a/several partial-closing trade(s) untill you clear out all of your position
            as if your positions are not locked:
                use buy or sell, not close_all

            when calculating margin, user can calculate using locked(lspos) or net position upon user's will
        """

        _str = r'''                       
                           _ooOoo_
                          o8888888o
                          88" . "88
                          (| -_- |)
                          O\  =  /O
                       ____/`---'\____
                     .'  \\|     |//  `.
                    /  \\|||  :  |||//  \
                   /  _||||| -:- |||||-  \
                   |   | \\\  -  /// |   |
                   | \_|  ''\---/''  |   |
                   \  .-\__  `-`  ___/-. /
                 ___`. .'  /--.--\  `. . __
              ."" '<  `.___\_<|>_/___.'  >'"".
             | | :  `- \`.;`\ _ /`;.`/ - ` : | |
             \  \ `-.   \_ __\ /__ _/   .-` /  /
        ======`-.____`-.___\_____/___.-`____.-'======
                           `=---='
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    佛祖保佑       策略赚钱

        '''
        print(_str)
        self.dm = data_manager
        self.instruments = self.dm.ins_info
        self.optmetas = optmetas
        self.fillthennext = fillthennext
        self.pos = {k: 0 for k in self.dm.ins_info.keys()}  # initialize pos=0 for all insturments

        # update pos for if position is rolled from previous month
        if rollpos:
            self.pos.update({k: sum([i.size if i.side == 'long' else -i.size for i in v]) for k, v in rollpos.items()})

        self.factordata = factordata
        self.pnl = pd.DataFrame(index=factordata.index)
        self.timelist = list(factordata.index)
        self.timepointer = -1
        self.stop = False
        self.datetime = None
        self._str_date = ''
        self.endtime = self.factordata.index[-1]

        # initialize trading status for all instruments
        self.trades = dict()
        if rollpos:
            self.trades.update(rollpos)
        self.rollpos = defaultdict(list)

        # stores delay_p
        if delay_p:
            if type(delay_p) == int:
                self.delay_p_dict = {k: delay_p for k in self.dm.ins_info.keys()}
            elif type(delay_p) == dict:
                self.delay_p_dict = delay_p
            else:
                raise TypeError('wrong delay_p type. Can ONLY input int or dict')
        else:
            self.delay_p_dict = {k: 0 for k in self.dm.ins_info.keys()}

        # self.mult_dict = {}
        # for k, v in self.dm.ins_info.items():
        #     if 'mult' not in v:
        #         raise ValueError(f'must provide mult for {k}')
        #     else:
        #         self.mult_dict[k] = self.dm.get_mult(k)

        # dicts to store size of unfilled buy or sell orders, no size info is needed for close_all, because close ALL
        self.unfilled_buysize_dict = {k: 0 for k in self.dm.ins_info.keys()}
        self.unfilled_sellsize_dict = {k: 0 for k in self.dm.ins_info.keys()}
        self.unfilled_closeall_dict = {k: False for k in self.dm.ins_info.keys()}

        # dicts to store fill prices of unfilled orders if provided buy user, all prices must be positive
        self.buyprice_dict = {k: None for k in self.dm.ins_info.keys()}
        self.sellprice_dict = {k: None for k in self.dm.ins_info.keys()}
        self.closeprice_dict = {k: {} for k in self.dm.ins_info.keys()}

        # dicts to store delay/unfilled status for all orders
        self.delay_dict = {k: 0 for k in self.dm.ins_info.keys()}
        self.unfilled_dict = {}

        self.last_traded_time_dict = {k: None for k in self.dm.ins_info.keys()}
        if rollpos:
            self.last_traded_time_dict.update({k: min([i.entrytime for i in v]) for k, v in rollpos.items()})

    # ------------------------------------------------------------
    # ------------ user interactive funcs ------------------------
    # ------------------------------------------------------------

    def log(self, txt):
        print(f"{self.datetime} => {txt}")

    def get_close(self, dataname):
        """

        :param dataname: -> str, name of the instrument you wish to check
        :return: close price of dataname
        """
        return self.dm.get_data(dataname).loc[self.datetime, 'close']

    def get_quote(self, dataname, colname):
        """

        :param dataname: -> str, name of the instrument you wish to check
        :param colname:-> str, name of the column you wish to get
        :return: the [colname] of dataname
        """
        return self.dm.get_data(dataname).loc[self.datetime, colname]

    def get_factor(self, colname):
        """

        :param colname:-> str, name of the column you wish to get from factor data
        :return: the [colname] of factor
        """
        return self.row[colname]

    def get_all_datanames(self):
        """
        :return: names of all instruments added to the engine
        """
        return list(self.instruments.keys())

    def get_pos(self, dataname):
        """

        :param dataname: -> str, name of the instrument you wish to check
        :return: net pos of dataname
        """
        return self.pos[dataname]

    def get_curr_pos(self):
        """

        :return: {dataname: netpos, for all datanames with non-zero pos}
        """
        return {k: v for k, v in self.get_all_instrumentpos().items() if v}

    def get_all_instrumentpos(self):
        """

        :return:  {dataname: netpos, for all datanames}
        """
        return {k: self.pos[k] for k in self.instruments.keys()}

    def get_lspos(self, dataname):
        """

        :param dataname: -> str, name of the instrument you wish to check
        :return: {'long': net_long_size, 'short': net_short_size}
        """
        long = 0
        short = 0
        for position in self.trades[dataname]:
            if not position.closed:
                if position.side == 'long':
                    long += position.size
                else:
                    short += position.size
        return {'long': long, 'short': short}

    def get_mult(self, dataname):
        """

        :param dataname: -> str, name of the instrument you wish to check
        :return: multiplier of the instrument

        """
        return self.dm.get_mult(data_name=dataname)

    def get_pnl_bytrade(self, trade: Position, use_price_col='close'):
        """
        返回一个 final pnl，不返回中间过程。

        :param trade:
        :param use_price_col:
        :return:
        """
        all_data = self.dm.get_data(trade.data_name)[use_price_col]
        # 只取头尾
        start_end = pd.concat([all_data[[all_data.index[0]]], all_data[[all_data.index[-1]]]])
        return trade.calc_pnl(start_end, cum_pnl=False).values[-1]

    def get_margincalculator(self, dataname):
        """

        :param dataname: -> str, name of the instrument you wish to check
        :return: margin calculator of the instrument
        """
        return self.dm.get_margin_calculator(dataname)

    def all_filled(self):
        """

        :return: -> boolian, wether or not all orders for all instruments have been filled
        """
        # todo need testing
        return not self.unfilled_dict

    def filled(self, dataname):
        """

        :param dataname: -> str, name of the instrument you wish to check
        :return: -> boolian, wether or not all orders have been filled
        """
        return not self.unfilled_dict[dataname]

    def unfilled(self, dataname):
        """

        :param dataname: -> str, name of the instrument you wish to check
        :return: -> boolian, wether or not there's unfilled order
        """
        return self.unfilled_dict[dataname]

    def unfilledbuy(self, dataname):
        """

        :param dataname: -> str, name of the instrument you wish to check
        :return: -> size of unfilled buy order
        """
        return self.unfilled_buysize_dict[dataname]

    def unfilledsell(self, dataname):
        """

        :param dataname: -> str, name of the instrument you wish to check
        :return: -> size of unfilled sell order
        """
        return self.unfilled_sellsize_dict[dataname]

    def get_all_unfilledinstruments(self):
        return list(self.unfilled_dict.keys())

    # ------------------------------------------------------------
    # ---------------------- engine funcs ------------------------
    # ------------------------------------------------------------

    def run(self):
        print('--------BACKTEST BEGIN---------')
        while not self.stop:
            self._next()
        print('\nBACKTEST FINISHED, begin postprocessing ...\n')
        return self._postprocess()

    def _next(self):
        self.timepointer += 1
        if self.timepointer < len(self.timelist):

            self.datetime = self.timelist[self.timepointer]
            self._str_date = self.datetime.strftime('%Y%m%d')
            self.prevbar_datetime = None if self.timepointer == 0 else self.timelist[self.timepointer - 1]
            if self.timepointer < len(self.timelist) - 1:
                self.nextbar_datetime = self.timelist[self.timepointer + 1]
            else:
                self.nextbar_datetime = None

            self.row = self.factordata.loc[self.timelist[self.timepointer]]

            self.update_mult()

            if not self.fillthennext:
                self.next(self.row)

            # carry out trades
            for dataname in self.get_all_unfilledinstruments():
                if self.unfilled_buysize_dict[dataname]:
                    self.delay_dict[dataname] -= 1
                    if not self.delay_dict[dataname]:
                        self._buy(dataname, self.unfilled_buysize_dict[dataname], self.buyprice_dict[dataname])

                if self.unfilled_sellsize_dict[dataname]:
                    self.delay_dict[dataname] -= 1
                    if not self.delay_dict[dataname]:
                        self._sell(dataname, self.unfilled_sellsize_dict[dataname], self.sellprice_dict[dataname])

                if self.unfilled_closeall_dict[dataname]:
                    self.delay_dict[dataname] -= 1
                    if not self.delay_dict[dataname]:
                        self._close_all(dataname, self.closeprice_dict[dataname])

            if self.fillthennext:
                self.next(self.row)

        else:
            self.stop = True

    def next(self, row):

        """
        define your strategy here
        NOTE: MUST and ONLY define update_mult() define your update_mult() if you're trading etf_opt
              update mult for your etf_opt positions at the beginning of each day
              also update self.mult_dict
        :return:
        """
        pass

    def on_trade(self, dataname, side, size, price, openclose):
        """
        define your on_trade func here
        :param side:
        :param size:
        :param price:
        :return:
        """
        warnings.warn('critical function not defined by user.')

    def update_mult(self):

        if self.prevbar_datetime is None or self.datetime.date() != self.prevbar_datetime.date():
            # 更新 ins dict 中的所有mult ，以及在trade 中有仓位的mult
            for data_item in self.instruments:
                if self.instruments[data_item]['type'] == 'etf_opt':
                    opt_meta: ETFOptContractMeta = self.optmetas.get_contract_meta(data_item)
                    # 如果合约还没有挂牌，直接跳过
                    if opt_meta.get_start_date() > self._str_date:
                        continue
                    tdy_mult = opt_meta.get_contract_size(self._str_date)
                    if self.instruments[data_item]['mult'] != tdy_mult:
                        self.instruments[data_item]['mult'] = tdy_mult
                        if data_item in self.trades:
                            for p in self.trades[data_item]:
                                if not p.closed:
                                    p.update_mult(tdy_mult)

    def set_pos(self, targetpos, targetprice=None, delay_p=None):
        """

        :param targetpos: {dataname: targetpos, }
        :param targetprice: {dataname: targetprice, }
        :return:
        """
        currentpos = self.get_curr_pos()

        clear_pos = currentpos.keys() - targetpos
        new_pos = targetpos.keys() - currentpos
        change_pos = [i for i in targetpos.keys() & currentpos if targetpos[i] != currentpos[i]]

        # if targetprice:
        for dataname in clear_pos:
            self.close_all(dataname, targetprice[dataname] if targetprice is not None else None, delay_p)
        for dataname in new_pos:
            if targetpos[dataname] > 0:
                self.buy(dataname, targetpos[dataname], targetprice[dataname] if targetprice is not None else None,
                         delay_p)
            if targetpos[dataname] < 0:
                self.sell(dataname, abs(targetpos[dataname]), targetprice[dataname] if targetprice is not None else None,
                          delay_p)
        for dataname in change_pos:
            pos_change = targetpos[dataname] - currentpos[dataname]
            if pos_change > 0:
                self.buy(dataname, pos_change, targetprice[dataname] if targetprice is not None else None, delay_p)
            if pos_change < 0:
                self.sell(dataname, abs(pos_change), targetprice[dataname] if targetprice is not None else None,
                          delay_p)

    def buy(self, dataname, size, price=None, delay_p=None):
        delay_p = self.delay_p_dict[dataname] if delay_p is None else delay_p
        if not delay_p:
            self._buy(dataname, size, price)
        else:
            self.unfilled_buysize_dict[dataname] = size
            self.delay_dict[dataname] = delay_p
            self.buyprice_dict[dataname] = price
            self.unfilled_dict[dataname] = True

    def sell(self, dataname, size, price=None, delay_p=None):
        delay_p = self.delay_p_dict[dataname] if delay_p is None else delay_p
        if not delay_p:
            self._sell(dataname, size, price)
        else:
            self.unfilled_sellsize_dict[dataname] = size
            self.delay_dict[dataname] = delay_p
            self.sellprice_dict[dataname] = price
            self.unfilled_dict[dataname] = True

    def close_all(self, dataname, closeprice=None, delay_p=None):
        delay_p = self.delay_p_dict[dataname] if delay_p is None else delay_p
        if not delay_p:
            self._close_all(dataname, closeprice)
        else:
            self.unfilled_closeall_dict[dataname] = True
            self.delay_dict[dataname] = delay_p
            self.closeprice_dict[dataname] = closeprice
            self.unfilled_dict[dataname] = True

    def _buy(self, dataname, size, price=None):
        if price is None:
            row = self.dm.get_data(dataname).loc[self.timelist[self.timepointer]]
            price = row['buyprice']

            if 'slipfix' in self.dm.get_data(dataname).columns:
                price += row['slipfix']
            elif 'slippct' in self.dm.get_data(dataname).columns:
                price *= (1 + row['slippct'])
            elif 'slipfix' in self.instruments[dataname].keys():
                price += self.instruments[dataname]['slipfix']
            elif 'slippct' in self.instruments[dataname].keys():
                price *= (1 + self.instruments[dataname]['slippct'])

        pos = Position(dataname, 'long', self.datetime, price, size, self.get_mult(dataname))
        if dataname not in self.trades:
            self.trades[dataname] = []
        self.trades[dataname].append(pos)
        if 'opt' in self.instruments[dataname]['type']:
            tradingcode = self.optmetas.get_contract_meta(dataname).trading_code
            strike = self.optmetas.get_contract_meta(dataname).get_strike(self._str_date)
            print(f'{self.datetime} => buying {tradingcode} at K = {strike} size = {size} at {price}')
        else:
            print(f'{self.datetime} => buying {dataname} size = {size} at {price}')

        self.last_traded_time_dict[dataname] = self.datetime
        self.pos[dataname] += size
        self.buyprice_dict[dataname] = None
        self.unfilled_buysize_dict[dataname] = 0
        if dataname in self.unfilled_dict:
            del self.unfilled_dict[dataname]

        self.on_trade(dataname, 'buy', size, price, 'open')

    def _sell(self, dataname, size, price=None):
        if price is None:
            row = self.dm.get_data(dataname).loc[self.timelist[self.timepointer]]
            price = row['sellprice']

            if 'slipfix' in self.dm.get_data(dataname).columns:
                price += row['slipfix']
            elif 'slippct' in self.dm.get_data(dataname).columns:
                price *= (1 - row['slippct'])
            elif 'slipfix' in self.instruments[dataname].keys():
                price += self.instruments[dataname]['slipfix']
            elif 'slippct' in self.instruments[dataname].keys():
                price *= (1 - self.instruments[dataname]['slippct'])
        else:
            price = -price

        pos = Position(dataname, 'short', self.datetime, price, size, self.get_mult(dataname))
        if dataname not in self.trades:
            self.trades[dataname] = []
        self.trades[dataname].append(pos)

        if 'opt' in self.instruments[dataname]['type']:
            tradingcode = self.optmetas.get_contract_meta(dataname).trading_code
            strike = self.optmetas.get_contract_meta(dataname).get_strike(self._str_date)
            print(f'{self.datetime} => selling {tradingcode} at K = {strike} size = {size} at {-price}')
        else:
            print(f'{self.datetime} => selling {dataname} size = {size} at {-price}')

        self.last_traded_time_dict[dataname] = self.datetime
        self.pos[dataname] -= size
        self.sellprice_dict[dataname] = None
        self.unfilled_sellsize_dict[dataname] = 0
        if dataname in self.unfilled_dict:
            del self.unfilled_dict[dataname]

        self.on_trade(dataname, 'sell', size, -price, 'open')

    def _close_all(self, dataname, closeprice=None):

        if closeprice is None:
            row = self.dm.get_data(dataname).loc[self.timelist[self.timepointer]]

            buy_close_price = row['buyprice']
            if 'slipfix' in self.dm.get_data(dataname).columns:
                buy_close_price += row['slipfix']
            elif 'slippct' in self.dm.get_data(dataname).columns:
                buy_close_price *= (1 + row['slippct'])
            elif 'slipfix' in self.instruments[dataname].keys():
                buy_close_price += self.instruments[dataname]['slipfix']
            elif 'slippct' in self.instruments[dataname].keys():
                buy_close_price *= (1 + self.instruments[dataname]['slippct'])
            buy_close_price *= -1

            sell_close_price = row['sellprice']
            if 'slipfix' in self.dm.get_data(dataname).columns:
                sell_close_price += row['slipfix']
            elif 'slippct' in self.dm.get_data(dataname).columns:
                sell_close_price *= (1 - row['slippct'])
            elif 'slipfix' in self.instruments[dataname].keys():
                sell_close_price += self.instruments[dataname]['slipfix']
            elif 'slippct' in self.instruments[dataname].keys():
                sell_close_price *= (1 - self.instruments[dataname]['slippct'])
            sell_close_price *= -1

        else:
            buy_close_price = -closeprice['buyprice']
            sell_close_price = closeprice['sellprice']

        closedshort = 0
        closedlong = 0
        for position in self.trades[dataname]:
            if not position.closed:
                if position.side == 'long':
                    position.close_pos(self.datetime, sell_close_price)
                    closedlong += position.size
                else:
                    position.close_pos(self.datetime, buy_close_price)
                    closedshort += position.size

        if closedlong:
            if 'opt' in self.instruments[dataname]['type']:
                tradingcode = self.optmetas.get_contract_meta(dataname).trading_code
                strike = self.optmetas.get_contract_meta(dataname).get_strike(self._str_date)
                print(f'{self.datetime} => selling close {tradingcode} at K = {strike} size = {closedlong} at {sell_close_price}')
            else:
                print(f'{self.datetime} => selling close {dataname}, size = {closedlong} at {sell_close_price}')
            self.on_trade(dataname, 'sell', closedlong, sell_close_price, 'close')

        if closedshort:
            if 'opt' in self.instruments[dataname]['type']:
                tradingcode = self.optmetas.get_contract_meta(dataname).trading_code
                strike = self.optmetas.get_contract_meta(dataname).get_strike(self._str_date)
                print(f'{self.datetime} => buying close {tradingcode} at K = {strike} size = {closedshort} at {-buy_close_price}')
            else:
                print(f'{self.datetime} => buying close {dataname}, size = {closedshort} at {-buy_close_price}')
            self.on_trade(dataname, 'buy', closedshort, -buy_close_price, 'close')

        self.last_traded_time_dict[dataname] = self.datetime
        self.pos[dataname] = 0
        self.closeprice_dict[dataname] = {}
        self.unfilled_closeall_dict[dataname] = False
        if dataname in self.unfilled_dict:
            del self.unfilled_dict[dataname]

    def _postprocess(self):
        self.process_rollpos()
        self.calc_pnl()
        return self.postprocess()

    def postprocess(self):
        """
        define your post process here
        :return:
        """
        warnings.warn('critical function not defined by user.')

    def process_rollpos(self):
        """
        unclosed trades needs to copy from trades，realize pnl and roll to next term.

        :return:
        """
        for dataname, trades in self.trades.items():
            if trades:
                for i in range(len(trades)):
                    # roll 仓时需要新创建 roll pos。目的是将本term内的pnl转化为realized pnl，然后在下一期重新计算。
                    roll_pos = copy(trades[i])
                    if not roll_pos.closed:
                        if roll_pos.side == 'long':
                            roll_pos.entryprice = self.get_close(dataname)
                            self.trades[dataname][i].term_endprice = self.get_close(dataname)
                        else:
                            roll_pos.entryprice = -self.get_close(dataname)
                            self.trades[dataname][i].term_endprice = -self.get_close(dataname)
                        roll_pos.pnl = {}
                        self.rollpos[dataname].append(roll_pos)

    # def calc_pnl(self):
    #     for dataname, trades in self.trades.items():
    #         trade_pnl = [pd.Series(trades[i].pnl,name=f'trade_{dataname}_{i}') for i in range(len(trades))]
    #         self.pnl = pd.concat([self.pnl] + trade_pnl, axis=1)
    #     self.pnl = self.pnl.ffill().fillna(0)
    #     self.pnl['pnl'] = self.pnl.apply(lambda x: x.sum(), axis=1)

    def calc_pnl_bytrade(self, trade: Position, use_price_col='close', set_series_name=None) -> pd.Series:
        """
        计算每个时点的pnl，返回一个 时序 pnl

        :param trade:
        :param use_price_col:
        :param set_series_name:
        :return:
        """
        if set_series_name:
            return trade.calc_pnl(self.dm.get_data(trade.data_name)[use_price_col], cum_pnl=True).rename(
                set_series_name)
        return trade.calc_pnl(self.dm.get_data(trade.data_name)[use_price_col], cum_pnl=True)

    def calc_pnl(self, use_price_col='close', include_closed: bool = True):
        """
        计算 pnl

        :param use_price_col:
        :param include_closed: 是否包括 closed 仓位。一般为 False
        :return:
        """
        for dataname, trades in self.trades.items():
            trade_pnl = [self.calc_pnl_bytrade(trade=trades[i], set_series_name=f'trade_{dataname}_{i}',
                                               use_price_col=use_price_col)
                         for i in range(len(trades)) if (include_closed or (not trades[i].closed))]
            self.pnl = pd.concat([self.pnl] + trade_pnl, axis=1, )
        self.pnl = self.pnl.ffill().fillna(0)
        self.pnl['pnl'] = self.pnl.apply(lambda x: x.sum(), axis=1)
