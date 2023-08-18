import pandas as pd


class Position:
    def __init__(self, data_name,side, datetime, price, size, mult):
        self.side = side
        # self._check_price(price)
        self.data_name = data_name
        self.entrytime = datetime
        self.size = size
        self.mult = mult
        self.closetime = None
        self.closeprice = None
        self.entryprice = price
        # self.pnl = None
        self.closed = False
        self.term_endprice = None
        self.init_entryprice = price  # 记录初始成交价格。后续roll 仓时，entry price 会被更新
        self.kwargs = {}

    def __str__(self):
        if self.closed:
            return f'{self.data_name} {self.side} {self.size} closed {self.entrytime.strftime("%Y%m%d %H:%M:%S")}'
        else:
            return f'{self.data_name} {self.side} {self.size} active {self.entrytime.strftime("%Y%m%d %H:%M:%S")} {self.closetime.strftime("%Y%m%d %H:%M:%S")}'

    def _check_price(self, price):
        if self.side == 'short':
            if price >= -1e-20:
                raise ValueError('Convention：short 仓位的价格需要是非正数')
        if self.side == 'long':
            if price <= 1e-20:
                raise ValueError('Convention：long 仓位的价格需要是非负数')

    def calc_pnl(self,price_series:pd.Series,cum_pnl:bool) -> pd.Series:
        """

        :param price_series: 用于计价的价格序列，必须满足时间升序！一般为 close. 不要求 series 的 datetime 连续，可以只输入要计算的时点，而不输入整个时序。函数内会用pos时间做切割
        :param cum_pnl: 返回 累计pnl，还是每个时间段的分段pnl
        :return:
        """
        # 注意！！！ 本函数中的 price series 中，基本上会出现 重复的index。
        # 因此，需要一个反常的操作：use loc and index cautiously; use iloc and int index as an alternative
        if len(price_series) == 0:
            raise ValueError('计价时间序列至少要包含一个价格！不能为空series')

        # 如果平仓，则 切片只取到平仓时点。
        price_s = price_series[self.entrytime:self.closetime] if self.closetime else price_series[self.entrytime:]
        price_s = price_s if self.side == 'long' else -price_s

        # if len(price_s) > 0:
        #     self._check_price(price_s.values[0])

        # price with entry: 第一个bar 从 entry时刻到 close 有pnl ，所以 一定要加。这样也会导致 series 的 开头两个timestamp 一致
        price_with_entry = pd.concat([pd.Series({self.entrytime:self.entryprice}),price_s])
        # 需要特别注意：如果给的price series 包含了 平仓时间的bar，则 平仓时间的 price 需要替换成 close price！（即 计价序列在开平时需要替换成成交价）
        _prc_series_for_pnl = price_with_entry

        if self.closeprice:
            # 如果序列包含closetime 则做价格替换
            if self.closetime in price_with_entry.index:
                # 极端情况下，可能会有两个index 都是close time（最后一个bar 成交然后马上平）。所以这里直接用 iloc[len-1] ，取后一个总是正确的
                # 细节：iloc 是按 整数索引取，不是按 定义的index 取，因此 .iloc[:end] 不包括 end (loc会包括)。因此这里用iloc 自动去掉了最后一个
                price_with_exit = pd.concat([price_with_entry.iloc[:len(price_with_entry)-1],pd.Series({self.closetime:self.closeprice})])
            # 如果序列不包含 close time，则 需要添加
            else:
                price_with_exit = pd.concat([price_with_entry,pd.Series({self.closetime:self.closeprice})])
            _prc_series_for_pnl = price_with_exit

        # 第一个元素为 nan，默认不取
        pnl_series = ((_prc_series_for_pnl.diff() * self.size * self.mult).iloc[1:]).fillna(0.)
        return pnl_series.cumsum() if cum_pnl else pnl_series

    def close_pos(self, datetime, price):
        # self.update_pnl(datetime, price)
        self.closed = True
        self.closetime = datetime
        self.closeprice = price

    def update_mult(self, mult):
        self.mult = mult

