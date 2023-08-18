import numpy as np

from optengine.SimpleMultiDataBackTest_Engine import MultiDataBTEngine
import pandas as pd
from data_manager import DataManager
from optcal import EuOptCalculator, AmOptCalculator


class Strat1(MultiDataBTEngine):
    def __init__(self, factordata, data_manager: DataManager, YYMM, delay_p, optmeta, rollterm_day,
                 delta_th, Klvel, dtc, rollpos=None, fillthennext=False):

        """
        迭代策略方案一：当n天历史波动率的SMA_fast < SMA_slow，当天收盘前卖出m档的ATM Strangle。
        若信号连续触发，则判断
        1）组合delta是否超过delta_th，如果是则roll到新的m档ATM strangle
        2）是否roll仓日期，如果是则roll到次月的ATM strangle。

        :param factordata:
        :param data_manager:
        :param YYMM:
        :param delay_p:
        :param optmeta:
        :param rollterm_day:
        :param dtc:
        :param rollpos:
        :param fillthennext:
        """
        super().__init__(factordata, data_manager, optmetas=optmeta, delay_p=delay_p, rollpos=rollpos,
                         fillthennext=fillthennext)

        # trading params
        self.opensignal = False
        self.C_contract = None
        self.P_contract = None
        self.delta_th = delta_th
        self.Klvel = Klvel

        # time params
        self.dtc = dtc
        self.YYMM = YYMM
        self.fullyearcode = f'20{YYMM}'
        self.exp_date = self.factordata.iloc[-1]['str_date']
        self.rollterm_day = rollterm_day

        if rollpos:
            begin_strdate = self.factordata.index[0].strftime('%Y%m%d')
            for dataname in rollpos.keys():
                cp = self.optmetas.get_contract_meta(dataname).get_cp_type()
                if cp == 'C':
                    self.C_contract = dataname
                    C_tradingcode = self.optmetas.get_contract_meta(dataname).trading_code
                    K = self.optmetas.get_contract_meta(dataname).get_strike(begin_strdate)
                    print(f'rolled from prevYYMM {C_tradingcode} at K = {K}')
                else:
                    self.P_contract = dataname
                    K = self.optmetas.get_contract_meta(dataname).get_strike(begin_strdate)
                    P_tradingcode = self.optmetas.get_contract_meta(dataname).trading_code
                    print(f'rolled from prevYYMM {P_tradingcode} at K = {K}')

    def next(self, row):

        print(f'{self.datetime} => ttm = {self.get_ttm()}')

        self.opensignal = self.get_factor('shortvol')

        # self.opensignal = True

        if self.opensignal:
            if not self.C_contract and not self.P_contract:
                print(
                    f"{self.datetime} => sell open ATM strangle at Klvel={self.Klvel}, ulapx = {self.get_factor('close')}")
                C, P = self.get_strangle()
                self.set_pos({C: -1, P: -1})
                # self.set_pos({C: 1, P: 1})

            else:
                # 有持仓，如果delta暴露过大，roll仓
                pos_delta = self.get_portfolio_delta()
                C_contract_ttm = self.get_ctrct_ttm(self.C_contract)
                P_contract_ttm = self.get_ctrct_ttm(self.C_contract)
                if abs(pos_delta) > self.delta_th or C_contract_ttm <= self.rollterm_day or P_contract_ttm <= self.rollterm_day:
                    print(
                        f"{self.datetime} => C_ttm={C_contract_ttm}, P_ttm={P_contract_ttm}, pos_delta={pos_delta}, roll pos")
                    C, P = self.get_strangle()
                    self.set_pos({C: -1, P: -1})
                    # self.set_pos({C: 1, P: 1})
        else:
            # 若有持仓，平仓
            if self.C_contract and self.P_contract:
                print(f"{self.datetime} => close out strangle")
                self.set_pos({})

    def get_strangle(self):
        fullyearcode = f"20{self.row['term']}"
        date = self.row['str_date']
        # ulaprice = self.row[f'open']
        ulaprice = self.row[f'close']

        if self.get_ttm() <= self.rollterm_day:
            fullyearcode = self.optmetas.get_dueyymm_series(date)[1]

        C_K_list = self.optmetas.get_otm_strikes(fullyearcode, date, ulaprice, 100, alter_times=0)['C']
        P_K_list = self.optmetas.get_otm_strikes(fullyearcode, date, ulaprice, 100, alter_times=0)['P']

        if C_K_list:
            C_K = C_K_list[self.Klvel] if self.Klvel < len(C_K_list) else C_K_list[-1]
        else:
            C_K = P_K_list[-1]

        if P_K_list:
            P_K = P_K_list[-self.Klvel] if self.Klvel < len(P_K_list) else P_K_list[0]
        else:
            P_K = C_K_list[0]

        C = self.optmetas.get_contract_by_strike(date, fullyearcode, 'C', C_K, 0).contract_code
        P = self.optmetas.get_contract_by_strike(date, fullyearcode, 'P', P_K, 0).contract_code

        return C, P

    def get_ttm(self):
        return self.dtc.get_days_diff(self.get_factor('str_date'), self.exp_date)

    def get_ctrct_ttm(self, contract):
        return self.dtc.get_days_diff(self.get_factor('str_date'),
                                      self.optmetas.get_contract_meta(contract).get_exp_date())

    def get_portfolio_delta(self):

        if not self.C_contract and not self.P_contract:
            return 0

        ud_prc = self.get_factor('close')
        r = 0
        div = 0
        ttm = (self.get_ctrct_ttm(self.C_contract) - 1 + 0.0000001) / 240
        CK = float(self.optmetas.get_contract_meta(self.C_contract).get_strike(self._str_date))
        PK = float(self.optmetas.get_contract_meta(self.P_contract).get_strike(self._str_date))

        C_vol = EuOptCalculator.get_imp_vol('C', ud_prc, CK, ttm, r, div, self.get_quote(self.C_contract, 'close'))
        P_vol = EuOptCalculator.get_imp_vol('P', ud_prc, PK, ttm, r, div, self.get_quote(self.P_contract, 'close'))
        C_delta = EuOptCalculator.get_greeks('C', ud_prc, CK, ttm, r, div, C_vol)['delta']
        P_delta = EuOptCalculator.get_greeks('P', ud_prc, PK, ttm, r, div, P_vol)['delta']

        return C_delta * self.pos[self.C_contract] + P_delta * self.pos[self.P_contract]

    def on_trade(self, dataname, side, size, price, openclose):
        if openclose == 'open':
            if self.optmetas.get_contract_meta(dataname).get_cp_type() == 'C':
                self.C_contract = dataname

            if self.optmetas.get_contract_meta(dataname).get_cp_type() == 'P':
                self.P_contract = dataname

        else:
            if self.optmetas.get_contract_meta(dataname).get_cp_type() == 'C':
                self.C_contract = None

            if self.optmetas.get_contract_meta(dataname).get_cp_type() == 'P':
                self.P_contract = None

    def postprocess(self):
        return self.pnl, self.rollpos, self.trades


class Strat2(Strat1):
    def __init__(self, factordata, data_manager: DataManager, YYMM, delay_p, optmeta, rollterm_day,
                 delta_th, Klvel, dtc, rollpos=None, fillthennext=False, kwargs=None):
        """
        迭代策略方案二：当n天历史波动率的SMA_fast < SMA_slow，当天收盘前卖出m档的ATM Strangle。
        若信号连续触发，则判断
        1）组合delta是否超过delta_th，如果是则平掉delta大的合约，roll回到n档虚值合约
        2）是否roll仓日期，如果是则roll到次月的ATM strangle。

        :param factordata:
        :param data_manager:
        :param YYMM:
        :param delay_p:
        :param optmeta:
        :param rollterm_day:
        :param delta_th:
        :param Klvel:
        :param dtc:
        :param rollpos:
        :param fillthennext:
        """
        super().__init__(factordata, data_manager, YYMM, delay_p, optmeta, rollterm_day,
                         delta_th, Klvel, dtc, rollpos, fillthennext)

    def next(self, row):

        print(f'{self.datetime} => ttm = {self.get_ttm()}')

        self.opensignal = self.get_factor('shortvol')

        # self.opensignal = True

        if self.opensignal:
            if not self.C_contract and not self.P_contract:
                print(
                    f"{self.datetime} => sell open ATM strangle at Klvel={self.Klvel}, ulapx = {self.get_factor('close')}")
                C, P = self.get_strangle()
                self.set_pos({C: -1, P: -1})
                # self.set_pos({C: 1, P: 1})

            else:
                # 有持仓，如果delta暴露过大，roll仓
                pos_delta = self.get_portfolio_delta()
                C_contract_ttm = self.get_ctrct_ttm(self.C_contract)
                P_contract_ttm = self.get_ctrct_ttm(self.C_contract)
                if abs(pos_delta) > self.delta_th or C_contract_ttm <= self.rollterm_day or P_contract_ttm <= self.rollterm_day:
                    print(
                        f"{self.datetime} => C_ttm={C_contract_ttm}, P_ttm={P_contract_ttm}, pos_delta={pos_delta}, roll pos")
                    C, P = self.get_strangle()
                    self.set_pos({C: -1, P: -1})
                    # self.set_pos({C: 1, P: 1})
        else:
            # 若有持仓，平仓
            if self.C_contract and self.P_contract:
                print(f"{self.datetime} => close out strangle")
                self.set_pos({})

    def get_strangle(self):
        fullyearcode = f"20{self.row['term']}"
        date = self.row['str_date']
        # ulaprice = self.row[f'open']
        ulaprice = self.row[f'close']

        if self.get_ttm() <= self.rollterm_day:
            fullyearcode = self.optmetas.get_dueyymm_series(date)[1]

        C_K_list = self.optmetas.get_otm_strikes(fullyearcode, date, ulaprice, 100, alter_times=0)['C']
        P_K_list = self.optmetas.get_otm_strikes(fullyearcode, date, ulaprice, 100, alter_times=0)['P']

        if C_K_list:
            C_K = C_K_list[self.Klvel] if self.Klvel < len(C_K_list) else C_K_list[-1]
        else:
            C_K = P_K_list[-1]

        if P_K_list:
            P_K = P_K_list[-self.Klvel] if self.Klvel < len(P_K_list) else P_K_list[0]
        else:
            P_K = C_K_list[0]

        C = self.optmetas.get_contract_by_strike(date, fullyearcode, 'C', C_K, 0).contract_code
        P = self.optmetas.get_contract_by_strike(date, fullyearcode, 'P', P_K, 0).contract_code

        if self.C_contract and self.P_contract:
            C_delta = self.get_ctrct_delta(self.C_contract)
            P_delta = self.get_ctrct_delta(self.P_contract)

            if abs(C_delta) > abs(P_delta) and self.get_ctrct_ttm(self.P_contract) > self.rollterm_day:
                P = self.P_contract
            elif abs(C_delta) < abs(P_delta) and self.get_ctrct_ttm(self.P_contract) > self.rollterm_day:
                C = self.C_contract

        return C, P

    def get_ctrct_delta(self, contract):
        this_meta = self.optmetas.get_contract_meta(contract)
        if not self.C_contract and not self.P_contract:
            return 0

        ud_prc = self.get_factor('close')
        # fullyearcode = self.get_factor('opt_1m')
        # ud_prc = self.get_quote(self.optmetas.ula_fut_meta.get_contract_meta_by_fyc(fullyearcode).contract_code,
        #                         'close')

        r = 0
        div = 0
        ttm = (self.get_ctrct_ttm(self.C_contract) - 1 + 0.0000001) / 240
        # CK = float(self.optmetas.get_contract_meta(self.C_contract).get_strike(self._str_date))
        # PK = float(self.optmetas.get_contract_meta(self.P_contract).get_strike(self._str_date))
        #
        # vol = EuOptCalculator.get_imp_vol('C', ud_prc, CK, ttm, r, div, self.get_quote(contract, 'close'))
        # delta = AmOptCalculator.get_greeks('C', ud_prc, PK, ttm, r, div, vol)[0]

        k = float(this_meta.get_strike(self._str_date))
        vol = EuOptCalculator.get_imp_vol(this_meta.cp_type, ud_prc, k, ttm, r, div, self.get_quote(contract, 'close'))
        delta = EuOptCalculator.get_greeks(this_meta.cp_type, ud_prc, k, ttm, r, div, vol)['delta']
        # if this_meta.opt_type != 'comm_opt':
        #     vol = EuOptCalculator.get_imp_vol(this_meta.cp_type, ud_prc, k, ttm, r, div, self.get_quote(contract, 'close'))
        #     delta = EuOptCalculator.get_greeks(this_meta.cp_type, ud_prc, k, ttm, r, div, vol)[0]
        # else:
        #     vol = AmOptCalculator.get_imp_vol(this_meta.cp_type, ud_prc, k, ttm, r, div, self.get_quote(contract, 'close'))
        #     delta = AmOptCalculator.get_greeks(this_meta.cp_type, ud_prc, k, ttm, r, div, vol)[0]
        return delta


class Strat3(Strat2):
    def __init__(self, factordata, data_manager: DataManager, YYMM, delay_p, optmeta, rollterm_day,
                 delta_th, Klvel, rollinK_day, dtc, rollpos=None, fillthennext=False):

        """
        基于Strat2, 加入新逻辑：在到期日前rollinK_day当天，把Klvl缩进一档，模拟尾部行权价档位的delta decay的行为
        :param factordata:
        :param data_manager:
        :param YYMM:
        :param delay_p:
        :param optmeta:
        :param rollterm_day:
        :param delta_th:
        :param Klvel:
        :param rollinK_day:
        :param dtc:
        :param rollpos:
        :param fillthennext:
        """
        super().__init__(factordata, data_manager, YYMM, delay_p, optmeta, rollterm_day,
                         delta_th, Klvel, dtc, rollpos, fillthennext)

        self.rollinK_day = rollinK_day

    def next(self, row):

        print(f'{self.datetime} => ttm = {self.get_ttm()}')

        if self.get_ttm() == self.rollinK_day:
            self.Klvel = (self.Klvel - 1) if self.Klvel != 0 else self.Klvel
            print(f'{self.datetime} => roll in Klvl to {self.Klvel} at ttm = {self.get_ttm()}')

        self.opensignal = self.get_factor('shortvol')

        # self.opensignal = True

        if self.opensignal:
            if not self.C_contract and not self.P_contract:
                print(
                    f"{self.datetime} => sell open ATM strangle at Klvel={self.Klvel}, ulapx = {self.get_factor('close')}")
                C, P = self.get_strangle()
                self.set_pos({C: -1, P: -1})
                # self.set_pos({C: 1, P: 1})

            else:
                # 有持仓，如果delta暴露过大，roll仓
                pos_delta = self.get_portfolio_delta()
                contract_ttm = self.get_ctrct_ttm(self.C_contract)
                if abs(pos_delta) > self.delta_th or contract_ttm <= self.rollterm_day:
                    print(
                        f"{self.datetime} => contract_ttm={contract_ttm},pos_delta={pos_delta}, roll pos at Klvl = {self.Klvel}")
                    C, P = self.get_strangle()
                    self.set_pos({C: -1, P: -1})
                    # self.set_pos({C: 1, P: 1})
        else:
            # 若有持仓，平仓
            if self.C_contract and self.P_contract:
                print(f"{self.datetime} => close out strangle")
                self.set_pos({})


class Strat4(Strat2):
    def __init__(self, factordata, data_manager: DataManager, YYMM, delay_p, optmeta, rollterm_day,
                 delta_th, Klvel, dtc, rollpos=None, fillthennext=False):

        """
        基于Strat2, 加入新逻辑：shortvol=True，则开仓，直到tp=True，则平仓。
        :param factordata:
        :param data_manager:
        :param YYMM:
        :param delay_p:
        :param optmeta:
        :param rollterm_day:
        :param delta_th:
        :param Klvel:
        :param dtc:
        :param rollpos:
        :param fillthennext:
        """
        super().__init__(factordata, data_manager, YYMM, delay_p, optmeta, rollterm_day,
                         delta_th, Klvel, dtc, rollpos, fillthennext)

        self.closesignal = False

    def next(self, row):
        print(f'{self.datetime} => ttm = {self.get_ttm()}')

        self.opensignal = self.get_factor('shortvol')
        self.closesignal = self.get_factor('closevshort')

        # self.opensignal = True

        # 没有持仓，判断是否开仓
        if not self.C_contract and not self.P_contract:
            if self.opensignal:
                print(
                    f"{self.datetime} => sell open ATM strangle at Klvel={self.Klvel}, ulapx = {self.get_factor('close')}")
                C, P = self.get_strangle()
                self.set_pos({C: -1, P: -1})
                # self.set_pos({C: 1, P: 1})

        # 有持仓，判断是否平仓
        else:
            if not self.closesignal:
                # 有持仓，如果delta暴露过大，或者到roll仓日期，roll仓
                pos_delta = self.get_portfolio_delta()
                contract_ttm = self.get_ctrct_ttm(self.C_contract)
                if abs(pos_delta) > self.delta_th or contract_ttm <= self.rollterm_day:
                    print(f"{self.datetime} => contract_ttm={contract_ttm},pos_delta={pos_delta}, roll pos")
                    C, P = self.get_strangle()
                    self.set_pos({C: -1, P: -1})
                    # self.set_pos({C: 1, P: 1})
            else:
                print(f"{self.datetime} => close out strangle")
                self.set_pos({})


class Strat5(MultiDataBTEngine):
    def __init__(self, factordata, data_manager: DataManager, YYMM, delay_p, optmeta, rollterm_day,
                 delta_th, Klvel, dtc, rollpos=None, fillthennext=False, kwargs=None):  # todo input kwargs to debug.

        """
        同Strat2，used on 商品

        与Strat2的区别是：etf or idx 期权的ula价格是factor中的close，而商品需要去读取期权仓位对应期货term的合约的close

        :param factordata:
        :param data_manager:
        :param YYMM:
        :param delay_p:
        :param optmeta:
        :param rollterm_day:
        :param dtc:
        :param rollpos:
        :param fillthennext:
        """
        super().__init__(factordata, data_manager, optmetas=optmeta, delay_p=delay_p, rollpos=rollpos,
                         fillthennext=fillthennext)

        # trading params
        self.opensignal = False
        self.C_contract = None
        self.P_contract = None
        self.delta_th = delta_th
        self.Klvel = Klvel
        self.kwargs = kwargs  # todo: delete kwargs. Input this to debug
        # time params
        self.dtc = dtc
        self.YYMM = YYMM
        self.fullyearcode = f'20{YYMM}'
        self.exp_date = self.factordata.iloc[-1]['str_date']
        self.rollterm_day = rollterm_day
        self._holding_info = []
        if rollpos:
            begin_strdate = self.factordata.index[0].strftime('%Y%m%d')
            for dataname in rollpos.keys():
                cp = self.optmetas.get_contract_meta(dataname).get_cp_type()
                if cp == 'C':
                    self.C_contract = dataname
                    C_tradingcode = self.optmetas.get_contract_meta(dataname).trading_code
                    K = self.optmetas.get_contract_meta(dataname).get_strike(begin_strdate)
                    print(f'rolled from prevYYMM {C_tradingcode} at K = {K}')
                else:
                    self.P_contract = dataname
                    K = self.optmetas.get_contract_meta(dataname).get_strike(begin_strdate)
                    P_tradingcode = self.optmetas.get_contract_meta(dataname).trading_code
                    print(f'rolled from prevYYMM {P_tradingcode} at K = {K}')

    def _record_holding_info(self):
        """
        记录当日的期权持仓与当日ula price 的 变化。可以用作debug. 本函数是 next 进来之后的第一步，可以看作是 timepointer移动到今日之后且
        在今日的其他任何交易之前的时点；此时 time pointer 指向 today，但所有的仓位，还在昨日

        :return:
        """
        last_time = self.timepointer - 1
        # last_bar_opt_pos = [[d, self.pos[d]] for d in self.pos if self.pos[d] != 0 and d in self.optmetas.metas]
        last_bar_opt_pos = [[pos_code, self.pos[pos_code]] for pos_code in [
            self.C_contract, self.P_contract] if pos_code is not None]
        if len(last_bar_opt_pos) == 0:
            return
        if last_time < 0:
            pos_str = []
            for opt_pos in last_bar_opt_pos:
                data_name = opt_pos[0]
                opt_trading_code = self.optmetas.get_contract_meta(data_name).trading_code
                pos_str.append(opt_trading_code)
                pos_str.append(str(opt_pos[1]))
            pos_str = ' '.join(pos_str)
            info = f'starts with {pos_str}'
        else:
            last_date = self.factordata.index[last_time]
            last_ula_cls = self.factordata.loc[last_date, 'close_adj']

            this_ula_cls = self.factordata.loc[self.factordata.index[self.timepointer], 'close_adj']

            pos_str = []
            for opt_pos in last_bar_opt_pos:
                data_name = opt_pos[0]
                opt_trading_code = self.optmetas.get_contract_meta(data_name).trading_code
                pos_str.append(opt_trading_code)
                pos_str.append(str(opt_pos[1]))
                pos_str.append(str(round(self.get_ctrct_delta(data_name), 2) * opt_pos[1]))
                pos_str.append(
                    str(round((self.dm.get_data(data_name).loc[self.datetime, 'close'] -
                               self.dm.get_data(data_name).loc[last_date, 'close']) * opt_pos[1] * self.get_mult(
                        data_name) / self.kwargs['initial_cash'] * 100, 2)) + '%')
            pos_str = ' '.join(pos_str)
            # info = f'{last_date.date()} {pos_str} {last_ula_cls}->{this_ula_cls} {round(100*(this_ula_cls/last_ula_cls - 1),1)}%'
            info = f'{last_date.date()} {pos_str} {self.factordata.loc[last_date, "close"]} {round(100 * (this_ula_cls / last_ula_cls - 1), 1)}%'
        self._holding_info.append(info)

    def next(self, row):
        # self._record_holding_info()
        if self.C_contract and self.P_contract:
            print(
                f'{self.datetime} => C_ttm = {self.get_ctrct_ttm(self.C_contract)} P_ttm = {self.get_ctrct_ttm(self.P_contract)}')
        else:
            print(
                f'{self.datetime} => no position, ttm = {self.get_ttm()}')
        self.opensignal = self.get_factor('shortvol')

        self.opensignal = True

        if self.opensignal:
            if not self.C_contract and not self.P_contract:
                fullyearcode = self.get_factor('opt_1m')
                ulaprice = self.get_quote(
                    self.optmetas.ula_fut_meta.get_contract_meta_by_fyc(fullyearcode).contract_code,
                    'close')
                print(
                    f"{self.datetime} => sell open ATM strangle at Klvel={self.Klvel}, ulapx = {ulaprice}")

                C, P = self.get_strangle()


                # # butterfly
                # Klvel = self.Klvel
                # self.Klvel = 99
                # _C, _P = self.get_strangle()
                # self.set_pos({C: -1, P: -1, _C: 1, _P: 1})
                # self.Klvel = Klvel

                self.set_pos({C: -1, P: -1})
                # self.set_pos({C: -2, P: -2})
                # self.set_pos({C: -5, P: -5})
                # self.set_pos({C: 1, P: 1})

            else:
                # 有持仓，如果delta暴露过大，roll仓
                pos_delta = self.get_portfolio_delta()
                C_contract_ttm = self.get_ctrct_ttm(self.C_contract)
                P_contract_ttm = self.get_ctrct_ttm(self.P_contract)
                if abs(pos_delta) > self.delta_th or C_contract_ttm <= self.rollterm_day or P_contract_ttm <= self.rollterm_day:
                    print(
                        f"{self.datetime} => C_ttm={C_contract_ttm}, P_ttm={P_contract_ttm}, pos_delta={pos_delta}, roll pos")
                    C, P = self.get_strangle()
                    # # butterfly
                    # Klvel = self.Klvel
                    # self.Klvel = 99
                    # _C, _P = self.get_strangle()
                    # self.set_pos({C: -1, P: -1, _C: 1, _P: 1})
                    # self.Klvel = Klvel

                    self.set_pos({C: -1, P: -1})
                    # self.set_pos({C: -2, P: -2})
                    # self.set_pos({C: -5, P: -5})
                    # self.set_pos({C: 1, P: 1})
        else:
            # 若有持仓，平仓
            if self.C_contract and self.P_contract:
                print(f"{self.datetime} => close out strangle")
                self.set_pos({})

    def get_strangle(self):
        fullyearcode = self.get_factor('opt_1m')
        date = self.row['str_date']
        # ulaprice = self.row[f'open']
        # ulaprice = self.row[f'close']

        if self.get_ttm() <= self.rollterm_day:
            fullyearcode = self.get_factor('opt_2m')
            if fullyearcode is None:
                raise ValueError('!!!Need to debug here. Can not find next main to roll pos!!!')

        ulaprice = self.get_quote(self.optmetas.ula_fut_meta.get_contract_meta_by_fyc(fullyearcode).contract_code,
                                  'close')

        C_K_list = self.optmetas.get_otm_strikes(fullyearcode, date, ulaprice, 100, alter_times=0)['C']
        P_K_list = self.optmetas.get_otm_strikes(fullyearcode, date, ulaprice, 100, alter_times=0)['P']

        if C_K_list:
            C_K = C_K_list[self.Klvel] if self.Klvel < len(C_K_list) else C_K_list[-1]
        else:
            C_K = P_K_list[-1]

        if P_K_list:
            P_K = P_K_list[-self.Klvel - 1] if self.Klvel < len(P_K_list) else P_K_list[0]
        else:
            P_K = C_K_list[0]

        C = self.optmetas.get_contract_by_strike(date, fullyearcode, 'C', C_K, 0).contract_code
        P = self.optmetas.get_contract_by_strike(date, fullyearcode, 'P', P_K, 0).contract_code

        if self.C_contract and self.P_contract:
            C_delta = self.get_ctrct_delta(self.C_contract)
            P_delta = self.get_ctrct_delta(self.P_contract)

            if abs(C_delta) > abs(P_delta) and self.get_ctrct_ttm(self.P_contract) > self.rollterm_day:
                P = self.P_contract
            elif abs(C_delta) < abs(P_delta) and self.get_ctrct_ttm(self.C_contract) > self.rollterm_day:
                C = self.C_contract

        return C, P

    def get_ctrct_delta(self, contract):
        this_meta = self.optmetas.get_contract_meta(contract)
        if not self.C_contract and not self.P_contract:
            return 0

        # ud_prc = self.get_factor('close')
        fullyearcode = self.get_factor('opt_1m')
        ud_prc = self.get_quote(self.optmetas.ula_fut_meta.get_contract_meta_by_fyc(fullyearcode).contract_code, 'close')

        r = 0
        div = 0
        ttm = (self.get_ctrct_ttm(self.C_contract) - 1 + 0.0000001) / 240

        k = float(this_meta.get_strike(self._str_date))
        vol = EuOptCalculator.get_imp_vol(this_meta.cp_type, ud_prc, k, ttm, r, div, self.get_quote(contract, 'close'))
        delta = EuOptCalculator.get_greeks(this_meta.cp_type, ud_prc, k, ttm, r, div, vol)['delta']

        return delta

    def get_ttm(self):
        return self.dtc.get_days_diff(self.get_factor('str_date'), self.exp_date)

    def get_ctrct_ttm(self, contract):
        return self.dtc.get_days_diff(self.get_factor('str_date'),
                                      self.optmetas.get_contract_meta(contract).get_exp_date())

    def get_portfolio_delta(self):

        if not self.C_contract and not self.P_contract:
            return 0

        return self.get_ctrct_delta(self.C_contract) * self.pos[self.C_contract] + self.get_ctrct_delta(
            self.P_contract) * self.pos[self.P_contract]

    def on_trade(self, dataname, side, size, price, openclose):
        if openclose == 'open':
            if self.optmetas.get_contract_meta(dataname).get_cp_type() == 'C':
                self.C_contract = dataname

            if self.optmetas.get_contract_meta(dataname).get_cp_type() == 'P':
                self.P_contract = dataname

        else:
            if self.optmetas.get_contract_meta(dataname).get_cp_type() == 'C':
                self.C_contract = None

            if self.optmetas.get_contract_meta(dataname).get_cp_type() == 'P':
                self.P_contract = None

    def postprocess(self):
        # print(np.array(self._holding_info))  # todo delete test part
        return self.pnl, self.rollpos, self.trades


class Strat6(Strat5):
    def __init__(self, factordata, data_manager: DataManager, YYMM, delay_p, optmeta, rollterm_day,
                 delta_th, Klvel, dtc, rollpos=None, fillthennext=False):

        """
        基于Strat5, 加入新逻辑：shortvol=True，则开仓，直到tp=True，则平仓。
        :param factordata:
        :param data_manager:
        :param YYMM:
        :param delay_p:
        :param optmeta:
        :param rollterm_day:
        :param delta_th:
        :param Klvel:
        :param dtc:
        :param rollpos:
        :param fillthennext:
        """
        super().__init__(factordata, data_manager, YYMM, delay_p, optmeta, rollterm_day,
                         delta_th, Klvel, dtc, rollpos, fillthennext)

        self.closesignal = False

    def next(self, row):
        if self.C_contract and self.P_contract:
            print(
                f'{self.datetime} => C_ttm = {self.get_ctrct_ttm(self.C_contract)} P_ttm = {self.get_ctrct_ttm(self.P_contract)}')
        else:
            print(
                f'{self.datetime} => no position, ttm = {self.get_ttm()}')

        self.opensignal = self.get_factor('shortvol')
        self.closesignal = self.get_factor('closevshort')

        self.opensignal = True

        # 没有持仓，判断是否开仓
        if not self.C_contract and not self.P_contract:
            if self.opensignal:
                fullyearcode = self.get_factor('opt_1m')
                ulaprice = self.get_quote(
                    self.optmetas.ula_fut_meta.get_contract_meta_by_fyc(fullyearcode).contract_code, 'close')
                print(f"{self.datetime} => sell open ATM strangle at Klvel={self.Klvel}, ulapx = {ulaprice}")
                C, P = self.get_strangle()
                self.set_pos({C: -1, P: -1})
                # self.set_pos({C: 1, P: 1})

        # 有持仓，判断是否平仓
        else:
            if not self.closesignal:
                # 有持仓，如果delta暴露过大，或者到roll仓日期，roll仓
                pos_delta = self.get_portfolio_delta()
                C_contract_ttm = self.get_ctrct_ttm(self.C_contract)
                P_contract_ttm = self.get_ctrct_ttm(self.P_contract)
                if abs(pos_delta) > self.delta_th or C_contract_ttm <= self.rollterm_day or P_contract_ttm <= self.rollterm_day:
                    print(
                        f"{self.datetime} => C_ttm={C_contract_ttm}, P_ttm={P_contract_ttm}, pos_delta={pos_delta}, roll pos")
                    C, P = self.get_strangle()
                    self.set_pos({C: -1, P: -1})
                    # self.set_pos({C: 1, P: 1})
            else:
                print(f"{self.datetime} => close out strangle")
                self.set_pos({})