import numpy as np

from optengine.SimpleMultiDataBackTest_Engine import MultiDataBTEngine
import pandas as pd
from data_manager import DataManager
from optcal import EuOptCalculator, AmOptCalculator


class BuyOnEntrySig(MultiDataBTEngine):
    def __init__(self, factordata, data_manager: DataManager, YYMM, delay_p, optmeta, rollterm_day,
                 delta_th, Klvel, dtc, rollpos=None, fillthennext=False, commodity=False):

        """

        仅开仓信号触发下的买权策略 （无平仓信号）

        若有仓位：
            在开盘判断：
                组合delta是否超过delta_th，如果是则roll到新的Klvl档ATM strangle，月份选择根据rollterm_day决定

            若信号触发，在收盘判断：
                1）组合delta是否超过delta_th
                2）是否为roll仓日期，

                二者有一为True，则roll到新的ATM strangle，月份选择根据rollterm_day决定。否则不做交易

            若无信号触发：
                在收盘平仓

        若无仓位：
            若信号触发：
                收盘价买入Klvl档的ATM Strangle。

                注意！！！！！！本策略delay_p必须为0！！！！！！！

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
        # 注意！！！！！！本策略delay_p必须为0！！！！！！！
        if delay_p != 0:
            raise ValueError('delay_p must equal 0 for this strategy')

        super().__init__(factordata, data_manager, optmetas=optmeta, delay_p=delay_p, rollpos=rollpos,
                         fillthennext=fillthennext)

        # trading params
        self.opensignal = False
        self.C_contract = None
        self.P_contract = None
        self.delta_th = delta_th
        self.Klvel = Klvel
        self.commodity = commodity

        # time params
        self.dtc = dtc
        self.YYMM = YYMM
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
        """

        仅开仓信号触发下的买权策略 （无平仓信号）

        若有仓位：
            在开盘判断：
                组合delta是否超过delta_th，如果是则roll到新的Klvl档ATM strangle，月份选择根据rollterm_day决定

            若信号触发，在收盘判断：
                1）组合delta是否超过delta_th
                2）是否roll仓日期，

                二者有一为True，则roll到新的ATM strangle，月份选择根据rollterm_day决定。否则不做交易

            若无信号触发：
                在收盘平仓

        若无仓位：
            若信号触发：
                收盘价买入Klvl档的ATM Strangle。


        :param row:
        :return:
        """
        print(f'{self.datetime} => ttm = {self.get_ttm()}')

        self.opensignal = self.get_factor('entrysignal')

        # self.opensignal = True

        # 若有仓位：
        if self.C_contract and self.P_contract:

            C_contract_ttm = self.get_ctrct_ttm(self.C_contract)
            P_contract_ttm = self.get_ctrct_ttm(self.C_contract)

            # 在开盘判断：组合delta是否超过delta_th，如果是则roll到新的Klvl档ATM strangle，月份选择根据rollterm_day决定
            # 注意！！开盘交易
            pos_delta = self.get_portfolio_delta(_open=True)
            if abs(pos_delta) > self.delta_th:
                self.log(f"C_ttm={C_contract_ttm}, P_ttm={P_contract_ttm}, pos_delta={pos_delta} at open, roll pos")
                C, P = self.get_strangle(_open=True)
                C_slip_fixed = self.dm.get_item(C, 'slipfix')
                P_slip_fixed = self.dm.get_item(P, 'slipfix')
                targetprice = {C: self.get_quote(C, 'open') + C_slip_fixed,
                               P: self.get_quote(P, 'open') + P_slip_fixed}
                if C != self.C_contract:
                    slip = self.dm.get_item(self.C_contract, 'slipfix')
                    targetprice.update({self.C_contract: {'buyprice': self.get_quote(self.C_contract, 'open') + slip,
                                                          'sellprice': self.get_quote(self.C_contract, 'open') - slip}})
                if P != self.P_contract:
                    slip = self.dm.get_item(self.P_contract, 'slipfix')
                    targetprice.update({self.P_contract: {'buyprice': self.get_quote(self.P_contract, 'open') + slip,
                                                          'sellprice': self.get_quote(self.P_contract, 'open') - slip}})

                self.set_pos({C: 1, P: 1}, targetprice=targetprice)

            # 若信号触发，在收盘判断：
            # 1）组合delta是否超过delta_th
            # 2）是否roll仓日期，
            # 二者有一为True，则roll到新的ATM strangle，月份选择根据rollterm_day决定。否则不做交易
            if self.opensignal:

                # 注意！！收盘交易
                pos_delta = self.get_portfolio_delta()
                if abs(pos_delta) > self.delta_th or C_contract_ttm <= self.rollterm_day or P_contract_ttm <= self.rollterm_day:
                    self.log(
                        f"C_ttm={C_contract_ttm}, P_ttm={P_contract_ttm}, pos_delta={pos_delta} at close, roll pos")
                    C, P = self.get_strangle()
                    self.set_pos({C: 1, P: 1})

            # 若无信号触发, 在收盘平仓
            else:
                self.log(f"close out strangle")
                self.set_pos({})

        # 若无仓位, 若信号触发：收盘价买入Klvl档的ATM Strangle。
        else:
            if self.opensignal:
                # 注意！！收盘交易
                C, P = self.get_strangle()
                self.set_pos({C: 1, P: 1})

    def get_strangle(self, _open=False):
        _date = self.row['str_date']
        fullyearcode = self.get_factor('opt_1m') if self.commodity else self.optmetas.get_dueyymm_series(_date)[0]

        # ulaprice = self.row[f'open']
        # ulaprice = self.row[f'close']

        if self.get_ttm() <= self.rollterm_day:
            if self.commodity:
                fullyearcode = self.get_factor('opt_2m')
                if fullyearcode is None:
                    raise ValueError('!!!Need to debug here. Can not find next main to roll pos!!!')
            else:
                fullyearcode = self.optmetas.get_dueyymm_series(_date)[1]

        _price = 'close' if not _open else 'open'
        # ulaprice = self.get_factor(_price)

        if self.commodity:
            ulaprice = self.get_quote(self.optmetas.ula_fut_meta.get_contract_meta_by_fyc(fullyearcode).contract_code,
                                      _price)
        else:
            ulaprice = self.get_factor(_price)

        C_K_list = self.optmetas.get_otm_strikes(fullyearcode, _date, ulaprice, 100, alter_times=0)['C']
        P_K_list = self.optmetas.get_otm_strikes(fullyearcode, _date, ulaprice, 100, alter_times=0)['P']

        if C_K_list:
            C_K = C_K_list[self.Klvel] if self.Klvel < len(C_K_list) else C_K_list[-1]
        else:
            C_K = P_K_list[-1]

        if P_K_list:
            P_K = P_K_list[-self.Klvel - 1] if self.Klvel < len(P_K_list) else P_K_list[0]
        else:
            P_K = C_K_list[0]

        C = self.optmetas.get_contract_by_strike(_date, fullyearcode, 'C', C_K, 0).contract_code
        P = self.optmetas.get_contract_by_strike(_date, fullyearcode, 'P', P_K, 0).contract_code

        return C, P

    def get_portfolio_delta(self, _open=False):

        if not self.C_contract and not self.P_contract:
            return 0

        return self.get_ctrct_delta(self.C_contract, _open) * self.pos[self.C_contract] + self.get_ctrct_delta(
            self.P_contract, _open) * self.pos[self.P_contract]

    def get_ctrct_delta(self, contract, _open=False):
        this_meta = self.optmetas.get_contract_meta(contract)
        if not self.C_contract and not self.P_contract:
            return 0

        # ud_prc = self.get_factor('close')
        _date = self.row['str_date']
        fullyearcode = self.optmetas.get_contract_meta(contract).full_year_code

        _price = 'close' if not _open else 'open'
        # ud_prc = self.get_factor(_price)

        if self.commodity:
            ud_prc = self.get_quote(self.optmetas.ula_fut_meta.get_contract_meta_by_fyc(fullyearcode).contract_code,
                                    _price)
        else:
            ud_prc = self.get_factor(_price)

        r = 0
        div = 0
        ttm = (self.get_ctrct_ttm(self.C_contract) - 1 + 0.0000001) / 240 if not _open else self.get_ctrct_ttm(
            self.C_contract) / 240

        k = float(this_meta.get_strike(self._str_date))
        vol = EuOptCalculator.get_imp_vol(this_meta.cp_type, ud_prc, k, ttm, r, div, self.get_quote(contract, _price))
        delta = EuOptCalculator.get_greeks(this_meta.cp_type, ud_prc, k, ttm, r, div, vol)['delta']

        return delta

    def get_ttm(self):
        return self.dtc.get_days_diff(self.get_factor('str_date'), self.exp_date)

    def get_ctrct_ttm(self, contract):
        return self.dtc.get_days_diff(self.get_factor('str_date'),
                                      self.optmetas.get_contract_meta(contract).get_exp_date())

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


class BuyOnEntryExitSig(BuyOnEntrySig):
    def __init__(self, factordata, data_manager: DataManager, YYMM, delay_p, optmeta, rollterm_day,
                 delta_th, Klvel, dtc, rollpos=None, fillthennext=False, commodity=False):
        """

        开仓+平仓信号触发下的买权策略。除开平仓外，其他交易/对冲逻辑同 BuyOnEntrySig

        若有仓位：
            在开盘判断：
                组合delta是否超过delta_th，如果是则roll到新的Klvl档ATM strangle，月份选择根据rollterm_day决定

            若出场信号未触发，在收盘判断：
                1）组合delta是否超过delta_th
                2）是否为roll仓日期，

                二者有一为True，则roll到新的ATM strangle，月份选择根据rollterm_day决定。否则不做交易

            若出场信号触发：
                在收盘平仓

        若无仓位：
            若信号触发：
                收盘价买入Klvl档的ATM Strangle。

                注意！！！！！！本策略delay_p必须为0！！！！！！！

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
        :param commodity:
        """
        super().__init__(factordata, data_manager, YYMM, delay_p, optmeta, rollterm_day,
                         delta_th, Klvel, dtc, rollpos, fillthennext, commodity=commodity)
        self.closesignal = False

    def next(self, row):
        """
        开仓+平仓信号触发下的买权策略。除开平仓外，其他交易/对冲逻辑同 BuyOnEntrySig

        若有仓位：
            在开盘判断：
                组合delta是否超过delta_th，如果是则roll到新的Klvl档ATM strangle，月份选择根据rollterm_day决定

            若出场信号未触发，在收盘判断：
                1）组合delta是否超过delta_th
                2）是否为roll仓日期，

                二者有一为True，则roll到新的ATM strangle，月份选择根据rollterm_day决定。否则不做交易

            若出场信号触发：
                在收盘平仓

        若无仓位：
            若开仓信号触发：
                收盘价买入Klvl档的ATM Strangle。

        :param row:
        :return:
        """
        print(f'{self.datetime} => ttm = {self.get_ttm()}')

        self.opensignal = self.get_factor('entrysignal')
        self.closesignal = self.get_factor('exitsignal')

        # self.opensignal = True

        # 若有仓位：
        if self.C_contract and self.P_contract:

            C_contract_ttm = self.get_ctrct_ttm(self.C_contract)
            P_contract_ttm = self.get_ctrct_ttm(self.C_contract)

            # 在开盘判断：组合delta是否超过delta_th，如果是则roll到新的Klvl档ATM strangle，月份选择根据rollterm_day决定
            # 注意！！开盘交易
            pos_delta = self.get_portfolio_delta(_open=True)
            if abs(pos_delta) > self.delta_th:
                self.log(f"C_ttm={C_contract_ttm}, P_ttm={P_contract_ttm}, pos_delta={pos_delta} at open, roll pos")
                C, P = self.get_strangle(_open=True)
                C_slip_fixed = self.dm.get_item(C, 'slipfix')
                P_slip_fixed = self.dm.get_item(P, 'slipfix')
                targetprice = {C: self.get_quote(C, 'open') + C_slip_fixed,
                               P: self.get_quote(P, 'open') + P_slip_fixed}
                if C != self.C_contract:
                    slip = self.dm.get_item(self.C_contract, 'slipfix')
                    targetprice.update({self.C_contract: {'buyprice': self.get_quote(self.C_contract, 'open') + slip,
                                                          'sellprice': self.get_quote(self.C_contract, 'open') - slip}})
                if P != self.P_contract:
                    slip = self.dm.get_item(self.P_contract, 'slipfix')
                    targetprice.update({self.P_contract: {'buyprice': self.get_quote(self.P_contract, 'open') + slip,
                                                          'sellprice': self.get_quote(self.P_contract, 'open') - slip}})
                self.set_pos({C: 1, P: 1}, targetprice=targetprice)

            # 若出场信号未触发，在收盘判断：
            # 1）组合delta是否超过delta_th
            # 2）是否roll仓日期，
            # 二者有一为True，则roll到新的ATM strangle，月份选择根据rollterm_day决定。否则不做交易
            if not self.closesignal:
                # 注意！！收盘交易
                pos_delta = self.get_portfolio_delta()
                if abs(pos_delta) > self.delta_th or C_contract_ttm <= self.rollterm_day or P_contract_ttm <= self.rollterm_day:
                    self.log(
                        f"C_ttm={C_contract_ttm}, P_ttm={P_contract_ttm}, pos_delta={pos_delta} at close, roll pos")
                    C, P = self.get_strangle()
                    self.set_pos({C: -1, P: -1})

            # 若出场信号触发, 在收盘平仓
            else:
                self.log(f"close out strangle")
                self.set_pos({})

        # 若无仓位, 若信号触发：收盘价买入Klvl档的ATM Strangle。
        else:
            if self.opensignal:
                # 注意！！收盘交易
                C, P = self.get_strangle()
                self.set_pos({C: 1, P: 1})


class BuyOnEntrySig_2(BuyOnEntrySig):
    def __init__(self, factordata, data_manager: DataManager, YYMM, delay_p, optmeta, rollterm_day,
                 delta_th, Klvel, dtc, rollpos=None, fillthennext=False, commodity=False):
        """

        与 BuyOnEntrySig 的区别是：
            当需要对冲时，选择delta更大的合约进行roll仓，而delta小的合约则不再roll，除非到了rollterm_day

        仅开仓信号触发下的买权策略 （无平仓信号）

        若有仓位：
            在开盘判断：
                组合delta是否超过delta_th，如果是则roll到新的Klvl档ATM strangle，月份选择根据rollterm_day决定

            若信号触发，在收盘判断：
                1）组合delta是否超过delta_th
                2）是否为roll仓日期，

                二者有一为True，则roll到新的ATM strangle，月份选择根据rollterm_day决定。否则不做交易

            若无信号触发：
                在收盘平仓

        若无仓位：
            若信号触发：
                收盘价买入Klvl档的ATM Strangle。

                注意！！！！！！本策略delay_p必须为0！！！！！！！

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
        :param commodity:
        """
        super().__init__(factordata, data_manager, YYMM, delay_p, optmeta, rollterm_day,
                         delta_th, Klvel, dtc, rollpos, fillthennext, commodity=commodity)

    def get_strangle(self, _open=False):
        _date = self.row['str_date']
        fullyearcode = self.get_factor('opt_1m') if self.commodity else self.optmetas.get_dueyymm_series(_date)[0]

        # ulaprice = self.row[f'open']
        # ulaprice = self.row[f'close']

        if self.get_ttm() <= self.rollterm_day:
            if self.commodity:
                fullyearcode = self.get_factor('opt_2m')
                if fullyearcode is None:
                    raise ValueError('!!!Need to debug here. Can not find next main to roll pos!!!')
            else:
                fullyearcode = self.optmetas.get_dueyymm_series(_date)[1]

        _price = 'close' if not _open else 'open'
        # ulaprice = self.get_factor(_price)

        if self.commodity:
            ulaprice = self.get_quote(self.optmetas.ula_fut_meta.get_contract_meta_by_fyc(fullyearcode).contract_code,
                                      _price)
        else:
            ulaprice = self.get_factor(_price)

        C_K_list = self.optmetas.get_otm_strikes(fullyearcode, _date, ulaprice, 100, alter_times=0)['C']
        P_K_list = self.optmetas.get_otm_strikes(fullyearcode, _date, ulaprice, 100, alter_times=0)['P']

        if C_K_list:
            C_K = C_K_list[self.Klvel] if self.Klvel < len(C_K_list) else C_K_list[-1]
        else:
            C_K = P_K_list[-1]

        if P_K_list:
            P_K = P_K_list[-self.Klvel - 1] if self.Klvel < len(P_K_list) else P_K_list[0]
        else:
            P_K = C_K_list[0]

        C = self.optmetas.get_contract_by_strike(_date, fullyearcode, 'C', C_K, 0).contract_code
        P = self.optmetas.get_contract_by_strike(_date, fullyearcode, 'P', P_K, 0).contract_code

        if self.C_contract and self.P_contract:
            C_delta = self.get_ctrct_delta(self.C_contract, _open=_open)
            P_delta = self.get_ctrct_delta(self.P_contract, _open=_open)

            if abs(C_delta) > abs(P_delta) and self.get_ctrct_ttm(self.P_contract) > self.rollterm_day:
                P = self.P_contract
            elif abs(C_delta) < abs(P_delta) and self.get_ctrct_ttm(self.C_contract) > self.rollterm_day:
                C = self.C_contract

        return C, P


class BuyOnEntryExitSig_2(BuyOnEntrySig):
    def __init__(self, factordata, data_manager: DataManager, YYMM, delay_p, optmeta, rollterm_day,
                 delta_th, Klvel, dtc, rollpos=None, fillthennext=False, commodity=False):
        """

        开仓+平仓信号触发下的买权策略。除开平仓外，其他交易/对冲逻辑同 BuyOnEntrySig

        与 BuyOnEntryExitSig 的区别是：
            当需要对冲时，选择delta更大的合约进行roll仓，而delta小的合约则不再roll，除非到了rollterm_day

        若有仓位：
            在开盘判断：
                组合delta是否超过delta_th，如果是则roll到新的Klvl档ATM strangle，月份选择根据rollterm_day决定

            若出场信号未触发，在收盘判断：
                1）组合delta是否超过delta_th
                2）是否为roll仓日期，

                二者有一为True，则roll到新的ATM strangle，月份选择根据rollterm_day决定。否则不做交易

            若出场信号触发：
                在收盘平仓

        若无仓位：
            若信号触发：
                收盘价买入Klvl档的ATM Strangle。

                注意！！！！！！本策略delay_p必须为0！！！！！！！

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
        :param commodity:
        """
        super().__init__(factordata, data_manager, YYMM, delay_p, optmeta, rollterm_day,
                         delta_th, Klvel, dtc, rollpos, fillthennext, commodity=commodity)
        self.closesignal = False

    def next(self, row):
        """
        开仓+平仓信号触发下的买权策略。除开平仓外，其他交易/对冲逻辑同 BuyOnEntrySig

        若有仓位：
            在开盘判断：
                组合delta是否超过delta_th，如果是则roll到新的Klvl档ATM strangle，月份选择根据rollterm_day决定

            若出场信号未触发，在收盘判断：
                1）组合delta是否超过delta_th
                2）是否为roll仓日期，

                二者有一为True，则roll到新的ATM strangle，月份选择根据rollterm_day决定。否则不做交易

            若出场信号触发：
                在收盘平仓

        若无仓位：
            若开仓信号触发：
                收盘价买入Klvl档的ATM Strangle。

        :param row:
        :return:
        """
        print(f'{self.datetime} => ttm = {self.get_ttm()}')

        self.opensignal = self.get_factor('entrysignal')
        self.closesignal = self.get_factor('exitsignal')

        # self.opensignal = True

        # 若有仓位：
        if self.C_contract and self.P_contract:

            C_contract_ttm = self.get_ctrct_ttm(self.C_contract)
            P_contract_ttm = self.get_ctrct_ttm(self.C_contract)

            # 在开盘判断：组合delta是否超过delta_th，如果是则roll到新的Klvl档ATM strangle，月份选择根据rollterm_day决定
            # 注意！！开盘交易
            pos_delta = self.get_portfolio_delta(_open=True)
            if abs(pos_delta) > self.delta_th:
                self.log(f"C_ttm={C_contract_ttm}, P_ttm={P_contract_ttm}, pos_delta={pos_delta} at open, roll pos")
                C, P = self.get_strangle(_open=True)
                C_slip_fixed = self.dm.get_item(C, 'slipfix')
                P_slip_fixed = self.dm.get_item(P, 'slipfix')
                targetprice = {C: self.get_quote(C, 'open') + C_slip_fixed,
                               P: self.get_quote(P, 'open') + P_slip_fixed}
                if C != self.C_contract:
                    slip = self.dm.get_item(self.C_contract, 'slipfix')
                    targetprice.update({self.C_contract: {'buyprice': self.get_quote(self.C_contract, 'open') + slip,
                                                          'sellprice': self.get_quote(self.C_contract, 'open') - slip}})
                if P != self.P_contract:
                    slip = self.dm.get_item(self.P_contract, 'slipfix')
                    targetprice.update({self.P_contract: {'buyprice': self.get_quote(self.P_contract, 'open') + slip,
                                                          'sellprice': self.get_quote(self.P_contract, 'open') - slip}})

                self.set_pos({C: 1, P: 1}, targetprice=targetprice)

            # 若出场信号未触发，在收盘判断：
            # 1）组合delta是否超过delta_th
            # 2）是否roll仓日期，
            # 二者有一为True，则roll到新的ATM strangle，月份选择根据rollterm_day决定。否则不做交易
            if not self.closesignal:
                # 注意！！收盘交易
                pos_delta = self.get_portfolio_delta()
                if abs(pos_delta) > self.delta_th or C_contract_ttm <= self.rollterm_day or P_contract_ttm <= self.rollterm_day:
                    self.log(
                        f"C_ttm={C_contract_ttm}, P_ttm={P_contract_ttm}, pos_delta={pos_delta} at close, roll pos")
                    C, P = self.get_strangle()
                    self.set_pos({C: -1, P: -1})

            # 若出场信号触发, 在收盘平仓
            else:
                self.log(f"close out strangle")
                self.set_pos({})

        # 若无仓位, 若信号触发：收盘价买入Klvl档的ATM Strangle。
        else:
            if self.opensignal:
                # 注意！！收盘交易
                C, P = self.get_strangle()
                self.set_pos({C: 1, P: 1})

    def get_strangle(self, _open=False):
        _date = self.row['str_date']
        fullyearcode = self.get_factor('opt_1m') if self.commodity else self.optmetas.get_dueyymm_series(_date)[0]

        # ulaprice = self.row[f'open']
        # ulaprice = self.row[f'close']

        if self.get_ttm() <= self.rollterm_day:
            if self.commodity:
                fullyearcode = self.get_factor('opt_2m')
                if fullyearcode is None:
                    raise ValueError('!!!Need to debug here. Can not find next main to roll pos!!!')
            else:
                fullyearcode = self.optmetas.get_dueyymm_series(_date)[1]

        _price = 'close' if not _open else 'open'
        # ulaprice = self.get_factor(_price)

        if self.commodity:
            ulaprice = self.get_quote(self.optmetas.ula_fut_meta.get_contract_meta_by_fyc(fullyearcode).contract_code,
                                      _price)
        else:
            ulaprice = self.get_factor(_price)

        C_K_list = self.optmetas.get_otm_strikes(fullyearcode, _date, ulaprice, 100, alter_times=0)['C']
        P_K_list = self.optmetas.get_otm_strikes(fullyearcode, _date, ulaprice, 100, alter_times=0)['P']

        if C_K_list:
            C_K = C_K_list[self.Klvel] if self.Klvel < len(C_K_list) else C_K_list[-1]
        else:
            C_K = P_K_list[-1]

        if P_K_list:
            P_K = P_K_list[-self.Klvel - 1] if self.Klvel < len(P_K_list) else P_K_list[0]
        else:
            P_K = C_K_list[0]

        C = self.optmetas.get_contract_by_strike(_date, fullyearcode, 'C', C_K, 0).contract_code
        P = self.optmetas.get_contract_by_strike(_date, fullyearcode, 'P', P_K, 0).contract_code

        if self.C_contract and self.P_contract:
            C_delta = self.get_ctrct_delta(self.C_contract, _open=_open)
            P_delta = self.get_ctrct_delta(self.P_contract, _open=_open)

            if abs(C_delta) > abs(P_delta) and self.get_ctrct_ttm(self.P_contract) > self.rollterm_day:
                P = self.P_contract
            elif abs(C_delta) < abs(P_delta) and self.get_ctrct_ttm(self.C_contract) > self.rollterm_day:
                C = self.C_contract

        return C, P
