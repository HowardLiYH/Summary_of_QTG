from collections import OrderedDict
import datetime
import hashlib
import hmac
import urllib3
import requests
import json
import pandas as pd
import numpy as np
from typing import List, Dict
import time


class BybitHedgeClient:
    def __init__(self, key, secret, env="test", alertClient=None):

        self.key = key
        self.secret = secret
        self.env = env

        urlBase = "api"
        self.url = f"http://{urlBase}.bybit.yijinin.biz/"
        if env == "test":
            urlBase = "api-testnet"
            self.url = f"https://{urlBase}.bytick.com/"
        self.headers = {"Content-Type": "application/json"}

        self.alertClient = alertClient

    def set_trade_info(self, min_qty, max_qty, qty_round_num, tick_size):
        self.min_qty = min_qty
        self.max_qty = max_qty
        self.qty_round_num = qty_round_num
        self.tick_size = tick_size

    def hash_secret(self, sign):
        secretKey = str.encode(self.secret)
        hash = hmac.new(secretKey, sign.encode("utf-8"), hashlib.sha256)
        signature = hash.hexdigest()
        return signature

    @property
    def cur_timestamp(self):
        return str(int(datetime.datetime.now().timestamp() * 1000) -10000)

    def create_body(self, params):
        """Add auth, time info and preprocess."""
        params["api_key"] = self.key
        params["timestamp"] = self.cur_timestamp
        params["recv_window"] = "93800000000"
        params = OrderedDict(sorted(params.items()))

        sign = ""
        for key in sorted(params.keys()):
            v = params[key]
            if isinstance(params[key], bool):
                if params[key]:
                    v = "true"
                else:
                    v = "false"
            sign += key + "=" + str(v) + "&"
        sign = sign[:-1]
        signature = self.hash_secret(sign)

        sign_real = {"sign": signature}

        body = dict(params, **sign_real)

        urllib3.disable_warnings()

        return body

    def send_request(self, params={}, api=None, method="get",result = 1):
        assert method in ["get", "post"]
        body = self.create_body(params)
        url = self.url + api.lstrip("/")
        curlStr = url + "?" + ("&").join([f"{k}={v}" for k, v in body.items()])

        s = requests.session()
        s.keep_alive = False
        # TODO: add failover
        if method == "get":
            res = requests.get(curlStr)
        else:
            res = requests.post(
                url, data=json.dumps(body), headers=self.headers, verify=False
            )
        #print(res.text)
        # print(res.code)
        # TODO: con_retry not correct should use others
        # res = gen_util.con_retry(func, retry=3, pause=10)(body)

#        self.scan_response(res, info=curlStr + str(body))

        # TODO: put FO here
        if result:

            resDic = json.loads(res.text)["result"]
        else :
            resDic = json.loads(res.text)
        return resDic

    def scan_response(self, response, info=None):
        """Alert """
        # TODO: add alert here
        if response.status_code == 404:
            logger.warning(f"API issue, meet 404, need human check, detail: {info}")
        elif response.status_code == 403:
            logger.warning("Request too many times")
        elif response.status_code == 200:  # normal case
            resDic = json.loads(response.text)
            if resDic["ret_code"] != 0 or resDic["ret_msg"] != "OK":
                info = f"Incorrect Response, need human check, detail: {resDic}"
                logger.warning(info)
            else:
                pass
        return

    """Below are based on API."""

    def filter_list_of_dict(self, posL: List, filterDic):
        """Filter list of dictionary.
        Example: filter_position(res, {"symbol": "ETHUSDT", "side": "Buy"})
        """
        for k, v in filterDic.items():
            posL = filter(lambda x: filterDic[k] == x[k], posL)
        return list(posL)

    def create_export_info(
        self, infoDic={"symbol": "ETHUSDT"}, infoList=["position", "trade"]
    ):
        """Export essential info after every trade.
        1. Position info.
        """
        info = ""
        # 1 position
        if "position" in infoList:
            posDF = self.get_cur_position()
            posInfo = posDF.to_markdown()
            info += "Postion info: \n"
            info += posInfo + "\n" * 3

        # 2 recent 5 trade
        if "trade" in infoList:
            symbol = infoDic["symbol"]
            tradeDF = self.get_recent_order(symbol, 5)
            tradeInfo = tradeDF.to_markdown()
            info += "Trade info: \n"
            info += tradeInfo + "\n" * 3

        return info

    def get_cur_price(self, symbol):
        """Get current price."""
        for i in range(5):
            try:
                api = '/derivatives/v3/public/recent-trade'
                if symbol[-1] == 'T':
                    res = self.send_request({"category": "linear", "symbol": symbol, "limit": 5}, api, method="get")
                    p = float(res['list'][0]["price"])
                elif symbol[-1] == 'D':
                    res = self.send_request({"category": "inverse", "symbol": symbol, "limit": 5}, api, method="get")
                    p = float(res['list'][0]["price"])
                return p
            except:
                print(f'第{i + 1}次查询价格时报错')
                time.sleep(1)

    def get_recent_order(self, symbol, topn=50, status="Filled"):
        """Get topn order."""
        myapi = "/contract/v3/private/order/list"
        length = int(topn/50) + 1

        myparams = {"symbol": symbol, "limit": 50, "orderStatus":status}

        while True:
            try:
                trade = self.send_request(params=myparams, api=myapi, method="get")
                tradeDF = pd.DataFrame(trade["list"])
                break
            except:
                time.sleep(60)
        i = 1
        while i < length:
            try:
                myparams["page"] = i+1
                trade = self.send_request(params=myparams, api=myapi, method="get")
                tradeDF2 = pd.DataFrame(trade["list"])
                tradeDF = tradeDF.append(tradeDF2)
                time.sleep(1)
                i = i+1
            except:
                break
        tradeDF.index = range(len(tradeDF))
        tradeDF = tradeDF[:topn]
        return tradeDF

    def get_myorder_status(self, symbol, myorder_id):
        try:
            myapi = "/contract/v3/private/order/list"
            for i in range(5):
                print(f'第{i + 1}次查询order')
                try:
                    trade = self.send_request(
                        {"symbol": symbol, "orderId": myorder_id}, api=myapi, method="get"
                    )
                    myorderDF = pd.DataFrame(trade['list'], index=[0])
                    return myorderDF
                except:
                    time.sleep(0.1)
        except:
            print('get_myorder_status时候出错')

    def get_my_position(self, symbol):
        myapi = "/contract/v3/private/position/list"
        i = 0
        while i < 5:
            try:
                ans = self.send_request(params={"symbol": symbol}, api=myapi, method="get")
                p1 = ans['list'][0]['positionIdx']
                if p1 == 1:
                    cur_position_long = float(ans['list'][0]['size'])
                    cur_position_short = float(ans['list'][1]['size'])
                elif p1 == 2:
                    cur_position_long = float(ans['list'][1]['size'])
                    cur_position_short = float(ans['list'][0]['size'])
                elif p1 == 0:
                    cur_position_long = float(ans['list'][0]['size']) if ans['list'][0]['side'] == 'Buy' else 0
                    cur_position_short = float(ans['list'][0]['size']) if ans['list'][0]['side'] == 'Sell' else 0
                cur_position = cur_position_long - cur_position_short
                print(f"cur_position : {cur_position}")
                return (cur_position, cur_position_long, cur_position_short)
            except:
                print(f'第{i+1}次查询仓位')
                i = i+1
        return([])

    def get_my_position_info(self, symbol):
        i = 0
        while i < 5:
            try:
                ans = self.send_request(params={"symbol": symbol}, api="/contract/v3/private/position/list", method="get")
                return(ans['list'])
            except:
                print(f'第{i+1}次查询仓位')
                i = i+1
        return([])

    def send_limit_order(self, qty, price, symbol, side, action):
        qty = np.round(qty, self.qty_round_num)
        price = np.round(price/self.tick_size)*self.tick_size
        price = np.round(price,4)
        myapi = "/contract/v3/private/order/create"

        '如果symbol是usd 那么qty最小单位是100'
        if action == 'open':
            reduce_only = bool(0)
            close_on_trigger = bool(0)
        elif action == 'close':
            reduce_only = bool(1)
            close_on_trigger = bool(1)
        if action == 'open' and side == 'Buy':
            positionIdx = 1
        elif action == 'open' and side == 'Sell':
            positionIdx = 2
        elif action == 'close' and side == 'Buy':
            positionIdx = 2
        elif action == 'close' and side == 'Sell':
            positionIdx = 1

        '市价单也就是吃单版 暂时可能用不到'
        params = {"symbol": symbol,
                  "side": side,
                  "positionIdx":positionIdx,
                  "orderType": "Limit",
                  "qty": str(qty),
                  "timeInForce": "GoodTillCancel",
                  "reduceOnly": reduce_only,
                  "price": str(price),
                  "closeOnTrigger": close_on_trigger}

        for i in range(5):
            print(f'第{i + 1}次尝试下limit单')
            try:
                ans = self.send_request(
                    params, api=myapi, method="post"
                )
                print(ans)
                orderid = ans['orderId']
                print(f'挂单{side} 以{price}的价格{side}入{qty}个')
                print(f'挂单{side} 的单号是{orderid}')
                return orderid
            except:
                time.sleep(0.1)

        return (0)

    def send_cancel_order(self,symbol, orderID):
        myapi = "/contract/v3/private/order/cancel"
        params = {"symbol": symbol, "orderId": orderID}

        for i in range(5):
            try:
                ans = self.send_request(
                    params, api=myapi, method="post",result=0
                )
                return ans['orderId']
            except:
                time.sleep(0.1)

    def send_cancel_all_order(self, symbol):
        for i in range(5):
            try:
                myapi = "/contract/v3/private/order/cancel-all"
                params = {"symbol": symbol}
                ans = self.send_request(
                    params, api=myapi, method="post"
                )
                print('所有单子都被撤掉')
                return ans['list']
            except:
                time.sleep(0.1)

    # data structure is quiet different from v2
    def get_orderbook(self,symbol):
        '正反向通用'
        if symbol[-1] == 'T':
            category = 'linear'
        elif symbol[-1] == 'D':
            category = 'inverse'
        for i in range(5):
            try:
                myapi = "/derivatives/v3/public/order-book/L2"
                params = {"category": category, "symbol": symbol}
                ans = self.send_request(params, api=myapi, method="get")
                return(ans)
            except:
                print(f'第{i+1}次查询盘口')
                i = i+1

    def get_predicted_fundingrate(self,symbol):
        myapi = "/private/linear/funding/predicted-funding"
        params = {"symbol":symbol}
        ans = self.send_request(params,api = myapi,method = "get")

        return(ans)


    def get_prev_fundingrate(self,symbol):
        myapi = "/private/linear/funding/prev-funding"
        params = {"symbol":symbol}
        ans = self.send_request(params,api = myapi,method = "get")

        return(ans)

    def get_contract_wallet_balances(self,symbol):
        '正反向通用'
        myapi = "/contract/v3/private/account/wallet/balance"
        if symbol:
            params = {"coin":symbol}
        else :
            params = {}
        ans = self.send_request(params,api = myapi,method = "get")
        return(ans['list'])

    def get_risk_limit_info(self,symbol):
        myapi = "/derivatives/v3/public/risk-limit/list"
        params = {"symbol": symbol}
        ans = self.send_request(params,myapi,"get")
        return(ans)

    def set_risk_limit(self,symbol,risk_id):
        myapi = '/contract/v3/private/position/set-risk-limit'
        try:
            params = {"symbol":symbol,'positionIdx':1,'risk_id':risk_id}
            ans1 = self.send_request(params,myapi,"post")
        except:
            ans1 = []

        try:
            params = {"symbol":symbol,'positionIdx':2,'risk_id':risk_id}
            ans2 = self.send_request(params,myapi,"post")
        except:
            ans2 = []

        return([ans1,ans2])

    def set_leverage(self,symbol,leverage):
        api = '/contract/v3/private/position/set-leverage'
        params = {"symbol":symbol,'buy_leverage':str(leverage),'sell_leverage':str(leverage)}
        ans = self.send_request(params,api,"post")
        return(ans)

    def get_public_info(self,symbol):
        try:
            ans = self.send_request(params={"category": 'linear', "symbol": symbol},
                                    api="/derivatives/v3/public/instruments-info", method="get")['list']
            with open("public_trading_info.json", 'w') as f:
                json.dump(ans, f)
        except:
            with open("public_trading_info.json", 'r') as f:
                ans = json.load(f)
        for categry in ans:
            if categry["symbol"] == symbol:
                return categry
        print("no such symbol")
        return(None)

    def get_history_price(self,symbol,timestamp,interval=1):
        "利用k线数据读取指定时间的开盘价作为历史数据"
        myapi = "/derivatives/v3/public/kline"
        if symbol[-1] == 'T':
            category = "linear"
        elif symbol[-1] == 'D':
            category = "inverse"
        params = {"category": category, "symbol": symbol, "interval":str(interval), "start":int(timestamp), "end": int(timestamp + interval*60*1000),"limit":1}
        ans = self.send_request(
            params, api=myapi, method="get"
        )
        price = float(ans['list'][0][1])
        return price

    def get_history_price_v2(self,symbol,timestamp,limit,interval=1):
        "利用k线数据读取指定时间的开盘价作为历史数据"
        myapi = "/derivatives/v3/public/kline"
        if symbol[-1] == 'T':
            category = "linear"
        elif symbol[-1] == 'D':
            category = "inverse"
        params = {"category": category, "symbol": symbol, "interval": interval, "start": int(timestamp),
                  "end": int(timestamp + interval * 60 * 1000), "limit": 1}
        ans = self.send_request(
            params, api=myapi, method="get"
        )

        return ans

