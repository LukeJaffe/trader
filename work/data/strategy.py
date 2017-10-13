#!/usr/bin/env python3

import datetime

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)

class Strategy:
    def __init__(self, stock_data=None, start_fund=None, trade_fee=None, 
        min_active=None, max_active=None,
        fund_interval=None, fund_amount=None,
        init_rule=None, init_num=None, init_return_interval=None,
        sell_rule=None, buy_rule=None,
        start_date=None, end_date=None):
        self.stock_date = stock_data
        self.balance = start_fund
        self.trade_fee = trade_fee
        self.fund_interval = fund_interval
        self.fund_amount = fund_amount
        self.start_date = start_date
        self.end_date = end_date
        self.init_rule = init_rule
        self.sell_rule = sell_rule
        self.buy_rule = buy_rule

    def _init(self):
        # Pick init_num stocks at random
        if self.init_rule == 'random':
            pass
        # Pick the init_num top stocks by return over the init_return_interval
        elif self.init_rule == 'best_return':
            pass
        else:
            raise Exception('Invalid initialization rule: {}'.format(
                self.init_rule))

    def _fund(self, curr_date):
        if self.fund_interval is not None:
            if curr_date - self.start_date % self.fund_interval == 0:
                self.balance += fund_amount

    def _sell(self):
        if self.sell_rule == 'basic':
            pass
        else:
            raise Exception('Invalid sell rule: {}'.format(self.sell_rule))

    def _buy(self):
        if self.buy_rule == 'basic':
            pass
        else:
            raise Exception('Invalid buy rule: {}'.format(self.buy_rule))

    def _check_open(self, curr_date):
        if curr_date.weekday() in [0, 1, 2, 3, 4]:
            return True
        else:
            return False

    def _trade(self, curr_date):
        self._fund(curr_date)
        self._sell()
        self._buy()

    def backtest(self):
        self._init()
        for curr_date in daterange(self.start_date, self.end_date):
            if self._check_open(curr_date):
                print(curr_date)        
                self._trade(curr_date)    

if __name__=='__main__':
    start_date = datetime.datetime(2000, 1, 1)
    end_date = datetime.datetime(2000, 2, 1)
    strategy = Strategy(start_fund=10000, trade_fee=10, 
        min_active=1, max_active=10, 
        init_rule='random', sell_rule='basic', buy_rule='basic',
        start_date=start_date, end_date=end_date)
    strategy.backtest()  
