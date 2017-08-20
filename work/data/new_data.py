#!/usr/bin/env python3

# Filters:
# start_date

# Params:
# history_period
# predict_period
# n_day_average

# Metadata:
# 3190 tickers
# 14,980,800 entries

import os
import sys

import csv
import collections
import datetime
import numpy as np
import progressbar
import argparse

from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='wiki_10.csv')
parser.add_argument('--history_period', type=int, default=100)
parser.add_argument('--predict_period', type=int, default=50)
parser.add_argument('--avg_period', type=int, default=7)
args = parser.parse_args()

out_dir = os.path.splitext(args.input_path)[0]
data_file = os.path.join('data_{}_{}_{}.npy'.format(
    args.history_period, args.predict_period, args.avg_period))
data_path = os.path.join(out_dir, data_file)
label_file = os.path.join('labels_{}_{}_{}.npy'.format(
    args.history_period, args.predict_period, args.avg_period))
label_path = os.path.join(out_dir, label_file)

class WikiHeader:
    TICKER = 'ticker'
    DATE = 'date'
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'
    CLOSE = 'close'
    VOLUME = 'volume'
    EX_DIVIDEND = 'ex-dividend'
    SPLIT_RATIO = 'split_ratio'
    ADJ_OPEN = 'adj_open'
    ADJ_HIGH = 'adj_high'
    ADJ_LOW = 'adj_low'
    ADJ_CLOSE = 'adj_close'
    ADJ_VOLUME = 'adj_volume'

PARSE_DICT = {
    WikiHeader.OPEN: lambda x: float(x),
    WikiHeader.HIGH: lambda x: float(x),
    WikiHeader.LOW: lambda x: float(x),
    WikiHeader.CLOSE: lambda x: float(x),
    WikiHeader.VOLUME: lambda x: float(x),
    WikiHeader.EX_DIVIDEND: lambda x: float(x),
    WikiHeader.SPLIT_RATIO: lambda x: float(x),
    WikiHeader.ADJ_OPEN: lambda x: float(x),
    WikiHeader.ADJ_HIGH: lambda x: float(x),
    WikiHeader.ADJ_LOW: lambda x: float(x),
    WikiHeader.ADJ_CLOSE: lambda x: float(x),
    WikiHeader.ADJ_VOLUME: lambda x: float(x)
}

class DataFilter(object):

    def __init__(self, data_dict, thresh_date=None):
        # Store the data dict
        self.data_dict = data_dict

        # Cull entries before the threshold date
        if thresh_date is not None:
            for ticker,entry_dict in sorted(self.data_dict.items()):
                for curr_date in sorted(entry_dict):
                    if (curr_date - thresh_date).days < 0:
                        del self.data_dict[ticker][curr_date]

        # Interpolate data for missing dates
        for ticker,entry_dict in sorted(self.data_dict.items()):
            prev_date, prev_entry = None, None
            for curr_date,curr_entry in sorted(entry_dict.items()):
                if prev_date is not None:
                    delta_days = (curr_date - prev_date).days
                    if delta_days > 1:
                        for i in range(1, delta_days):
                            new_entry = {}
                            new_date = prev_date + datetime.timedelta(i)
                            for k in prev_entry.keys():
                                diff = (curr_entry[k] - prev_entry[k])/delta_days
                                new_entry[k] = prev_entry[k]+diff*i
                            self.data_dict[ticker][new_date] = new_entry
                prev_date, prev_entry = curr_date, curr_entry

    def get_start(self, ticker):
        return sorted(self.data_dict[ticker])[0]

    def k_day_avg(self, ticker, date, field, k):
        avg_start = date - datetime.timedelta(k//2)
        tot = 0.0
        for i in range(k):
            d = avg_start+datetime.timedelta(i)
            try:
                tot += self.data_dict[ticker][d][field]
            except KeyError:
                return None
        else:
            avg = tot/k
            return avg

    def get_data(self, ticker, start_date,
        history_period, predict_period, avg_period,
        fields=[WikiHeader.OPEN, WikiHeader.CLOSE, WikiHeader.VOLUME],
        avg_field=WikiHeader.CLOSE):
        """
        Parameters:
            (str) ticker: stock ticker
            (datetime) start_date:  start of period
            (int) history_period: period of historical data
            (int) predict_period: period of gap before prediction
            (int) avg_period: number of days to average over
        Returns:
            (np.array(<float>)) data: historical data
            (int) label: prediction [-1, +1]
            (datetime) end_date: date of last day in predict average
        """
        # Get historical data
        data = np.zeros((len(fields), history_period))
        for i in range(1, history_period):
            dt = start_date+datetime.timedelta(i)
            if dt in self.data_dict[ticker]:
                for j,f in enumerate(fields):
                    data[j][i] = self.data_dict[ticker][dt][f]

        # Get k-day averages for prediction
        history_avg = self.k_day_avg(ticker, 
            start_date+datetime.timedelta(history_period), 
            avg_field, avg_period)
        predict_avg = self.k_day_avg(ticker, 
            start_date+datetime.timedelta(history_period+predict_period), 
            avg_field, avg_period)

        if history_avg is None or predict_avg is None:
            return None
        else:
            # Get prediction from k-day averages
            label = 1 if predict_avg > history_avg else -1

            # Get the datetime of the last day in the predict average
            end_date = (start_date + 
                datetime.timedelta(history_period+predict_period+avg_period//2))

            return data, label, end_date


data_dict = collections.defaultdict(
    lambda : collections.defaultdict(dict)
)
with open(args.input_path, 'r') as csvfile:
    dictreader = csv.DictReader(csvfile)
    for row in dictreader:
        ticker = row[WikiHeader.TICKER]
        date = datetime.datetime.strptime(
            row[WikiHeader.DATE], '%Y-%m-%d')
        for k,f in PARSE_DICT.items():
            data_dict[ticker][date][k] = f(row[k])

tickers = data_dict.keys()
thresh_date = datetime.datetime(2000, 1, 1)
data_filter = DataFilter(data_dict, thresh_date=thresh_date)

data_list = []
label_list = []
with progressbar.ProgressBar(max_value=len(tickers)) as bar:
    for i,ticker in enumerate(tickers):
        start_dt = data_filter.get_start(ticker)
        p,m = 0, 0
        while True:
            result = data_filter.get_data(ticker, start_dt, 
                args.history_period, args.predict_period, args.avg_period)
            if result is None:
                break
            else:
                data, label, start_dt = result
                data_list.append(data)
                label_list.append(label)
                if label == 1:
                    p += 1
                else:
                    m += 1
        #print('{}\t+{}\t-{}'.format(ticker, p, m))
        bar.update(i)
data_arr = np.array(data_list)
label_arr = np.array(label_list)
print(data_arr.shape, label_arr.shape)

# Create out dir if it doesn't exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Save off data
np.save(data_path, data_arr)
np.save(label_path, label_arr)
