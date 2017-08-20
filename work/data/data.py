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
import enum

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

class WikiHeader(enum.Enum):
    TICKER = 0
    DATE = 1
    OPEN = 2
    HIGH = 3
    LOW = 4
    CLOSE = 5
    VOLUME = 6
    EX_DIVIDEND = 7
    SPLIT_RATIO = 8
    ADJ_OPEN = 9
    ADJ_HIGH = 10
    ADJ_LOW = 11
    ADJ_CLOSE = 12
    ADJ_VOLUME = 13

class DataFilter(object):

    def __init__(self, data_dict):
        self.data_dict = data_dict

    def get_start(self, ticker):
        return self.data_dict[ticker][0][0]

    def get_data(self, ticker, start_date, 
        history_period, predict_period, avg_period):
        """
        Parameters:
            (str) ticker: stock ticker
            (datetime) start_date:  start of period
            (int) history_period: period of historical data
            (int) predict_period: period of gap before prediction
            (int) avg_period: number of days to average over
        Returns:
            (list[float]) history: historical data
            (int) label: prediction [-1, +1]
            (datetime) end_date: date of last day in predict average
        """
        raw = self.data_dict[ticker]
        dt_list, cp_list = zip(*raw)
        try:
            idx = dt_list.index(start_date)
        except ValueError:
            # Interpolate to get start date 
            # Find nearest dates before and after
            new_start = start_date
            while True:
                try:
                    start_idx = dt_list.index(new_start)
                except ValueError:
                    new_start -= datetime.timedelta(days=1)
                else:
                    _, start_close = raw[start_idx]
                    break

            new_end = start_date
            while True:
                try:
                    end_idx = dt_list.index(new_end)
                except ValueError:
                    new_end += datetime.timedelta(days=1)
                else:
                    _, end_close = raw[end_idx]
                    break

            delta = (new_end - new_start).days
            close_delta = (end_close - start_close)/(delta - 1)
            day_delta = (start_date - new_start).days
            curr_close = end_close + day_delta*close_delta
            raw.insert(end_idx, (start_date, curr_close))
            idx = end_idx
                
        prev_date = start_date
        history = []
        while(len(history) < history_period+predict_period+avg_period//2):
            try:
                curr_date, curr_close = raw[idx]
            except IndexError:
                return None
            delta = (curr_date - prev_date).days
            # If there was more than one day between elements
            if delta > 1:
                # Interpolate data for missing days
                close_delta = (curr_close - prev_close)/(delta - 1)
                for i in range(1, delta-1):
                    history.append(prev_close+i*close_delta) 
            history.append(curr_close)
            prev_date, prev_close = curr_date, curr_close
            idx += 1

        # Get history average
        start_records = history[history_period-avg_period//2:history_period+avg_period//2+avg_period%2]
        start_avg = sum(start_records)/len(start_records)
        # Get predict average
        end_records = history[predict_period-avg_period//2:predict_period+avg_period//2+avg_period%2]
        end_avg = sum(end_records)/len(end_records)
        # Get label
        label = 1 if start_avg < end_avg else -1
        # Get end date + 1 (start date of next query)
        end_date = start_date + datetime.timedelta(days=history_period+predict_period+avg_period//2) 

        return history[:history_period], label, end_date


data_dict = collections.defaultdict(list)
with open(args.input_path, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    headers = next(datareader)
    for row in datareader:
        dt = datetime.datetime.strptime(row[WikiHeader.DATE.value], '%Y-%m-%d')
        try:
            close = float(row[WikiHeader.CLOSE.value])
        except:
            pass
        else:
            data_dict[row[WikiHeader.TICKER.value]].append((dt, close))

tickers = data_dict.keys()
data_filter = DataFilter(data_dict)
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
