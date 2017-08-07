#!/usr/bin/env python3

import quandl
from IPython import embed

quandl.ApiConfig.api_key = 'h2ho2wNxjkegKK9gtdDu'
data = quandl.get_table('WIKI/PRICES', ticker='FB')
embed()
