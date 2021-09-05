import json
import config, requests
import alpaca_trade_api as tradeapi
from datetime import datetime
from Stock_Trader_Alligator import *


me = Stock_Trader(False, 4)
me.give_stock('AAPL')
data = open('data/AAPL/AAPL_2021-6-30.txt','r').read().split('\n')
close = []
data.pop()
for i in data:
  i = i.split(',')
  close.append(float(i[4]))

me.create_SMMA('AAPL', close)
print(me.SMMA)
#print(me.SMMA_offset)

new_close = close[13:]

for i in new_close:
  me.update_SMMA('AAPL',i)
  print()
  print(me.SMMA)
  #print(me.SMMA_offset)

# a = 'asdf,fdsa'
# print(a.split(','))
# l = [1,2,3,4,5,6,7]
# for i in l[1::]:
#   print(i)
# me = Stock_Trader(False)

# me.get_watchlist()
# me.print_date()
# BARS_URL = 'https://data.alpaca.markets/v2/stocks/OCGN/bars'

# minuteBars = BARS_URL + '?start=2021-05-03T07:30:27.265251004-04:00&end=2021-05-03T15:00:27.265251004-04:00&limit=200&timeframe=1Min'
# order = config.BASE_URL + '/v2/orders'
# clock = 'https://paper-api.alpaca.markets/v2/clock'

# a = {
#   "AAPL" : ('r', 'r', 'r', (0,0,0,0))
# }

# print(a["AAPL"])
# print(a["AAPL"][0])
# print(a["AAPL"][3])
# a["AAPL"][1] = 'r'
# source_code = open('data.txt','r').read().split('\n')
# for i in source_code:
#   a, b = i.split(',')
#   print(a)
#   print(b)
# stock_memory = {
#   "AAPL" : (False, [1,2,3,4,5,6], (23,18))
# }

# stock = 'AAPL'
# r = requests.get(config.CLOCK, headers=config.HEADERS)
# curr_time = r.json()['timestamp']

# time_hour = int(curr_time[11:13])
# time_min = int(curr_time[14:16])
# print(curr_time)
# print(stock_memory[stock][2][0] - time_hour)
# print(60 - time_min + stock_memory[stock][2][1])
# print(stock_memory[stock][2][1] - time_min)

# #15 minutes has been reached
# if(stock_memory[stock][2][0] - time_hour > 1 or
#   (stock_memory[stock][2][0] - time_hour == 1 and 60 - time_min + stock_memory[stock][2][1]) >= 15 or
#   (stock_memory[stock][2][0] - time_hour == 0 and stock_memory[stock][2][1] - time_min >= 15)):
#   print("Correct")
# else:
#   print("Need More Time")

# r = requests.get(config.POSITION, headers=config.HEADERS)
# for i in r.json():
#   print(i['symbol'] + " " + i['qty'])


# r = requests.get(clock, headers=config.HEADERS)
# time_end = r.json()['timestamp']
# #print(int(time_end[14]))
# time_start = '{}{}{}'.format(time_end[0:11], str(int(time_end[11:13]) - 4).zfill(2), time_end[13::])
# time_end = '{}{}:{}{}'.format(time_end[0:11], str(int(time_end[11:13])).zfill(2), str(int(time_end[14:16])-1).zfill(2), time_end[16::])

# print(time_start)
# print(time_end)

# minuteBars = BARS_URL + '?start={}&end={}&timeframe=1Min'.format(time_start,time_end)
# now = datetime.now()
 
# # dd/mm/YY H:M:S
# dt_string = now.strftime("%Y-%m-%dT%H:%M:00Z")
# print("date and time =", dt_string)
# r = requests.get(config.WATCHLIST, headers=config.HEADERS)
# b = r.json()
# print(len(b['assets']))
# for i in b['assets']:
#   print(i['symbol'])




#r = requests.get(config.POSITION, headers=config.HEADERS)
# r = requests.get(config.ORDER, headers=config.HEADERS)
# b = json.loads(r.content)
# for i in b:
#   print("\n")
#   print(i)






# r = requests.get(minuteBars, headers=config.HEADERS)
# data = r.json()['bars']
# testfile =  open("samplefile.txt", "a")
# for i in data:
#   testfile.write('{},{}\n'.format(i['t'],i['c']))
# testfile.close()


# print(json.dumps(r.json(),indent=2))
# testfile =  open("OCGN_hist.txt", "w")
# for i in r.json()["bars"]:
#   testfile.write(str(i['c']) + '\n')
# testfile.close()
# data = {
#   "symbol": "MSFT",
#   "notional": 200,
#   "side": "buy",
#   "type": "market",
#   "time_in_force": "day"
# }

# r = requests.post(order, json=data, headers=config.HEADERS)

# print(json.loads(r.content))





# for data in r.json()['bars']:
#   timedata = data['t'][0:10] + '-' + data['t'][11:16]
#   closedata = data['c']
#   highdata = data['h']
#   lowdata = data['l']
#   opendata = data['o']
#   adjdata = 0
#   volumedata = data['v']
#   testfile =  open("OCGNData.txt", "a")
#   testfile.write(timedata + ',' + str(closedata) + ',' + str(highdata)+ ',' + str(lowdata)+ ',' + str(opendata)+ ',' + str(adjdata) + ',' + str(volumedata) + '\n')
#   testfile.close()

# data = json.loads(json.dumps(r.json()))
# for i in data['MSFT']:
#   ts = int(i['t'])
#   print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))


# api = tradeapi.REST(
#         'PKN4SBCEB2XTGPG43UIW',
#         'wboLhIVjVmwwKSkTmrQur6iMdUMYSxl2WA1u3zAz',
#         'https://paper-api.alpaca.markets'
#     )

# api.submit_order(
#     symbol='AAPL',
#     qty=1,
#     side='buy',
#     type='market',
#     time_in_force='gtc'
# )

# # Get our account information.
# account = api.get_account()
# clock = api.get_clock()
# print(clock.timestamp)

# # Check if our account is restricted from trading.
# if account.trading_blocked:
#     print('Account is currently restricted from trading.')

# # Check how much money we can use to open new positions.
# print('${} is available as buying power.'.format(account.buying_power))
# print('${}'.format(account.portfolio_value))

# # Get daily price data for AAPL over the last 5 trading days.
# barset = api.get_barset('AAPL', 'minute', limit=5)
# #print(barset)

# bigbird = api.get_bars('AAPL', 'day', timeframe='1Min', limit=5)
# bigbird = bigbird._raw
# data = json.loads(json.dumps(bigbird))
# for i in data['AAPL']:
#   ts = int(i['t'])
#   print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
#print(type(bigbird._raw))