import json
import config, requests
import alpaca_trade_api as tradeapi
from datetime import datetime
#from Stock_Trader import *
import time



# r = requests.get(config.CLOCK, headers=config.HEADERS)
# time_end = r.json()['timestamp']
# print(time_end)
o_time = (2021,9,17)
num_weeks = 4
months = [31,28,31,30,31,30,31,31,30,31,30,31]
stock = 'AAPL'
year = o_time[0]
month = o_time[1]
day = o_time[2]
ctime_start = '{}-{}-{}T9:30:00-04:00'.format(year,str(month).zfill(2),str(day).zfill(2))
ctime_end = '{}-{}-{}T16:00:00-04:00'.format(year,str(month).zfill(2),str(day).zfill(2))

#stock trader
# Boa = Stock_Trader(False,4)
# wallet = 500
# last = 0

def increment_date(year, month, day):
  week_date = datetime.strptime('{} {} {}'.format(day, month, year), '%d %m %Y').weekday()
  if(week_date == 4):
    day += 3
  else:
    day += 1

  if(months[month-1] < day):
    day = day - months[month-1]
    month += 1
    if(month > 12):
      month = 1
      year += 1

  return (year,month,day)

#2021-05-09
# year = 2021
# month = 6
# day = 30
# print('{}-{}-{}'.format(year,month,day))
# for i in range(0,80):
#   year,month,day = increment_date(year,month,day)
#   print('{}-{}-{}'.format(year,month,day))
# print(increment_date(2021, 5, 9))


##get callibration data
# bar_url = 'https://data.alpaca.markets/v2/stocks/{}/bars?start={}&end={}&timeframe=1Min'.format(stock,ctime_start,ctime_end)
# r = requests.get(bar_url, headers=config.HEADERS)
# #print(r.content)
# #write data
# testfile =  open("data/{}/callibration_{}.txt".format(stock,stock), "w")
# for i in r.json()["bars"]:
#   #ohlc
#   testfile.write('{},{},{},{},{}\n'.format(i['t'],i['o'],i['h'],i['l'],i['c']))
# testfile.close()
# print(stock)

#new start of time data
for i in range(1,5 * num_weeks):
  if(i % 190 == 0):
    time.sleep(65)
  year,month,day = increment_date(year,month,day)
  ctime_start = '{}-{}-{}T9:30:00-04:00'.format(year,str(month).zfill(2),str(day).zfill(2))
  ctime_end = '{}-{}-{}T15:59:00-04:00'.format(year,str(month).zfill(2),str(day).zfill(2))
  # print(ctime_start)
  # print(ctime_end)
  # print()

  #get testing data
  bar_url = 'https://data.alpaca.markets/v2/stocks/{}/bars?start={}&end={}&timeframe=1Min'.format(stock,ctime_start,ctime_end)
  r = requests.get(bar_url, headers=config.HEADERS)
  #print(r.content)
  #print('{}-{}-{}'.format(year,month,day))
  testfile =  open("data/{}/{}_{}-{}-{}.txt".format(stock,stock,year,month,day), "w")
  for i in r.json()["bars"]:
    #ohlc
    testfile.write('{},{},{},{},{}\n'.format(i['t'],i['o'],i['h'],i['l'],i['c']))
  testfile.close()


#Start of Trading Data


#read from calibration data and update
# ohlc = []
# callibration = open("data/{}/callibration_{}.txt".format(stock,stock)).read().split('\n')
# callibration = callibration[0:-1]
# for j in callibration:
#   #print(j.split(','))
#   date,openp,highp,lowp,closep = j.split(',')
#   ohlc.append((date,float(openp),float(highp),float(lowp),float(closep)))


# # Boa.fill_HI_data(stock, ohlc)
# # Boa.fill_MA_20_data(stock, ohlc)
# # Boa.fill_MA_50_data(stock, ohlc)
# # Boa.fill_WMA_14(stock, ohlc)
# # Boa.fill_RSI_data(stock, ohlc)

# # print(Boa.stocks_WMA_14[stock][0])
# #backtest
# print(wallet)

# year = o_time[0]
# month = o_time[1]
# day = o_time[2]



# curr_ohlc = 0
# previous_close = 0

# #file to write back test result to
# testfile =  open("data/{}_BTResults.txt".format(stock), "w")
# #Hold AH
# action = 'hold'
# num_hold = 0
# for i in range(1, 5 * num_weeks):
#   #Sell AH
#   #action = 'hold'
#   #num_hold = 0

#   year,month,day = increment_date(year,month,day)
#   ctime_start = '{}-{}-{}T9:30:00-04:00'.format(year,str(month).zfill(2),str(day).zfill(2))
#   ctime_end = '{}-{}-{}T16:00:00-04:00'.format(year,str(month).zfill(2),str(day).zfill(2))

#   testfile.write('{}-{}-{}'.format(year,month,day))

#   temp = open("data/{}/{}_{}-{}-{}.txt".format(stock,stock,year,month,day))
#   histdata = temp.read().split('\n')
#   #Disable when 5min
#   histdata = histdata[0:-1]

#   #loop over day data
#   for j in histdata:
#     date,openp,highp,lowp,closep = j.split(',')
#     new_ohlc = (date,float(openp),float(highp),float(lowp),float(closep))


#     if(action == 'buy'):
#       #testfile.write(',{},'.format(closep))
#       wallet -= float(openp)
#       num_hold = 1
#     elif(action == 'sell'):
#       #testfile.write('{}\n'.format(closep))
#       wallet += float(openp)
#       num_hold = 0

#     # Boa.update_MA_20(stock, float(closep))
#     # Boa.update_MA_50(stock, float(closep))
#     # Boa.update_HI(stock, new_ohlc)
#     # Boa.update_RSI(stock, float(closep), previous_close)
#     # Boa.update_WMA(stock, float(closep))
#     action = Boa.evaluation(stock)
    
#     # print(action)
#     # print(Boa.stocks_RSI[stock])
#     # print(Boa.stocks_SMA_20[stock])
#     # print(Boa.stocks_HI_Bars[stock])
#     # print(num_hold)
#     # print(Boa.paperqty)
#     # print(wallet)
#     previous_close = float(closep)


#   #after one day
#   temp.close()
#   #print(num_hold)

#   #Sell after day
#   # if(num_hold == 1):
#   #   wallet += float(closep)
#   #Boa.set_paperqty(0)

#   #holding over AH
#   if(num_hold == 1):
#     testfile.write(',{}\n'.format(round(wallet + float(closep),2)))
#     #print(wallet + float(closep))
#   else:
#     testfile.write(',{}\n'.format(round(wallet,2)))
#     #print(wallet)
#   #print(wallet)
  
# testfile.close()