import config, os, sys
import websocket, json, requests

#Filenames
# - historic data hist_TICKER.txt    (start of day - erased and refilled for yesterday's info)

#Global variables
#(avg_gain, avg_loss, RSI)
stocks_RSI = {}
stocks_SMA_20 = {}
stocks_SMA_50 = {}

stonks = []


#Websocket Function
#Authorizes login and initializes subscriptions
def on_open(ws):
  print("opened")
  auth_data = {"action": "auth","key": config.API_KEY,"secret": config.SECRET_KEY}
  ws.send(json.dumps(auth_data))
  #update stonks
  get_watchlist()
  #print(stonks)
  for i in stonks:
    print(i)
    print("MA_20: {}".format(stocks_SMA_20[i]))
    print("MA_50: {}".format(stocks_SMA_50[i]))
    print("RSI: {}".format(stocks_RSI[i][2]))
    print()
  print("end")
  # listen_message = {"action": "subscribe","bars": ["AAPL"]}
  # ws.send(json.dumps(listen_message))

#Websocket Function
#Handles all trading and updates trade information
def on_message(ws, message):
  #check watchlist for changes
  get_watchlist()

  ticker = json.loads(message)[0]['S']
  #ticker = message[15:19]
  found = False
  #search through all tickers
  for i in stonks:
    if(i == ticker):
      found = True

  if(found):
    
    data = json.loads(message)
    # print(data)

    #data for graphing
    # timedata = data[0]['t'][0:10] + '-' + data[0]['t'][11:16]
    closedata = data[0]['c']
    highdata = data[0]['h']
    lowdata = data[0]['l']
    opendata = data[0]['o']
    # adjdata = 0
    # volumedata = data[0]['v']
    # testfile =  open("graph_{}.txt".format(ticker), "a")
    # testfile.write(timedata + ',' + str(closedata) + ',' + str(highdata)+ ',' + str(lowdata)+ ',' + str(opendata)+ ',' + str(adjdata) + ',' + str(volumedata) + '\n')
    # testfile.close()

    #data for history and SMA/RSI
    testfile =  open("hist_{}.txt".format(ticker), "a")
    testfile.write(str(closedata) + '\n')
    testfile.close()

    #adjust SMA values
    close_data = load_hist(ticker)
    stocks_SMA_20[ticker] = SMA(20,close_data)
    stocks_SMA_50[ticker] = SMA(50,close_data)
    Update_RSI(ticker, close_data[-1], close_data[-2])

    #show updated values
    # print("SMA_20: {}".format(stocks_SMA_20[ticker]))
    # print("SMA_50: {}".format(stocks_SMA_50[ticker]))
    # print("RSI: {}".format(stocks_RSI[ticker]))

    
    #conduct evaluation for stock
    print(ticker)
    print("SMA_20: {}".format(stocks_SMA_20[ticker]))
    print("SMA_50: {}".format(stocks_SMA_50[ticker]))
    evaluation(ticker)
    print()
  else:
    print(message)

def on_close(ws):
    print("closed connection")


#Moving Average
def SMA(size, close):
  total = 0
  for i in range(size):
    total += close[-i-1]
  average = total / size
  return average

#updates moving average without constant addition
def Update_SMA(total, size, close):
  return total + ((close - total) / size)

#Need to conduct 250 times to ensure accurate
def Update_RSI(stock,curr, prev):
  avg_gain = stocks_RSI[stock][0]
  avg_loss = stocks_RSI[stock][1]
  num = curr - prev
  if(num > 0):
    avg_gain = ((avg_gain * 13) + num) / 14
    avg_loss = (avg_loss * 13) / 14
  else:
    avg_loss = ((avg_loss * 13) + abs(num)) / 14
    avg_gain = (avg_gain * 13) / 14
  rs = avg_gain / avg_loss
  stocks_RSI[stock] = (avg_gain, avg_loss, 100 - (100 / (1+rs)))


#Conducted the first time
def RSI(stock, size, close):
  loss = 0
  gain = 0
  for i in range(1,size+1):
    num = close[i] - close[i-1]
    if(num > 0):
      gain += num
    else:
      loss += abs(num)

  avg_loss = loss / size
  avg_gain = gain / size
  rs = avg_gain / avg_loss
  stocks_RSI[stock] = (avg_gain, avg_loss, 100 - (100 / (1+rs)))

#Fill all historic data text files for one stock
#Due to subscription, can only gather data 15 minutes from current time
def fill_hist(stock):
  #time from 4 hours previous to current time
  r = requests.get(config.CLOCK, headers=config.HEADERS)
  time_end = r.json()['timestamp']
  #make sure time_end isn't subtracting anything
  time_start = '{}{}{}'.format(time_end[0:11], str(int(time_end[11:13]) - 2).zfill(2), time_end[13::])

  

  #need to account for required 15min time delay
  time_end = '{}{}:{}{}'.format(time_end[0:11], str(int(time_end[11:13])).zfill(2), str(int(time_end[14:16])-15).zfill(2), time_end[16::])

  #get data
  bar_url = 'https://data.alpaca.markets/v2/stocks/{}/bars?start={}&end={}&timeframe=1Min'.format(stock,time_start,time_end)
  r = requests.get(bar_url, headers=config.HEADERS)
  #print(r.content)
  #write data
  testfile =  open("hist_{}.txt".format(stock), "w")
  for i in r.json()["bars"]:
    testfile.write('{}\n'.format(i['c']))
  testfile.close()


#Set RSI and SMA values
def set_RSI_SMA(stock):
  #loop through all stocks
  #for stock in stocks:
  #read all close data
  close_data = load_hist(stock)

  #calculate SMA data
  stocks_SMA_20[stock] = SMA(20,close_data)
  stocks_SMA_50[stock] = SMA(50,close_data)

  #calculate RSI data
  #print(close_data[0:])
  RSI(stock, 14, close_data[0:15])
  for i in range(15,len(close_data)):
    Update_RSI(stock, close_data[i], close_data[i-1])


#Load History Data
def load_hist(stock):
  #read all close data
  source_code = open('hist_{}.txt'.format(stock),'r').read()
  close_data = source_code.split('\n')
  #remove last value
  close_data = close_data[0:len(close_data)-1]
  #convert to float
  for num in range(len(close_data)):
    close_data[num] = float(close_data[num])
  return close_data


#Make an order buy/sell
def make_order(stock, side):
  #TODO: Implemnt some sort of risk value to qty/notional values
  data = {
  "symbol": stock,
  "qty": 1,
  "side": side,
  "type": "market",
  "time_in_force": "day"
  }

  r = requests.post(config.ORDER, json=data, headers=config.HEADERS)


#Get current positions
#return quantity of given stock
def get_positions(stock):
  r = requests.get(config.POSITION, headers=config.HEADERS)
  qty = 0
  for i in r.json():
    if(i['symbol'] == stock):
      qty = i['qty']
  return int(qty)

#Get the current items on the watchlist
def get_watchlist():
  #get watchlist
  r = requests.get(config.WATCHLIST, headers=config.HEADERS)
  b = r.json()
  for i in b['assets']:
    stock = i['symbol']
    #if stock is not added then add data
    if(stock not in stonks):
      #add stock to global dictionaries and list
      stonks.append(stock)
      stocks_RSI[stock] = (0,0,0)
      stocks_SMA_20[stock] = 0
      stocks_SMA_50[stock] = 0
      #add data
      fill_hist(stock)
      set_RSI_SMA(stock)
      #add to stream
      listen_message = {"action": "subscribe","bars": [stock]}
      ws.send(json.dumps(listen_message))


#Decide if to buy, sell, hold, or do nothing
def evaluation(stock):
  risk = 1
  qty = get_positions(stock)
  #Alligator method
  #Buy
  if(stocks_SMA_20[stock] > stocks_SMA_50[stock] and qty == 0):
    make_order(stock, 'buy')
    print("Bought 1: {}".format(stock))
  #Sell
  elif(stocks_SMA_20[stock] < stocks_SMA_50[stock] and qty > 0):
    make_order(stock, 'sell')
    print("Sold 1: {}".format(stock))
  # else:
  #   print("NO ACTION")

#fill_hist(stonks[0])
# set_RSI_SMA(stonks)
# get_watchlist()
# print(stocks_SMA_20)
# print(stocks_SMA_50)
# print(stocks_RSI)
# print('done')
ws = websocket.WebSocketApp(config.SOCKET, on_open=on_open, on_message=on_message, on_close=on_close)
ws.run_forever()