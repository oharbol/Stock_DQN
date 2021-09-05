import config, os, sys
import websocket, json, requests

#Filenames
# - historic data hist_TICKER.txt    (start of day - erased and refilled for yesterday's info)

#Global variables
#(avg_gain, avg_loss, RSI)
stocks_RSI = {}
stocks_SMA_20 = {}
stocks_SMA_50 = {}
#heigin ashi bars
#(string, string, string, (float,float,float,float))
#(3rd to Last, 2nd to Last, Last, ohlc)
stocks_HI_Bars = {}

stonks = []
#15 minutes have passed, close data, time started (hour,min)
#(boolean,[Int],(Int,Int)))
stock_memory = {}

#Websocket Function
#Authorizes login and initializes subscriptions
def on_open(ws):
  print("opened")
  auth_data = {"action": "auth","key": config.API_KEY,"secret": config.SECRET_KEY}
  ws.send(json.dumps(auth_data))
  #update stonks
  get_watchlist()
  #print(stonks)
  # for i in stonks: #Move this to set_RSI_SMA
  #   print(i)
  #   print("MA_20: {}".format(stocks_SMA_20[i]))
  #   print("MA_50: {}".format(stocks_SMA_50[i]))
  #   print("RSI: {}".format(stocks_RSI[i][2]))
  #   print()
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

  if(found): #add the boolean value from stock_memory[stock][0]
    data = json.loads(message)

    #data for graphing
    # timedata = data[0]['t'][0:10] + '-' + data[0]['t'][11:16]
    closedata = data[0]['c']
    highdata = data[0]['h']
    lowdata = data[0]['l']
    opendata = data[0]['o']
    # testfile =  open("graph_{}.txt".format(ticker), "a")
    # testfile.write(timedata + ',' + str(closedata) + ',' + str(highdata)+ ',' + str(lowdata)+ ',' + str(opendata)+ ',' + str(adjdata) + ',' + str(volumedata) + '\n')
    # testfile.close()

    #data for history and SMA/RSI
    testfile =  open("hist_{}.txt".format(ticker), "a")
    testfile.write('{},{},{},{}\n'.format(opendata,highdata,lowdata,closedata))
    testfile.close()

    #adjust SMA values
    # close_data = load_hist(ticker)
    #stocks_SMA_20[ticker] = SMA(20,close_data) # MAYBE I SHOULDN"T HAVE ENABLED THIS!!!!!
    # stocks_SMA_50[ticker] = SMA(50,close_data)
    # Update_RSI(ticker, close_data[-1], close_data[-2])
    ohlc = (opendata,highdata,lowdata,closedata)
    update_HI(ticker, ohlc)

    #conduct evaluation for stock
    # print(ticker)
    # print("SMA_20: {}".format(stocks_SMA_20[ticker]))
    # print("SMA_50: {}".format(stocks_SMA_50[ticker]))
    evaluation(ticker)
    #print()
  #going to contain dead_time function
  #elif(found):
    #dead_time(stock,json.loads(message)[0]['c'])
    

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
  time_start = '{}{}{}'.format(time_end[0:11], str(int(time_end[11:13]) - 6).zfill(2), time_end[13::])

  temp_min = int(time_end[14:16]) - 15
  temp_hr = int(time_end[11:13])
  #time in previous hour
  if(temp_min < 0):
    time_end = '{}{}:{}{}'.format(time_end[0:11], str(temp_hr-1).zfill(2), str(temp_min + 60).zfill(2), time_end[16::])
  #time in current hour
  else:
    #need to account for required 15min time delay
    time_end = '{}{}:{}{}'.format(time_end[0:11], str(temp_hr).zfill(2), str(temp_min).zfill(2), time_end[16::])

  #get data
  bar_url = 'https://data.alpaca.markets/v2/stocks/{}/bars?start={}&end={}&timeframe=1Min'.format(stock,time_start,time_end)
  r = requests.get(bar_url, headers=config.HEADERS)
  #print(r.content)
  #write data
  testfile =  open("hist_{}.txt".format(stock), "w")
  for i in r.json()["bars"]:
    #ohlc
    testfile.write('{},{},{},{}\n'.format(i['o'],i['h'],i['l'],i['c']))
  testfile.close()
  print(stock)


#Set RSI and SMA values
def set_RSI_SMA(stock):
  #loop through all stocks
  #for stock in stocks:
  #read all close data
  ohlc = load_hist(stock)
  close_data = []
  for i in ohlc:
    close_data.append(i[3])
  #calculate SMA data
  stocks_SMA_20[stock] = SMA(20,close_data)
  #stocks_SMA_50[stock] = SMA(50,close_data)
  print("MA_20: {}".format(stocks_SMA_20[stock]))
  #calculate RSI data
  
  # RSI(stock, 14, close_data[0:15])
  # for i in range(15,len(close_data)):
  #   Update_RSI(stock, close_data[i], close_data[i-1])


#Evaluates HI bar data
def set_HI(stock, last):
  # Get this tested and working!!!!!!!!!!!!!
  # ohlc = load_hist(stock)[-4]
  # stocks_HI_Bars[stock] = ('r', 'r', 'r', (ohlc[0][0],ohlc[0][1],ohlc[0][2],ohlc[0][3]))
  # for i in range(1,4):
  #   update_HI(stock,(ohlc[i][0],ohlc[i][1],ohlc[i][2],ohlc[i][3]))

  stocks_HI_Bars[stock] = ('r', 'r', 'r', (last[0],last[1],last[2],last[3]))
  print(stocks_HI_Bars[stock])


def update_HI(stock, ohlc):
  ohlc_HI = stocks_HI_Bars[stock][3]
  openp = ohlc[0]
  highp = ohlc[1]
  lowp = ohlc[2]
  closep = ohlc[3]

  highi = max([highp,openp,closep])
  lowi = min([lowp,openp,closep])
  closei = (openp + highp + lowp+closep) / 4
  openi = (ohlc_HI[0]+ohlc_HI[3]) / 2
  color = 'r'
  if(closei > openi):
    color = 'g'

  if(stocks_HI_Bars[stock][2] == 'g' and color == 'r'):
    color = 'dr'
  elif(stocks_HI_Bars[stock][2] == 'r' and color == 'g'):
    color = 'dg'

  stocks_HI_Bars[stock] = (stocks_HI_Bars[stock][1],stocks_HI_Bars[stock][2],color, (openi,highi,lowi,closei))
  print(stock)
  print(stocks_HI_Bars[stock])

#Load History Data
def load_hist(stock):
  #read all close data
  source_code = open('hist_{}.txt'.format(stock),'r').read()
  close_data = source_code.split('\n')
  #remove last value
  close_data = close_data[0:len(close_data)-1]
  #convert to float
  # for num in range(len(close_data)):
  #   close_data[num] = float(close_data[num])
  ohlc = []
  for i in close_data:
    openc,highc,lowc,closec = i.split(',')
    ohlc.append((float(openc),float(highc),float(lowc),float(closec)))
  #return close_data
  return ohlc


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

  #new // TEST THIS!!!///
  #if(len(b['assets']) != len(stonks)):
  for i in b['assets']:
    stock = i['symbol']
    #if stock is not added then add data
    if(stock not in stonks):
      #add stock to global dictionaries and list
      stonks.append(stock)
      stocks_RSI[stock] = (0,0,0)
      stocks_SMA_20[stock] = 0
      stocks_SMA_50[stock] = 0
      stocks_HI_Bars[stock] = ('r','r','r',(0,0,0,0))
      #NEW!!!!
      # r = requests.get(config.CLOCK, headers=config.HEADERS)
      # curr_time = r.json()['timestamp']
      # stock_memory[stock] = (False, [], (int(curr_time[11:13]),int(curr_time[14:16])))
      #add data
      fill_hist(stock)
      #cannot set this value with the dead_time function active
      #set_RSI_SMA(stock) #delete this!!
      set_HI(stock, load_hist(stock)[-1])

      #add to stream
      listen_message = {"action": "subscribe","bars": [stock]}
      ws.send(json.dumps(listen_message))
      print("watchlist")


#Decide if to buy, sell, hold, or do nothing
def evaluation(stock):
  risk = 1
  qty = get_positions(stock)
  #Alligator method
  #Buy

  # if(stocks_SMA_20[stock] > stocks_SMA_50[stock] and qty == 0):
  #   make_order(stock, 'buy')
  #   print("Bought {}: {}\n".format(risk,stock))
  # #Sell
  # elif(stocks_SMA_20[stock] < stocks_SMA_50[stock] and qty > 0):
  #   make_order(stock, 'sell')
  #   print("Sold {}: {}\n".format(risk,stock))
  # else:
  #   print("NO ACTION")

  #HI Mehtod
  if(stocks_HI_Bars[stock][1] == 'dg' and stocks_HI_Bars[stock][2] == 'g' and qty == 0):
    make_order(stock, 'buy')
    print("Bought {}: {}\n".format(risk,stock))

  elif(((stocks_HI_Bars[stock][1] == 'dr' and stocks_HI_Bars[stock][2] == 'r') or
      (stocks_HI_Bars[stock][0] == 'dr' and stocks_HI_Bars[stock][2] == 'g' and stocks_HI_Bars[stock][2] == 'dr')) and
      qty > 0):

    make_order(stock, 'sell')
    print("Sold {}: {}\n".format(risk,stock))

  #HI with MA_20
  # close_price = stocks_HI_Bars[stock][3]
  # if(stocks_SMA_20[stock] < close_price and qty == 0):
  #   make_order(stock, 'buy')
  #   print("Bought {}: {}\n".format(risk,stock))

  # elif(stocks_SMA_20[stock] > close_price and qty == 0):
  #   make_order(stock, 'sell')
  #   print("Sold {}: {}\n".format(risk,stock))


ws = websocket.WebSocketApp(config.SOCKET, on_open=on_open, on_message=on_message, on_close=on_close)
ws.run_forever()

#functionality to handle adding data during 15 min "dead-time"
# def dead_time(stock, close):
#   #get current time
#   r = requests.get(config.CLOCK, headers=config.HEADERS)
#   curr_time = r.json()['timestamp']
#   time_hour = int(curr_time[11:13])
#   time_min = int(curr_time[14:16])
#   time_end = ""
#   time_start = ""

#   #15 minutes has been reached
#   if(stock_memory[stock][2][0] - time_hour > 1 or
#     (stock_memory[stock][2][0] - time_hour == 1 and 60 - time_min + stock_memory[stock][2][1]) >= 15 or
#     (stock_memory[stock][2][0] - time_hour == 0 and stock_memory[stock][2][1] - time_min >= 15)):
#   #if(time_hour - stock_memory[stock][2][0] > 1 or (time_hour - stock_memory[stock][2][0] < 2 and 60 - time_min + stock_memory[stock][2][1]) >= 15):
#     #gather 15 minute data from 30 minutes ago to 15 minutes ago
#     temp_min = int(curr_time[14:16]) - 15
#     temp_min_2 = int(curr_time[14:16]) - 30
#     temp_hr = int(curr_time[11:13])

#     #both start and end go past the hour
#     if(temp_min < 0):
#       time_end = '{}{}:{}{}'.format(curr_time[0:11], str(temp_hr-1).zfill(2), str(temp_min + 60).zfill(2), curr_time[16::])
#       time_start = '{}{}:{}{}'.format(curr_time[0:11], str(temp_hr-1).zfill(2), str(temp_min_2 + 60).zfill(2), curr_time[16::])
#     #only start goes past the hour
#     elif(temp_min_2 < 0):
#       time_end = '{}{}:{}{}'.format(curr_time[0:11], str(temp_hr).zfill(2), str(temp_min).zfill(2), curr_time[16::])
#       time_start = '{}{}:{}{}'.format(curr_time[0:11], str(temp_hr-1).zfill(2), str(temp_min_2 + 60).zfill(2), curr_time[16::])
#     #both times are within the hour
#     else:
#       time_end = '{}{}:{}{}'.format(curr_time[0:11], str(temp_hr).zfill(2), str(temp_min).zfill(2), curr_time[16::])
#       time_start = '{}{}:{}{}'.format(curr_time[0:11], str(temp_hr).zfill(2), str(temp_min_2).zfill(2), curr_time[16::])


  #   bar_url = 'https://data.alpaca.markets/v2/stocks/{}/bars?start={}&end={}&timeframe=1Min'.format(stock,time_start,time_end)
  #   r = requests.get(bar_url, headers=config.HEADERS)
  #   #write gathered history data onto hist_stock
  #   testfile =  open("hist_{}.txt".format(stock), "a")
  #   for i in r.json()["bars"]:
  #     testfile.write('{}\n'.format(i['c']))
    
  #   #write data from deadtime_stock onto hist_stock
  #   for j in stock_memory[stock][1]:
  #     testfile.write('{}\n'.format(j))
  #   testfile.close()
  #   #call:  set_RSI_SMA(stock)
  #   set_RSI_SMA(stock)
  #   #set the stock memory to true
  #   stock_memory[stock][0] = True

  # else:
  #   print("Data added at time")
  #   print(curr_time)
  #   stock_memory[stock][1].append(close)