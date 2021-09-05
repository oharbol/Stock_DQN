import config, os, sys
import websocket, json, requests
from datetime import datetime

class Stock_Trader:

  #Constructor
  #stream: Boolean
  def __init__(self, stream, strat):
    self.stream = stream
    self.strat = strat
    #(avg_gain, avg_loss, RSI)
    # self.stocks_RSI = {}
    # self.stocks_SMA_20 = {}
    # self.stocks_SMA_50 = {}
    # self.stocks_WMA_14 = {}
    self.stocks_SMMA = {}
    #heigin ashi bars
    #(string, string, string, (float,float,float,float))
    #(3rd to Last, 2nd to Last, Last, ohlc)
    # self.stocks_HI_Bars = {}
    #(boolean,[Int],(Int,Int)))
    self.stock_memory = {}
    self.stonks = []
    self.paperqty = 0

    if(stream):
      print("opened")
      auth_data = {"action": "auth","key": config.API_KEY,"secret": config.SECRET_KEY}
      ws.send(json.dumps(auth_data))

      self.get_watchlist()

  #Paper trading function
  #Takes a given stock and adds empty data to global
  def give_stock(self, stock):
    self.stonks.append(stock)
    self.stocks_SMMA[stock] = (0,0,0, {})
    # self.stocks_RSI[stock] = (0,0,0)
    # self.stocks_SMA_20[stock] = (0 , [])
    # self.stocks_SMA_50[stock] = (0 , [])
    # self.stocks_WMA_14 = (0 , [])
    # self.stocks_HI_Bars[stock] = ('r','r','r',(0,0,0,0))


  #Get the current items on the watchlist
  def get_watchlist(self):
    #get watchlist
    r = requests.get(config.WATCHLIST, headers=config.HEADERS)
    b = r.json()

    for i in b['assets']:
      stock = i['symbol']
      #if stock is not added then add data
      if(stock not in self.stonks):
        #add stock to global dictionaries and list
        self.stonks.append(stock)
        self.stocks_RSI[stock] = (0,0,0)
        self.stocks_SMA_20[stock] = 0
        self.stocks_SMA_50[stock] = 0
        self.stocks_HI_Bars[stock] = ('r','r','r',(0,0,0,0))

        self.print_info(stock)


  #Fill all historic data text files for one stock
  #Due to subscription, can only gather data 15 minutes from current time

  #ohlc_list needs to be length 2 or greater
  def fill_HI_data(self, stock, ohlc_list):
    last = ohlc_list[0]
    self.stocks_HI_Bars[stock] = ('r', 'r', 'r', (last[1],last[2],last[3],last[4]))

    for i in ohlc_list[1::]:
      self.update_HI(stock, i)
    print(self.stocks_HI_Bars[stock])
  

  #sets the MA_20 data
  def fill_MA_20_data(self, stock, ohlc_list):
    last = ohlc_list[-21:-1]
    close_data = []
    total = 0
    for i in last:
      total += i[4]
      close_data.append(i[4])

    self.stocks_SMA_20[stock] = (total / 20, close_data)
    print( self.stocks_SMA_20[stock])

  #sets the MA_50 data
  def fill_MA_50_data(self, stock, ohlc_list):
    last = ohlc_list[-51:-1]
    close_data = []
    total = 0
    for i in last:
      total += i[4]
      close_data.append(i[4])

    self.stocks_SMA_50[stock] = (total / 50, close_data)
    print( self.stocks_SMA_50[stock])


  #Conducted the first time
  def fill_RSI_data(self, stock, ohlc_list):
    loss = 0
    gain = 0
    for i in range(1,15):
      num = ohlc_list[i][4] - ohlc_list[i-1][4]
      if(num > 0):
        gain += num
      else:
        loss += abs(num)

    avg_loss = loss / 14
    avg_gain = gain / 14
    rs = avg_gain / avg_loss
    self.stocks_RSI[stock] = (avg_gain, avg_loss, 100 - (100 / (1+rs)))
    
    for i in range(14+1, len(ohlc_list)-13):
      self.update_RSI(stock, ohlc_list[i][4], ohlc_list[i-1][4])
    print(self.stocks_RSI[stock])


  #Weighted Moving Average
  def fill_WMA_14(self, stock, ohlc_list):
    denom = 105
    total = 0
    new_list = []
    for i in range(1, 15):
      meme = ohlc_list[-i][4]
      total += (meme * (15 - i)) / denom
      new_list.append(meme)
    self.stocks_WMA_14[stock] = (total, new_list)




  #Updates the values of the HI each time
  def update_HI(self, stock, ohlc):
    ohlc_HI = self.stocks_HI_Bars[stock][3]
    openp = ohlc[1]
    highp = ohlc[2]
    lowp = ohlc[3]
    closep = ohlc[4]

    highi = max([highp,openp,closep])
    lowi = min([lowp,openp,closep])
    closei = (openp + highp + lowp+closep) / 4
    openi = (ohlc_HI[0]+ohlc_HI[3]) / 2
    color = 'r'
    if(closei > openi):
      color = 'g'

    if(self.stocks_HI_Bars[stock][2] == 'g' and color == 'r'):
      color = 'dr'
    elif(self.stocks_HI_Bars[stock][2] == 'r' and color == 'g'):
      color = 'dg'

    self.stocks_HI_Bars[stock] = (self.stocks_HI_Bars[stock][1],self.stocks_HI_Bars[stock][2],color, (openi,highi,lowi,closei))
    #print(self.stocks_HI_Bars[stock])


  def update_MA_20(self, stock, closep):
    vals = self.stocks_SMA_20[stock][1]
    vals.append(closep)
    vals.remove(vals[0])
    self.stocks_SMA_20[stock] = (sum(vals) / 20,vals)


  def update_MA_50(self, stock, closep):
    vals = self.stocks_SMA_50[stock][1]
    vals.append(closep)
    vals.remove(vals[0])
    self.stocks_SMA_50[stock] = (sum(vals) / 50,vals)


  def update_RSI(self, stock, curr, prev):
    avg_gain = self.stocks_RSI[stock][0]
    avg_loss = self.stocks_RSI[stock][1]
    num = curr - prev
    if(num > 0):
      avg_gain = ((avg_gain * 13) + num) / 14
      avg_loss = (avg_loss * 13) / 14
    else:
      avg_loss = ((avg_loss * 13) + abs(num)) / 14
      avg_gain = (avg_gain * 13) / 14
    rs = avg_gain / avg_loss
    self.stocks_RSI[stock] = (avg_gain, avg_loss, 100 - (100 / (1+rs)))

  def update_WMA(self, stock, closep):
    ohlc = self.stocks_WMA_14[stock][1]
    ohlc.append(closep)
    ohlc.remove(ohlc[0])
    denom = 105
    total = 0
    for i in range(1, 15):
      total += (ohlc[-i] * (15 - i)) / denom
    self.stocks_WMA_14[stock] = (total, ohlc)


  #Decide if to buy, sell, hold, or do nothing
  def evaluation(self, stock):
    if(self.stream):
      if(self.strat == 1):
        self.strat_1(stock)
      else:
        self.strat_2(stock)

    else:
      if(self.strat == 1):
        return self.paperstrat_1(stock)
      elif(self.strat == 2):
        return self.paperstrat_2(stock)
      elif(self.strat == 3):
        return self.paperstrat_3(stock)
      elif(self.strat == 4):
        return self.paperstrat_4(stock)


  #Alligator method
  def strat_1(self, stock):
    risk = 1
    qty = self.get_positions(stock)
    
    #Buy
    if(self.stocks_SMA_20[stock] > self.stocks_SMA_50[stock] and qty == 0):
      self.make_order(stock, 'buy')
      print("Bought {}: {}\n".format(risk,stock))
    #Sell
    elif(self.stocks_SMA_20[stock] < self.stocks_SMA_50[stock] and qty > 0):
      self.make_order(stock, 'sell')
      print("Sold {}: {}\n".format(risk,stock))



  #HI Method
  def strat_2(self, stock):
    risk = 1
    qty = get_positions(stock)

    if(self.stocks_HI_Bars[stock][1] == 'dg' and self.stocks_HI_Bars[stock][2] == 'g' and qty == 0):
      make_order(stock, 'buy')
      print("Bought {}: {}\n".format(risk,stock))

    elif(((self.stocks_HI_Bars[stock][1] == 'dr' and self.stocks_HI_Bars[stock][2] == 'r') or
        (self.stocks_HI_Bars[stock][0] == 'dr' and self.stocks_HI_Bars[stock][2] == 'g' and self.stocks_HI_Bars[stock][2] == 'dr')) and
        qty > 0):

      make_order(stock, 'sell')
      print("Sold {}: {}\n".format(risk,stock))



  #Alligator Method
  def paperstrat_1(self, stock):
    #Buy
    if(self.stocks_SMA_20[stock][0] > self.stocks_SMA_50[stock][0] and self.paperqty == 0):
      return 'buy'
    #Sell
    elif(self.stocks_SMA_20[stock][0] < self.stocks_SMA_50[stock][0] and self.paperqty > 0):
      return 'sell'

    return 'hold'


  #HI Method
  def paperstrat_2(self,stock):
    #buy
    if(((self.stocks_HI_Bars[stock][1] == 'dg' and self.stocks_HI_Bars[stock][2] == 'g') or
      (self.stocks_HI_Bars[stock][1] == 'g' and self.stocks_HI_Bars[stock][2] == 'g')) and 
      self.paperqty == 0):
      self.paperqty = 1
      return 'buy'

    #sell
    elif(((self.stocks_HI_Bars[stock][1] == 'dr' and self.stocks_HI_Bars[stock][2] == 'r') or
        (self.stocks_HI_Bars[stock][0] == 'dr' and self.stocks_HI_Bars[stock][2] == 'g' and self.stocks_HI_Bars[stock][2] == 'dr')) and
        self.paperqty > 0):

      self.paperqty = 0
      return 'sell'

    return 'hold'

  #custon V1.0
  def paperstrat_3(self, stock):
    if(((self.stocks_HI_Bars[stock][1] == 'dg' and self.stocks_HI_Bars[stock][2] == 'g') or
      (self.stocks_HI_Bars[stock][1] == 'g' and self.stocks_HI_Bars[stock][2] == 'g')) and
      self.stocks_RSI[stock][2] > 50 and self.stocks_HI_Bars[stock][3][3] > self.stocks_SMA_20[stock][0] and self.paperqty == 0):
      self.paperqty = 1
      return 'buy'

    elif((self.stocks_RSI[stock][2] < 50 or self.stocks_HI_Bars[stock][3][3] < self.stocks_SMA_20[stock][0]) and self.paperqty > 0):
      self.paperqty = 0
      return 'sell'

    return 'hold'

  #custon V1.1
  def paperstrat_4(self, stock):
    if(self.stocks_RSI[stock][2] > 50 and self.stocks_HI_Bars[stock][3][3] > self.stocks_WMA_14[stock][0] and self.paperqty == 0):
      self.paperqty = 1
      return 'buy'

    elif((self.stocks_RSI[stock][2] < 50 or self.stocks_HI_Bars[stock][3][3] < self.stocks_WMA_14[stock][0]) and self.paperqty > 0):
      self.paperqty = 0
      return 'sell'

    return 'hold'


  def set_paperqty(self, qty):
    self.paperqty = qty

  #Make an order buy/sell
  def make_order(self, stock, side):
    data = {
    "symbol": stock,
    "notional": notional,
    "side": side,
    "type": "market",
    "time_in_force": "day"
    }

    r = requests.post(config.ORDER, json=data, headers=config.HEADERS)

  #Returns the quantity of the stock
  def get_positions(self, stock):
    r = requests.get(config.POSITION, headers=config.HEADERS)
    qty = 0
    for i in r.json():
      if(i['symbol'] == stock):
        qty = i['qty']
    return int(qty)


  def print_info(self,stock):
    print(stock)
    print(self.stocks_HI_Bars[stock])

  def print_date(self):
    today = datetime.now()
    print(today.strftime("%H:%M"))
    print(datetime.today().weekday())