import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.animation as animation
#import matplotlib.finance
from mpl_finance import candlestick_ohlc
from matplotlib import style

import numpy as np
import urllib.request
import datetime as dt
import time

style.use('fivethirtyeight')

# def animate(i):
#   graph_data = open('samplefile.txt','r').read()
#   lines = graph_data.split('\n')
#   xs = []
#   ys = []
#   for line in lines:
#     if len(line) > 1:
#       x,y = line.split(',')
#       xs.append(x)
#       ys.append(y)
#   ax1.clear()
#   ax1.plot(xs,ys)

# ani = animation.FuncAnimation(fig, animate, interval=1000)
# plt.show()
def SMA(size, close):
  total = 0
  for i in range(size):
    total += close[-i-1]
  average = total / size
  return average

def RSI(size, close):
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
  return 100 - (100 / (1+rs))


def bytespdate2num(fmt, encoding='utf-8'):
  strconverter = mdates.strpdate2num(fmt)
  def bytesconverter(b):
    s = b.decode(encoding)
    return strconverter(s)
  return bytesconverter

fig = plt.figure()

def graph_data(i):

  #fig = plt.figure()
  ax1 = plt.subplot2grid((1,1),(0,0))

  # Unfortunately, Yahoo's API is no longer available
  # feel free to adapt the code to another source, or use this drop-in replacement.
  #stock_price_url = 'https://pythonprogramming.net/yahoo_finance_replacement'

  #source_code = urllib.request.urlopen(stock_price_url).read().decode()
  source_code = open('hist_AAPL.txt','r').read()
  stock_data = []
  split_source = source_code.split('\n')

  for line in split_source[1:]:
      split_line = line.split(',')
      if len(split_line) == 7:
          if 'values' not in line:
              stock_data.append(line)

  #stock_data = stock_data[-100::]
  closep, highp, lowp, openp, adj_closep= np.loadtxt(stock_data,delimiter=',',unpack=True)

  #print(stock_data)
  #print(SMA(20,closep))
  #print(RSI(14,closep[-15::]))
  xlen = len(stock_data)
  x = 0
  y = xlen
  ohlc = []
  while x < y:
    #heikin-ashi formula
    highi = max([highp[x],openp[x],closep[x]])
    lowi = min([lowp[x],openp[x],closep[x]])
    closei = (openp[x] + highp[x] + lowp[x]+closep[x]) / 4
    if(x != 0):
      openi = (openp[x-1]+closep[x-1]) / 2

    
    else:
      openi = (openp[x]+closep[x]) / 2
    
    openp[x] = openi
    closep[x] = closei

    append_me = openi, highi, lowi, closei

    #append_me = date[x], openp[x], highp[x], lowp[x], closep[x], volume[x]
    ohlc.append(append_me)
    x+=1

  ax1.clear()
  # date, closep, highp, lowp, openp, adj_closep, volume = np.loadtxt(stock_data,delimiter=',',unpack=True)

  # dateconv = np.vectorize(dt.datetime,fromtimestamp)
  # date = dateconv(date)

  

  # ax1.plot_date(date,closep,'-',label='price')
  # ax1.plot([],[],linewidth=5,label='loss',color='r',alpha=0.5)
  # ax1.plot([],[],linewidth=5,label='gain',color='g',alpha=0.5)
  # ax1.axhline(closep[0],color='k',linewidth=5)
  # ax1.fill_between(date,closep, closep[0], where=(closep > closep[0]), facecolor='g',alpha=0.5)
  # ax1.fill_between(date,closep, closep[0], where=(closep < closep[0]), facecolor='r',alpha=0.5)
  # for label in ax1.xaxis.get_ticklabels():
  #   label.set_rotation(45)
  # ax1.grid(True)
  # # ax1.xaxis.label.set_color('c')
  # # ax1.yaxis.label.set_color('r')
  # #ax1.set_yticks([0,25,50,75,100])
  # ax1.spines['left'].set_color('c')
  # ax1.spines['right'].set_visible(False)
  # ax1.spines['top'].set_visible(False)

  # ax1.spines['left'].set_linewidth(5)

  # ax1.tick_params(axis='x',colors='#f06215')

  candlestick_ohlc(ax1, ohlc,width=0.0005,colorup='#77d879',colordown='#db3f3f')

  #ax1.plot(date,closep)

  for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(45)

  ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d-%H:%M'))
  
  #ax1.xaxis.set_major_locator(mticker.MultipleLocator(1))
  ax1.xaxis.set_major_locator(mticker.MaxNLocator(5))
  ax1.grid(True)

  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.title('AAPL')
  #plt.legend()
  plt.subplots_adjust(left=0.09,bottom=0.20,right=0.94,top=0.90,wspace=0.2,hspace=0)
  #print(stock_data)


ani = animation.FuncAnimation(fig, graph_data, interval=10000)
plt.show()


#time.sleep(5000)
#print("Hello")
#graph_data('TSLA')


































# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib import style

# style.use('fivethirtyeight')

# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)

# def animate(i):
#     graph_data = open('samplefile.txt','r').read()
#     lines = graph_data.split('\n')
#     xs = []
#     ys = []
#     for line in lines:
#         if len(line) > 1:
#             x, y = line.split(',')
#             xs.append(x)
#             ys.append(y)
#     ax1.clear()
#     ax1.plot(xs,ys)
    
# ani = animation.FuncAnimation(fig, animate, interval=1000)
# plt.show()