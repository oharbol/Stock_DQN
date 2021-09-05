import json
import requests
import alpaca_trade_api as tradeapi
from datetime import datetime
import csv


#Take a file name,

#Loop through all values in the data set
for i in range(0, 1430):
    print('yes')
fields = ['date', 'open','high', 'low', 'close', 'rsi']
data = open('./AAPL/callibration_AAPL.txt','r').read().split('\n')
data.pop()
j = data[0].split(',')
i = data[1].split(',')
print(j)
with open("Test.csv", 'w', newline= '') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(fields)
    for i in data:
        j = i.split(',')
        writer.writerow(j)
