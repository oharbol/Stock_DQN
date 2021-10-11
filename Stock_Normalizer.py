import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy
import json
import requests
import alpaca_trade_api as tradeapi
from datetime import datetime
import csv

#Used to organized and normalize data from previous collections into txt files
#Should only use ___ methods during streaming to calculate 
class Stock_Normalizer:
    #Initialize with time data with start date being Jan 5, 2016
    def __init__(self, stock = "AAPL"):
        #Data used for transfering textfiles to csv
        self.stock = stock
        #Year,Month,Date
        self.o_time = (2016,1,6) #(2016,1,5)
        self.e_time = (2021,10,11)
        #self.num_weeks = 9
        self.months = [31,28,31,30,31,30,31,31,30,31,30,31]
        self.stock = stock
        self.year = self.o_time[0]
        self.month = self.o_time[1]
        self.day = self.o_time[2]
        self.start_date = "{}-{}-{}".format(self.year,self.month,self.day)

        #Indicator data
        self.SMMA = {}
        self.SMMA_offset = {}
        self.RSI = {}

        #EMA
        self.CONST_9 = 9
        self.CONST_12 = 12
        self.CONST_26 = 26

        self.N_9 = 2 / 10
        self.N_12 = 2 / 13
        self.N_26 = 2 / 27

        #[EMA_12, EMA_26]
        self.EMA = {}

        #MACD
        #integer
        self.MACD = {}
        #list
        self.EMA_MACD = {}
        #int
        self.MACD_Hist = {}

    #Increase the date by one day and adjust month and year as needed
    def __increment_date(self, year, month, day):
        week_date = datetime.strptime('{} {} {}'.format(day, month, year), '%d %m %Y').weekday()
        if(week_date == 4):
            day += 3
        else:
            day += 1

        if(self.months[month-1] < day):
            day = day - self.months[month-1]
            month += 1
            if(month > 12):
                month = 1
                year += 1
        return (year,month,day)

    def Convert_TXT_CSV(self, stock, filename):
        curr_date = self.o_time
        #fields = ['date', 'open','high', 'low', 'close', 'rsi']
        #Loop until at end of text files
        while(curr_date != self.e_time):
        #for k in range(0,56):
            try:
                with open("./data/{}.csv".format(filename), 'a', newline= '') as csvfile:
                    writer = csv.writer(csvfile)
                    #writer.writerow(fields)
                    with open('./data/{}/{}_{}-{}-{}.txt'.format(stock, stock, curr_date[0], curr_date[1], curr_date[2])) as f:
                        j = f.readline()
                        #Loop until end of file
                        while(j != ""):
                            j = j.strip().split(",")
                            j[0] = self.Convert_Time(j[0])
                            writer.writerow(j)
                            j = f.readline()
            except FileNotFoundError:
                print("Skipped Day")
                curr_date = self.__increment_date(curr_date[0], curr_date[1], curr_date[2])


            curr_date = self.__increment_date(curr_date[0], curr_date[1], curr_date[2])
    
    def Convert_Time(self, time_t):
        return time_t[11:16].replace(":","")

    #Take the raw ohlc data and add the rsi and alligator indicators to it
    #open, high, low, close, lips, teeth, jaw, rsi
    def AddIndicators(self, stock, filename_read, filename_write):
        #Make sure all dictionaries are 0
        self.InitializeEmpty(stock)

        with open("./data/{}.csv".format(filename_write), 'a', newline= '') as csvfile:
            writer = csv.writer(csvfile)
            with open("./data/{}.csv".format(filename_read), newline= '') as f:
                reader = csv.reader(f, delimiter=' ')
                #value to keep track of
                #counter = 0
                prev = 0
                close_list = []
                hist_ema_list = []

                for counter, row in enumerate(reader):
                    close_f = float(row[0].split(",")[3])

                    if(counter <= 25):
                        close_list.append(close_f)

                    #Add EMA counter for ema_12 and ema_26
                    if(counter == 25):
                        self.Add_EMA(stock, close_list[-12:], self.CONST_12, 0)
                        self.Add_EMA(stock, close_list, self.CONST_26, 1)
                    
                    
                    if(counter >= 26):
                        #update ema_12 and ema_26 counters
                        self.Update_EMA(stock, close_f, self.N_12, 0)
                        self.Update_EMA(stock, close_f, self.N_26, 1)
                        #calclate macd
                        self.Update_MACD(stock) 

                        if(counter <= 34):
                            #add last macd
                            hist_ema_list.append(self.MACD[stock])

                            if(counter == 34):
                                self.Add_HIST_EMA(stock, hist_ema_list)
                                self.Update_MACDHist(stock)
                        
                        else:
                            self.Update_HIST_EMA(stock, self.N_9)
                            self.Update_MACDHist(stock)
                    
                    if(counter >= 15):
                        self.Update_SMMA(stock, close_f)
                        #print(self.SMMA_offset)
                        self.update_RSI(stock, close_f, prev)
                        
                        prev = close_f

                    #otherwise add row data to ohlc_list
                    elif(counter == 14):
                        
                        

                        #when 15 values have been counted, call AddRSIandAli

                        #rsi
                        self.Add_RSI(stock, close_list)
                        #print(self.RSI)
                        #pop so there are only 13 values for alligator indicator
                        #close_list.pop(0)
                        #close_list.pop(0)
                        self.Add_Alligator(stock, close_list[-13:])
                        #print(self.MACD_Hist[stock])

                        # temp = row[0]+","+str(self.SMMA[stock][0])+","+str(self.SMMA[stock][1])+","+str(self.SMMA[stock][2])+","+str(self.RSI[stock][2])+","+str(self.MACD_Hist[stock])
                        # writer.writerow(temp.split(","))

                        #keep track of previous
                        prev = close_f
                    
                    if(counter >= 34):
                        temp = row[0]+","+str(self.SMMA[stock][0])+","+str(self.SMMA[stock][1])+","+str(self.SMMA[stock][2])+","+str(self.RSI[stock][2])+","+str(self.MACD_Hist[stock])
                        writer.writerow(temp.split(","))
            

    #Takes stock data from csv file and writes it to another csv
    def Normalize_Stock(self, data_filename, norm_filename):
        #df = pd.read_csv('./data/{}.csv'.format(data_filename))
        names = ['open', 'high', 'low', 'close', 'lips', 'teeth', 'jaw', 'rsi']
        df = pd.read_csv('./data/{}.csv'.format(data_filename), names = names)
        print(df.head)
        array = df.values
        X = array[0:552948] #Magic number BAD!! TODO: Find way to read total number of lines in csv
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaledX = scaler.fit_transform(X)
        numpy.set_printoptions(precision=3)
        with open("./data/{}.csv".format(norm_filename), 'a', newline= '') as csvfile:
            writer = csv.writer(csvfile)
            for i in rescaledX:
                writer.writerow(i)


    #Need to read the first 13 lines of the callibration data file
    #Callibration data is either the callibration file or the
    # previous day of data during live trading
    def Add_Alligator(self, stock, close_list):
        #First time calling SMMA which is just a moving average
        #Lips, teeth, jaw
        #(5,8,13) smoothed by 3,5,8
        #last = ohlc_list[0:13]
        #close_data = []
        #get the close data
        # for i in last:
        #   close_data.append(float(i[4]))
        #lips
        lips = sum(close_list[8:]) / 5
        #teeth
        teeth = sum(close_list[5:]) / 8
        #jaw
        jaw = sum(close_list) / 13

        self.SMMA[stock] = (lips, teeth, jaw)
        self.SMMA_offset[stock] = ([lips,lips,lips], [teeth,teeth,teeth,teeth,teeth], [jaw,jaw,jaw,jaw,jaw,jaw,jaw,jaw])


    #The method "Add_Alligator" must be called before this one
    #Automatically updates the next offset of Alligator data
    #Updates the current Alligator data
    def Update_SMMA(self, stock, close):
        #last = self.SMMA[stock]
        vals = self.SMMA_offset[stock]
        lip_list = vals[0]
        teeth_list = vals[1]
        jaw_list = vals[2]

        #calculate lips
        lips = (lip_list[2]*4+close)/5
        #calculate teeth
        teeth = (teeth_list[4]*7+close)/8
        #calculate jaw
        jaw = (jaw_list[7]*12+close)/13
        
        #append values to self.SMMA
        lip_list.append(lips)
        teeth_list.append(teeth)
        jaw_list.append(jaw)

        #remove first value
        #also the current SMMA values
        lips = lip_list.pop(0)
        teeth = teeth_list.pop(0)
        jaw = jaw_list.pop(0)
        self.SMMA[stock] = (lips, teeth, jaw)
        self.SMMA_offset[stock] = (lip_list, teeth_list, jaw_list)

    #intialize dictionaries so that no errors occur
    def InitializeEmpty(self, stock):
        self.SMMA[stock] = ([0,0,0], [0,0,0,0,0], [0,0,0,0,0,0,0,0])
        self.SMMA_offset[stock] = (0,0,0)
        self.RSI[stock] = (0,0,0)

        self.EMA_MACD[stock] = 0
        self.MACD[stock] = 0
        self.MACD_Hist[stock] = 0

        self.EMA[stock] = [0,0]

    #Conducted the first time
    def Add_RSI(self, stock, close_list):
        loss = 0
        gain = 0
        for i in range(1,15):
            num = close_list[i] - close_list[i-1]
            if(num > 0):
                gain += num
            else:
                loss += abs(num)

            avg_loss = loss / 14
            avg_gain = gain / 14
            rs = avg_gain / avg_loss
            self.RSI[stock] = (avg_gain, avg_loss, 100 - (100 / (1+rs)))
        
        for i in range(14+1, len(close_list)-13):
            self.update_RSI(stock, close_list[i], close_list[i-1])
            print(self.stocks_RSI[stock])

    def update_RSI(self, stock, curr, prev):
        avg_gain = self.RSI[stock][0]
        avg_loss = self.RSI[stock][1]
        num = curr - prev
        if(num > 0):
            avg_gain = ((avg_gain * 13) + num) / 14
            avg_loss = (avg_loss * 13) / 14
        else:
            avg_loss = ((avg_loss * 13) + abs(num)) / 14
            avg_gain = (avg_gain * 13) / 14
        rs = avg_gain / avg_loss
        self.RSI[stock] = (avg_gain, avg_loss, 100 - (100 / (1+rs)))

    #Conducts a SMA of a list of close data
    def Add_EMA(self, stock, list_close, n, index):
        self.EMA[stock][index] = sum(list_close) / n

    def Update_EMA(self, stock, close, k, index):
        self.EMA[stock][index] = (close - self.EMA[stock][index]) * k + self.EMA[stock][index]
    
    def Add_HIST_EMA(self, stock, list_ema):
        self.EMA_MACD[stock] = sum(list_ema) / self.CONST_9
    
    def Update_HIST_EMA(self, stock, k):
        self.EMA_MACD[stock] = (self.MACD[stock] - self.EMA_MACD[stock]) * k + self.EMA_MACD[stock]

    def Update_MACD(self, stock):
        self.MACD[stock] = self.EMA[stock][0] - self.EMA[stock][1]

    def Update_MACDHist(self, stock):
        self.MACD_Hist[stock] = self.MACD[stock] - self.EMA_MACD[stock]


norm = Stock_Normalizer()
#norm.Convert_TXT_CSV("AAPL", "AAPL_rawtest")
norm.AddIndicators("AAPL", "AAPL_rawtest", "AAPL_completetest")
#norm.Normalize_Stock("AAPL_complete", "AAPL_norm")

# f = open("./data/CompleteTest.csv", newline= '')
# print(next(f).split(","))
# open_f, high_f, low_f, close_f, lips, teeth, jaw, rsi = next(f).split(",")
# print(open_f)
# print(high_f)
# print(low_f)
# print(close_f)
# print(lips)
# print(teeth)
# print(jaw)
# print(float(rsi))
# k = float(j.split(",")[-1])
# print(k + 1)
# j = next(f)
# print(j)
# f.close