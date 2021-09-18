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
        self.o_time = (2021,6,11) #(2016,1,5)
        self.e_time = (2021,6,30)
        #self.num_weeks = 9
        self.months = [31,28,31,30,31,30,31,31,30,31,30,31]
        self.stock = stock
        self.year = self.o_time[0]
        self.month = self.o_time[1]
        self.day = self.o_time[2]
        self.start_date = "{}-{}-{}".format(self.year,self.month,self.day)
        self.SMMA = {}
        self.SMMA_offset = {}
        self.RSI = {}

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
        #while(curr_date != self.e_time):
        for k in range(0,15):
            try:
                with open("./data/{}.csv".format(filename), 'a', newline= '') as csvfile:
                    writer = csv.writer(csvfile)
                    #writer.writerow(fields)
                    with open('./data/{}_{}-{}-{}.txt'.format(stock, curr_date[0], curr_date[1], curr_date[2])) as f:
                        j = f.readline()
                        #Loop until end of file
                        while(j != ""):
                            j = j.strip()
                            writer.writerow(j.split(",")[1:])
                            j = f.readline()
            except FileNotFoundError:
                print("Skipped Day")
                curr_date = self.__increment_date(curr_date[0], curr_date[1], curr_date[2])


            curr_date = self.__increment_date(curr_date[0], curr_date[1], curr_date[2])

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
                counter = 0
                prev = 0
                close_list = []
                for row in reader:
                    #only need to update based on new 
                    if(counter >= 15):
                        self.Update_SMMA(stock, float(row[0].split(",")[3]))
                        self.update_RSI(stock, float(row[0].split(",")[3]), prev)
                        temp = row[0]+","+str(self.SMMA[stock][0])+","+str(self.SMMA[stock][1])+","+str(self.SMMA[stock][2])+","+str(self.RSI[stock][2])
                        writer.writerow(temp.split(","))
                        prev = float(row[0].split(",")[3])

                    #otherwise add row data to ohlc_list
                    else:
                        close_f = float(row[0].split(",")[3])
                        close_list.append(close_f)

                        #when 15 values have been counted, call AddRSIandAli
                        if(counter == 14):
                            #rsi
                            self.Add_RSI(stock, close_list)
                            print(self.RSI)
                            #pop so there are only 13 values for alligator indicator
                            close_list.pop(0)
                            close_list.pop(0)
                            self.Add_Alligator(stock, close_list)
                            temp = row[0]+","+str(self.SMMA[stock][0])+","+str(self.SMMA[stock][1])+","+str(self.SMMA[stock][2])+","+str(self.RSI[stock][2])
                            writer.writerow(temp.split(","))

                            #keep track of previous
                            prev = float(row[0].split(",")[3])

                    #increment each line
                    counter += 1
            

    #Takes stock data from csv file and writes it to another csv
    def Normalize_Stock(self, data_filename, norm_filename):
        df = pd.read_csv('{}.csv'.format(data_filename))
        names = ['open', 'high', 'low', 'close'] #TODO: Add rsi, and alligator bars
        df = pd.read_csv('{}.csv'.format("test"), names= names)
        print(df.head)
        array = df.values
        X = array[0:534090] #Magic number BAD!! TODO: Find way to read total number of lines in csv
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaledX = scaler.fit_transform(X)
        numpy.set_printoptions(precision=3)
        with open("{}.csv".format(norm_filename), 'a', newline= '') as csvfile:
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
        lips = sum(close_list[8:13]) / 5
        #teeth
        teeth = sum(close_list[5:13]) / 8
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


norm = Stock_Normalizer()
#norm.Convert_TXT_CSV("AAPL", "NewTest")
norm.AddIndicators("AAPL", "NewTest", "CompleteTest")