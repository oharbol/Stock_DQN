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
        self.o_time = (2016,1,5)
        self.e_time = (2021,6,30)
        #self.num_weeks = 9
        self.months = [31,28,31,30,31,30,31,31,30,31,30,31]
        self.stock = stock
        self.year = self.o_time[0]
        self.month = self.o_time[1]
        self.day = self.o_time[2]
        self.start_date = "{}-{}-{}".format(self.year,self.month,self.day)
        #self.ctime_start = '{}-{}-{}T9:30:00-04:00'.format(self.year,str(self.month).zfill(2),str(self.day).zfill(2))
        #self.ctime_end = '{}-{}-{}T16:00:00-04:00'.format(self.year,str(self.month).zfill(2),str(self.day).zfill(2))

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
            try:
                with open("./data/{}.csv".format(filename), 'a', newline= '') as csvfile:
                    writer = csv.writer(csvfile)
                    #writer.writerow(fields)
                    with open('./data/{}/{}_{}-{}-{}.txt'.format(stock, stock, curr_date[0], curr_date[1], curr_date[2])) as f:
                        j = f.readline()
                        #Loop until end of file
                        while(j != ""):
                            j = j.strip()
                            writer.writerow(j.split(","))
                            #print(j.split(","))
                            j = f.readline()
            except FileNotFoundError:
                print("Skipped Day")
                curr_date = self.__increment_date(curr_date[0], curr_date[1], curr_date[2])


            curr_date = self.__increment_date(curr_date[0], curr_date[1], curr_date[2])

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

    #TODO: Add functionality to add rsi and (3)SMMA alligator to raw data
    # def Add_RSIandAli():
    #     #First time calling SMMA which is just a moving average
    #     #Lips, teeth, jaw
    #     #(5,8,13) smoothed by 3,5,8
    #     def create_SMMA(self, stock, ohlc_list):
    #         last = ohlc_list[0:13]
    #         close_data = []
    #         #get the close data
    #         # for i in last:
    #         #   close_data.append(float(i[4]))
    #         #lips
    #         lips = sum(last[0:5]) / 5
    #         #teeth
    #         teeth = sum(last[0:8]) / 8
    #         #jaw
    #         jaw = sum(last) / 13
    #         #append values to self.SMMA
    #         vals = self.SMMA_offset[stock]
    #         lip_list = vals[0]
    #         teeth_list = vals[1]
    #         jaw_list = vals[2]

    #         self.SMMA[stock] = (lips, teeth, jaw)
    #         self.SMMA_offset[stock] = ([lips,lips,lips], [teeth,teeth,teeth,teeth,teeth], [jaw,jaw,jaw,jaw,jaw,jaw,jaw,jaw])


    #     def update_SMMA(self, stock, close):
    #         last = self.SMMA[stock]
    #         vals = self.SMMA_offset[stock]
    #         lip_list = vals[0]
    #         teeth_list = vals[1]
    #         jaw_list = vals[2]

    #         #calculate lips
    #         lips = (lip_list[2]*4+close)/5
    #         #calculate teeth
    #         teeth = (teeth_list[4]*7+close)/8
    #         #calculate jaw
    #         jaw = (jaw_list[7]*12+close)/13
            
    #         #append values to self.SMMA
    #         lip_list.append(lips)
    #         teeth_list.append(teeth)
    #         jaw_list.append(jaw)

    #         lips = lip_list.pop(0)
    #         teeth = teeth_list.pop(0)
    #         jaw = jaw_list.pop(0)
    #         self.SMMA[stock] = (lips, teeth, jaw)
    #         self.SMMA_offset[stock] = (lip_list, teeth_list, jaw_list)
