from datetime import datetime

class Data_Handler:
    
    def __init__(self):
        #set of all stocks
        self.stocks = {}
        #current stock being observed
        #self.current_stock = ""

        #dictionaries:
        
        #all data found 30 days before observed day
        #used to determine min-max
        #-1: earliest, 0: latest
        #[OHLC, ...]
        self.previous_data = {}

        #used to normalize data
        #(Min, Max)
        self.min_max = {}

        #Year,Month,Date
        # self.o_time = (2016,1,6) #(2016,1,5)
        # self.e_time = (2021,10,11)
        #self.num_weeks = 9
        # self.months = [31,28,31,30,31,30,31,31,30,31,30,31]
        # self.year = self.o_time[0]
        # self.month = self.o_time[1]
        # self.day = self.o_time[2]
        # self.start_date = "{}-{}-{}".format(self.year,self.month,self.day)

    #Adds a stock to the global set
    def Add_Stock(self, stock):
        self.stocks.add(stock)

        #initialize all dictionaries with each stock
        self.previous_data[stock] = []
        self.min_max[stock] = (float('inf'), float('-inf'))
    
    #Attempts to remove a stock from the global set
    def Remove_Stock(self, stock):
        try :
            self.stocks.remove(stock)
            self.previous_data.pop(stock)
            self.min_max.pop(stock)
        except KeyError:
            print("A non-existant stock was attempted to be removed.")

    #Sets the stock to be observed and performed upon
    # def Set_Current_Stock(self, stock):
    #     self.stock = stock
    
    #Adds a single line of OHLC data to use for callibration
    def Add_Callibration_Data(self, stock, ohlc):
        min,max = self.min_max[stock]

        #check for new Min-Max
        if(ohlc[2] < min):
            min = ohlc[2]
            self.min_max[stock] = (min,max)
        if(ohlc[1] > max):
            max = ohlc[1]
            self.min_max[stock] = (min,max)
        
        #add ohlc data to previous month data list
        self.previous_day[stock].append(ohlc)
    
    #Removes the latest 390 data points from the monthly data
    def Remove_Callibration_Data(self, stock):
        #flag to reevaluate entire month for new Min-Max
        flag = False
        min,max = self.min_max[stock]
        #loop through latest day's data (index 0 to 390)
        for ohlc in self.previous_data[stock][0:390]:
            if(not flag):
                if(ohlc[2] == min):
                    flag = True
                if(ohlc[1] == max):
                    flag = True
            
            #remove ohlc from list
            self.previous_data[stock].pop(0)
        
        if(flag):
            self.Evaluate_Min_Max(stock)
        
    #Evaluates the min-max value from the monthly data
    def Evaluate_Min_Max(self, stock):
        #set arbitrary min and max
        min,max = float('inf'), float('-inf')
        for ohlc in self.previous_data[stock]:
            if(ohlc[2] < min):
                min = ohlc[2]
            if(ohlc[1] > max):
                max = ohlc[1]
        
        self.previous_data[stock] = (min,max)

    