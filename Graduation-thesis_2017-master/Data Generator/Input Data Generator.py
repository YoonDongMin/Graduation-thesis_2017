import numpy as np, numpy.random
import random
import numpy as np
import math
import xlwt
import pandas as pd
from tempfile import TemporaryFile # import module
np.random.seed(0) # set the seed / Make next random number predicting
book = xlwt.Workbook() # can save excel file
sheet1 = book.add_sheet('sheet1') # can save excel file_sheet

# MOQ = Maximum order quantity
# week = Order_data = 7 day data

def while_True(week, MOQ, size=None): # function to make Order_data(as len(week))

    def loop(): # function to make Order_data(as len(1 week))

        Order_A_Little = 0.1 # Probability of order the Moderate quantity will be ordered
        Do_Not_Order = 0.85 # Probability of order the zeros quantity will be ordered
        Order_A_Max = 0.05 # Probability of order the maximum quantity will be ordered
        How_Many_Loop = MOQ - 1 # Probability p Space range
        MP_LOOP = Order_A_Little / How_Many_Loop # Dispersion of Probability Values
        p = [] # Generate Probability List p

        def probability_loop(): # Function to Complete Probability list p
            for i in range(How_Many_Loop):
                p.insert(1, MP_LOOP)
            p.insert(0, Do_Not_Order)
            p.insert(How_Many_Loop + 1, Order_A_Max)
        probability_loop()

        def Check_List(): # Function to Make Probability List sum(p) == 1
            while True:
                if sum(p) < 1:
                    p[1] = p[1] + 0.0000000000000001
                if sum(p) == 1:
                    break
                elif sum(p) > 1:
                        p[1] = p[1] - 0.0000000000000001
                if sum(p) == 1:
                    break
        Check_List()

# GOD = Generating Order Data

        def GOD(): # Function to Using Random Functions 7 Generating Order Data
            global list_1
            list_1 = sorted(np.hstack(np.random.choice(MOQ+1, size=(1, 7), p=p)))
            return list_1
#
        while True: # Condition of Order Data (7 days) Generation
            if MOQ > sum(GOD()):
                # Condition : The sum of Order Data (7 days) must always be less than the maximum order quantity.
                result_1 = list_1
                return result_1
            else:
                GOD()

    final_list = [] # Generate Final Order Data List
    for i in range(week): # Perform function GOD() as many as week arguments and insert the Value to Final Order Data List
        final_list.append(loop())
    final_list = np.hstack(final_list).tolist()
    return final_list

list_date_year = [] # Daily index value List
list_date_month =[] # 4-weekly index value List
list_date_week = [] # Weekly index value List

def year_day():
    for i in range(1, 4):
        for j in range(1, 366):
            list_date_year.append(j)
    return list_date_year

def year_month():
    for i in range(1, 4):
        for j in range(1,32) :#1월
            list_date_month.append(j)
        for j in range(1,29) :#2월
            list_date_month.append(j)
        for j in range(1,32) :#3월
            list_date_month.append(j)
        for j in range(1,31) :#4월
            list_date_month.append(j)
        for j in range(1,32) :#5월
            list_date_month.append(j)
        for j in range(1,31) :#6월
            list_date_month.append(j)
        for j in range(1,32) :#7월
            list_date_month.append(j)
        for j in range(1,32) :#8월
            list_date_month.append(j)
        for j in range(1,31) :#9월
            list_date_month.append(j)
        for j in range(1,32) :#10월
            list_date_month.append(j)
        for j in range(1,31) :#11월
            list_date_month.append(j)
        for j in range(1,32) :#12월
            list_date_month.append(j)
    return list_date_month

def year_week(): # Function to Complete Weekly(1~7 repeat) index value List
    def week():
        for i in list(range(1, 8)):
            list_date_week.append(i)
    for i in range(100): # 200 : week(arguments) value
        week()
    return list_date_week

tempo2015 = pd.read_csv('tempo_2015.csv') # The temperatures data from  Busan Nampo-Dong, 2015
tempo2016 = pd.read_csv('tempo_2016.csv') # The temperatures data from  Busan Nampo-Dong, 2016

tempo2015 = tempo2015['tempo']
tempo2015.values
tempo2016 = tempo2016['tempo']
tempo2016.values

tempo2015 = [value for value in tempo2015 if not math.isnan(value)]
tempo2016 = [value for value in tempo2016 if not math.isnan(value)]

tempo_year = []

def tempo():
    for i in range(0, 8760, 24):
        tempo_year.append(round(np.mean(tempo2015[i:i + 23])))
    for i in range(0, 8784, 24):
        tempo_year.append(round(np.mean(tempo2016[i:i + 23])))
    return tempo_year

rain2015 = pd.read_csv('rain_2015.csv') # The rain data from  Busan Nampo-Dong, 2015
rain2016 = pd.read_csv('rain_2016.csv') # The rain data from  Busan Nampo-Dong, 2016

rain2015 = rain2015['rain']
rain2015.values
rain2016 = rain2016['rain']
rain2016.values

rain2015 = [value for value in rain2015 if not math.isnan(value)]
rain2016 = [value for value in rain2016 if not math.isnan(value)]

rain_year = []

def rain():
    for i in range(0, 8760, 24):
        rain_year.append(np.max(rain2015[i:i + 23]))
    for i in range(0, 8784, 24):
        rain_year.append(np.max(rain2016[i:i + 23]))
    return rain_year

for i,e in enumerate(year_day()): # Write Daily index value on Excel_sheet1
    sheet1.write(i,0,e)

for i,e in enumerate(year_month()): # Write 4-weekly(1~28 repeat) index value on Excel_sheet1
    sheet1.write(i,1,e)

for i,e in enumerate(year_week()): # Write Weekly(1~7 repeat) index value on Excel_sheet1
    sheet1.write(i,2,e)

for i,e in enumerate(tempo()): # Write The temperatures data from  Busan Nampo-Dong on Excel_sheet1
    sheet1.write(i,3,e)

for i,e in enumerate(rain()): # Write The rain data from  Busan Nampo-Dong on Excel_sheet1
    sheet1.write(i,4,e)

for i,e in enumerate(while_True(100, 50, size=None)): # Write The real sale data on Excel_sheet1
    sheet1.write(i,5,e)

for i,e in enumerate(while_True(100, 50, size=None)): # Write The real stock data on Excel_sheet1
    sheet1.write(i,6,e)


name = "DT.xls" # Excel file name
book.save(name)
book.save(TemporaryFile())