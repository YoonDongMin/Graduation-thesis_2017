import numpy as np, numpy.random
import random
import numpy as np
import math
import xlwt
from tempfile import TemporaryFile # import module
np.random.seed(0) # set the seed / Make next random number predicting
book = xlwt.Workbook() # can save excel file
sheet1 = book.add_sheet('sheet1') # can save excel file_sheet

# MSQ = Maximum Sale quantity
# week = Sale_data = 7 day data

def while_True(week, MSQ, size=None): # function to make Sale data(as len(week))

    def loop(): # function to make Sale data(as len(1 week))

        Order_A_Little = 0.9999 # Probability of sales the Moderate quantity will be sold
        Do_Not_Order = 0.00005 # Probability of sales the zeros quantity will be sold
        Order_A_Max = 0.00005 # Probability of sales the maximum quantity will be sold
        How_Many_Loop = MSQ - 1 # Probability p Space range
        MP_LOOP = Order_A_Little / How_Many_Loop # Dispersion of Probability Values
        p = [] # Generate Probability List p

        def probability_loop(): # Function to Complete Probability list p
            for i in range(How_Many_Loop):
                p.insert(1, MP_LOOP)
            p.insert(0, Do_Not_Order)
            p.insert(How_Many_Loop + 1, Order_A_Max)
        probability_loop()

        change_p_index = int((MSQ / 2) + 1)
        change_p = p[change_p_index:]  # max_value / 2 + 1 만큼 넣고 나머지 사이즈가 max_value /2가 되어야함
        p_array = 2 * np.array(change_p)
        p[change_p_index:] = np.zeros(change_p_index-1) #인덱싱은 위의 change_p만큼, np.zeros(max_value/2 만큼)
        p[1:int(change_p_index-1)] = sorted(p_array) #np.zeros의 크기만큼
        p.remove(0)
        p[0] = 0
        p_d = sum(p[(change_p_index-4):change_p_index]) / len(p[(change_p_index-4):])
        p[(change_p_index-4):change_p_index] = np.zeros(4)
        p = np.array(p)
        index = np.argwhere(p==0.0)
        p = np.delete(p, index)
        p = p.tolist()
        for i in range(change_p_index+3):
            p.insert(22, p_d)

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

# GSD = Generating Sale Data

        def GSD(): # Function to Using Random Functions 7 Generating Sale Data
            global list_1
            list_1 = np.hstack(np.random.choice(MSQ+1, size=(1, 7), p=p))
            return list_1

        while True: # Condition of Sale Data (7 days) Generation
            if MSQ > sum(GSD()):
                # Condition : The sum of order data (7 days) must always be less than the maximum sales quantity.
                result_1 = list_1
                return result_1
            else:
                GSD()

    final_list = [] # generate final sale data list

    for i in range(week): # Perform function GSD() as many as week arguments and insert the Value to Final Order Data List
        final_list.append(loop())
    final_list = np.hstack(final_list).tolist()
    return final_list

while True:
    if sum(while_True(200, 16, size=None)) < 3000:
        for i, e in enumerate(while_True(200, 16, size=None)):
            sheet1.write(i, 3, e)

        name = "DT3.xls"
        book.save(name)
        book.save(TemporaryFile())
        break
    else:
        continue
