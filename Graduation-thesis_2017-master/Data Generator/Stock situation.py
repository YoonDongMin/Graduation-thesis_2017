import numpy as np
import math
import xlwt
import xlrd

xy = np.loadtxt('DT.csv', delimiter=',', dtype=np.float32)
sale = np.loadtxt('Sale_Data.csv', delimiter=',')
OD = xy[0:700, [-1]] # OD is Order Data
SD = sale[0:700] # SD is Sale Data
Expiration_Data = 30

def Stock(OD, SD, Expiration_Data):
    import numpy as np
    import copy
    OD = np.reshape(OD, (len(OD))).tolist()
    SD = np.reshape(SD, (len(SD))).tolist()
    NSD = copy.deepcopy(OD)  # Order Data copy
    SALE = []
    for i in range(0, len(SD)+ 100):
        SALE.append(0)
    stock = []
    for i in range(0, len(SD) + 100):
        stock.append(0)
    Today = []

    for i in range(len(SD)+100):
        if NSD[i] - SD[i] == 0:
            SALE[i] = SALE[i] + round(SD[i])
            stock[i] = stock[i] + 0
            SD[i] = 0
            NSD[i] = 0
            SD.append(0)
            NSD.append(0)

        elif NSD[i] - SD[i] < 0:
            SALE[i] = SALE[i] + round(SD[i])
            stock[i] = stock[i] + 0
            SD[i] = 0
            NSD[i] = 0
            SD.append(0)
            NSD.append(0)

        elif NSD[i] - SD[i] > 0:
            SALE[i] = SALE[i] + round(SD[i])
            stock[i] = stock[i] + NSD[i] - SD[i]
            Today.append(NSD[i] - SD[i])
            NSD[i] = 0
            SD[i] = 0
            SD.append(0)
            NSD.append(0)

            for j in range(Expiration_Data - 1):
                if Today[0] == 0:
                    break
                elif Today[0] - SD[i+j+1] > 0:
                    SALE[i+j+1] = SALE[i+j+1] + round(SD[i+j+1])
                    stock[i+j+1] = stock[i+j+1] + (Today[0] - SD[i+j+1])
                    Today[0] = Today[0] - SD[i+j+1]
                    SD[i+j+1] = 0
                elif Today[0] - SD[i+j+1] == 0:
                    SALE[i+j+1] = SALE[i+j+1] + round(SD[i+j+1])
                    stock[i+j+1] = stock[i+j+1] + 0
                    SD[i + j + 1] = 0
                    Today[0] = 0
                elif Today[0] - SD[i+j+1] < 0:
                    SALE[i+j+1] = SALE[i+j+1] + round(Today[0])
                    stock[i+j+1] = stock[i+j+1] + 0
                    SD[i+j+1] = SD[i+j+1] - round(Today[0])
                    Today[0] = 0

                    if NSD[i+j+1] > SD[i+j+1]:
                        SALE[i+j+1] = SALE[i+j+1] + round(SD[i+j+1])
                        stock[i+j+1] = stock[i+j+1] + (NSD[i+j+1] - SD[i+j+1])
                        NSD[i+j+1] = NSD[i+j+1] - SD[i+j+1]
                        SD[i+j+1] = 0

                    elif NSD[i+j+1] < SD[i+j+1]:
#                        SALE[i+j+1] = SALE[i+j+1] + round(NSD[i+j+1])
                        stock[i + j + 1] = stock[i + j + 1] + 0
                        SD[i+j+1] = SD[i+j+1] - NSD[i+j+1]
                        NSD[i+j+1] = 0

                    elif NSD[i+j+1] == SD[i+j+1]:
                        SALE[i+j+1] = SALE[i+j+1] + round(SD[i+j+1])
                        stock[i + j + 1] = stock[i + j + 1] + 0
                        SD[i+j+1] = SD[i+j+1] - NSD[i+j+1]
                        NSD[i+j+1] = 0
                        SD[i+j+1] = 0

            Today = []
    stock = np.array(stock)
    stock = stock[0:700]
    return stock

import xlsxwriter
book = xlsxwriter.Workbook('S2.xlsx')
sheet1 = book.add_worksheet()

row = 0
col = 1

for i in (Stock(OD, SD, Expiration_Data)):
    sheet1.write(row, col, i)
    row += 1

book.close()