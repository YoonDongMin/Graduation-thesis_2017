import numpy as np
import math
import xlwt
import xlrd
import xlsxwriter

xy = np.loadtxt('DT1.csv', delimiter=',', dtype=np.float32)
sale = np.loadtxt('SD.csv', delimiter=',')
OD = xy[0:700, [-1]]
SD = sale[0:700]
Expiration_Data = 30

def Real_Sale(OD, SD, Expiration_Data):
    import numpy as np
    import copy
    OD = np.reshape(OD, (len(OD))).tolist()
    SD = np.reshape(SD, (len(SD))).tolist()
    NSD = copy.deepcopy(OD)  # Order Data copy
    Sale_Situation = []
    for i in range(0, len(SD)+200):
        Sale_Situation.append(0)
    Today = []

    for i in range(len(SD)+100):
        if NSD[i] - SD[i] == 0:
            Sale_Situation[i] = Sale_Situation[i] + round(NSD[i])
            SD[i] = 0
            NSD[i] = 0
            SD.append(0)
            NSD.append(0)

        elif NSD[i] - SD[i] < 0:
            Sale_Situation[i] = Sale_Situation[i] + round(NSD[i])
            SD[i] = 0
            NSD[i] = 0
            SD.append(0)
            NSD.append(0)

        elif NSD[i] - SD[i] > 0:
            Sale_Situation[i] = Sale_Situation[i] + round(SD[i])
            Today.append(NSD[i] - SD[i])
            NSD[i] = 0
            SD[i] = 0
            SD.append(0)
            NSD.append(0)

            for j in range(Expiration_Data - 1):
                if Today[0] == 0:
                    break
                elif Today[0] - SD[i+j+1] > 0:
                    Sale_Situation[i+j+1] = Sale_Situation[i+j+1] + round(SD[i+j+1])
                    Today[0] = Today[0] - SD[i+j+1]
                    SD[i+j+1] = 0
                elif Today[0] - SD[i+j+1] == 0:
                    Sale_Situation[i+j+1] = Sale_Situation[i+j+1] + round(SD[i+j+1])
                    SD[i + j + 1] = 0
                    Today[0] = 0
                elif Today[0] - SD[i+j+1] < 0:
                    Sale_Situation[i+j+1] = Sale_Situation[i+j+1] + round(Today[0])
                    SD[i+j+1] = SD[i+j+1] - round(Today[0])
                    Today[0] = 0

                    if NSD[i+j+1] > SD[i+j+1]:
                        Sale_Situation[i+j+1] = Sale_Situation[i+j+1] + round(SD[i+j+1])
                        NSD[i+j+1] = NSD[i+j+1] - SD[i+j+1]
                        SD[i+j+1] = 0

                    elif NSD[i+j+1] < SD[i+j+1]:
                        Sale_Situation[i+j+1] = Sale_Situation[i+j+1] + round(NSD[i+j+1])
                        SD[i+j+1] = SD[i+j+1] - NSD[i+j+1]
                        NSD[i+j+1] = 0

                    elif NSD[i+j+1] == SD[i+j+1]:
                        Sale_Situation[i+j+1] = Sale_Situation[i+j+1] + round(SD[i + j + 1])
                        SD[i+j+1] = SD[i+j+1] - NSD[i+j+1]
                        NSD[i+j+1] = 0
                        SD[i+j+1] = 0

            Today = []
    Sale_SituationSALE = np.array(Sale_Situation)
    Sale_Situation = Sale_Situation[0:700]
    return Sale_Situation

Real_Sale(OD, SD, Expiration_Data)

import xlsxwriter
book = xlsxwriter.Workbook('RS2.xlsx')
sheet1 = book.add_worksheet()

row = 0
col = 7

for i in (Real_Sale(OD, SD, Expiration_Data)):
    sheet1.write(row, col, i)
    row += 1

book.close()