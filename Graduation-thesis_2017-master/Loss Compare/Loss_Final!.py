import numpy as np

Expiration_Data = 30

def Loss_compare(Order_Data, Sale_Data , Expiration_data):

    import itertools
    import numpy as np
    import copy
    OD = np.reshape(Order_Data, (len(Order_Data))).tolist() # OD is Order Data
    SD = np.reshape(Sale_Data, (len(Sale_Data))).tolist() # SD is Sale Data
    OD_S = sum(OD)
    SD_S = sum(SD)
    NSD = copy.deepcopy(OD)  # Order Data copy
    SALE = []  # Sale progress
    OL = []  # Opportunity Loss
    DL = []  # Disposal loss
    DLL = []  # Disposal Loss result
    Today = []

    for i in range(len(OD)+100):  # for loop

        if NSD[0] - SD[0] == 0: # if Inventory on the day - Sale volume on the day = 0:
            SALE.append(SD[0])
            SD.remove(SD[0])
            NSD.remove(NSD[0])
            NSD.append(0)
            SD.append(0)

        elif NSD[0] - SD[0] < 0: # if Inventory on the day - Sale volume on the day < 0:
            OL.append(SD[0] - NSD[0])
            SALE.append(NSD[0])
            SD.remove(SD[0])
            NSD.remove(NSD[0])
            SALE = []
            NSD.append(0)
            SD.append(0)

        elif NSD[0] - SD[0] > 0: # if Inventory on the day - Sale volume on the day > 0:
            if Expiration_Data == 1:
                SALE.append(SD[0])
                DLL.append(NSD[0] - SD[0])
                SD.remove(SD[0])
                NSD.remove(NSD[0])
                NSD.append(0)
            else:
                SALE.append(SD[0])
                Today.append(NSD[0] - SD[0])
                NSD.remove(NSD[0])
                SD.remove(SD[0])
                NSD.append(0)

                for j in range(Expiration_Data-1):
                    if Today[0] == 0:
                        break

                    elif Today[0] - SD[j] > 0:
                        SALE.append(SD[j])
                        DL.append(Today[0] - SD[j])
                        Today[0] = (Today[0] - SD[j])
                        SD[j] = 0
                        SD.append(0)

                    elif Today[0] - SD[j] == 0:
                        SALE.append(SD[j])
                        DL.append(Today[0] - SD[j])
                        SD[j] = 0
                        Today[0] = 0
                        SD.append(0)

                    elif Today[0] - SD[j] < 0:
                        SALE.append(Today[0])
                        SD[j] = SD[j] - (Today[0])
                        Today[0] = 0
                        DL = [0]
                        if NSD[0] > SD[j]:
                            SALE.append(SD[j])
                            NSD[0] = NSD[0] - SD[j]
                            SD[j] = 0
                            SD.append(0)

                        elif NSD[0] < SD[j]:
                            SALE.append(NSD[0])
                            SD[j] = SD[j] - NSD[0]
                            OL.append(SD[j] - NSD[0])
                            NSD[0] = 0
                            SD.append(0)

                        elif NSD[0] == SD[j]:
                            SALE.append(SD[j])
                            SD[j] = SD[j] - NSD[0]
                            SD[j] = 0
                            NSD[0] = 0
                            SD.append(0)

                DLL.append(DL[-1])
                Today = []

    Disposal_Loss_real = sum(DLL)
    Opportunity_Loss = sum(OL)

    print("기회 Loss : ", Opportunity_Loss / SD_S, "%")
    print("폐기 Loss : ", Disposal_Loss_real / OD_S, "%")
    print("--------------------------------------------")