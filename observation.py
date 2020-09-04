import pandas as pd
import numpy as np
import os
from state import state_arr
from utils import symbol_calculator

current_path = os.getcwd()
input_path = f"{current_path}//data//observation//"
files = os.listdir(input_path)

for filename in files:

    df = pd.read_csv(input_path + str(filename)).dropna()

    original = df["reply_count"] * df["tweet_count"] * df["quote_count"]
    spreader = df["mention_count"] * df["retweet_count"]
    reputed = df["Fav_count"] * df["retweet_by_other_count"]

    '''ORIGINAL=IF(J2<=50,"A",IF(J2<=500,"B",IF(J2<=5000,"C","D")))
    SPREADER==IF(L2<=12,"A",IF(L2<=45,"B",IF(K2<=150,"C","D")))
    REPUTED=IF(K2<=100000,"A",IF(K2<=7000000,"B",IF(K2<=500000000,"C","D")))
    '''
    print("Generating observation matrix...")

    original_a = 50
    original_b = 500
    original_c = 5000

    spreader_a = 12
    spreader_b = 45
    spreader_c = 150

    reputed_a = 100000
    reputed_b = 7000000
    reputed_c = 500000000

    original_state = symbol_calculator(original, original_a, original_b, original_c)
    spreader_state = symbol_calculator(spreader, spreader_a, spreader_b, spreader_c)
    reputed_state = symbol_calculator(reputed, reputed_a, reputed_b, reputed_c)

    OBS = ["AAA", "AAB", "AAC", "AAD", "ABA", "ABB", "ABC", "ABD", "ACA", "ACB", "ACC", "ACD", "ADA", "ADB", "ADC",
           "ADD",
           "BAA", "BAB", "BAC", "BAD", "BBA", "BBB", "BBC", "BBD", "BCA", "BCB", "BCC", "BCD", "BDA", "BDB", "BDC",
           "BDD",
           "CAA", "CAB", "CAC", "CAD", "CBA", "CBB", "CBC", "CBD", "CCA", "CCB", "CCC", "CCD", "CDA", "CDB", "CDC",
           "CDD",
           "DAA", "DAB", "DAC", "DAD", "DBA", "DBB", "DBC", "DBD", "DCA", "DCB", "DCC", "DCD", "DDA", "DDB", "DDC",
           "DDD"]

    st_arr = []
    for i in range(len(original_state)):
        s = f"{original_state[i]}{spreader_state[i]}{reputed_state[i]}"
        st_arr.append(s)

    obs_arr = []
    for i in st_arr:
        if i in OBS:
            obs_arr.append(OBS.index(i))

    emm_dict = {}
    for i in range(len(state_arr) - 1):
        s = state_arr[i], obs_arr[i]
        if s in emm_dict:
            emm_dict[s] += 1
        else:
            emm_dict[s] = 1

    e = np.zeros((27, 64))
    for key in emm_dict.keys():
        e[key] = emm_dict[key]

    for i in range(len(e)):
        s = sum(e[i]) + 64
        for j in range(len(e[i])):
            e[i, j] = (e[i, j] + 1)
            e[i, j] = e[i, j] / s

    OBS_MATRIX = e
    d2 = OBS_MATRIX
    df2 = pd.DataFrame(data=d2)
    df2.to_csv(f"{current_path}//output//observation//{filename}_observation.csv")
    print("Observation matrix generated.")
