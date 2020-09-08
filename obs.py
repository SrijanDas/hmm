from utils import symbol_calculator
from state import state_arr
import numpy as np

obs_arr = []


def obs_matrix(df4):
    original = df4["reply_count"] * df4["tweet_count"] * df4["quote_count"]
    spreader = df4["mention_count"] * df4["retweet_count"]
    reputed = df4["Fav_count"] * df4["retweet_by_other_count"]

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

    global obs_arr
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

    for i in range(27):
        s = sum(e[i]) + 64
        for j in range(64):
            e[i, j] = (e[i, j] + 1)
            e[i, j] = e[i, j] / s

    return e
