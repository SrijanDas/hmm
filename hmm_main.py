import os
import pandas as pd
import numpy as np
from hmmlearn import hmm
import math
from utils import state_calculator, symbol_calculator

input_path = "F://hmm-master//hmm-master//data//observation//"
files = os.listdir(input_path)
state_path = "F://hmm-master//hmm-master//data//state//"

for filename in files:

    df2 = pd.read_csv(state_path + str(filename)).dropna()
    df = df2[:18]
    # print(df)
    df3 = pd.read_csv(input_path + str(filename)).dropna()
    df4 = df3[:24]
    # print(df4)
    print("\nState calculation.. for " + str(filename))
    LIKE_LOW_RANGE = 100
    LIKE_MID_RANGE = 1000

    RETWEET_LOW_RANGE = 0.75
    RETWEET_MID_RANGE = 1.75

    TWEET_LOW_RANGE = 0.7
    TWEET_MID_RANGE = 1.25

    like = df["sd_like"]
    retweet = df["sd_retweet"]
    tweet = df["sd_tweet"]

    like_state = state_calculator(like, LIKE_LOW_RANGE, LIKE_MID_RANGE)
    retweet_state = state_calculator(retweet, RETWEET_LOW_RANGE, RETWEET_MID_RANGE)
    tweet_state = state_calculator(tweet, TWEET_LOW_RANGE, TWEET_MID_RANGE)

    STATE = ['LLL', 'LLM', 'LLH', 'LML', 'LMM', 'LMH', 'LHL', 'LHM', 'LHH', 'MLL',
             'MLM', 'MLH', 'MML', 'MMM', 'MMH', 'MHL', 'MHM', 'MHH', 'HLL',
             'HLM', 'HLH', 'HML', 'HMM', 'HMH', 'HHL', 'HHM', 'HHH']
    print("Generating state transition  matrix... for " + str(filename))
    comparison_arr = []
    for i in range(0, len(like_state)):
        s = f"{like_state[i]}{retweet_state[i]}{tweet_state[i]}"
        comparison_arr.append(s)

    state_arr = []
    for i in comparison_arr:
        if i in STATE:
            state_arr.append(STATE.index(i))
    # print(state_arr)
    trans_dict = {}
    for i in range(len(state_arr) - 1):
        s = state_arr[i], state_arr[i + 1]
        if s in trans_dict:
            trans_dict[s] += 1
        else:
            trans_dict[s] = 1
    # print(trans_dict)
    a = np.zeros((27, 27))
    for key in trans_dict.keys():
        a[key] = trans_dict[key]
    # print(a[20])
    for i in range(27):
        s = sum(a[i]) + 27
        for j in range(27):
            a[i, j] = (a[i, j] + 1)
            a[i, j] = a[i, j] / s

    state_matrix=a
    data = pd.DataFrame(data=state_matrix)
    data.to_csv("F://hmm-master//hmm-master//output//state//"+str(filename)+"_state.csv")
    print("State trasition matrix generated succesfully\n")
    print("\nObservation table calculation .....for "+str(filename))

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
    obs_arr = []
    for i in st_arr:
        if i in OBS:
            obs_arr.append(OBS.index(i))
    # print(state_arr)
    # print(obs_arr[:18])

    emm_dict = {}
    for i in range(len(state_arr) - 1):
        s = state_arr[i], obs_arr[i]
        if s in emm_dict:
            emm_dict[s] += 1
        else:
            emm_dict[s] = 1
    # print(emm_dict)
    print("state observation transition matrix for   "+str(filename))
    e = np.zeros((27, 64))
    for key in emm_dict.keys():
        e[key] = emm_dict[key]

    for i in range(27):
        s = sum(e[i]) + 64
        for j in range(64):
            e[i, j] = (e[i, j] + 1)
            e[i, j] = e[i, j] / s
    #    print(e[i].sum())
    obs_matrix=e

    data2=pd.DataFrame(data=obs_matrix)
    data2.to_csv("F://hmm-master//hmm-master//output//observation//"+str(filename)+"_observation.csv")
    print("Observation matrix generated.\n")
    print("Ready for HMM model.....")
    model = hmm.MultinomialHMM(n_components=27)
    model.startprob_ = np.ones(27) / 27
    model.transmat_ = state_matrix
    model.emissionprob_ = obs_matrix

    logprob, seq = model.decode(np.array([obs_arr[18:]]).transpose())

    print("math.exp(logprob) = ", math.exp(logprob))
    print("seq = ", seq)
    df5 = pd.DataFrame(data=seq)
    df5.to_csv("F://hmm-master//hmm-master//output//final_seq//"+str(filename)+"_final_seq.csv")
    print(f"{filename} final sequence csv created.")


