import numpy as np
from utils import state_calculator

state_arr = []


def state_matrix(df):
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
    comparison_arr = []
    for i in range(0, len(like_state)):
        s = f"{like_state[i]}{retweet_state[i]}{tweet_state[i]}"
        comparison_arr.append(s)

    global state_arr
    for i in comparison_arr:
        if i in STATE:
            state_arr.append(STATE.index(i))

    trans_dict = {}
    for i in range(len(state_arr) - 1):
        s = state_arr[i], state_arr[i + 1]
        if s in trans_dict:
            trans_dict[s] += 1
        else:
            trans_dict[s] = 1

    a = np.zeros((27, 27))
    for key in trans_dict.keys():
        a[key] = trans_dict[key]

    for i in range(27):
        s = sum(a[i]) + 27
        for j in range(27):
            a[i, j] = (a[i, j] + 1)
            a[i, j] = a[i, j] / s

    return a
