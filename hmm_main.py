import os
import pandas as pd
import numpy as np
from hmmlearn import hmm
import math
from utils import state_calculator, symbol_calculator
from state import state_matrix
from obs import obs_matrix, obs_arr

obs_path = "D://project//hmm//data//observation//"
files = os.listdir(obs_path)
state_path = "D://project//hmm//data//state//"

output_path = "D://project//hmm//output//"

for filename in files:
    # -------------------: State :----------------------
    df2 = pd.read_csv(state_path + str(filename)).dropna()

    df = df2[:round(len(df2) * 0.8)]
    state_matrix_output = state_matrix(df)
    print("\nState calculation.. for " + str(filename))

    data = pd.DataFrame(data=state_matrix_output)
    data.to_csv(output_path + "state//" + str(filename) + "_state.csv")
    print("State transition matrix generated successfully\n")

    # -----------: Observation :-----------------
    print("\nObservation table calculation .....for " + str(filename))
    df3 = pd.read_csv(obs_path + str(filename)).dropna()
    df4 = df3[:round(len(df3) * 0.8)]

    obs_matrix_output = obs_matrix(df4)

    data2 = pd.DataFrame(data=obs_matrix_output)
    data2.to_csv(output_path + "observation//" + str(filename) + "_observation.csv")
    print("Observation matrix generated.\n")

    # ----: Final Seq :----
    print("Ready for HMM model.....")
    model = hmm.MultinomialHMM(n_components=27)
    model.startprob_ = np.ones(27) / 27
    model.transmat_ = state_matrix_output
    model.emissionprob_ = obs_matrix_output

    test_split = int(len(obs_arr) * 0.2)
    logprob, seq = model.decode(np.array([obs_arr[test_split:]]).transpose())

    print("math.exp(logprob) = ", math.exp(logprob))
    print("seq = ", seq)

    df5 = pd.DataFrame(data=seq)
    df5.to_csv(output_path + "final_seq//" + str(filename) + "_final_seq.csv")
    print(f"{filename} final sequence csv created.")