from hmmlearn import hmm
import numpy as np
import pandas as pd
import math
from state import STATE_MATRIX, filename
from observation import emm_dict, OBS_MATRIX
import os

current_path = os.getcwd()

model = hmm.MultinomialHMM(n_components=27)
model.startprob_ = np.ones(27) / 27
model.transmat_ = STATE_MATRIX
model.emissionprob_ = OBS_MATRIX

logprob, seq = model.decode(np.array([[25, 15, 11, 28, 25, 7,25,36,45,12,10,0,56,2,0,0,0,0,0,0,0,]]).transpose())

print("math.exp(logprob) = ", math.exp(logprob))
print("seq = ", seq)


d = seq
df = pd.DataFrame(data=d)
df.to_csv(f"{current_path}//output//final_seq//{filename}_final_seq.csv")
print(f"{filename} final sequence csv created.")