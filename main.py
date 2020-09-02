from hmmlearn import hmm
import numpy as np
import math
from state import STATE_MATRIX
from observation import emm_dict, OBS_MATRIX


model = hmm.MultinomialHMM(n_components=27)
model.startprob_ = np.ones(27) / 27
model.transmat_ = STATE_MATRIX
model.emissionprob_ = OBS_MATRIX

logprob, seq = model.decode(np.array([[0, 0, 0, 2, 1, 0]]).transpose())

print("math.exp(logprob) = ", math.exp(logprob))
print("seq = ", seq)
