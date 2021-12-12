import math
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
# config epoch stream
with open('tools/mmap_observation.pkl', 'rb') as f:
    mmap_observation = pickle.load(f)
n_config, n_epoch, n_stream = mmap_observation.shape
data = mmap_observation.reshape(n_config, n_epoch * n_stream)
score = np.zeros(n_epoch * n_stream)

for i in range(n_epoch * n_stream):
    y = data[:, i]
    x = np.arange(n_config).reshape(-1, 1)
    x = np.hstack((x, x[:, 0: 1] ** 2))
    x = np.hstack((x, x[:, 0: 1] ** 3))
    model = LinearRegression().fit(x, y)
    score[i] = model.score(x, y)
print(score.mean())
