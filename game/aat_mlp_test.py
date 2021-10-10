import pickle
import numpy as np
from typing import List
from chief_block_aat import run_games

data_dir = './training_data/'

model = pickle.load(open(data_dir + 'trained_mlp_aat.pickle', 'rb'))
scaler = pickle.load(open(data_dir + 'trained_mlp_scaler_aat.pickle', 'rb'))


def mlp_aat_prediction_func(x: List) -> float:
    x = np.array(x).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)

    return pred


run_games(train=False, aat_predict_func=mlp_aat_prediction_func)
