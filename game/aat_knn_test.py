import pickle
import numpy as np
from typing import List
from chief_block_aat import run_games

data_dir = './training_data/'


def distance_func(x, y):
    play_num_dist = abs(x[0] - y[0])
    round_num_dist = 2 * abs(x[1] - y[1])
    curr_avg_payoff_dist = 2 * abs(x[2] - y[2])
    human_static_during_round_dist = 2 * abs(x[3] - y[3])
    human_static_across_rounds_dist = 2 * abs(x[4] - y[4])
    human_captured_in_experts_dist = 5 * abs(x[5] - y[5])

    return sum([play_num_dist, round_num_dist, curr_avg_payoff_dist, human_static_during_round_dist, human_static_across_rounds_dist, human_captured_in_experts_dist])


model = pickle.load(open(data_dir + 'trained_knn_aat.pickle', 'rb'))
scaler = pickle.load(open(data_dir + 'trained_knn_scaler_aat.pickle', 'rb'))


def knn_aat_prediction_func(x: List) -> float:
    x = np.array(x).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)

    return pred


run_games(train=False, aat_predict_func=knn_aat_prediction_func)
