import pickle
import numpy as np
from typing import List, Tuple
from chief_block_aat import run_games
from new_chief_block_aat import run_games_with_new_chief

# Change this boolean flag if to switch between the old and new chief agent
use_new_chief = False

data_dir = './training_data/'
training_data_file = 'training_data_new_chief.pickle' if use_new_chief else 'training_data.pickle'

with open(data_dir + training_data_file, 'rb') as f:
    training_data = np.array(pickle.load(f))

print(training_data[0])
print(training_data[-1])


def distance_func(x, y):
    play_num_dist = abs(x[0] - y[0])
    round_num_dist = 2 * abs(x[1] - y[1])
    curr_avg_payoff_dist = 2 * abs(x[2] - y[2])
    human_static_during_round_dist = 2 * abs(x[3] - y[3])
    human_static_across_rounds_dist = 2 * abs(x[4] - y[4])
    human_captured_in_experts_dist = 5 * abs(x[5] - y[5])

    return sum([play_num_dist, round_num_dist, curr_avg_payoff_dist, human_static_during_round_dist, human_static_across_rounds_dist, human_captured_in_experts_dist])


trained_knn_file = 'trained_knn_aat_new_chief.pickle' if use_new_chief else 'trained_knn_aat.pickle'
trained_knn_scaler_file = 'trained_knn_scaler_aat_new_chief.pickle' if use_new_chief else 'trained_knn_scaler_aat.pickle'

model = pickle.load(open(data_dir + trained_knn_file, 'rb'))
scaler = pickle.load(open(data_dir + trained_knn_scaler_file, 'rb'))


# def knn_aat_prediction_func(x: List) -> float:
#     x = np.array(x).reshape(1, -1)
#     x_scaled = scaler.transform(x)
#     pred = model.predict(x_scaled)

#     return pred

def knn_aat_prediction_func(x: List) -> Tuple[List, List, List]:
    x = np.array(x).reshape(1, -1)
    x_scaled = scaler.transform(x)
    neighbor_distances, neighbor_indices = model.kneighbors(x_scaled, 15)

    predictions = []
    corrections = []
    distances = []

    for i in range(len(neighbor_indices[0])):
        neighbor_idx = neighbor_indices[0][i]
        neighbor_dist = neighbor_distances[0][i]
        predictions.append(training_data[neighbor_idx, -1])
        corrections.append(training_data[neighbor_idx, -2])
        distances.append(neighbor_dist)

    return predictions, corrections, distances


run_games_with_new_chief(
    train=False, aat_predict_func=knn_aat_prediction_func) if use_new_chief else run_games(train=False, aat_predict_func=knn_aat_prediction_func)
