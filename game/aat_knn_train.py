import pickle
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from chief_block_aat import run_games
from new_chief_block_aat import run_games_with_new_chief

# Change this boolean flag if to switch between the old and new chief agent
use_new_chief = False

run_games_with_new_chief(
    train=True) if use_new_chief else run_games(train=True)

print('Training KNN model...')

data_dir = './training_data/'
training_data_file = 'training_data_new_chief.pickle' if use_new_chief else 'training_data.pickle'

with open(data_dir + training_data_file, 'rb') as f:
    training_data = np.array(pickle.load(f))

print(training_data[0])
print(training_data[-1])

x = training_data[:, 0:-2]
y = training_data[:, -1]

print('X train shape: ' + str(x.shape))
print('Y train shape: ' + str(y.shape))

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


def distance_func(x, y):
    play_num_dist = abs(x[0] - y[0])
    round_num_dist = 2 * abs(x[1] - y[1])
    curr_avg_payoff_dist = 2 * abs(x[2] - y[2])
    human_static_during_round_dist = 2 * abs(x[3] - y[3])
    human_static_across_rounds_dist = 2 * abs(x[4] - y[4])
    human_captured_in_experts_dist = 5 * abs(x[5] - y[5])

    return sum([play_num_dist, round_num_dist, curr_avg_payoff_dist, human_static_during_round_dist, human_static_across_rounds_dist, human_captured_in_experts_dist])


# parameters = {'weights': ['distance'],
#               'n_neighbors': [15],
#               'metric': [distance_func]}

# model = GridSearchCV(KNeighborsRegressor(), parameters)
# model.fit(x_scaled, y)

# print('Best score: ' + str(model.best_score_))
# print('Best parameters: ' + str(model.best_params_))

model = NearestNeighbors(n_neighbors=15, metric=distance_func)
model.fit(x_scaled)

trained_knn_file = 'trained_knn_aat_new_chief.pickle' if use_new_chief else 'trained_knn_aat.pickle'
trained_knn_scaler_file = 'trained_knn_scaler_aat_new_chief.pickle' if use_new_chief else 'trained_knn_scaler_aat.pickle'

with open(data_dir + trained_knn_file, 'wb') as f:
    pickle.dump(model, f)

with open(data_dir + trained_knn_scaler_file, 'wb') as f:
    pickle.dump(scaler, f)
