import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np

data_dir = './training_data/'

with open(data_dir + 'training_data.pickle', 'rb') as f:
    training_data = np.array(pickle.load(f))

x = training_data[:, 0:-1]
y = training_data[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


def distance_func(x, y):
    play_num_dist = abs(x[0] - y[0])
    round_num_dist = 2 * abs(x[1] - y[1])
    curr_avg_payoff_dist = 2 * abs(x[2] - y[2])
    human_static_during_round_dist = 2 * abs(x[3] - y[3])
    human_static_across_rounds_dist = 2 * abs(x[4] - y[4])
    human_captured_in_experts_dist = 5 * abs(x[5] - y[5])

    return sum([play_num_dist, round_num_dist, curr_avg_payoff_dist, human_static_during_round_dist, human_static_across_rounds_dist, human_captured_in_experts_dist])


# parameters = {'weights': ('uniform', 'distance'),
#               'n_neighbors': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
#               'metric': [distance_func, 'minkowski']}
parameters = {'weights': ['distance'],
              'n_neighbors': [15],
              'metric': [distance_func]}

model = GridSearchCV(KNeighborsRegressor(), parameters)
model.fit(x_train_scaled, y_train)

print('Best score: ' + str(model.best_score_))
print('Best parameters: ' + str(model.best_params_))

results = (y_test - model.predict(x_test_scaled))**2
print(np.mean(results), np.max(results), np.min(results), np.var(results))

with open(data_dir + 'trained_knn_aat.pickle', 'wb') as f:
    pickle.dump(model, f)

with open(data_dir + 'trained_knn_scaler_aat.pickle', 'wb') as f:
    pickle.dump(scaler, f)
