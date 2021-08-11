import pickle
from sklearn.neighbors import KNeighborsRegressor
import random

k = 5
n_samples = 4500

data_dir = './training_data/'

with open(data_dir + 'training_data.pickle', 'rb') as f:
	training_data = pickle.load(f)

# randomly select samples
if n_samples < len(training_data):
	sample_idxs = random.sample(list(range(len(training_data))), n_samples)
	training_data_filtered = [training_data[sample_idx] for sample_idx in sample_idxs]
else:
	training_data_filtered = training_data.copy()

# format the data into inputs and outputs
X = [example[0:-1] for example in training_data_filtered]
Y = [example[-1] for example in training_data_filtered]

# fit a KNN regression
knn = KNeighborsRegressor(n_neighbors=k)
knn.fit(X, Y)

print(knn.predict([[1, 8, 0.75], [0.2, 8, 0.75]]))