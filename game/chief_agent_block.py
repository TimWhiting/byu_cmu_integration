import copy
import numpy as np
import os

from simple_rl.agents import Agent, QLearningAgent, RandomAgent
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class CloneStructure(nn.Module):
	def __init__(self, input_size, hidden_layer_size, dropout_rate, n_actions):
		super(CloneStructure, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(input_size, hl_size),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(hl_size, hl_size),
			nn.ReLU(),
			nn.Dropout(p=0.3),
			nn.Linear(hl_size,n_actions)
			)

	def forward(self, x):
		return self.network(x)


class PlayerPoolWithClones(object):
	def __init__(self, agents, n_states, hidden_layer_size, dropout_rate, n_actions, train_data_file, model_parameter_file):
		self.agents = agents
		self.clones = dict()
		self.n_states = n_states
		self.n_actions = n_actions

		for a in agents:
			clone_structure = CloneStructure(n_states,hidden_layer_size,dropout_rate,n_actions)
			optimizer = torch.optim.SGD(clone_structure.parameters(), lr=0.001, momentum=0.9)
			self.clones[a] = (clone_structure, optimizer)

	def get_probs(self, state):
		probs = {}

		for a in self.agents:
			probs[a.name] = (nn.functional.softmax(self.clones[a][0](state))).numpy()

		return probs

	def train_clones(self):
		if os.state(self.model_parameter_file).st_size > 0:
			print("Models already trained")
			saved_models = torch.load(model_parameter_file)

			for a in self.agents:
				self.clones[a.name][0].load_state_dict(saved_models[a.name])
				self.clones[a.name][1].load_state_dict(saved_models[a.name + "_optim"])

			return

		for a in self.agents:
			train_x = []
			train_y = []

			if os.stat(self.train_data_file).st_size == 0:
				##########################
				# Generate training data #
				##########################

			else:
				######################
				# Load training data #
				######################



			agent_clone, agent_optim = self.clones[a.name]
			data_set = TensorDataset(torch.Tensor(train_x),torch.Tensor(train_y))
			data_loader = DataLoader(data_set, batch_size=20, shuffle=True)

			for e in range(10):
				for batch, (X,y) in enumerate(data_loader):
					pred = model(X)
					loss = nn.functional.cross_entropy(pred,y)

					agent_optim.zero_grad()
			        loss.backward()
			        agent_optim.step()

		print("training done")

