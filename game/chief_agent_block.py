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
			self.clones[a.name] = (clone_structure, optimizer)

	def get_probs(self, state):
		probs = {}
		input_encoding = self.state_to_input(state)

		for a in self.agents:
			probs[a.name] = (nn.functional.softmax(self.clones[a][0](input_encoding))).numpy()

		return probs

	def state_to_input(self, state):
		pass

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



class Chief_Agent_BlockGame(Agent):
	def __init__(self, name, actions, playerpoolwc):
		super().__init__(name, actions)
		self.agent_mapping = dict()
		self.playerpoolwc = playerpoolwc
		counter = 0

		for a in playerpoolwc.clones:
			self.agent_mapping[a] = counter
			counter += 1

		self.num_agents = counter
		self.bayesian_inference_distribution = np.ones(counter)/counter

		self.epsilon = 0.3

	def act(self, state, reward):
		explore = np.random.random() <= self.epsilon

		if explore:
			return np.random.choice(actions)
		else:
			return self._maximize(state, self.get_predicted_action(state))

	def _maximize(self, state, action):
		pass

	def reset(self):
		pass

	def get_predicted_action(self, state):
		preds = np.zeros(len(self.actions))
		probs = self.playerpoolwc.get_probs(state)

		for a in selfplayerpoolwc.clones:
			preds += self.bayesian_inference_distribution[self.agent_mapping[a]]*probs[a]

		return np.argmax(preds)

	def make_prob(self, probs):
		distr = probs/np.sum(probs)
		distr[-1] = 1 - distr[:-1]
		return distr

	def bayes_update(self, state, action):
		action_probs = self.playerpoolwc.get_probs(state)
		agent_probs = np.zeros(self.num_agents)

		for a in self.playerpoolwc.clones:
			agent_probs[self.agent_mapping[a]] = action_probs[a][action]

		self.bayesian_inference_distribution = self.make_prob(agent_probs*self.bayesian_inference_distribution)