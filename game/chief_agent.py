import copy
import numpy as np

from simple_rl.agents import Agent, QLearningAgent, RandomAgent

class PlayerPool(Object):
	def __init__(self, agents, sample_size=100, probs_method='sampling'):
		self.size = len(agents)
		self.agents = copy.deepcopy(agents)
		self.sample_size = sample_size
		self.probs_method = probs_method # 'sampling' or 'qvalues'

	def get_size(self):
		return self.size

	def get_probs(self, state, action):
		if probs_method == 'sampling':
			return self._get_probs_sampling(state, action)
		elif probs_method == 'qvalues':
			return self._get_probs_qvalues(state, action)

	def _get_probs_sampling(self, state, action):
		probs = []

		for a in self.agents:
			num_matches = 0

			for i in range(self.sample_size):
				q = a.policy(state)
				num_matches += 1*(q == action)

			probs.append(num_matches)

		return np.array(probs)

	def _get_probs_qvalues(self, state, action):
		# Assumes that all agents are QLearningAgents
		probs = []

		for a in self.agents:
			p = a.get_action_distr(state)[action]
			probs.append(p)

		return np.array(probs)


class ChiefAgent(Agent):
	def __init__(self, player_pool, bayesian_prior=None):
		self.player_pool = player_pool # Needs to be of type PlayerPool
		self.pool_size = player_pool.get_size()
		self.current_MLE_values = np.zeros(self.pool_size)
		self.moves_recorded = 0

		if bayesian_prior:
			self.current_bayesian_values = bayesian_prior

			if len(bayesian_prior != self.pool_size):
				raise Exception("Bayesian prior length doesn't match agent pool")
		else:
			self.current_bayesian_values = np.ones(self.pool_size)/self.pool_size

	def _probability_update(self, state, action):
		# State --> the state from the teammates perspective
		# Action --> the action taken by our teammate from this state
		probs = self.player_pool.get_probs(state, action)

		self.current_MLE_values = (probs + self.current_MLE_values*self.moves_recorded)/(self.moves_recorded + 1)
		self.current_bayesian_values = probs*self.current_bayesian_values/(np.dot(probs, self.current_bayesian_values))
		self.moves_recorded += 1

