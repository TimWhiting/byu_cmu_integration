import copy
import numpy as np

from simple_rl.agents import Agent, QLearningAgent, RandomAgent

class PlayerPool(object):
	def __init__(self, agents, sample_size=100, probs_method='sampling'):
		self.size = len(agents)
		self.agents = copy.deepcopy(agents)
		self.sample_size = sample_size
		self.probs_method = probs_method # 'sampling' or 'qvalues'

	def get_size(self):
		return self.size

	def get_agent_action(self, idx, state):
		return self.agents[idx].policy(state)

	def get_probs(self, state, action):
		if self.probs_method == 'sampling':
			return self._get_probs_sampling(state, action)
		elif self.probs_method == 'qvalues':
			return self._get_probs_qvalues(state, action)

	def _get_probs_sampling(self, state, action):
		probs = []

		for a in self.agents:
			num_matches = 0

			for i in range(self.sample_size):
				q = a.policy(state)
				num_matches += 1*(q == action)

			probs.append(num_matches/self.sample_size)

		return np.array(probs)

	def _get_probs_qvalues(self, state, action):
		# Assumes that all agents are QLearningAgents
		probs = []

		for a in self.agents:
			distr = a.get_action_distr(state)

			for i, prob in enumerate(distr):
				if (a.actions[i] == action):
					p = prob
					break

			probs.append(p)

		return np.array(probs)


class ChiefAgent(Agent):
	def __init__(self, actions, name, player_pool, mirrored_player_pool, gamma=0.99, bayesian_prior=None, likelihood_threshold=0.3, partner_idx=1):
		Agent.__init__(self, name=name, actions=actions, gamma=gamma)

		self.player_pool = player_pool # Needs to be of type PlayerPool
		self.mirrored_player_pool = mirrored_player_pool
		self.pool_size = self.player_pool.get_size()
		self.current_MLE_values = np.zeros(self.pool_size)
		self.moves_recorded = 0
		self.likelihood_threshold = likelihood_threshold
		self.bayesian_prior_init = bayesian_prior
		self.prev_state = None
		self.partner_idx = partner_idx
		self.exploration_prob = 0.7
		self.exploration_decrement = 0.05
		self.exploration_min = 0.3

		if bayesian_prior:
			self.current_bayesian_values = bayesian_prior

			if len(bayesian_prior != self.pool_size):
				raise Exception("Bayesian prior length doesn't match agent pool")
		else:
			self.current_bayesian_values = np.ones(self.pool_size)/self.pool_size

	def record_teammate_action(self, state, action):
		# State --> the state from the teammates perspective
		# Action --> the action taken by our teammate from this state
		probs = self.player_pool.get_probs(state, action)
		self.current_MLE_values = (probs + self.current_MLE_values*self.moves_recorded)/(self.moves_recorded + 1)
		denom = np.dot(probs, self.current_bayesian_values)
		self.current_bayesian_values = probs*self.current_bayesian_values/np.clip(denom, .0001, 1)

		if sum(self.current_bayesian_values) == 0:
			self.current_bayesian_values = np.ones(self.pool_size)/self.pool_size

		self.moves_recorded += 1

	def get_proposed_model_idx(self):
		proposed_model_idx = np.argmax(self.current_bayesian_values)

		if (self.current_MLE_values[proposed_model_idx] < self.likelihood_threshold):
			proposed_model_idx = np.argmax(self.current_MLE_values)

		return proposed_model_idx

	def act(self, state, reward):
		if self.prev_state:
			self.record_teammate_action(self.prev_state, self.actions[state.selection[self.partner_idx]])
			self.prev_state = state
		else:
			self.prev_state = state

		return self._best_response(self.get_proposed_model_idx(), state)

	def _best_response(self, proposed_model_idx, state):
		mirrored_action = self.mirrored_player_pool.get_agent_action(proposed_model_idx, state)

		if (np.random.random() < self.exploration_prob):
			return np.random.choice(self.actions)

		self.exploration_prob = max(self.exploration_prob - self.exploration_decrement, self.exploration_min)

		# returning action from mirrored agent
		return mirrored_action

	def get_predicted_action(self, state):
		return self.player_pool.get_agent_action(self.get_proposed_model_idx(), state)

	def reset(self):
		# don't need to reset playerpool, since we never update the agents inside it
		self.current_MLE_values = np.zeros(self.pool_size)
		self.moves_recorded = 0

		if self.bayesian_prior_init:
			self.current_bayesian_values = bayesian_prior
		else:
			self.current_bayesian_values = np.ones(self.pool_size)/self.pool_size



class ChiefAgentWithAAT(ChiefAgent):
	def __init__(self, estimation_model, output_assumptions, actions, name, player_pool, mirrored_player_pool, gamma=0.99, bayesian_prior=None, likelihood_threshold=0.3, partner_idx=1):
		ChiefAgent.__init__(self, actions, name, player_pool, mirrored_player_pool, gamma, bayesian_prior, likelihood_threshold, partner_idx)

		self.output_assumptions = output_assumptions
		self.alignment_profile = []
		self.estimation_model = estimation_model

	def performance_estimation(self):
		# fill in with function of baseline and alignment profile
		baseline_estimation = self.current_MLE_values[self.get_proposed_model_idx()]
		
		if (len(self.alignment_profile) == 0):
			return baseline_estimation

		feature_1 = len(self.alignment_profile)
		feature_2 = np.mean(np.array(self.alignment_profile), axis=0)[0]

		print("baseline:",round(baseline_estimation,3),"||| number of rounds:",feature_1,"||| average accuracy",round(feature_2,3))

		return self.estimation_model(baseline_estimation, feature_1, feature_2)

	def _add_to_profile(self, state, action, result):
		profile_item = []

		# output assumptions are lambda state, action, result: true/false

		for a in self.output_assumptions:
			profile_item.append(int(a(self, state, action, result)))

		self.alignment_profile.append(profile_item)

	def act(self, state, reward):
		prev_state = self.prev_state
		prev_action = self.actions[state.selection[self.partner_idx]]
		curr_state = state

		self._add_to_profile(prev_state, prev_action, curr_state)

		return ChiefAgent.act(self, state, reward)

	def reset(self):
		self.output_assumptions = []
		self.alignment_profile = []

		ChiefAgent.reset(self)