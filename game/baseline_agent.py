import copy
import numpy as np

from simple_rl.agents import Agent
from collections import defaultdict

class BaselineAgent(Agent):
	def __init__(self, actions, name, gamma=0.99, partner_idx=1):
		Agent.__init__(self, name=name, actions=actions, gamma=gamma)

		self.prev_state = None
		self.partner_idx = partner_idx
		self.partner_recorder = defaultdict(lambda : defaultdict(int)) # state -> action -> count


	def record_teammate_action(self, state, action):
		# State --> the state from the teammates perspective
		# Action --> the action taken by our teammate from this state
		self.partner_recorder[state][action] += 1

	def act(self, state, reward):
		if self.prev_state:
			self.record_teammate_action(self.prev_state, self.actions[state.selection[self.partner_idx]])
			self.prev_state = state
		else:
			self.prev_state = state

		return np.random.choice(self.actions)

	def get_predicted_action(self, state):
		d = self.partner_recorder[state]
		try:
			return max(d, key=d.get)
		except:
			return np.random.choice(self.actions)

	def reset(self):
		self.prev_state = None
		self.partner_recorder = defaultdict(defaultdict(int))		