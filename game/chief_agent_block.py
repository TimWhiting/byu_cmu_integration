import copy
import numpy as np
import os
import json
from collections import defaultdict

from simple_rl.agents import Agent, QLearningAgent, RandomAgent
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CloneStructure(nn.Module):
    def __init__(self, input_size, hidden_layer_size, dropout_rate, n_actions):
        super(CloneStructure, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_layer_size, n_actions),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.network(x)
        return x


class PlayerPoolWithClones(object):
    def __init__(self, agents, n_states, hidden_layer_size, dropout_rate, n_actions, train_data_file, model_parameter_file, markov_game_mdp):
        self.agents = agents
        self.clones = dict()
        self.n_states = n_states  # Will be 10 for Block game
        self.n_actions = n_actions  # Will be 9 for Block game
        self.action_encoding = dict()
        self.markov_game_mdp = markov_game_mdp
        self.train_data_file = train_data_file
        self.model_parameter_file = model_parameter_file

        for a in agents:
            clone_structure = CloneStructure(
                n_states, hidden_layer_size, dropout_rate, n_actions)
            optimizer = torch.optim.SGD(
                clone_structure.parameters(), lr=0.001, momentum=0.9)
            self.clones[a.name] = (clone_structure, optimizer)

        for i, action in enumerate(markov_game_mdp.get_actions()):
            self.action_encoding[action] = i

    def get_probs(self, state):
        probs = {}
        input_encoding = self.state_to_input(state)

        for a in self.agents:
            probs[a.name] = (nn.functional.softmax(
                self.clones[a][0](input_encoding))).numpy()

    for a in playerpoolwc.clones:
        self.agent_mapping[a] = counter
        counter += 1

    def state_to_input(self, state):
        f = state.features()
        return f[0] + [f[1]]

    def action_to_output(self, action):
        return self.action_encoding[action]

    def resave_model_params(self):
        all_params_dict = dict()

        for c in self.clones:
            all_params_dict[c] = self.clones[c][0]
            all_params_dict[c + "_optim"] = self.clones[c][1]

        torch.save(all_params_dict, self.model_parameter_file)

	def train_clones(self):
		#############################
		# Loading any existing data #
		#############################
		with open(self.train_data_file, "r") as f:
			try:
				d = json.load(f)
			except Exception:
				d = {}

		try:
            model_params = torch.load(self.model_parameter_file)
        except Exception:
            model_params = {}

		######################################
		# Getting training data and training #
		######################################
		for agent_blueprint in self.agents:
			if agent_blueprint.name in model_params:
				print(agent_blueprint.name, " already has trained parameters. Loading now...")
				self.clones[agent_blueprint.name][0].load_state_dict(model_params[agent_blueprint.name])
				self.clones[agent_blueprint.name][1].load_state_dict(model_params[agent_blueprint.name + "_optim"])
				continue

			if not (agent_blueprint.name in d): # If no training data exists for this agent
				for teammate_blueprint in self.agents:
					x_addition, y_addition = self._gen_data(agent_blueprint, teammate_blueprint)
					train_x += x_addition
					train_y += y_addition

				d[agent_blueprint.name] = (train_x, train_y)
			else:
				train_x, train_y = d["train X", "train Y"]

			self._train_loop_clone(agent_blueprint.name, train_x, train_y)

		##################################
		# Update files with any new data #
		##################################
		with open(self.train_data_file, "w") as f:
			json.dump(d,f)

		self.resave_model_params()
		print("training done")

	def _gen_data(self, agent_blueprint, teammate_blueprint, episodes=200):
		data_x, data_y = [], []

		if np.random.random() < 0.5:
			agent1 = copy.deepcopy(agent_blueprint)
			agent2 = copy.deepcopy(teammate_blueprint)
			agent_ind = 0
		else:
			agent1 = copy.deepcopy(teammate_blueprint)
			agent2 = copy.deepcopy(agent_blueprint)
			agent_ind = 1

		agent1.name = "a1"
		agent2.name = "a2"
		agent_dict = {}

		for a in [agent1,agent2]:
			agent_dict[a.name] = a

		for episode in range(episodes):
			reward_dict = defaultdict(str)
			action_dict = {}

			# Compute initial state/reward.
			state = self.markov_game_mdp.get_init_state()

			for step in range(30):

				# Compute each agent's policy.
				for a in agent_dict.values():
					agent_reward = reward_dict[a.name]
					agent_action = a.act(state, agent_reward)
					action_dict[a.name] = agent_action

				# Terminal check.
				if state.is_terminal():
					break

				if (state.turn == agent_ind):
					data_x.append(self.state_to_input(state))
					data_y.append(self.action_to_output(action_dict[list(action_dict.keys())[state.turn]]))

				# Execute in MDP.
				reward_dict, next_state = self.markov_game_mdp.execute_agent_action(action_dict)

				# Update pointer.
				state = next_state

			# Reset the MDP, tell the agent the episode is over.
			self.markov_game_mdp.reset()

		return (data_x, data_y)

	def _train_loop_clone(self, agent_name, train_x, train_y, update_models=True, epochs=10):
		agent_clone, agent_optim = self.clones[agent_name]
		
		data_set = TensorDataset(torch.Tensor(train_x),torch.Tensor(train_y))
		data_loader = DataLoader(data_set, batch_size=20, shuffle=True)
		loss_fn = nn.NLLLoss()

		for e in range(epochs):
			avg_loss = 0

			for batch, (X,y) in enumerate(data_loader):
				pred = agent_clone(X)
				loss = loss_fn(pred, y.to(torch.long))
				avg_loss += loss

				if update_models:
					agent_optim.zero_grad()
					loss.backward()
					agent_optim.step()

			if update_models:
				print(agent_name, " - ", e, ":", avg_loss/batch)
			else:
				print(agent_name, ":", avg_loss/batch)
				print(torch.argmax(pred,axis=1),y)
				break

		if update_models:
			self.clones[agent_name] = (agent_clone, agent_optim)

	def validate_clones(self):
		for agent_blueprint in self.agents:
			test_x, test_y = [], []

			for teammate_blueprint in self.agents:
				test_x_addition, test_y_addition = self._gen_data(agent_blueprint, teammate_blueprint, episodes=500)
				test_x += test_x_addition
				test_y += test_y_addition

			self._train_loop_clone(agent_blueprint.name, test_x, test_y, update_models=False)


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
            return self._maximize(state)

    # Going to try to maximize our immediate reward - opponents predicted next reward
    def _maximize(self, state):
		D = dict()
		player_ind = state.turn

		for a in state.valid_moves():
			r = 0

			for sample in range(50):
				r += self._simulate_game(state, a, player_ind)

			D[a] = r/50

        return max(D, key=D.get)

    # The CHIEF will always assume that it is the state.turn agent (if not doesn't matter because game won't care about CHIEF's action)
    def _simulate_game(self, state_input, initial_action, chief_ind):
    	game = self.playerpoolwc.markov_game_mdp
       	action_tracker = initial_action
    	chief_reward = 0

		chief_name = "c"
		other_player_name = "p"

		reward_dict = defaultdict(str)
		action_dict = {}
		state = state_input

		for step in range(30):
			if chief_ind = 0:
				action_dict[chief_name] = action_tracker
				action_dict[other_player_name] = self.get_predicted_action(state)
			else:
				action_dict[other_player_name] = self.get_predicted_action(state)
				action_dict[chief_name] = action_tracker

			# Terminal check.
			if state.is_terminal():
				break

			# Execute in MDP.
			reward_dict, next_state = self.markov_game_mdp.execute_agent_action(action_dict)
			chief_reward += reward_dict[chief_name]

			# Update pointer.
			state = next_state

			if state.turn == chief_ind:
				action_tracker = np.random.choice(state.valid_moves())

		return chief_reward


    def reset(self):
        # This will never need to reset any parameters, since they can just be used over multiple games
        return

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

        self.bayesian_inference_distribution = self.make_prob(
            agent_probs*self.bayesian_inference_distribution)
