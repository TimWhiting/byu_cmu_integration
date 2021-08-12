from chief_agent import PlayerPool, ChiefAgent, ChiefAgentWithAAT
from baseline_agent import BaselineAgent
from alternator import *
from simple_rl.agents import QLearningAgent, FixedPolicyAgent
from copy import deepcopy
from collections import defaultdict
from simple_rl.run_experiments import play_markov_game
import matplotlib.pyplot as plt
from tabulate import tabulate
from numpy import random
import os
import pickle

markov_game = AlternatorMDP()

# policy1 = (lambda x: ACTIONS[x.selection[1]) # return teammate's previous action
# policy2 = (lambda x: ACTIONS[(x.selection[1] + 1)%3]) # return cycling through own moves

## VISUALIZATIONS (Look at PyGame) TODO: Arnav / Mike


def create_agents(other_player):
	this_player = (other_player + 1) % 2

	rand = FixedPolicyAgent(policy=(lambda x: random.choice(ACTIONS)), name='Random')

	### Fixed Policy Agents
	# Minimax Agent
	minimax = FixedPolicyAgent(policy=(lambda x: ACTIONS[1]), name='Minimax')
	# Maximize Both Player's Payoffs
	max_welfare1 = FixedPolicyAgent(policy=(lambda x: ACTIONS[(2-x.selection[this_player]) % 3] if x.selection[this_player] != -1 else ACTIONS[0]), name='MaxWelfare')
	max_welfare2 = FixedPolicyAgent(policy=(lambda x: ACTIONS[(2-x.selection[this_player]) % 3] if x.selection[this_player] != -1 else ACTIONS[2]), name='MaxWelfare2')
	# Maximize Own Payoffs
	max_self = FixedPolicyAgent(policy=(lambda x: ACTIONS[0]), name='MaxSelf')
	# Maximize Their Payoffs
	max_other = FixedPolicyAgent(policy=(lambda x: ACTIONS[2]), name='MaxOther')

	### Adaptive Agents TODO: Tim / Najma
	# Tit for Tat
	tit_for_tat = FixedPolicyAgent(policy=(lambda x: ACTIONS[x.selection[other_player]] if x.selection[other_player] != -1 else ACTIONS[random.randint(0, 2)]), name='Tit for Tat')
	# Tit for 2 Tats

	# Agent that adapts over time to become more efficient
	efficiency = FixedPolicyAgent(policy=(lambda x:  minimax.policy(x) if x.round < 5 else tit_for_tat.policy(x) if x.round < 10 else max_welfare1.policy(x)), name='Efficiency')

	# Ficticious Play Agent

	return {
		"random":rand,
		"minimax":minimax,
		"max_welfare1":max_welfare1,
		"max_welfare2":max_welfare2,
		"max_self":max_self,
		"max_other":max_other,
		"tit_for_tat":tit_for_tat,
		"efficiency":efficiency
	}

human_idx = 1
pool_agents = create_agents(1 - human_idx) # other player from human point of view is us
player_pool = PlayerPool(list(pool_agents.values()), sample_size=10)
mirrored_agents = create_agents(human_idx) # other player from our point of view is human
mirrored_pool = PlayerPool(list(mirrored_agents.values()), sample_size=10)

n_iterations = 20
epsilons = [0, 0.2, 0.3, 0.5, 0.7]

data_dir = './training_data/'
os.makedirs(data_dir, exist_ok=True)

training_data = []
for iter in range(n_iterations):
	for human_strategy in pool_agents.keys():
		print('Strategy: {}'.format(human_strategy))

		for epsilon in epsilons:
			print('Epsilon: {}'.format(epsilon))

			human_teammate = deepcopy(pool_agents[human_strategy])

			output_assumptions = [(lambda chief,s,a,next_s: chief.get_predicted_action(s) == chief.actions[next_s.selection[chief.partner_idx]])]
			estimation_model = lambda baseline,num_rounds,average_accuracy : (baseline + num_rounds*average_accuracy)/(num_rounds + 1)

			chief_player = ChiefAgentWithAAT(estimation_model=estimation_model, output_assumptions=output_assumptions, actions=markov_game.get_actions(), name="chief", player_pool=player_pool, mirrored_player_pool=mirrored_pool, partner_idx=human_idx, likelihood_threshold=0.6)

			probabilities_over_time = dict(zip(list(pool_agents.keys()), [[]]*len(pool_agents)))
			# print(probabilities_over_time)
			correct_predictions_over_time = []
			total_correct = 0
			average_accuracy_till_now = []

			# print("Setup done")
			# print("Index of teammate:", human_idx)

			# execution
			if human_idx == 0:
				agents = [human_teammate, chief_player]
			else:
				agents = [chief_player, human_teammate]

			def format_nums(L):
				LL = []

				for l in L:
					LL.append(round(l,3))

				return LL

			baseline_player = BaselineAgent(markov_game.get_actions(), "baseline", partner_idx=human_idx)
			baseline_accuracy_over_time = []
			total_baseline = 0


			markov_game.reset()
			state = markov_game.get_init_state()

			res = 0
			step_num = 40

			for step in range(step_num):
				action_dict = dict()
				reward_dict = defaultdict(float)

				prediction = chief_player.get_predicted_action(state)

				for a in agents:
					agent_reward = reward_dict[a.name]
					agent_action = a.act(state, agent_reward)

					if a.name != "chief" and random.random() < epsilon: # adding noise
						agent_action = random.choice(["A","B","C"])

					action_dict[a.name] = agent_action

				baseline_player.act(state, agent_reward)
				baseline_prediction = baseline_player.get_predicted_action(state)
				total_baseline += int(baseline_prediction == action_dict[human_teammate.name])
				baseline_accuracy_over_time.append(total_baseline/(len(correct_predictions_over_time) + 1))

				correct_val = int(prediction == action_dict[human_teammate.name])
				total_correct += correct_val
				correct_predictions_over_time.append(correct_val)
				average_accuracy_till_now.append(total_correct/len(correct_predictions_over_time))

				performance_features = chief_player.obtain_performance_features()
				# print("-----PERFORMANCE ASSESSMENT-----")
				# print("assessment value:",round(estimation_model(*performance_features), 3))

				reward_dict, next_state = markov_game.execute_agent_action(action_dict)

				state = next_state

				training_data.append(performance_features)

			final_performance = average_accuracy_till_now[-1]

			for i in range(1, step_num + 1):
				training_data[-i].append(final_performance)

data_dir = './training_data/'
os.makedirs(data_dir, exist_ok=True)

with open(data_dir + 'training_data.pickle', 'wb') as f:
	pickle.dump(training_data, f)

xvals = list(range(len(correct_predictions_over_time)))

plt.plot(xvals, correct_predictions_over_time, 'bo')
plt.gca().set_ylim(0,1)
plt.title("Correct or not for each step")
plt.show()
plt.plot(xvals, average_accuracy_till_now)
plt.plot(xvals, baseline_accuracy_over_time, color="red")
plt.gca().set_ylim(0,1)
plt.title("Average accuracy till each step")
plt.show()