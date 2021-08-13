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
from gamesBfAltAAT import gamesBf,gamesBfAAT

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

human_idx = 0
pool_agents = create_agents(1 - human_idx) # other player from human point of view is us
player_pool = PlayerPool(list(pool_agents.values()), sample_size=10)
mirrored_agents = create_agents(human_idx) # other player from our point of view is human
mirrored_pool = PlayerPool(list(mirrored_agents.values()), sample_size=10)

#gameStr="blocks2"
#me=human_idx
#bfhuman = gamesBf(gameStr, me)
#bfai = gamesBf(gameStr, (1-me))


epsilons = [0]

training_data = []
for human_strategy in pool_agents.keys():
	if human_strategy=="minimax":
		print("look here")
	print('Strategy: {}'.format(human_strategy))

	for epsilon in epsilons:
		print('Epsilon: {}'.format(epsilon))

		human_teammate = deepcopy(pool_agents[human_strategy])

		output_assumptions = [(lambda partnerAction,bfAction: partnerAction == bfAction)]

		estimation_model = lambda baseline,num_rounds,average_accuracy : (baseline + num_rounds*average_accuracy)/(num_rounds + 1)

		bf_player = gamesBfAAT(estimation_model=estimation_model, output_assumptions=output_assumptions, name="BF", me=0, gameStr="blocks2")


		#probabilities_over_time = dict(zip(list(pool_agents.keys()), [[]]*len(pool_agents)))
		probabilities_over_time = dict(
			{"s0": [], "s1": [], "s2": [], "s3": [], "s4": [], "s5": [], "s6": [], "s7": [], "s8": []})

		# print(probabilities_over_time)
		correct_predictions_over_time = []
		total_correct = 0
		average_accuracy_till_now = []

		# print("Setup done")
		# print("Index of teammate:", human_idx)

		# execution
		if human_idx == 0:
			agents = [human_teammate, bf_player]
		else:
			agents = [bf_player, human_teammate]

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

		player_names = []
		# Bfagents= dict({"s0":"Maximin","s1":"Best Response","s2":"Bouncer","s3":"Follower-pure CX","s4":"Leader-pure CX","s5":"Follower-alt CX- AZ","s6":"Leader-alt CX- AZ","s7":"Follower-pure AZ","s8":"Leader pure AZ"})
		Bfagents = {"s0": "Maximin", "s1": "Best Response", "s2": "Bouncer", "s3": "F-CX", "s4": "L-CX",
					"s5": "FA-CX-AZ", "s6": "LA-CX-AZ", "s7": "F-AZ", "s8": "L-AZ"}
		player_names = list(Bfagents.values())
		actionsmap = dict({0: "A", 1: "B", 2: "C"})
		playerStr = ""
		me=0
		observations_z0 = "None"
		observations_z1 = "None"
		# action0=str(random.randint(0,2))# the code has to model this action
		# action1=random.randint(0,2)#the code plays this action
		prediction = random.randint(0, 2)
		action1 = str(prediction)  # the code plays this action
		print("random", action1)

		# print("Player: "+ str(me)+" played: "+ action0)
		# print("Player: "+ str(me)+" played: "+ action1)
		bf_player.calculatePriorBelief("None", playerStr, me)  # initialProposal is the 1st S# proposal

		actionsmap = dict({0: "A", 1: "B", 2: "C"})

		for step in range(step_num):
			action_dict = dict()
			reward_dict = defaultdict(float)

			#####prediction = chief_player.get_predicted_action(state)
			partnerAction= None

			for a in agents:
				agent_reward = reward_dict[a.name]
				if a.name=="BF":
					agent_action = actionsmap.get(prediction)
					action1= str(prediction)
					bfAction= agent_action
				else:
					agent_action = a.act(state, agent_reward)
					partnerAction=agent_action
					action0=str(list(actionsmap.keys())[list(actionsmap.values()).index(agent_action)])

				if a.name != "BF" and random.random() < epsilon: # adding noise
					agent_action = random.choice(["A","B","C"])
					action0=str(list(actionsmap.keys())[list(actionsmap.values()).index(agent_action)])

				action_dict[a.name] = agent_action

			baseline_player.act(state, agent_reward)
			baseline_prediction = baseline_player.get_predicted_action(state)
			total_baseline += int(baseline_prediction == action_dict[human_teammate.name])
			baseline_accuracy_over_time.append(total_baseline/(len(correct_predictions_over_time) + 1))

			correct_val = int(actionsmap.get(prediction) == action_dict[human_teammate.name])
			total_correct += correct_val
			correct_predictions_over_time.append(correct_val)
			average_accuracy_till_now.append(total_correct/len(correct_predictions_over_time))

			#get partner action- partnerAction
			#get your prediction- prediction, bfAction
			#check alignment profile

			bf_player._add_to_profile(partnerAction, bfAction)
			performance_features = bf_player.obtain_performance_features()#send baseline
			# print("-----PERFORMANCE ASSESSMENT-----")
			# print("assessment value:",round(estimation_model(*performance_features), 3))

			reward_dict, next_state = markov_game.execute_agent_action(action_dict)

			state = next_state
			bf_player.calculateBelbarCurrentState(observations_z0, observations_z1, action0, action1, step, me, playerStr)
			prediction = bf_player.playAction  # this is int number

			predictionString = actionsmap.get(prediction)  # this is string C
			#print("Prediction by BF: ", prediction)
			action1 = str(prediction)  # this is string 2
			print("training initial", training_data)
			training_data.append(performance_features)
			print(training_data)

		final_performance = average_accuracy_till_now[-1]

		for i in range(1, step_num + 1):
			training_data[-i].append(final_performance)

data_dir = './training_data/'
os.makedirs(data_dir, exist_ok=True)

with open(data_dir + 'training_data_najma.pickle', 'wb') as f:
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