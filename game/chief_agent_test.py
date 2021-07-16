from chief_agent import PlayerPool, ChiefAgent
from alternator import *
from simple_rl.agents import QLearningAgent, FixedPolicyAgent
from copy import deepcopy
from collections import defaultdict
from simple_rl.run_experiments import play_markov_game


markov_game = AlternatorMDP()

policy1 = (lambda x: ACTIONS[x.selection[1]]) # return teammate's previous action
policy2 = (lambda x: ACTIONS[(x.selection[1] + 1)%3]) # return cycling through own moves

pool_agents = [FixedPolicyAgent(policy=policy1, name="agent1"), FixedPolicyAgent(policy=policy2, name="agent2")]
temp_teammates = deepcopy(pool_agents)

for a in temp_teammates:
	a.name += "teammate"


#  training
# for i in range(len(pool_agents)):
# 	agents = [pool_agents[i], temp_teammates[i]]
# 	print([a.actions for a in agents])

# 	for episode in range(20000):
# 		markov_game.reset()
# 		state = markov_game.get_init_state()

# 		for steps in range(200):
# 			action_dict = dict()
# 			reward_dict = defaultdict(float)

# 			for a in agents:
# 				agent_reward = reward_dict[a.name]
# 				agent_action = a.act(state, agent_reward)
# 				action_dict[a.name] = agent_action

# 			reward_dict, next_state = markov_game.execute_agent_action(action_dict)

# 			state = next_state

player_pool = PlayerPool(pool_agents, sample_size=1)

human_idx = 0
human_teammate = deepcopy(pool_agents[human_idx])
chief_player = ChiefAgent(actions=markov_game.get_actions(), name="chief", player_pool=player_pool, partner_idx=human_idx)


print("Setup done")
print("Index of teammate:", human_idx)

# execution
if human_idx == 0:
	agents = [human_teammate, chief_player]
else:
	agents = [chief_player, human_teammate]

for episode in range(10):
	markov_game.reset()
	state = markov_game.get_init_state()

	res = 0
	step_num = 10

	for steps in range(step_num):
		action_dict = dict()
		reward_dict = defaultdict(float)

		prediction = chief_player.get_predicted_action(state)

		for a in agents:
			agent_reward = reward_dict[a.name]
			agent_action = a.act(state, agent_reward)
			action_dict[a.name] = agent_action

		if prediction == action_dict[human_teammate.name]:
			res += 1

		reward_dict, next_state = markov_game.execute_agent_action(action_dict)

		state = next_state

	print("bayesian inference", chief_player.current_bayesian_values)
	print("MLE values", chief_player.current_MLE_values)

	print("Matches:", res, "out of", step_num)