from chief_agent import PlayerPool, ChiefAgent, ChiefAgentWithAAT
from alternator import *
from simple_rl.agents import QLearningAgent, FixedPolicyAgent
from copy import deepcopy
from collections import defaultdict
from simple_rl.run_experiments import play_markov_game
import matplotlib.pyplot as plt
import sys, pygame
from tabulate import tabulate
from pygame.locals import *
from baseline_agent import BaselineAgent
from numpy import random
import os
import pickle

markov_game = AlternatorMDP()
pygame.init() # https://www.pygame.org/docs/tut/PygameIntro.html

screen = pygame.display.set_mode([350, 320])
pygame.display.set_caption("Alternator Game")
font = pygame.font.Font(None, 20)

A_img = pygame.transform.scale(pygame.image.load("game/images/A.png"), (50,80))
A_rect = A_img.get_rect()
A_rect.topleft = 55, 230
B_img = pygame.transform.scale(pygame.image.load("game/images/B.png"), (50,80))
B_rect = B_img.get_rect()
B_rect.topleft = 135, 230
C_img = pygame.transform.scale(pygame.image.load("game/images/C.png"), (50,80))
C_rect = C_img.get_rect()
C_rect.topleft = 215, 230

buttons = {"A":(A_img, A_rect), "B":(B_img, B_rect), "C":(C_img, C_rect)}

def inside_rectang(rec, point):
	x,y = point

	return rec.left <= x <= (rec.left + rec.width) and rec.top <= y <= (rec.top + rec.height)

def action_choice(action_coords):
	x,y = action_coords

	for button in buttons:
		_, rect = buttons[button]

		if (inside_rectang(rect, (x,y))):
			return button

	return None


# policy1 = (lambda x: ACTIONS[x.selection[1]) # return teammate's previous action
# policy2 = (lambda x: ACTIONS[(x.selection[1] + 1)%3]) # return cycling through own moves

## VISUALIZATIONS (Look at PyGame) TODO: Arnav / Mike


def create_agents(other_player):
	this_player = (other_player + 1) % 2

	rand = FixedPolicyAgent(policy=(lambda x : ACTIONS[random.choice([0,1,2])]), name="Random")
	### Fixed Policy Agents
	# Minimax Agent
	minimax = FixedPolicyAgent(policy=(lambda x: ACTIONS[1]), name='Minimax')
	# Maximize Both Player's Payoffs
	max_welfare1 = FixedPolicyAgent(policy=(lambda x: {0:ACTIONS[2], 2:ACTIONS[0], 1:random.choice([ACTIONS[0], ACTIONS[2]]), -1:ACTIONS[0]}[x.selection[this_player]]), name='MaxWelfare')
	max_welfare2 = FixedPolicyAgent(policy=(lambda x: {0:ACTIONS[2], 2:ACTIONS[0], 1:random.choice([ACTIONS[0], ACTIONS[2]]), -1:ACTIONS[2]}[x.selection[this_player]]), name='MaxWelfare2')
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
pool_agents = create_agents(1 - human_idx)
player_pool = PlayerPool(list(pool_agents.values()), sample_size=10)

mirrored_agents = create_agents(human_idx)
mirrored_pool = PlayerPool(list(mirrored_agents.values()), sample_size=10)
output_assumptions = [(lambda chief,s,a,next_s: chief.get_predicted_action(s) == chief.actions[next_s.selection[chief.partner_idx]])]

with open("trained_chief_estimation_model", 'rb') as f:
	estimation_model = pickle.load(f)

chief_player = ChiefAgentWithAAT(estimation_model=estimation_model, output_assumptions=output_assumptions, actions=markov_game.get_actions(), name="chief", player_pool=player_pool, mirrored_player_pool=mirrored_pool, partner_idx=human_idx, likelihood_threshold=0.6)


probabilities_over_time = dict(zip(list(pool_agents.keys()), [[]]*len(pool_agents)))
correct_predictions_over_time = []
total_correct = 0
average_accuracy_till_now = []
performance_estimation_over_time = []

print("Setup done")
print("Index of teammate:", human_idx)


def format_nums(L):
	LL = []

	for l in L:
		LL.append(round(l,3))

	return LL

# execution
if human_idx == 0:
	agents = ["Human", chief_player]
else:
	agents = [chief_player, "Human"]

total_rewards = defaultdict(float)
reward_dict = defaultdict(float)

baseline_player = BaselineAgent(markov_game.get_actions(), "baseline", partner_idx=human_idx)
baseline_accuracy_over_time = []
total_baseline = 0

markov_game.reset()
state = markov_game.get_init_state()

res = 0
step_num = 50

player_names = []
for agent in chief_player.player_pool.agents:
	player_names.append(agent.name)

plt_pause = 0.2
x_vals_probabilities = list(range(len(player_names)))
plt.clf()
plt.title("Bayesian Probabilities over Agents")
plt.xlabel("Agents")
plt.ylabel("Probabilities")
plt.bar(x_vals_probabilities, chief_player.current_bayesian_values, tick_label=player_names)
plt.draw()
plt.pause(plt_pause) # plt requires a small delay to actually plot

agent_moves = []
human_moves = []

for steps in range(1, step_num + 2):
	action_dict = dict()

	prediction = chief_player.get_predicted_action(state)

	screen.fill([255,255,255])

	text_round = font.render("Round {}/{}".format(max(steps - 1, 1), step_num), 1, (5,5,5))
	text_round_rect = text_round.get_rect(centerx=170, centery=10)
	screen.blit(text_round, text_round_rect)

	text1 = font.render("your selection", 1, (5,5,5))
	text2 = font.render("opponent's selection", 1, (5,5,5))
	text1rect = text1.get_rect(centerx=255, centery=50)
	text2rect = text2.get_rect(centerx=65, centery=50)

	screen.blit(text1, text1rect)
	screen.blit(text2, text2rect)

	rewardtext1 = font.render("your reward: " + str(reward_dict["Human"]), 1, (5,5,5))
	rewardtext2 = font.render("their reward: " + str(reward_dict["chief"]), 1, (5,5,5))
	text1rect = text1.get_rect(centerx=235, centery=150)
	text2rect = text2.get_rect(centerx=65, centery=150)

	screen.blit(rewardtext1, text1rect)
	screen.blit(rewardtext2, text2rect)

	totalrewardtext1 = font.render("your total reward: " + str(total_rewards["Human"]), 1, (5,5,5))
	totalrewardtext2 = font.render("their total reward: " + str(total_rewards["chief"]), 1, (5,5,5))
	text1rect = text1.get_rect(centerx=235, centery=180)
	text2rect = text2.get_rect(centerx=65, centery=180)

	screen.blit(totalrewardtext1, text1rect)
	screen.blit(totalrewardtext2, text2rect)

	for b in buttons:
		img, rect = buttons[b]
		pygame.draw.rect(screen, (200,50,200), rect)
		screen.blit(img, rect)

	if state.selection[human_idx] != -1:
		img, rect = list(buttons.values())[state.selection[1 - human_idx]]
		new_rect = pygame.Rect(40, 70, 50, 80)
		screen.blit(img, new_rect)

		img, rect = list(buttons.values())[state.selection[human_idx]]
		new_rect = pygame.Rect(230, 70, 50, 80)
		screen.blit(img, new_rect)

	pygame.display.flip()

	if steps == step_num + 1:
		break

	agent_action = None
	for a in agents:
		if a == "Human":
			# get action
			chosen_action = None
			clicked = False

			while(chosen_action == None):
				for event in pygame.event.get():
					if event.type == MOUSEBUTTONDOWN:
						clicked = True
					if event.type == MOUSEBUTTONUP and clicked:
						mouse_pos = pygame.mouse.get_pos()
						chosen_action = action_choice(mouse_pos)
						# print(chosen_action, mouse_pos)
						clicked = False

					if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
						# Quit.
						pygame.display.quit()

				# allows you to dynamically resize the figure in a new window (helpful for viewing all labels)
				plt.draw()
				plt.pause(plt_pause)

			action_dict[a] = chosen_action
			human_moves.append(chosen_action)
		else:
			agent_reward = reward_dict[a.name]
			agent_action = a.act(state, agent_reward)
			action_dict[a.name] = agent_action
			agent_moves.append(agent_action)

	print("Agent: {}, Human: {}".format(agent_action, chosen_action))

	baseline_player.act(state, agent_reward)
	baseline_prediction = baseline_player.get_predicted_action(state)
	total_baseline += int(baseline_prediction == action_dict["Human"])
	baseline_accuracy_over_time.append(total_baseline/(len(correct_predictions_over_time) + 1))


	correct_val = int(prediction == action_dict["Human"])
	total_correct += correct_val
	correct_predictions_over_time.append(correct_val)
	average_accuracy_till_now.append(total_correct/len(correct_predictions_over_time))

	performance_estimation_over_time.append(chief_player.performance_estimation())

	reward_dict, next_state = markov_game.execute_agent_action(action_dict)

	total_rewards["Human"] += reward_dict["Human"]
	total_rewards["chief"] += reward_dict["chief"]

	state = next_state

	plt.clf()
	plt.title("Bayesian Probabilities over Agents")
	plt.xlabel("Agents")
	plt.ylabel("Probabilities")
	plt.bar(x_vals_probabilities, chief_player.current_bayesian_values, tick_label=player_names)
	plt.draw()
	plt.pause(plt_pause) # plt requires a small delay to actually plot


data_dir = './user_data/'
os.makedirs(data_dir, exist_ok=True)
n_participants = len(os.listdir(data_dir))

with open(data_dir + 'participant' + str(n_participants + 1) + '.pickle', 'wb') as f:
	pickle.dump((agent_moves, human_moves, correct_predictions_over_time, average_accuracy_till_now, baseline_accuracy_over_time), f)

xvals = list(range(len(correct_predictions_over_time)))

plt.figure(2)
plt.plot(xvals, correct_predictions_over_time, 'bo', markersize=12)
plt.gca().set_ylim(0,1)
plt.xlabel("Step")
plt.ylabel("Correctness of Prediction")
plt.xticks(ticks=xvals)
plt.yticks(ticks=[0, 1], labels=["False", "True"])
plt.title("Correct or not for each step")

plt.figure(3)
plt.plot(xvals, average_accuracy_till_now)
plt.plot(xvals, baseline_accuracy_over_time, color="red")
plt.legend(['CHIEF', 'Baseline'])
plt.title("Average accuracy till each step")
plt.xticks(ticks=xvals)
plt.gca().set_ylim(0,1)
plt.xlabel("Step")
plt.ylabel("Average accuracy")

plt.figure(4)
plt.plot(xvals, performance_estimation_over_time)
plt.plot(xvals, [average_accuracy_till_now[-1]]*len(xvals), color="red")
plt.gca().set_ylim(0,1)
plt.legend(['Performance Estimation', 'Actual End Performance'])
plt.title("Performance Estimation Over Time")
plt.xticks(ticks=xvals)
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.show()