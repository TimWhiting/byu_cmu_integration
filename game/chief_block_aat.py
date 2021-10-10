from old_chief_agent_block import PlayerPool, ChiefAgent
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
from block_game_agents_example import *


def create_agents(human=False, train=True):
    if human:
        if train:
            return {
                'minimax': minimax_agent,
                'max_welfare': max_welfare_agent,
                'max_other': max_other_agent,
                'random_action': random_action_agent,
                'random_policy': random_policy_agent,
                'efficient_cooperation': efficient_cooperation_agent,
            }
        else:
            return {
                'max_self': max_self_agent,
                'play_num_based': play_num_based_agent,
                'cooperative_or_greedy': cooperative_or_greedy_agent
            }

    else:
        return {
            "minimax": minimax_agent,
            "max_self": max_self_agent,
            "random_action": random_action_agent,
            "efficient_cooperation": efficient_cooperation_agent
        }


def run_games(train=False, aat_predict_func: 'function' = None):
    assert(train or aat_predict_func is not None)

    block_game = BlockGameMDP()

    chief_idx = 0
    human_idx = 1

    # 'Human' strategies to train on
    human_train_strategies = create_agents(human=True, train=train)

    # Pool for the chief agent
    pool_agents = create_agents()
    player_pool = PlayerPool(list(pool_agents.values()), sample_size=10)

    n_rounds = 50
    epsilons = [0, 0.2, 0.3, 0.5, 0.7]
    n_epochs = 100 if train else 1

    training_data = []
    test_data = []

    for epoch in range(n_epochs):
        print('Epoch: {}'.format(epoch))

        for human_strategy in human_train_strategies.keys():
            print('Strategy: {}'.format(human_strategy))

            human_teammate = deepcopy(human_train_strategies[human_strategy])

            chief_player = ChiefAgent(actions=block_game.get_actions(
            ), name="Chief", player_pool=player_pool, partner_idx=human_idx, likelihood_threshold=0.6)

            reward_map = {human_teammate.name: 0, chief_player.name: 0}
            current_training_data = []
            current_predictions = []

            for round_num in range(1, n_rounds + 1):
                block_game.reset()
                state = block_game.get_init_state()
                action_map = dict()

                while not state.is_terminal():
                    for agent in [human_teammate, chief_player]:
                        agent_reward = reward_map[agent.name]
                        agent_action = agent.act(
                            state, agent_reward, round_num - 1)
                        action_map[agent.name] = agent_action

                        if agent.name == 'Chief':
                            human_static_during_round = True if isinstance(
                                human_teammate, FixedPolicyBlockGameAgent) else not human_teammate.changes_during_round
                            human_static_across_rounds = True if isinstance(
                                human_teammate, FixedPolicyBlockGameAgent) else not human_teammate.changes_across_rounds
                            human_captured_in_experts = human_teammate.name in pool_agents.keys()
                            curr_tup = [state.get_play_num() + 1, round_num, agent_reward / round_num, int(
                                human_static_during_round), int(human_static_across_rounds), int(human_captured_in_experts), 0]

                            if train:
                                current_training_data.append(curr_tup)

                            else:
                                prediction = aat_predict_func(curr_tup[:-1])
                                current_predictions.append(prediction)

                    updated_rewards_map, next_state = block_game.execute_agent_action(
                        action_map)

                    for agent_name, new_reward in updated_rewards_map.items():
                        reward_map[agent_name] += new_reward

                    state = next_state

            avg_payoff = reward_map[chief_player.name] / n_rounds

            for tup in current_training_data:
                tup[-1] = avg_payoff

            squared_errors = []
            for pred in current_predictions:
                squared_errors.append((avg_payoff - pred) ** 2)

            training_data.extend(current_training_data)
            test_data.append((human_strategy, squared_errors))

    if train:
        data_dir = './training_data/'
        os.makedirs(data_dir, exist_ok=True)

        with open(data_dir + 'training_data.pickle', 'wb') as f:
            pickle.dump(training_data, f)

    else:
        for human_strategy, test_results in test_data:
            xvals = list(range(len(test_results)))

            plt.plot(xvals, test_results)
            # plt.gca().set_ylim(0, 1)
            plt.title(
                'Average Payoff Prediction Squared Errors for ' + str(human_strategy))
            plt.show()
            # plt.plot(xvals, average_accuracy_till_now)
            # plt.plot(xvals, baseline_accuracy_over_time, color="red")
            # plt.gca().set_ylim(0, 1)
            # plt.title("Average accuracy till each step")
            plt.show()
