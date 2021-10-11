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
                'efficient_cooperation': efficient_cooperation_agent
            }
        else:
            return {
                'max_self': max_self_agent,
                'play_num_based': play_num_based_agent,
                'cooperative_or_greedy': cooperative_or_greedy_agent,
                'min_welfare': min_welfare_agent,
                'greedy_until_negative': greedy_until_negative_agent,
                'q_learning': ql_agent,
                'random_action': random_action_agent,
                'efficient_cooperation': efficient_cooperation_agent
            }

    else:
        return {
            'minimax': minimax_agent,
            'max_self': max_self_agent,
            'random_action': random_action_agent,
            'efficient_cooperation': efficient_cooperation_agent,
        }


def run_games(train=False, aat_predict_func: 'function' = None):
    assert(train or aat_predict_func is not None)

    block_game = BlockGameMDP()

    chief_idx = 0
    human_idx = 1

    # 'Human' strategies to train on
    human_strategies = create_agents(human=True, train=train)

    # Pool for the chief agent
    pool_agents = create_agents()
    player_pool = PlayerPool(list(pool_agents.values()), sample_size=10)

    n_rounds = 50
    epsilons = [0, 0.2, 0.3, 0.5, 0.7]
    n_epochs = 100 if train else 50

    training_data = []
    test_data = []

    for epoch in range(1, n_epochs + 1):
        print('Epoch: {}'.format(epoch))

        for human_strategy in human_strategies.keys():
            print('Strategy: {}'.format(human_strategy))

            human_teammate = deepcopy(human_strategies[human_strategy])

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
                            state, agent_reward, round_num - 1) if not isinstance(agent, QLearningAgent) else agent.act(
                            state, agent_reward)
                        action_map[agent.name] = agent_action

                        if agent.name == 'Chief':
                            chiefs_proposed_model = agent.get_proposed_model()
                            predicted_payoff = chiefs_proposed_model.baseline_payoff if not isinstance(
                                chiefs_proposed_model, QLearningAgent) else -6.0874999999999995

                            human_captured_in_experts = max(
                                agent.current_bayesian_values) > 0.5
                            human_static_during_round = True if isinstance(
                                chiefs_proposed_model, FixedPolicyBlockGameAgent) else not chiefs_proposed_model.changes_during_round
                            human_static_across_rounds = True if isinstance(
                                chiefs_proposed_model, FixedPolicyBlockGameAgent) else not chiefs_proposed_model.changes_across_rounds

                            curr_tup = [state.get_play_num() + 1, round_num, (agent_reward / round_num) / predicted_payoff, int(
                                human_static_during_round), int(human_static_across_rounds), int(human_captured_in_experts), predicted_payoff]

                            if train:
                                current_training_data.append(curr_tup)

                            else:
                                prediction = aat_predict_func(
                                    curr_tup[:-1]) * predicted_payoff
                                current_predictions.append(prediction[0])

                    updated_rewards_map, next_state = block_game.execute_agent_action(
                        action_map)

                    for agent_name, new_reward in updated_rewards_map.items():
                        reward_map[agent_name] += new_reward

                    state = next_state

            avg_payoff = reward_map[chief_player.name] / n_rounds
            print('Avg payoff for ' + str(human_teammate.name) +
                  ': ' + str(avg_payoff))

            for tup in current_training_data:
                tup[-1] = avg_payoff / tup[-1]

            squared_errors = []
            average_payoffs = []
            for pred in current_predictions:
                squared_errors.append((avg_payoff - pred) ** 2)
                # squared_errors.append(pred)
                # average_payoffs.append(avg_payoff)

            training_data.extend(current_training_data)
            test_data.append((human_strategy, squared_errors))
            # test_data.append((human_strategy, squared_errors, average_payoffs))

    if train:
        data_dir = './training_data/'
        os.makedirs(data_dir, exist_ok=True)

        with open(data_dir + 'training_data.pickle', 'wb') as f:
            pickle.dump(training_data, f)

    else:
        for strategy_name in human_strategies.keys():
            test_results = []
            for human_strategy, results in test_data:
                if human_strategy == strategy_name:
                    test_results.append(results[:300])

            test_results = np.array(test_results).reshape(n_epochs, -1)
            xvals = list(range(test_results.shape[1]))
            mean_test_results = test_results.mean(axis=0)
            var_test_results = test_results.var(axis=0)

            plt.plot(xvals, mean_test_results)
            # plt.plot(xvals, var_test_results, color='red')
            # plt.plot(xvals, average_payoff, color='red')
            # plt.gca().set_ylim(0, 1)
            plt.title(
                'Average Payoff Prediction Squared Errors for ' + str(strategy_name))
            plt.show()
            # plt.plot(xvals, average_accuracy_till_now)
            # plt.plot(xvals, baseline_accuracy_over_time, color="red")
            # plt.gca().set_ylim(0, 1)
            # plt.title("Average accuracy till each step")
            plt.show()
