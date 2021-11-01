from chief_agent_block import Chief_Agent_BlockGame, PlayerPoolWithClones
from simple_rl.agents import QLearningAgent
from copy import deepcopy
import matplotlib.pyplot as plt
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
                # 'play_num_based': play_num_based_agent,
                # 'cooperative_or_greedy': cooperative_or_greedy_agent,
                # 'min_welfare': min_welfare_agent,
                # 'greedy_until_negative': greedy_until_negative_agent,
                # 'q_learning': ql_agent,
                # 'random_action': random_action_agent,
                # 'efficient_cooperation': efficient_cooperation_agent,
                # 'minimax': minimax_agent,
                # 'game_num_based': game_num_based_agent,
                # 'random_policy': random_policy_agent
            }

    else:
        return {
            'minimax': minimax_agent,
            'max_self': max_self_agent,
            'random_action': random_action_agent,
            'efficient_cooperation': efficient_cooperation_agent
        }


def run_games_with_new_chief(train=False, aat_predict_func: 'function' = None):
    assert(train or aat_predict_func is not None)

    block_game = BlockGameMDP()

    chief_idx = 0
    human_idx = 1

    # 'Human' strategies to train on
    human_strategies = create_agents(human=True, train=train)

    # Pool for the chief agent
    pool_agents = create_agents()
    player_pool = PlayerPoolWithClones(
        list(pool_agents.values()), 10, 30, 0.3, 9, "game/block_chief_testing_savedparams/traindata", "game/block_chief_testing_savedparams/params", BlockGameMDP())
    player_pool.train_clones()

    n_rounds = 50
    epsilons = [0, 0.2, 0.3, 0.5, 0.7]
    n_epochs = 100 if train else 50

    training_data = []
    test_data = []
    total_payoffs = []

    for epoch in range(1, n_epochs + 1):
        print('Epoch: {}'.format(epoch))

        for human_strategy in human_strategies.keys():
            print('Strategy: {}'.format(human_strategy))

            human_teammate = deepcopy(human_strategies[human_strategy])

            chief_player = Chief_Agent_BlockGame(
                name='Chief', actions=block_game.get_actions(), playerpoolwc=player_pool, player_ind=chief_idx)

            reward_map = {human_teammate.name: 0, chief_player.name: 0}
            current_training_data = []
            current_predictions = []

            for round_num in range(0, n_rounds):
                block_game.reset()
                state = block_game.get_init_state()
                action_map = dict()
                round_predictions = []
                round_proposed_total_payoffs = []

                while not state.is_terminal():
                    for agent in [human_teammate, chief_player]:
                        agent_reward = reward_map[agent.name]
                        agent_action = agent.act(
                            state, agent_reward, round_num) if not isinstance(agent, QLearningAgent) else agent.act(
                            state, agent_reward)
                        action_map[agent.name] = agent_action

                        if agent.name == 'Chief':
                            chiefs_proposed_model = agent.get_proposed_model(
                                state, agent_reward, round_num, agent_action)
                            chiefs_proposed_model = random_action_agent if chiefs_proposed_model is None else chiefs_proposed_model

                            proposed_avg_payoff = chiefs_proposed_model.baseline_payoff if not isinstance(
                                chiefs_proposed_model, QLearningAgent) else -6.0874999999999995
                            n_remaining_rounds = n_rounds - round_num
                            proposed_payoff_to_go = proposed_avg_payoff * n_remaining_rounds
                            proposed_total_payoff = agent_reward + proposed_payoff_to_go
                            proposed_payoff_diff = proposed_total_payoff - agent_reward

                            human_captured_in_experts = max(
                                agent.bayesian_inference_distribution) > 0.5
                            human_static_during_round = True if isinstance(
                                chiefs_proposed_model, FixedPolicyBlockGameAgent) else not chiefs_proposed_model.changes_during_round
                            human_static_across_rounds = True if isinstance(
                                chiefs_proposed_model, FixedPolicyBlockGameAgent) else not chiefs_proposed_model.changes_across_rounds

                            curr_tup = [state.get_play_num() + 1, round_num + 1, agent_reward / proposed_total_payoff if proposed_total_payoff != 0 else agent_reward / 0.000001, int(
                                human_static_during_round), int(human_static_across_rounds), int(human_captured_in_experts), proposed_total_payoff, proposed_total_payoff]

                            if train:
                                current_training_data.append(curr_tup)

                            else:
                                # prediction = aat_predict_func(
                                #     curr_tup[:-1]) * predicted_payoff
                                # round_predictions.append(prediction[0])
                                predictions, corrections, distances = aat_predict_func(
                                    curr_tup[:-2])

                                total_payoff_pred = 0
                                inverse_distance_sum = 0

                                for dist in distances:
                                    inverse_distance_sum += 1 / dist if dist != 0 else 1 / 0.000001

                                for i in range(len(predictions)):
                                    prediction_i = predictions[i]
                                    correction_i = corrections[i]
                                    distance_i = distances[i]
                                    inverse_distance_i = 1 / distance_i if distance_i != 0 else 1 / 0.000001
                                    distance_weight = inverse_distance_i / inverse_distance_sum

                                    total_payoff_pred += ((prediction_i *
                                                          correction_i) * distance_weight)
                                    # if (prediction_i * correction_i) * distance_weight > 10000:
                                    #     print('-----------------------')
                                    #     print(distance_i)
                                    #     print(inverse_distance_i)
                                    #     print(distance_weight)
                                    #     print(prediction_i)
                                    #     print(correction_i)
                                    #     print('-----------------------')

                                    # total_payoff_pred += (prediction_i *
                                    #                       distance_weight)

                                round_predictions.append(total_payoff_pred)

                    updated_rewards_map, next_state = block_game.execute_agent_action(
                        action_map)

                    for agent_name, new_reward in updated_rewards_map.items():
                        reward_map[agent_name] += new_reward

                    state = next_state

                if not train:
                    avg_round_predictions = sum(
                        round_predictions) / len(round_predictions)

                    current_predictions.append(avg_round_predictions)

            # avg_payoff = reward_map[chief_player.name] / n_rounds
            total_payoff = reward_map[chief_player.name]
            print('Avg payoff for ' + str(human_teammate.name) +
                  ': ' + str(total_payoff))

            total_payoffs.append(total_payoff)

            for tup in current_training_data:
                # tup[-1] = total_payoff / tup[-1]
                tup[-1] = total_payoff
                tup[-2] = total_payoff / \
                    tup[-2] if tup[-2] != 0 else total_payoff / 0.000001

            squared_errors = []
            for pred in current_predictions:
                # squared_errors.append((avg_payoff - pred) ** 2)
                # squared_errors.append(abs(total_payoff - pred) / 100)
                squared_errors.append(pred / 100)
                # average_payoffs.append(avg_payoff)

            training_data.extend(current_training_data)
            test_data.append((human_strategy, squared_errors))
            # test_data.append((human_strategy, squared_errors, average_payoffs))

    if train:
        data_dir = './training_data/'
        os.makedirs(data_dir, exist_ok=True)

        with open(data_dir + 'training_data_new_chief.pickle', 'wb') as f:
            pickle.dump(training_data, f)

    else:
        for strategy_name in human_strategies.keys():
            test_results = []
            for human_strategy, results in test_data:
                if human_strategy == strategy_name:
                    test_results.append(results)

            test_results = np.array(test_results).reshape(n_epochs, -1)
            xvals = list(range(test_results.shape[1]))
            mean_test_results = test_results.mean(axis=0)
            var_test_results = test_results.var(axis=0)

            print(mean_test_results)
            print(len(mean_test_results))

            plt.plot(xvals, mean_test_results)
            # plt.plot(xvals, var_test_results, color='red')
            plt.plot(xvals, [
                     (sum(total_payoffs) / len(total_payoffs)) / 100] * len(xvals), color='red')
            # plt.gca().set_ylim(0, 1)
            # plt.title(
            #     'AAT Prediction Errors (abs) for Chief vs. ' + str(strategy_name))
            plt.title(
                'AAT Predictions for Chief vs. ' + str(strategy_name))
            plt.xlabel('Round #')
            plt.ylabel('Total Payoff Predictions')
            plt.show()
            # plt.plot(xvals, average_accuracy_till_now)
            # plt.plot(xvals, baseline_accuracy_over_time, color="red")
            # plt.gca().set_ylim(0, 1)
            # plt.title("Average accuracy till each step")
            # plt.show()
