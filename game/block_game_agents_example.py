from block_game import ACTIONS, P1, P2, BlockGameState, BlockGameMDP
from block_game_agent import FixedPolicyBlockGameAgent, DynamicPolicyBlockGameAgent
from block_game_tree import BlockGameTree, AVAILABLE
import numpy as np
from simple_rl.agents import QLearningAgent
from simple_rl.run_experiments import play_markov_game
from typing import List, Tuple
from copy import deepcopy

block_game_tree = BlockGameTree()


# -------------------------------------------------------------------
# ------------------------ STATIC AGENTS ----------------------------
# -------------------------------------------------------------------

# Minimax agent (tries to minimize the maximum possible payoff of the other player)
def minimax_func(reward_action_pairs: List[Tuple[Tuple[float, float], int]], turn: int) -> Tuple[Tuple[float, float], int]:
    other_player = P1 if turn == P2 else P2
    return min(reward_action_pairs, key=lambda x: x[0][other_player])


minimax_agent = FixedPolicyBlockGameAgent(
    minimax_func, 'Minimax', block_game_tree, -30)


# Max self agent (tries to maximize their own payoff)
def max_self_func(reward_action_pairs: List[Tuple[Tuple[float, float], int]], turn: int) -> Tuple[Tuple[float, float], int]:
    return max(reward_action_pairs, key=lambda x: x[0][turn])


max_self_agent = FixedPolicyBlockGameAgent(
    max_self_func, 'MaxSelf', block_game_tree, 90)


# Max welfare agent (tries to maximize the total payoff of both players)
def max_welfare_func(reward_action_pairs: List[Tuple[Tuple[float, float], int]], turn: int) -> Tuple[Tuple[float, float], int]:
    return max(reward_action_pairs, key=lambda x: sum(x[0]))


max_welfare_agent = FixedPolicyBlockGameAgent(
    max_welfare_func, 'MaxWelfare', block_game_tree, 125)


# Max other agent (tries to maximize the payoff of the other player)
def max_other_func(reward_action_pairs: List[Tuple[Tuple[float, float], int]], turn: int) -> Tuple[Tuple[float, float], int]:
    other_player = P1 if turn == P2 else P2
    return max(reward_action_pairs, key=lambda x: x[0][other_player])


max_other_agent = FixedPolicyBlockGameAgent(
    max_other_func, 'MaxOther', block_game_tree, 125)


# Min welfare agent (tries to minimize the total payoff of both players)
def min_welfare_func(reward_action_pairs: List[Tuple[Tuple[float, float], int]], turn: int) -> Tuple[Tuple[float, float], int]:
    return min(reward_action_pairs, key=lambda x: sum(x[0]))


min_welfare_agent = FixedPolicyBlockGameAgent(
    min_welfare_func, 'MinWelfare', block_game_tree, -31.875)
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# ------------------------ DYNAMIC AGENTS ---------------------------
# -------------------------------------------------------------------

# Random action agent (randomly picks a playable action)
def random_action(state: BlockGameState, reward: float, episode_number: int) -> str:
    available_actions = [i for i, val in enumerate(
        state.blocks) if val == AVAILABLE]
    action_index = np.random.choice(available_actions)

    return ACTIONS[action_index]


random_action_agent = DynamicPolicyBlockGameAgent(
    random_action, 'RandomAction', True, False, -8.65)


# Random policy agent (randomly picks an action from the static agents)
def random_policy(state: BlockGameState, reward: float, episode_number: int) -> str:
    possible_agents = [minimax_agent, max_self_agent,
                       max_welfare_agent, max_other_agent]
    potential_actions = [agent.act(state, reward, episode_number)
                         for agent in possible_agents]
    action_index = np.random.choice(potential_actions)

    return ACTIONS[action_index]


random_policy_agent = DynamicPolicyBlockGameAgent(
    random_action, 'RandomPolicy', True, False, -3.5625)


# Play num agent (changes strategy based on the play number)
def play_num_based_policy(state: BlockGameState, reward: float, episode_number: int) -> str:
    curr_play_num = state.get_play_num()

    if curr_play_num <= 2:
        return max_self_agent.act(state, reward, episode_number)

    elif curr_play_num >= 3 and curr_play_num < 6:
        return max_welfare_agent.act(state, reward, episode_number)

    else:
        return ACTIONS[-1]


play_num_based_agent = DynamicPolicyBlockGameAgent(
    play_num_based_policy, 'PlayNumBased', True, False, 90.0)


# Game num agent (changes strategy based on the episode/game number)
def game_num_based_policy(state: BlockGameState, reward: float, episode_number: int) -> str:
    if episode_number < 15:
        return random_policy_agent.act(state, reward, episode_number)

    elif episode_number >= 15 and episode_number < 30:
        return max_self_agent.act(state, reward, episode_number)

    else:
        return max_self_agent.act(state, reward, episode_number)


game_num_based_agent = DynamicPolicyBlockGameAgent(
    game_num_based_policy, 'GameNumBased', True, True, -8.0625)


# Efficient cooperation agent (every other game it alternates between picking squares and triangles)
def efficient_cooperation_policy(state: BlockGameState, reward: float, episode_number: int) -> str:
    square_action_indices = [
        i for i, val in enumerate(ACTIONS) if 'square' in val and state.blocks[i] == AVAILABLE]
    triangle_action_indices = [
        i for i, val in enumerate(ACTIONS) if 'triangle' in val and state.blocks[i] == AVAILABLE]

    player1_picks_squares = episode_number % 2 == 0
    player2_picks_squares = not player1_picks_squares

    curr_turn = state.turn

    if ((curr_turn == P1 and player1_picks_squares) or (curr_turn == P2 and player2_picks_squares)) and len(square_action_indices) != 0:
        return ACTIONS[square_action_indices[0]]

    elif len(triangle_action_indices) != 0:
        return ACTIONS[triangle_action_indices[0]]

    else:
        return random_action_agent.act(state, reward, episode_number)


efficient_cooperation_agent = DynamicPolicyBlockGameAgent(
    efficient_cooperation_policy, 'EfficientCoop', False, True, 125.0)


# Cooperative or greedy agent (tries to achieve efficient cooperation, but will occasionally
# try to maximize its own payoff/be greedy)
def cooperative_or_greedy_policy(state: BlockGameState, reward: float, episode_number: int) -> str:
    defect_proba = 0.01
    possible_actions = [max_self_agent.act(
        state, reward, episode_number), efficient_cooperation_agent.act(state, reward, episode_number)]
    return np.random.choice(possible_actions, p=[defect_proba, 1 - defect_proba])


cooperative_or_greedy_agent = DynamicPolicyBlockGameAgent(
    cooperative_or_greedy_policy, 'CoopOrGreedy', True, True, 121.875)


# Greedy until negative agent (plays greedily until its reward is negative; after that, it
# will try to cooperate)
def greedy_until_negative_policy(state: BlockGameState, reward: float, episode_number: int) -> str:
    if reward < 0:
        return efficient_cooperation_agent.act(state, reward, episode_number)

    else:
        return max_self_agent.act(state, reward, episode_number)


greedy_until_negative_agent = DynamicPolicyBlockGameAgent(
    greedy_until_negative_policy, 'GreedUntilNegative', True, True, 90.0)

# Q-learning agent (tries to learn ideal actions over time)
ql_agent = QLearningAgent(actions=ACTIONS, name="QL")
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------

find_baseline_payoffs = False

if find_baseline_payoffs:
    n_rounds = 50
    block_game = BlockGameMDP()

    agents = [minimax_agent, max_self_agent, max_welfare_agent, max_other_agent, min_welfare_agent, random_action_agent, random_policy_agent,
              play_num_based_agent, game_num_based_agent, efficient_cooperation_agent, cooperative_or_greedy_agent, greedy_until_negative_agent, ql_agent]

    for agent in agents:
        player1 = deepcopy(agent)
        player2 = deepcopy(agent)
        player1.name = 'player1'
        player2.name = 'player2'

        reward_map = {player1.name: 0, player2.name: 0}

        for round_num in range(1, n_rounds + 1):
            block_game.reset()
            state = block_game.get_init_state()
            action_map = dict()

            while not state.is_terminal():
                for curr_agent in [player1, player2]:
                    agent_reward = reward_map[curr_agent.name]
                    agent_action = curr_agent.act(
                        state, agent_reward, round_num - 1) if not isinstance(curr_agent, QLearningAgent) else curr_agent.act(
                        state, agent_reward)
                    action_map[curr_agent.name] = agent_action

                updated_rewards_map, next_state = block_game.execute_agent_action(
                    action_map)

                for agent_name, new_reward in updated_rewards_map.items():
                    reward_map[agent_name] += new_reward

                state = next_state

        avg_payoff = ((reward_map[player1.name] / n_rounds) +
                      (reward_map[player2.name] / n_rounds)) / 2

        print('Baseline for ' + str(agent.name) + ' is: ' + str(avg_payoff))
