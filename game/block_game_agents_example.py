from block_game import ACTIONS, P1, P2, BlockGameState, BlockGameMDP
from block_game_agent import FixedPolicyBlockGameAgent, DynamicPolicyBlockGameAgent
from block_game_tree import BlockGameTree, AVAILABLE
import numpy as np
from simple_rl.agents import QLearningAgent
from simple_rl.run_experiments import play_markov_game
from typing import List, Tuple

block_game_tree = BlockGameTree()


# -------------------------------------------------------------------
# ------------------------ STATIC AGENTS ----------------------------
# -------------------------------------------------------------------

# Minimax agent (tries to minimize the maximum possible payoff of the other player)
def minimax_func(reward_action_pairs: List[Tuple[Tuple[float, float], int]], turn: int) -> Tuple[Tuple[float, float], int]:
    other_player = P1 if turn == P2 else P2
    return min(reward_action_pairs, key=lambda x: x[0][other_player])


minimax_agent = FixedPolicyBlockGameAgent(
    minimax_func, 'Minimax', block_game_tree)


# Max self agent (tries to maximize their own payoff)
def max_self_func(reward_action_pairs: List[Tuple[Tuple[float, float], int]], turn: int) -> Tuple[Tuple[float, float], int]:
    return max(reward_action_pairs, key=lambda x: x[0][turn])


max_self_agent = FixedPolicyBlockGameAgent(
    max_self_func, 'MaxSelf', block_game_tree)


# Max welfare agent (tries to maximize the totaly payoff of both players)
def max_welfare_func(reward_action_pairs: List[Tuple[Tuple[float, float], int]], turn: int) -> Tuple[Tuple[float, float], int]:
    return max(reward_action_pairs, key=lambda x: sum(x[0]))


max_welfare_agent = FixedPolicyBlockGameAgent(
    max_welfare_func, 'MaxWelfare', block_game_tree)


# Max other agent (tries to maximize the payoff of the other player)
def max_other_func(reward_action_pairs: List[Tuple[Tuple[float, float], int]], turn: int) -> Tuple[Tuple[float, float], int]:
    other_player = P1 if turn == P2 else P2
    return max(reward_action_pairs, key=lambda x: x[0][other_player])


max_other_agent = FixedPolicyBlockGameAgent(
    max_other_func, 'MaxOther', block_game_tree)
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
    random_action, 'RandomAction')


# Random policy agent (randomly picks an action from the static agents)
def random_policy(state: BlockGameState, reward: float, episode_number: int) -> str:
    possible_agents = [minimax_agent, max_self_agent,
                       max_welfare_agent, max_other_agent]
    potential_actions = [agent.act(state, reward) for agent in possible_agents]
    action_index = np.random.choice(potential_actions)

    return ACTIONS[action_index]


random_policy_agent = DynamicPolicyBlockGameAgent(
    random_action, 'RandomPolicy')


# Play num agent (changes strategy based on the play number)
def play_num_based_policy(state: BlockGameState, reward: float, episode_number: int) -> str:
    curr_play_num = state.get_play_num()

    if curr_play_num <= 2:
        return max_self_agent.act(state, reward)

    elif curr_play_num >= 3 and curr_play_num < 6:
        return max_welfare_agent.act(state, reward)

    else:
        return ACTIONS[-1]


play_num_based_agent = DynamicPolicyBlockGameAgent(
    play_num_based_policy, 'PlayNumBased')


# Game num agent (changes strategy based on the episode/game number)
def game_num_based_policy(state: BlockGameState, reward: float, episode_number: int) -> str:
    if episode_number < 15:
        return random_policy_agent.act(state, reward)

    elif episode_number >= 15 and episode_number < 30:
        return max_self_agent.act(state, reward)

    else:
        return max_self_agent.act(state, reward)


play_num_based_agent = DynamicPolicyBlockGameAgent(
    play_num_based_policy, 'PlayNumBased')


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
        return random_action_agent.act(state, reward)


efficient_cooperation_agent = DynamicPolicyBlockGameAgent(
    efficient_cooperation_policy, 'EfficientCoop')


# Cooperative or greedy agent (tries to achieve efficient cooperation, but will occasionally
# try to maximize its own payoff/be greedy)
def cooperative_or_greedy_policy(state: BlockGameState, reward: float, episode_number: int) -> str:
    defect_proba = 0.01
    possible_actions = [max_self_agent.act(
        state, reward), efficient_cooperation_agent.act(state, reward)]
    return np.random.choice(possible_actions, p=[defect_proba, 1 - defect_proba])


cooperative_or_greedy_agent = DynamicPolicyBlockGameAgent(
    cooperative_or_greedy_policy, 'CoopOrGreedy')


# Q-learning agent (tries to learn ideal actions over time)
ql_agent = QLearningAgent(actions=ACTIONS, name="q1")
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------


markov_game = BlockGameMDP()

# Run experiment
play_markov_game([efficient_cooperation_agent, cooperative_or_greedy_agent], markov_game,
                 instances=5, episodes=500, steps=30, open_plot=True)
