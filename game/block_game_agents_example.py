from block_game import ACTIONS, P1, P2, BlockGameState, BlockGameMDP
from block_game_agent import FixedPolicyBlockGameAgent, DynamicPolicyBlockGameAgent
from block_game_tree import BlockGameTree, AVAILABLE
import numpy as np
from simple_rl.run_experiments import play_markov_game
from typing import List, Tuple

block_game_tree = BlockGameTree()


# -------------------------------------------------------------------
# ------------------------ STATIC AGENTS ----------------------------
# -------------------------------------------------------------------
def minimax_func(reward_action_pairs: List[Tuple[Tuple[float, float], int]], turn: int) -> Tuple[Tuple[float, float], int]:
    other_player = P1 if turn == P2 else P2
    return min(reward_action_pairs, key=lambda x: x[0][other_player])


minimax_agent = FixedPolicyBlockGameAgent(
    minimax_func, 'Minimax', block_game_tree)


def max_self_func(reward_action_pairs: List[Tuple[Tuple[float, float], int]], turn: int) -> Tuple[Tuple[float, float], int]:
    return max(reward_action_pairs, key=lambda x: x[0][turn])


max_self_agent = FixedPolicyBlockGameAgent(
    max_self_func, 'MaxSelf', block_game_tree)


def max_welfare_func(reward_action_pairs: List[Tuple[Tuple[float, float], int]], turn: int) -> Tuple[Tuple[float, float], int]:
    return max(reward_action_pairs, key=lambda x: sum(x[0]))


max_welfare_agent = FixedPolicyBlockGameAgent(
    max_welfare_func, 'MaxWelfare', block_game_tree)


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
def random_action(state: BlockGameState) -> str:
    available_actions = [i for i, val in enumerate(
        state.blocks) if val == AVAILABLE]
    action_index = np.random.choice(available_actions)

    return ACTIONS[action_index]


random_action_agent = DynamicPolicyBlockGameAgent(
    random_action, 'RandomAction')


def random_policy(state: BlockGameState) -> str:
    possible_agents = [minimax_agent, max_self_agent,
                       max_welfare_agent, max_other_agent]
    potential_actions = [agent.act(state, 0) for agent in possible_agents]
    action_index = np.random.choice(potential_actions)

    return ACTIONS[action_index]


random_policy_agent = DynamicPolicyBlockGameAgent(
    random_action, 'RandomPolicy')


def play_num_based_policy(state: BlockGameState) -> str:
    curr_play_num = state.get_play_num()

    if curr_play_num <= 2:
        return max_self_agent.act(state, 0)

    elif curr_play_num >= 3 and curr_play_num < 6:
        return max_welfare_agent.act(state, 0)

    else:
        return ACTIONS[-1]


play_num_based_agent = DynamicPolicyBlockGameAgent(
    play_num_based_policy, 'PlayNumBased')
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------


markov_game = BlockGameMDP()

# Run experiment
play_markov_game([random_policy_agent, random_action_agent], markov_game,
                 instances=5, episodes=500, steps=30, open_plot=True)
