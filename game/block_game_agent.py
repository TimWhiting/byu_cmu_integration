from scipy.sparse import base
from block_game import ACTIONS, BlockGameState, P1, P2
from block_game_tree import BlockGameTree, BlockGameTreeNode
from collections import OrderedDict
from simple_rl.agents.AgentClass import Agent
from typing import Dict, Tuple


class BlockGameAgent(Agent):
    def __init__(self, name: str, baseline_payoff: float) -> None:
        Agent.__init__(self, name=name, actions=[])
        self.baseline_payoff = baseline_payoff


"""
Class that represents a fixed policy agent in the block game

Parameters:
eval_func (function): Function used for training the agent's policy.  The function should take
    in a List[Tuple[Tuple[float, float], int]] representing the different (reward, action) pairs
    that are possible to choose from, as well as an int representing the current turn.  The
    function should return a Tuple[Tuple[float, float], int] representing the best (reward, action)
    pair to pick
name (str): The name of the agent/what policy it's using
block_game_tree (BlockGameTree): Tree representing all of the possible game states

"""


class FixedPolicyBlockGameAgent(BlockGameAgent):
    def __init__(self, eval_func: 'function', name: str, block_game_tree: BlockGameTree, baseline_payoff: float) -> None:
        BlockGameAgent.__init__(
            self, name=name, baseline_payoff=baseline_payoff)
        self.eval_func = eval_func
        self.name = name
        self.state_to_action_map: dict[str, 'str'] = {}
        self._train(block_game_tree)

        def policy(state: BlockGameState, reward, episode_number) -> 'str':
            return self.state_to_action_map[str(state)]

        self.policy = policy

    def _train(self, block_game_tree: BlockGameTree):
        state_map = OrderedDict(
            sorted(block_game_tree.state_id_map.items(), reverse=True))
        ideal_reward_map: Dict[BlockGameTreeNode, Tuple[float, float]] = {}
        for tree_node in state_map.values():
            if tree_node.state.is_terminal():
                p1_reward = tree_node.state.reward(P1)
                p2_reward = tree_node.state.reward(P2)

                ideal_reward_map[tree_node] = (p1_reward, p2_reward)
                self.state_to_action_map[str(tree_node.state)] = ACTIONS[-1]

            else:
                reward_action_pairs = [(ideal_reward_map[tree_node.action_to_children_map[i]], i)
                                       for i in tree_node.action_to_children_map.keys()]
                ideal_reward, ideal_action = self.eval_func(
                    reward_action_pairs, tree_node.state.turn)

                ideal_reward_map[tree_node] = ideal_reward
                self.state_to_action_map[str(
                    tree_node.state)] = ACTIONS[ideal_action]

    def act(self, state: BlockGameState, reward, episode_number):
        return self.policy(state, reward, episode_number)

    def __str__(self) -> str:
        return str(self.name)


"""
Class that represents a dynamic policy agent in the block game

Parameters:
policy (function): Function used for picking an action given a state.  The function should take
    in a BlockGameState, reward (float), and episode/game number (int) and should return a str 
    representing the action to pick (should be a valid action in the ACTIONS list found in 
    block_game.py) 
name (str): The name of the agent/what policy it's using

"""


class DynamicPolicyBlockGameAgent(BlockGameAgent):
    def __init__(self, policy: 'function', name: str, changes_during_round: bool, changes_across_rounds: bool, baseline_payoff: float) -> None:
        BlockGameAgent.__init__(
            self, name=name, baseline_payoff=baseline_payoff)
        self.policy = policy
        self.name = name
        self.episode_to_update = self.episode_number
        self.changes_during_round = changes_during_round
        self.changes_across_rounds = changes_across_rounds

    def act(self, state: BlockGameState, reward, episode_number):
        if reward == '':
            reward = 0

        action = self.policy(state, float(reward), self.episode_number)

        if state.is_terminal():
            if self.episode_number != self.episode_to_update:
                self.episode_to_update = self.episode_number

            else:
                self.episode_number += 1

        return action

    def __str__(self) -> str:
        return str(self.name)
