#!/usr/bin/env python3

# Python imports.
import random
import sys
import itertools
import numpy as np
from copy import copy

# Other imports
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
from simple_rl.mdp.StateClass import State
from simple_rl.agents import QLearningAgent, FixedPolicyAgent
from simple_rl.run_experiments import play_markov_game
P1 = 0
P2 = 1
ACTIONS = ["red square", "blue square", "purple square", "red triangle",
           "blue triangle", "purple triangle", "red circle", "blue circle", "purple circle"]


class BlockGameState(State):
    ''' Abstract State class '''

    AVAILABLE = 3

    def __init__(self):
        self.blocks = [BlockGameState.AVAILABLE for _ in range(0, 9)]
        self.turn = P1

    def get_play_num(self):
        num_p1_actions = self.blocks.count(P1)
        num_p2_actions = self.blocks.count(P2)

        return sum([num_p1_actions, num_p2_actions])

    def features(self):
        '''
        Summary
            Used by function approximators to represent the state.
            Override this method in State subclasses to have functiona
            approximators use a different set of features.
        Returns:
            (iterable)
        '''
        return np.array([self.blocks, self.turn]).flatten()

    def valid_moves(self):
        if self.is_terminal():
            return []
        return [ACTIONS[idx] for (idx, availablility) in enumerate(self.blocks) if availablility == self.AVAILABLE]

    def get_data(self):
        return [self.blocks, self.turn]

    def get_num_feats(self):
        return len(self.blocks()) + 1

    def is_terminal(self):
        return self.blocks.count(self.AVAILABLE) == 3

    def __hash__(self):
        # print(str([self.blocks, self.turn]))
        return hash(str([self.blocks, self.turn]))

    def __str__(self):
        return "s." + str(self.blocks) + '.turn.' + str(self.turn)

    def __eq__(self, other):
        if isinstance(other, State):
            return self.blocks == other.blocks and self.turn == other.turn
        return False

    def __getitem__(self, index):
        if index < 9:
            return self.blocks[index]
        else:
            return self.turn

    def __len__(self):
        return len(self.data) + 1

    def next(self, action_0, action_1):
        act0 = ACTIONS.index(action_0) if action_0 is not None else None
        act1 = ACTIONS.index(action_1) if action_1 is not None else None
        state = BlockGameState()
        if self.turn == P1:
            if self.blocks[act0] == self.AVAILABLE:
                state.blocks = copy(self.blocks)
                state.blocks[act0] = P1
                state.turn = P2
                return state
            else:
                pass
        if self.turn == P2:
            if self.blocks[act1] == self.AVAILABLE:
                state.blocks = copy(self.blocks)
                state.blocks[act1] = P2
                state.turn = P1
                return state
            else:
                pass
        return self

    REWARDS = [75, 65, 60, 25, 15, 10, 15, 5, 0]

    def reward(self, player):
        reward = 0
        playerBlocks = []
        for (idx, block) in enumerate(self.blocks):
            if block == player:
                reward += self.REWARDS[idx]
                playerBlocks.append(idx)
        # All the same shape
        shapes = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        if anyOf(shapes, lambda x: allOf(x, lambda y: y in playerBlocks)):
            return reward
        # All the same color
        colors = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
        if anyOf(colors, lambda x: allOf(x, lambda y: y in playerBlocks)):
            return reward
        # Mixed sets
        mixed = [[0, 4, 8], [0, 5, 7], [1, 3, 8],
                 [1, 5, 6], [2, 3, 7], [2, 4, 6]]
        if anyOf(mixed, lambda x: allOf(x, lambda y: y in playerBlocks)):
            return reward
        else:
            return -reward / 4


def anyOf(iter, fcn):
    for x in iter:
        if fcn(x):
            return True
    return False


def allOf(iter, fcn):
    for x in iter:
        if not fcn(x):
            return False
    return True


class BlockGameMDP(MarkovGameMDP):
    ''' Class for a Block Game '''

    def __init__(self):
        state = BlockGameState()
        MarkovGameMDP.__init__(
            self, ACTIONS, self._transition_func, self._reward_func, init_state=state)

    def _reward_func(self, state, action_dict, next_state=None):
        '''
        Args:
            state (State)
            action (dict of actions)

        Returns
            (float)
        '''
        actions = list(action_dict.keys())
        agent_a, agent_b = actions[P1], actions[P2]
        action_a, action_b = action_dict[agent_a], action_dict[agent_b]

        reward_dict = {}
        next_state = state.next(action_a, action_b)
        # print(state)

        # print(next_state)
        if next_state.is_terminal():
            reward_dict[agent_a], reward_dict[agent_b] = next_state.reward(
                P1), next_state.reward(P2)
            return reward_dict  # TODO
        else:
            reward_dict[agent_a], reward_dict[agent_b] = 0, 0
            return reward_dict

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action_dict (str)

        Returns
            (State)
        '''
        if state.is_terminal():
            return state
        actions = list(action.keys())
        agent_a, agent_b = actions[P1], actions[P2]
        action_a, action_b = action[agent_a], action[agent_b]
        # print(action_a, action_b)
        return state.next(action_a, action_b)

    def __str__(self):
        return "block game"

    def end_of_instance(self):
        return self.get_curr_state().is_terminal()


def main(open_plot=True):
    # Setup MDP, Agents.
    markov_game = BlockGameMDP()
    ql_agent = QLearningAgent(actions=markov_game.get_actions(), name="q1")
    fixed_agent = QLearningAgent(actions=markov_game.get_actions(), name="q2")

    # Run experiment and make plot.
    play_markov_game([ql_agent, fixed_agent], markov_game,
                     instances=5, episodes=500, steps=30, open_plot=open_plot)


if __name__ == "__main__":
    # main(open_plot=not sys.argv[-1] == "no_plot")
    val = BlockGameState()
    print(val.valid_moves())
    val = val.next(ACTIONS[0], None)
    print(val.valid_moves())
    val = val.next(None, ACTIONS[4])
    print(val.valid_moves())
