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
ACTIONS = ["red square", "blue square", "purple square"]
payoff_matrix = [[{0: 0, 1: 0}, {0: 35, 1: 70}, {0: 100, 1: 40}], [{0: 70, 1: 35 }, {0: 10, 1: 10}, {0: 45, 1: 30}], [{0: 40, 1: 100}, {0: 30, 1 : 45}, {0: 40, 1: 40}]]

class AlternatorState(State):
    ''' Abstract State class '''

    def __init__(self):
        self.selection = [-1, -1]

    def features(self):
        '''
        Summary
            Used by function approximators to represent the state.
            Override this method in State subclasses to have functiona
            approximators use a different set of features.
        Returns:
            (iterable)
        '''
        return self.selection

    def get_data(self):
        return self.selection

    def get_num_feats(self):
        return 2

    def is_terminal(self):
        return self.selection[0] != -1 and self.selection[1] != -1

    def __hash__(self):
        return hash(str(self.selection))

    def __str__(self):
        return "s." + str(self.selection) 

    def __eq__(self, other):
        if isinstance(other, State):
            return self.selection == other.selection
        return False

    def __getitem__(self, index):
        if index < 2:
          return self.selection[index]
        return -1

    def __len__(self):
        return len(self.selection) 

    def next(self, action_0, action_1):
        act0 = ACTIONS.index(action_0)
        act1 = ACTIONS.index(action_1)
        state = AlternatorState()
        state.selection[0] = act0
        state.selection[1] = act1
        return state

    def reward(self, player):
        return payoff_matrix[self.selection[0]][self.selection[1]][player]

class AlternatorMDP(MarkovGameMDP):
    ''' Class for a Block Game '''

    def __init__(self):
        state = AlternatorState()        
        MarkovGameMDP.__init__(self, ACTIONS, self._transition_func, self._reward_func, init_state=state)

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
          reward_dict[agent_a], reward_dict[agent_b] = next_state.reward(P1), next_state.reward(P2)
          return reward_dict # TODO
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
        return "alternator game"

    def end_of_instance(self):
        return self.get_curr_state().is_terminal()
        

def main(open_plot=True):
    # Setup MDP, Agents.
    markov_game = AlternatorMDP()
    ql_agent = QLearningAgent(actions=markov_game.get_actions(), name="q1")
    fixed_agent = QLearningAgent(actions=markov_game.get_actions(), name="q2")

    # Run experiment and make plot.
    play_markov_game([ql_agent, fixed_agent], markov_game, instances=5, episodes=500, steps=30, open_plot=open_plot) 

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")