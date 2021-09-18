from block_game import BlockGameState, P1, P2
from copy import copy
from typing import Dict, Tuple

AVAILABLE = 3


"""
Class that represents a node in the block game tree

Parameters:
state_id (int): Int representing the state ID
state (BlockGameState): The state of the game
action_to_children_map (Dict[int, 'BlockGameTreeNode']): Map/dict of action to children nodes/states
visited (bool): Boolean flag for if the node has been visited (used from breadth-first search)

"""


class BlockGameTreeNode:
    def __init__(self, state_id: int, state: BlockGameState, action_to_children_map: Dict[int, 'BlockGameTreeNode'], visited: bool = False) -> None:
        self.state_id = state_id
        self.state = state
        self.action_to_children_map = action_to_children_map
        self.visited = visited


"""
Class that represents a node in the block game tree

Parameters:
head_node (BlockGameTreeNode): The top of the game tree
state_id_map (Dict[int, BlockGameTreeNode]): Map/dict of state ID to game tree node

"""


class BlockGameTree:
    def __init__(self) -> None:
        # Generate the tree by recursively creating children, starting at the very first state of the game
        self.head_node: BlockGameTreeNode = self._create_children(
            BlockGameState())

        self.state_id_map: Dict[int, BlockGameTreeNode] = {}

        # Use breadth-first search to label each state id in the tree (top of the tree is smallest, bottom is largest)
        self._bfs(self.head_node)

    def _create_children(self, parent_state: BlockGameState) -> BlockGameTreeNode:
        if parent_state.is_terminal():
            return BlockGameTreeNode(0, parent_state, {})

        new_player = P1 if parent_state.turn == P2 else P2
        blocks = parent_state.blocks

        action_state_map = {}
        for i in range(len(blocks)):
            if blocks[i] == AVAILABLE:
                new_state = BlockGameState()
                new_state.blocks = copy(blocks)
                new_state.blocks[i] = parent_state.turn
                new_state.turn = new_player
                action_state_map[i] = new_state

        new_action_children_map = {}
        for action, state in action_state_map.items():
            child_node = self._create_children(state)
            new_action_children_map[action] = child_node

        return BlockGameTreeNode(0, parent_state, new_action_children_map)

    def _bfs(self, head_node: BlockGameTreeNode) -> None:
        state_id = 0
        # Create a queue for BFS
        queue = []

        # Mark the source node as visited and enqueue it
        queue.append(head_node)
        head_node.visited = True
        head_node.state_id = state_id
        self.state_id_map[state_id] = head_node

        while queue:
            # Dequeue a node from the queue
            new_node = queue.pop(0)

            # Get all children nodes of the dequeued node new_node. If a child node has not been visited, mark and enqueue it
            for child_node in new_node.action_to_children_map.values():
                if not child_node.visited:
                    state_id += 1
                    queue.append(child_node)
                    child_node.visited = True
                    child_node.state_id = state_id
                    self.state_id_map[state_id] = child_node
