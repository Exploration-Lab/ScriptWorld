import pandas as pd
import json
import os
import gym
import networkx as nx
import numpy as np
import random
import pickle
from gym import spaces
from tqdm import tqdm
import torch

def set_seed(seed: int = 21) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

class ScriptWorldEnv(gym.Env):
    """
    A custom OpenAI Gym environment for script-based scenarios.
    
    This environment simulates various scenarios (e.g., "baking a cake") as a graph-based
    world where the agent needs to take correct actions to reach the goal state.
    """

    def __init__(
        self,
        scenario='baking a cake',
        num_actions=5,
        allowed_wrong_actions=5,
        hop=1,
        seed=42,
        disclose_state_node=True,
    ):
        self.scenario = scenario
        set_seed(seed)
        
        # Initialize scenario graph and related attributes
        self.scenario_graph = self._create_graph(scenario)
        self.compact_graph = self._create_compact_graph(self.scenario_graph)
        self.state_node = self.scenario_graph["nodes"][0]["id"]
        self.state = self.state_node.partition("_")[0]
        
        # Define scenario-specific attributes
        self.sample_scenarios = {
            "flying in an airplane": 8,
            "repairing a flat bicycle tire": 6,
            "borrowing a book from the library": 5,
            "riding on a bus": 5,
            "getting a hair cut": 9,
            "planting a tree": 4,
            "going grocery shopping": 8,
            "baking a cake": 6,
            "going on a train": 4,
            "taking a bath": 6,
        }
        self.sampling_dist = self.sample_scenarios[self.scenario]

        # Initialize environment state
        self.disclose_state_node = disclose_state_node
        self.trajectory = []
        self.total_reward = 0
        self.wrong_action_count = 0
        self.completion_percentage = 0
        self.sequence_number = 0
        self.hop = hop
        self.done = False
        self.quest = self.scenario_graph["nodes"][0]["quest"]
        self.num_actions = num_actions
        self.allowed_wrong_actions = allowed_wrong_actions

        # Set up action and observation spaces
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Box(low=-10000, high=10000, shape=(num_actions,))

        # Initialize graph node numbers
        self._initialize_graph_numbers()

        # Load sibling dictionary
        self.sib_dict = self._load_sibling_dict()

        action_choices = []
        for d in self.scenario_graph["links"]:
            if d["source"] == self.state_node:
                action_choices.append(d["target"])

        self.state_node = random.choice(action_choices)
        self.forbidden = []
        self._update_forbidden_actions()

        # Initialize action spaces and forbidden actions
        self.action_spaces = self._initialize_action_spaces()
        
    def step(self, action):
        """
        Take a step in the environment based on the chosen action.

        Args:
            action (int): The index of the chosen action.

        Returns:
            tuple: (observation, reward, done, info)
        """
        self.done = False
        self.prev_state = self.state_node.partition("_")[0]

        # Find the correct action for the current state
        correct_action = next(node["action"] for node in self.scenario_graph["nodes"] if node["id"] == self.state_node)

        if self.action_spaces[action] == correct_action:
            # Correct action taken
            self._handle_correct_action()
        else:
            # Wrong action taken
            self._handle_wrong_action()

        # Update state and check for terminal conditions
        self.state = self.state_node.partition("_")[0]
        self._update_completion_percentage()
        self._check_terminal_conditions()

        # Generate new action spaces
        self.action_spaces = self._generate_new_action_spaces()

        # Prepare the observation
        observation = self._get_observation()

        return (
            observation,
            self.reward,
            self.done,
            {
                "completion_percentage": self.completion_percentage,
                "trajectory": self.trajectory,
            }
        )

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            list: The initial observation.
        """
        self.state_node = self.scenario_graph["nodes"][0]["id"]
        self.wrong_action_count = 0
        self.done = False
        self.completion_percentage = 0
        self.trajectory = []
        self.total_reward = 0

        # Choose a random starting state
        possible_starts = [d["target"] for d in self.scenario_graph["links"] if d["source"] == self.state_node]
        self.state_node = random.choice(possible_starts)
        self.state = self.state_node.partition("_")[0]

        # Generate initial action spaces
        self.action_spaces = self._generate_new_action_spaces()

        return self._get_observation()

    def _create_graph(self, scenario):
        """Load the scenario graph from a JSON file."""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = scenario + ".json"
        with open(os.path.join(dir_path, "json", file), "r") as f:
            data = f.read()
        return json.loads(json.loads(data))

    def _create_compact_graph(self, graph):
        """Create a compact version of the scenario graph."""
        G = nx.DiGraph()
        for node in graph["nodes"]:
            if node["id"] == "Victory":
                G.add_node(node["id"])
                continue
            if node["id"].endswith("_l"):
                base_id = node["id"][:-2]
                G.add_node(base_id)
                for link in graph["links"]:
                    if link["source"] == node["id"]:
                        target = "Victory" if link["target"] == "Victory" else link["target"][:-2]
                        G.add_node(target)
                        G.add_edge(base_id, target)
            elif node["id"].endswith("_e"):
                base_id = node["id"][:-2]
                G.add_node(base_id)
                split_ways = sum(1 for link in graph["links"] if link["source"] == node["id"])
                G.nodes[base_id]["split_ways"] = split_ways

        for node in G.nodes:
            G.nodes[node]["no"] = 0
        return G

    def _initialize_graph_numbers(self):
        """Initialize node numbers in the compact graph."""
        self.compact_graph.nodes[self.state]["no"] = 0
        self._dfs_init(self.compact_graph, self.state, 1)
        self._dfs_max(self.compact_graph, self.state, 1)
        self.victory_node_number = self.compact_graph.nodes["Victory"]["no"]

    def _dfs_init(self, graph, node, number):
        """Depth-first search to initialize node numbers."""
        if node == "Victory":
            self.compact_graph.nodes[node]["no"] = number
            return
        for neighbor in self.compact_graph[node]:
            self.compact_graph.nodes[neighbor]["no"] = number
            self._dfs_init(graph, neighbor, number + 1)

    def _dfs_max(self, graph, node, number):
        """Depth-first search to maximize node numbers."""
        if node == "Victory":
            self.compact_graph.nodes[node]["no"] = max(number, self.compact_graph.nodes[node]["no"])
            return
        for neighbor in self.compact_graph[node]:
            self.compact_graph.nodes[neighbor]["no"] = max(number, self.compact_graph.nodes[neighbor]["no"])
            self._dfs_max(graph, neighbor, number + 1)

    def _load_sibling_dict(self):
        """Load the sibling dictionary from a pickle file."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.join(current_dir, "sib_dict.pickle")
        with open(fname, "rb") as f:
            data = pickle.load(f)
        return data[self.scenario]

    def _initialize_action_spaces(self):
        """Initialize the action spaces for the current state."""
        self.action_spaces = []
        for node in self.scenario_graph["nodes"]:
            if node["id"] == self.state_node:
                self.action_spaces.append(node["action"])
                break
        return self._add_wrong_actions(self.action_spaces)

    def _add_wrong_actions(self, action_spaces):
        """Add incorrect actions to the action spaces."""
        count = 0
        while count < self.num_actions - 1:
            node = random.choice(self.scenario_graph["nodes"])
            if self._is_valid_wrong_action(node):
                count += 1
                action_spaces.append(node["action"])
        random.shuffle(action_spaces)
        return action_spaces

    def _is_valid_wrong_action(self, node):
        """Check if a node is a valid wrong action."""
        if node["id"].partition("_")[0] in self.forbidden:
            return False
        if abs(self.compact_graph.nodes[node["id"].partition("_")[0]]["no"] - self.compact_graph.nodes[self.state]["no"]) <= self.sampling_dist:
            return False
        if node["id"] == "Victory" or node["id"].partition("_")[0] == self.state_node.partition("_")[0]:
            return False
        if node["type"] != "slot":
            return False
        if node["action"] in self.action_spaces or node["action"] in self.forbidden_actions:
            return False
        return True

    def _update_forbidden_actions(self):
        """Update the list of forbidden actions."""
        self.forbidden.extend(self.compact_graph.successors(self.state_node.partition("_")[0]))
        self.forbidden.extend(self.compact_graph.predecessors(self.state_node.partition("_")[0]))
        self.forbidden_actions = [
            node["action"]
            for node in self.scenario_graph["nodes"]
            if node["id"].partition("_")[0] in self.forbidden and node["id"] != "Victory" and node["type"] == "slot"
        ]

    def _handle_correct_action(self):
        """Handle the case when the correct action is taken."""
        self.trajectory.append([self.state_node, 0])
        self.reward = 0
        self.wrong_action_count = 0
        self._move_to_next_state()

    def _handle_wrong_action(self):
        """Handle the case when a wrong action is taken."""
        self.wrong_action_count += 1
        self.reward = -1
        self.total_reward -= 1
        self.trajectory.append([self.state_node, -1])
        self._update_forbidden_after_wrong_action()
        self._move_after_wrong_action()

    def _move_to_next_state(self):
        """Move to the next state after a correct action."""
        for edge in self.scenario_graph["links"]:
            if edge["source"] == self.state_node:
                self.state_node = edge["target"]
                break
        if self.state_node.endswith("_l"):
            self._handle_leaf_node()

    def _handle_leaf_node(self):
        """Handle the case when the current state is a leaf node."""
        possible_targets = [edge["target"] for edge in self.scenario_graph["links"] if edge["source"] == self.state_node]
        self.state_node = random.choice(possible_targets)
        if self.state_node != "Victory":
            possible_targets = [edge["target"] for edge in self.scenario_graph["links"] if edge["source"] == self.state_node]
            self.state_node = random.choice(possible_targets)

    def _update_forbidden_after_wrong_action(self):
        """Update forbidden states after a wrong action."""
        current_state = self.state_node.partition("_")[0]
        try:
            self.forbidden = self.sib_dict[current_state][self.prev_state]
            self.forbidden = list(set(self.forbidden))
        except KeyError:
            self.forbidden = []

    def _move_after_wrong_action(self):
        """Move to a new state after a wrong action."""
        current_state = self.state_node.partition("_")[0]
        if self.hop != -1:
            if self.hop == 0:
                self.state_node = current_state + "_e"
            elif self.hop == 1:
                try:
                    sibling_nodes = list(self.sib_dict[current_state].keys())
                    random.shuffle(sibling_nodes)
                    current_state = random.choice(sibling_nodes)
                except KeyError:
                    pass
                self.state_node = current_state + "_e"
            possible_targets = [edge["target"] for edge in self.scenario_graph["links"] if edge["source"] == self.state_node]
            self.state_node = random.choice(possible_targets)

    def _update_completion_percentage(self):
        """Update the completion percentage of the scenario."""
        self.completion_percentage = (self.compact_graph.nodes[self.state]["no"] / self.victory_node_number) * 100

    def _check_terminal_conditions(self):
        """Check if the episode has reached a terminal state."""
        if self.state_node == "Victory" or self.wrong_action_count == self.allowed_wrong_actions:
            self.done = True
            if self.wrong_action_count == self.allowed_wrong_actions:
                self.reward -= 5
                self.total_reward -= 5
            else:
                self.reward += 10
                self.total_reward += 10
            self.trajectory.append([self.state_node, self.reward])

    def _generate_new_action_spaces(self):
        """Generate new action spaces for the current state."""
        if self.done:
            return []
        action_spaces = []
        for node in self.scenario_graph["nodes"]:
            if node["id"] == self.state_node and node["type"] == "slot":
                action_spaces.append(node["action"])
                break
        return self._add_wrong_actions(action_spaces)

    def _get_observation(self):
        """Get the current observation based on the environment state."""
        if self.disclose_state_node:
            return [self.state_node] + self.action_spaces
        else:
            return self.action_spaces