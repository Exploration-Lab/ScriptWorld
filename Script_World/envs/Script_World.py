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
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class ScriptWorldEnv(gym.Env):
    def __init__(
        self,
        scn='baking a cake',
        no_of_actions=5,
        allowed_wrong_actions=5,
        hop=1,
        seed=42,
        disclose_state_node=True,
        
    ):  # e.g scn = bake a cake
        self.scn = scn
        set_seed(seed)
        self.scn_graph = self.create_graph(scn)
        self.disclose_state_node = disclose_state_node
        self.cg = self.create_compact_graph(self.scn_graph)
        self.state_node = self.scn_graph["nodes"][0]["id"]
        self.state = self.state_node.partition("_")[0]
        self.sample_scn = {
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
        self.sampling_dist = self.sample_scn[self.scn]  

       

        self.t = 0
        g = self.cg
  
        self.traj = []
        self.R = 0
       
        self.cg.nodes[self.state]["no"] = 0
        self.dfs_init(g, self.state, 1)
        self.dfs_max(g, self.state, 1)
        self.Victory_node = self.cg.nodes["Victory"]["no"]
        self.wc = allowed_wrong_actions
        self.sib = {}
        self.w_count = 0
        self.per_clp = 0
        self.seq_no = 0
        self.hop = hop
        # self.seed  = seed
        self.done = False
        self.quest = self.scn_graph["nodes"][0]["quest"]
        self.no_of_actions = no_of_actions
        self.action_spaces = []
        if disclose_state_node :
          self.observation_space=spaces.Box(low=-10000,high=10000,shape=((no_of_actions),))
          self.action_space=spaces.Discrete(no_of_actions)
            
        else :
          self.observation_space=spaces.Box(low=-10000,high=10000,shape=((no_of_actions),))
          self.action_space=spaces.Discrete(no_of_actions)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.join(current_dir, "sib_dict.pickle")
        with open(fname, "rb") as f:
            data = pickle.load(f)

        self.sib_dict = data[scn]

        A = []
        for d in self.scn_graph["links"]:

            if d["source"] == self.state_node:
                A.append(d["target"])

        self.state_node = random.choice(A)
        for node in self.scn_graph["nodes"]:

            if node["id"] == self.state_node:
                self.action_spaces.append(node["action"])

                break

        count = 0
        Wrong_samples = []
        self.forbidden = []
        self.forbidden.extend(self.cg.successors(self.state_node.partition("_")[0]))
        self.forbidden.extend(self.cg.predecessors(self.state_node.partition("_")[0]))
        self.forbidden_actions = [
            node["action"]
            if node["id"].partition("_")[0] in self.forbidden
            and node["id"] != "Victory"
            and node["type"] == "slot"
            else None
            for node in list(self.scn_graph["nodes"])
        ]

        while count < self.no_of_actions - 1:

            node = random.choice(self.scn_graph["nodes"])
            if node["id"].partition("_")[0] in self.forbidden:
                continue
            
            if (
                abs(
                    self.cg.nodes[node["id"].partition("_")[0]]["no"]
                    - self.cg.nodes[self.state]["no"]
                )
                <= self.sampling_dist
            ):
                continue
            if node["id"] == "Victory":
                continue
            if node["id"].partition("_")[0] in Wrong_samples:
                continue
            if (
                node["id"].partition("_")[0] != self.state_node.partition("_")[0]
                and node["type"] == "slot"
            ):
                if (
                    node["action"] in self.action_spaces
                    or node["action"] in self.forbidden_actions
                ):
                    continue
                count += 1
                Wrong_samples.append(node["id"].partition("_")[0])
                self.action_spaces.append(node["action"])
        random.shuffle(self.action_spaces)

    def step(self, action):

        

        self.done = False
        self.prev_state = self.state_node.partition("_")[0]

        for node in self.scn_graph["nodes"]:

            if node["id"] == self.state_node:
                self.right = node["action"]

                break

       
        if self.action_spaces[action] == self.right:

            self.traj.append([self.state_node, 0])
            self.reward = 0
            self.w_count = 0
            for edge in self.scn_graph["links"]:
                if edge["source"] == self.state_node:
                    self.state_node = edge["target"]
                    break

            if (
                self.state_node[-1] == "l" and self.state_node[-2] == "_"
            ):  # it is a leave node,need more transition ,random seed will be used

                p_t = []
                
                for edge in self.scn_graph["links"]:
                    if edge["source"] == self.state_node:
                        p_t.append(edge["target"])
                        
                
                self.state_node = random.choice(p_t)

                # on entry node now
                if self.state_node != "Victory":
                    # percentage following

                    p_t = []
                    for edge in self.scn_graph["links"]:

                        if edge["source"] == self.state_node:
                            p_t.append(edge["target"])

                    
                    self.state_node = random.choice(p_t)
            try:
                # banning siblings
                
                self.forbidden = self.sib_dict[self.state_node.partition("_")[0]][
                    self.prev_state
                ]

                self.forbidden = list(set(self.forbidden))

            except:

                self.forbidden = []

        else:
            self.w_count += 1
            self.reward = -1
            self.R -= 1
            self.traj.append([self.state_node, -1])
            n_cg = self.state_node.partition("_")[0]

            try:
                self.forbidden = list(self.sib_dict[n_cg].keys())
            except:
                self.forbidden = []

            # no = self.cg.nodes[n_cg]['no']
            # no = max(0,no-self.hop)
            if self.hop != -1:
                
                if self.hop == 0:
                    self.state_node = n_cg + "_e"

                elif self.hop == 1:

                    try:
                        SN = list(self.sib_dict[n_cg].keys())
                        random.shuffle(SN)
                        n_cg = random.choice(SN)

                    except:
                        pass

                    self.state_node = n_cg + "_e"

                # now on entry node

                p_t = []
                for edge in self.scn_graph["links"]:

                    if edge["source"] == self.state_node:

                        p_t.append(edge["target"])

                
                self.state_node = random.choice(p_t)

        self.state = self.state_node.partition("_")[0]
        self.prev = self.per_clp
        self.per_clp = (
            self.cg.nodes[self.state_node.partition("_")[0]]["no"]
            / self.cg.nodes["Victory"]["no"]
        )
        if self.state_node == "Victory" or self.w_count == self.wc:  # Terminal state

            self.done = True

            if self.w_count == self.wc:
                self.reward += -5
                self.R += -5

            else:
                self.reward += 10
                self.R += 10
           
            self.traj.append([self.state_node, self.reward])
           
        self.completion_percentage = (
            self.cg.nodes[self.state]["no"] / self.Victory_node
        ) * 100
        
        # new action space

        self.action_spaces = []
        if not self.done:
            A = []
            for node in self.scn_graph["nodes"]:

                if node["id"] == self.state_node and node["type"] == "slot":
                    self.action_spaces.append(node["action"])
                    break
            
            count = 0
            Wrong_samples = []  # keep check of diversity of -ve samples
             # checking of non unique mapping of compact nodes to actions
            self.forbidden.extend(self.cg.successors(self.state_node.partition("_")[0]))
            self.forbidden.extend(
                self.cg.predecessors(self.state_node.partition("_")[0])
            )
            self.forbidden_actions = [
                node["action"]
                if node["id"].partition("_")[0] in self.forbidden
                and node["id"] != "Victory"
                and node["type"] == "slot"
                else None
                for node in list(self.scn_graph["nodes"])
            ]

            while count < self.no_of_actions - 1:
                node = random.choice(self.scn_graph["nodes"])
                if node["id"].partition("_")[0] in self.forbidden:
                    continue

                if (
                    abs(
                        self.cg.nodes[node["id"].partition("_")[0]]["no"]
                        - self.cg.nodes[self.state]["no"]
                    )
                    <= self.sampling_dist
                ):
                    continue
                if node["id"] == "Victory":
                    continue
                if node["id"].partition("_")[0] in Wrong_samples:
                    continue
                if (
                    node["id"].partition("_")[0] != self.state_node.partition("_")[0]
                    and node["type"] == "slot"
                ):
                    if (
                        node["action"] in self.action_spaces
                        or node["action"] in self.forbidden_actions
                    ):
                        continue
                    count += 1
                    Wrong_samples.append(node["id"].partition("_")[0])
                    self.action_spaces.append(node["action"])
       
        random.shuffle(self.action_spaces)

       
        if self.disclose_state_node:
            
            return (
                
                [self.state_node] + self.action_spaces,
                self.reward,
                self.done,
                {
                    "per_clp": self.completion_percentage,
                    "traj": self.traj,
                }  
            )
        else:
            return (
               
                self.action_spaces,
                self.reward,
                self.done,
                {
                    "per_clp": self.completion_percentage,
                    "traj": self.traj,
                },  
            )

    def reset(self):

        self.state_node = self.scn_graph["nodes"][0]["id"]
        self.w_count = 0
        self.done = False
        self.action_spaces = []
        self.per_clp = 0
        self.traj = []
        self.R = 0
        A = []
        for d in self.scn_graph["links"]:

            if d["source"] == self.state_node:
                A.append(d["target"])
        # accounting for siblings
        self.Siblings = []
        for S in self.cg.edges:

            if S[0] == self.state_node.partition("_")[0]:
                self.Siblings.append(S[1])

        
        self.state_node = random.choice(A)
        self.state = self.state_node.partition("_")[0]
        for node in self.scn_graph["nodes"]:

            if node["id"] == self.state_node:
                
                self.action_spaces.append(node["action"])
                break

        count = 0
        self.forbidden = []

        self.forbidden.extend(self.cg.successors(self.state_node.partition("_")[0]))
        self.forbidden.extend(self.cg.predecessors(self.state_node.partition("_")[0]))
       
        self.forbidden_actions = [
            node["action"]
            if node["id"].partition("_")[0] in self.forbidden
            and node["id"] != "Victory"
            and node["type"] == "slot"
            else None
            for node in list(self.scn_graph["nodes"])
        ]
        while count < self.no_of_actions - 1:
            node = random.choice(self.scn_graph["nodes"])
            if node["id"].partition("_")[0] in self.forbidden:
                continue
            if (
                abs(
                    self.cg.nodes[node["id"].partition("_")[0]]["no"]
                    - self.cg.nodes[self.state]["no"]
                )
                <= self.sampling_dist
            ):
                continue
            if node["id"] == "Victory":
                continue
            if node["id"] != self.state_node and node["type"] == "slot":
                if (
                    node["action"] in self.action_spaces
                    or node["action"] in self.forbidden_actions
                ):
                    continue
                count += 1
                self.action_spaces.append(node["action"])

       

        random.shuffle(self.action_spaces)

       
        if self.disclose_state_node:

           
            return [self.state_node] + self.action_spaces

        else:
           
            return self.action_spaces

    def create_graph(self, scn):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = scn + ".json"
        with open(os.path.join(dir_path, "json", file), "r") as myfile:
            data = myfile.read()

        json_str = json.loads(data)
        df = json.loads(json_str)

        return df

    def create_compact_graph(self, D):
        g = D
        G = nx.DiGraph()
        for n in g["nodes"]:
            # leave node
            if n["id"] == "Victory":
                G.add_node(n["id"])
                continue

            if n["id"][-1] == "l" and n["id"][-2] == "_":

                G.add_node(n["id"][:-2])
                for d in g["links"]:

                    if d["source"] == n["id"]:

                        if d["target"] == "Victory":
                            G.add_node(d["target"])
                            G.add_edge(n["id"][:-2], d["target"])
                            continue
                        G.add_node(d["target"][:-2])
                        G.add_edge(n["id"][:-2], d["target"][:-2])

            if n["id"][-1] == "e" and n["id"][-2] == "_":
                count = 0
                for d in g["links"]:
                    if d["source"] == n["id"]:
                        G.add_node(n["id"][:-2])
                        count += 1
                        G.nodes[n["id"][:-2]]["split_ways"] = count

        for i in G.nodes:
            G.nodes[i]["no"] = 0
        return G

    def dfs_init(self, g, n, t):

        if n == "Victory":
            self.cg.nodes[n]["no"] = t

            return

        else:

            for i in self.cg[n].keys():
                self.cg.nodes[i]["no"] = t
                self.dfs_init(g, i, t + 1)

    def dfs_max(self, g, n, t):

        if n == "Victory":
            self.cg.nodes[n]["no"] = max(t, self.cg.nodes[n]["no"])

            return
        else:
            for i in self.cg[n].keys():
                self.cg.nodes[i]["no"] = max(t, self.cg.nodes[i]["no"])
                self.dfs_max(g, i, t + 1)

