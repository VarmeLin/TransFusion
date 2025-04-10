from typing import Union, Dict, List
import math
from pipeline.pregenerate import Pregenerate, PregenerateInput
from config.arch import ArchConfig
from pathlib import Path

#  [B, D, M, P, S]
class MCTS_Node:
    def __init__(self, name: str, value: int, parent=Union["MCTS_Node", None]):
        self.name = name
        self.value = value
        self.parent: "MCTS_Node" = parent
        self.children = []
        self.visits = 0
        self.total_cost = 0

    def is_leaf(self):
        return len(self.children) == 0

    def average_cost(self):
        return self.total_cost / self.visits if self.visits > 0 else float("inf")

    def uct(self, exploration_weight=1.4):
        if self.visits == 0:
            return float("inf")

        if self.parent == None:
            return -self.average_cost()

        return -self.average_cost() + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

    def backpropagate(self, cost):
        self.visits += 1
        self.total_cost += cost

        if self.parent != None:
            self.parent.backpropagate(cost)

    def is_root(self):
        return False

class MCTS_Root(MCTS_Node):
    def __init__(self):
        super().__init__("ROOT", None, None)

    def is_root(self):
        return True

class MCTS:
    def __init__(
        self,
        candidates: List[Dict[str, Dict[str, int]]],
        arch_config: ArchConfig,
        model: str,
        seq_len: str
    ):
        self.candidates = candidates
        self.root = MCTS_Root()
        self.pregenerate = Pregenerate()
        self.arch_config = arch_config
        self.model = model
        self.seq_len = seq_len
        self.best_energy = float("inf")
        self.best_leaf = None
        self.best_factors = None

        for factors in candidates:
            self.build_from_factors(factors)

    def update_best_leaf(self, factors: Dict[str, Dict[str, int]], leaf: MCTS_Node, energy: float):
        if energy < self.best_energy:
            self.best_energy = energy
            self.best_factors = factors
            self.best_leaf = leaf

    def build_from_factors(self, factors: Dict[str, Dict[str, int]]) -> MCTS_Node:
        facs = {
            "B": factors["L3"]["B"],
            "D": factors["L3"]["D"],
            "M": factors["L3"]["M"],
            "P": factors["L3"]["P"],
            "S": factors["L3"]["S"]
        }

        parent: MCTS_Node = self.root
        for dim, val in facs.items():
            node = MCTS_Node(dim, val, parent)
            parent.children.append(node)
            parent = node

        pre_rst = self.pregenerate.read_from_file(self.arch_config, self.model, self.seq_len, factors)
        if pre_rst != None:
            energy = pre_rst["energy"]
            parent.backpropagate(energy)
            self.update_best_leaf(factors, parent, energy)

    def select(self):
        node: MCTS_Node = self.root
        while not node.is_leaf():
            node = max(node.children, key=lambda child: child.uct())
        return node

    def get_leaf_factors(self, leaf: MCTS_Node):
        _facs = {}
        node = leaf
        while not node.is_root():
            _facs[node.name] = node.value
            node = node.parent

        dims = ["B", "D", "M", "P", "S"]
        factors = None
        for _factors in self.candidates:
            if all([_factors["L3"][dim] == _facs[dim] for dim in dims]):
                factors = _factors

        assert factors != None
        return factors

    def search(self, iterations = 20):
        for i in range(iterations):
            print(f"MCTS Search iter: {i} arch_config: {self.arch_config.name}, model: {self.model}, seq_len: {self.seq_len}")
            leaf = self.select()
            factors = self.get_leaf_factors(leaf)

            pre_rst = self.pregenerate.read_from_file(
                self.arch_config,
                self.model,
                self.seq_len,
                factors
            )

            if pre_rst == None:
                inp = PregenerateInput(
                    self.arch_config,
                    self.model,
                    self.seq_len,
                    outdir=Path(__file__).parent / "mcts.out"
                )
                pre_rst = self.pregenerate.pregenerate_factors(
                    inp,
                    factors,
                    True
                )

            cost = pre_rst["energy"]
            leaf.backpropagate(cost)
            self.update_best_leaf(factors, leaf, cost)

        return self.best_factors


