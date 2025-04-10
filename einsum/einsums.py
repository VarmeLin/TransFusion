from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Einsums:
    name: str
    names: List[str]
    dimensions: Dict[str, List[str]]
    dependency: Dict[str, List[str]]
    compute_cost: Dict[str, Dict[str, int]]
    inputs: Dict[str, List[str]]
    block_inputs: List[str]
    block_outputs: List[str]
    keep_in_dram: List[str]
    global_parameters: List[str]