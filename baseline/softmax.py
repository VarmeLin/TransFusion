from baseline.base import Baseline, FusedBaselineOptions, BaselineOutputs
from einsum.core import Einsum, Einsums
from einsum.softmax import SOFTMAX_EINSUMS
from functools import reduce
from dataclasses import asdict, dataclass
from config.arch import PE
from typing import List, Dict
from pathlib import Path

class Softmax(Baseline):
    def __init__(self, base_input: FusedBaselineOptions):
        super().__init__(base_input)

        self.name = "Softmax"

        if not base_input.fused:
            self.dram_keepin = ["SQK", "SA"]

    def get_target_einsums(self) -> Einsums:
        return SOFTMAX_EINSUMS

    def build_factors(self, dimensions: List[str], pe: PE) -> Dict[str, Dict[str, int]]:
        instance = asdict(self.instance)
        factors = {
            "DRAM": {d: instance[d] for d in dimensions},
            "L3": {d: 1 for d in dimensions},
            "PE": {d: 1 for d in dimensions},
            "reg_file": {d: 1 for d in dimensions}
        }

        factors["DRAM"]["M"] = 1
        factors["L3"]["M"] = instance["M"]

        factors["DRAM"]["N"] = 1
        factors["PE"]["N"] = instance["N"]

        return factors

    def get_einsum_pe(self, einsum: Einsum) -> PE:
        return PE.ONE_D

    def get_total_latency(self, latency: Dict[str, float]) -> float:
        return sum(latency.values())
