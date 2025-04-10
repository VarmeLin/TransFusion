from baseline.base import Baseline, BaselineOptions, BaselineOutputs
from einsum.core import Einsum, Einsums
from einsum.unfused import UNFUSED_EINSUMS
from functools import reduce
from dataclasses import asdict
from config.arch import PE
from typing import List, Dict
from pathlib import Path

class Unfused(Baseline):
    def __init__(self, base_input: BaselineOptions):
        super().__init__(base_input)

        self.name = "Unfused"

    def get_target_einsums(self) -> Einsums:
        return UNFUSED_EINSUMS

    def build_factors(self, dimensions: List[str], pe: PE) -> Dict[str, Dict[str, int]]:
        instance = asdict(self.instance)
        factors = {
            "DRAM": {d: instance[d] for d in dimensions},
            "L3": {d: 1 for d in dimensions},
            "PE": {d: 1 for d in dimensions},
            "PE_col": {d: 1 for d in dimensions},
            "reg_file": {d: 1 for d in dimensions}
        }

        if "E" in dimensions:  #  QK
            factors["DRAM"]["P"] = instance["P"] // self.arch_config.mesh_2d
            factors["PE"]["P"] = self.arch_config.mesh_2d
            factors["DRAM"]["N"] = 1
            factors["PE_col"]["N"] = self.arch_config.mesh_2d
        elif "F" in dimensions:  #  AV
            factors["DRAM"]["P"] = instance["P"] // self.arch_config.mesh_2d
            factors["PE"]["P"] = self.arch_config.mesh_2d
            factors["DRAM"]["N"] = 1
            factors["PE_col"]["N"] = self.arch_config.mesh_2d

        return factors

    def get_einsum_pe(self, einsum: Einsum) -> PE:
        return PE.TWO_D

    def get_total_latency(self, latency: Dict[str, float]) -> float:
        return sum(latency.values())
