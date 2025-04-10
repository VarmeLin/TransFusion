from baseline.base import Baseline, FusedBaselineOptions, BaselineOutputs
from einsum.core import Einsum, Einsums
from einsum.flat import FLAT_EINSUMS
from functools import reduce
from dataclasses import asdict
from config.arch import PE
from typing import List, Dict
from pathlib import Path
from baseline.fusemax import FuseMax

class Flat(FuseMax):
    def __init__(self, base_input: FusedBaselineOptions):
        super().__init__(base_input)

        self.name = "Flat"

        if not base_input.fused:
            self.dram_keepin = ["FQ", "FK", "FV", "FAV"]

    def get_target_einsums(self) -> Einsums:
        return FLAT_EINSUMS

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
            # factors["DRAM"]["P"] = instance["P"] // self.arch_config.mesh_2d
            # factors["PE"]["P"] = self.arch_config.mesh_2d

            P1_size = self.l3_P * self.arch_config.mesh_2d
            if P1_size < instance["P"]:
                factors["DRAM"]["P"] = instance["P"] // P1_size
            else:
                factors["DRAM"]["P"] = 1
            factors["L3"]["P"] = self.l3_P
            factors["PE"]["P"] = self.arch_config.mesh_2d


            factors["DRAM"]["N"] = 1
            factors["PE_col"]["N"] = self.arch_config.mesh_2d
            factors["DRAM"]["E"] = 1
            factors["L3"]["E"] = instance["E"]
            
            if self.fused:
                factors["DRAM"]["H"] = 1
                factors["L3"]["H"] = instance["H"]
            else:
                factors["DRAM"]["H"] = instance["H"]
                factors["L3"]["H"] = 1
        elif "F" in dimensions:  #  AV
            l3_F = max(1, instance["F"] // self.arch_config.mesh_2d)

            P1_size = self.l3_P * self.arch_config.mesh_2d
            if P1_size < instance["P"]:
                factors["DRAM"]["P"] = instance["P"] // P1_size
            else:
                factors["DRAM"]["P"] = 1
            factors["L3"]["P"] = int(self.l3_P*self.arch_config.mesh_2d)

            # factors["DRAM"]["P"] = instance["P"] // self.arch_config.mesh_2d
            # factors["PE"]["P"] = self.arch_config.mesh_2d
            factors["DRAM"]["N"] = 1
            factors["PE_col"]["N"] = instance["N"]
            factors["DRAM"]["F"] = 1
            factors["L3"]["F"] = l3_F
            factors["PE"]["F"] = instance["F"] // l3_F
            # factors["DRAM"]["F"] = 1
            # factors["L3"]["F"] = instance["F"]
            if self.fused:
                factors["DRAM"]["H"] = 1
                factors["L3"]["H"] = instance["H"]
            else:
                factors["DRAM"]["H"] = instance["H"]
                factors["L3"]["H"] = 1

        return factors

    def get_einsum_pe(self, einsum: Einsum) -> PE:
        return PE.TWO_D

    def get_total_latency(self, latency: Dict[str, float]) -> float:
        return sum(latency.values())
