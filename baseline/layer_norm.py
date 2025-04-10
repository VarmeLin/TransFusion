from einsum.core import *
from dataclasses import dataclass, asdict
from config.arch import PE
from typing import Dict
from pathlib import Path
from baseline.base import FusedBaselineOptions, BaselineOutputs
from baseline.fusemax import FuseMax
from itertools import product

class LayerNorm(FuseMax):
    def __init__(self, base_input: FusedBaselineOptions):
        super().__init__(base_input)
        self.name = "LayerNorm"

        if not base_input.fused:
            self.dram_keepin = ["AV", "NR"]

    def get_target_einsums(self) -> Einsums:
        return ANORM_EINSUMS
    
    def build_factors(self, dimensions: List[str], pe: PE) -> Dict[str, Dict[str, int]]:
        factors = super().build_factors(dimensions, pe)
        
        if "H" in dimensions and "F" in dimensions:
            pe_H, pe_F = self.assign_HEF(
                factors["L3"]["H"], 
                factors["L3"]["F"],
                self.arch_config.mesh_1d // factors["PE"]["P"]
            )
            factors["PE"]["H"] = pe_H
            factors["PE"]["F"] = pe_F
            factors["L3"]["H"] = factors["L3"]["H"] // pe_H
            factors["L3"]["F"] = factors["L3"]["F"] // pe_F
        
        if self.fused:
            return factors

        instance = asdict(self.instance)
        _factors = {
            "DRAM": {d: instance[d] for d in dimensions},
            "L3": {d: 1 for d in dimensions},
            "PE": {d: 1 for d in dimensions},
            "reg_file": {d: 1 for d in dimensions}
        }
        _factors["DRAM"]["P"] = self.instance.P // factors["PE"]["P"]
        _factors["PE"]["P"] = factors["PE"]["P"]
        if "H" in dimensions and "F" in dimensions:
            _factors["DRAM"]["H"] = self.instance.H // factors["PE"]["H"]
            _factors["PE"]["H"] = factors["PE"]["H"]
            _factors["DRAM"]["F"] = self.instance.F // factors["PE"]["F"]
            _factors["PE"]["F"] = factors["PE"]["F"]
        return _factors

    def get_einsum_pe(self, einsum: Einsum) -> PE:
        return PE.ONE_D

    def get_total_latency(self, latency: Dict[str, float]) -> float:
        return sum(latency.values())