from einsum.core import *
from dataclasses import dataclass, asdict
from typing import Dict
from pathlib import Path
from baseline.base import FusedBaselineOptions, BaselineOutputs
from baseline.fusemax import FuseMax
from config.arch import PE

class FFN(FuseMax):
    def __init__(self, base_input: FusedBaselineOptions):
        super().__init__(base_input)
        self.name = "FFN"

        self.model = base_input.model

        if not base_input.fused:
            self.dram_keepin = ["NR", "FFNT"]

        mesh_2d = base_input.arch_config.mesh_2d
        F = self.instance.F
        H = self.l3_H
        P = self.l3_P * mesh_2d
        NR_BT_FFN_space = (H * F * (1 + P + P)) * 2

        l3_size = self.arch_config.l3_size * 1024 * 1024
        avail_l3 = l3_size - NR_BT_FFN_space - self.cache_space

        #  FFN, WF, B, WFT
        used_space_per_s = (P + F*H + 1 + F*H) * 2

        proposed_S = max(1, avail_l3 // (used_space_per_s * mesh_2d))
        self.l3_S = min(proposed_S, self.instance.S // mesh_2d)
        for factor in self.factors(self.instance.S // mesh_2d):
            if factor <= self.l3_S:
                self.l3_S = factor
                break

        self.einsums_2d = ["FFN", "FFNT"]
        
        self.l3_S = min([self.l3_S, self.get_max_l3_s_in_transformer()])
        
    def get_max_l3_s_in_transformer(self):
        for l3_S in self.factors(self.instance.S):
            l3_pe_factors = {
                "B": 1,
                "D": 1,
                "H": self.l3_H,
                "E": self.instance.E,
                "F": self.instance.F,
                "M": 1,
                "N": self.instance.N,
                "P": self.l3_P * self.arch_config.mesh_2d,
                "S": l3_S * self.arch_config.mesh_2d
            }
            einsums_group = [load_ffn_einsums_for_model(self.model)]
            einsums_max_occpuy = max([
                Einsum.load_from_einsums(name, einsums).get_l3_occupy(l3_pe_factors)
                for einsums in einsums_group
                for name in einsums.names
            ])
            if einsums_max_occpuy < self.arch_config.l3_size * (2**20):
                return l3_S
        return 1

    def get_target_einsums(self) -> Einsums:
        return load_ffn_einsums_for_model(self.model)

    def build_factors(self, dimensions: List[str], pe: PE) -> Dict[str, Dict[str, int]]:
        factors = super().build_factors(dimensions, pe)

        factors["DRAM"]["S"] = self.instance.S // (self.l3_S*self.arch_config.mesh_2d)
        if pe == PE.TWO_D:
            factors["L3"]["S"] = self.l3_S
            factors["PE_col"]["S"] = self.arch_config.mesh_2d
        elif pe == PE.ONE_D:
            pe_S, _ = self.assign_HEF(self.l3_S *self.arch_config.mesh_2d, 1, self.arch_config.mesh_1d // factors["PE"]["P"])
            factors["L3"]["S"] = self.l3_S * self.arch_config.mesh_2d // pe_S
            factors["PE"]["S"] = pe_S
        else:
            raise Exception(f"Invalid PE {pe}")
        
        if self.fused:
            return factors

        instance = asdict(self.instance)
        _factors = {
            "DRAM": {d: instance[d] for d in dimensions},
            "L3": {d: 1 for d in dimensions},
            "PE": {d: 1 for d in dimensions},
            "reg_file": {d: 1 for d in dimensions}
        }
        if pe == PE.TWO_D:
            _factors["PE_col"] = {d: 1 for d in dimensions}
            _factors["DRAM"]["P"] = self.instance.P // self.arch_config.mesh_2d
            _factors["PE"]["P"] = self.arch_config.mesh_2d
            _factors["DRAM"]["S"] = self.instance.S // self.arch_config.mesh_2d
            _factors["PE_col"]["S"] = self.arch_config.mesh_2d
        elif pe == PE.ONE_D:
            _factors["DRAM"]["P"] = self.instance.P // self.arch_config.mesh_1d
            _factors["PE"]["P"] = self.arch_config.mesh_1d
        else:
            raise Exception(f"Invalid PE {pe}")
        return _factors

    def get_total_latency(self, latency: Dict[str, float]) -> float:
        return sum(latency.values())