from einsum.core import *
from dataclasses import dataclass, asdict
from typing import Dict
from pathlib import Path
from baseline.base import FusedBaselineOptions, BaselineOutputs
from baseline.fusemax import FuseMax
from config.arch import PE
from itertools import product

class QKV(FuseMax):
    def __init__(self, base_input: FusedBaselineOptions):
        super().__init__(base_input)
        self.name = "QKV"

        if not base_input.fused:
            self.dram_keepin = ["Q", "BK", "BV"]

        mesh_2d = base_input.arch_config.mesh_2d
        E = self.instance.E
        F = self.instance.F
        H = self.l3_H
        P = self.l3_P * mesh_2d
        N = self.instance.N
        Q_BK_BV_space = (H*E*P + H*E*N + H*F*N) * 2

        l3_size = self.arch_config.l3_size * 1024 * 1024
        avail_l3 = l3_size - Q_BK_BV_space - self.cache_space

        #  INP, WQ, INP, WK, INP, WV
        used_space_per_d = (P + H*E + N + H*E + N + H*F) * 2

        self.l3_D = min(avail_l3 // used_space_per_d, self.instance.D)
        for factor in self.factors(self.instance.D):
            if factor <= self.l3_D:
                self.l3_D = factor
                break
            
        self.l3_D = min([self.get_max_l3_d_in_transformer(), self.l3_D])

    def get_max_l3_d_in_transformer(self):
        l3_D_factors = self.factors(self.instance.D)
        max_l3_D = 1
        for l3_D in l3_D_factors:
            l3_pe_factors = {
                "B": 1,
                "D": l3_D,
                "H": self.l3_H,
                "E": self.instance.E,
                "F": self.instance.F,
                "M": 1,
                "N": self.instance.N,
                "P": self.l3_P * self.arch_config.mesh_2d,
                "S": 1
            }
            einsums_group = [QKV_EINSUMS]
            einsums_max_occpuy = max([
                Einsum.load_from_einsums(name, einsums).get_l3_occupy(l3_pe_factors)
                for einsums in einsums_group
                for name in einsums.names
            ])
            if einsums_max_occpuy < self.arch_config.l3_size * (2**20):
                if l3_D > max_l3_D:
                    max_l3_D = l3_D
        return max_l3_D

    def get_target_einsums(self) -> Einsums:
        return QKV_EINSUMS

    def build_factors(self, dimensions: List[str], pe: PE) -> Dict[str, Dict[str, int]]:
        factors = super().build_factors(dimensions, pe)

        instance = asdict(self.instance)
        factors["DRAM"]["D"] = self.instance.D // self.l3_D
        factors["DRAM"]["H"] = instance["H"] // self.l3_H
        factors["L3"]["D"] = self.l3_D

        pecol_factors = {d: 1 for d in dimensions}
        if "E" in dimensions:
            pe_H, pe_E = self.assign_HEF(self.l3_H, instance["E"], self.arch_config.mesh_2d)
            factors["L3"]["E"] = instance["E"] // pe_E
            factors["L3"]["H"] = self.l3_H // pe_H

            pecol_factors["E"] = pe_E
            pecol_factors["H"] = pe_H
        elif "F" in dimensions:
            pe_H, pe_F = self.assign_HEF(self.l3_H, instance["F"], self.arch_config.mesh_2d)
            factors["L3"]["F"] = instance["F"] // pe_F
            factors["L3"]["H"] = self.l3_H // pe_H

            pecol_factors["F"] = pe_F
            pecol_factors["H"] = pe_H

        if "P" in dimensions:
            factors["PE_col"] = pecol_factors
        elif "N" in dimensions:
            factors["PE"] = pecol_factors

        if self.fused:
            return factors
        
        instance = asdict(self.instance)
        _factors = {
            "DRAM": {d: instance[d] for d in dimensions},
            "L3": {d: 1 for d in dimensions},
            "PE": {d: 1 for d in dimensions},
            "PE_col": {d: 1 for d in dimensions},
            "reg_file": {d: 1 for d in dimensions}
        }
        if "E" in dimensions:
            _factors["DRAM"]["E"] = self.instance.E // pecol_factors["E"]
            _factors["PE_col"]["E"] = pecol_factors["E"]
            _factors["DRAM"]["H"] = self.instance.H // pecol_factors["H"]
            _factors["PE_col"]["H"] = pecol_factors["H"]
        elif "F" in dimensions:
            _factors["DRAM"]["F"] = self.instance.F // pecol_factors["F"]
            _factors["PE_col"]["F"] = pecol_factors["F"]
            _factors["DRAM"]["H"] = self.instance.H // pecol_factors["H"]
            _factors["PE_col"]["H"] = pecol_factors["H"]
        
        if "P" in dimensions:
            _factors["DRAM"]["P"] = self.instance.P // self.arch_config.mesh_2d
            _factors["PE"]["P"] = self.arch_config.mesh_2d
        elif "N" in dimensions:
            _factors["DRAM"]["N"] = 1
            _factors["PE"]["N"] = self.instance.N
        return _factors
            

    def get_einsum_pe(self, einsum: Einsum) -> PE:
        return PE.TWO_D

    def get_total_latency(self, latency: Dict[str, float]) -> float:
        return sum(latency.values())