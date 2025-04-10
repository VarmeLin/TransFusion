from baseline.base import Baseline, FusedBaselineOptions, BaselineOutputs
from einsum.core import *
from functools import reduce
from dataclasses import asdict
from config.arch import PE
from typing import List, Dict
from pathlib import Path

class FuseMax(Baseline):
    def __init__(self, base_input: FusedBaselineOptions):
        super().__init__(base_input)

        self.name = "FuseMax"
        self.model = base_input.model

        self.l3_bypass = ["SLN", "SPNV"]

        self.fused = base_input.fused
        if not base_input.fused:
            self.dram_keepin = ["Q", "BK", "BV", "AV"]
            self.l3_H = 1
        else:
            self.l3_H = self.instance.H
        
        mesh_2d = self.arch_config.mesh_2d
        bk_bv_space = (mesh_2d * self.instance.E * 2 \
            + mesh_2d * self.instance.F * 2) * self.l3_H

        self.cache_space = mesh_2d * 2  #  Extra cache space to support communicate
                                        #  between 1D and 2D

        l3_size = self.arch_config.l3_size * 1024 * 1024
        avail_l3 = l3_size - bk_bv_space - self.cache_space

        #  Q, RM, RD, RNV
        used_space_per_p = (self.instance.E + 1 + 1 + self.instance.F) * 2 * self.l3_H

        proposed_P1 = max(1, avail_l3 // (used_space_per_p * mesh_2d))
        self.l3_P = min(proposed_P1, self.instance.P // mesh_2d)

        #  Do not use improper factorization.
        for factor in self.factors(self.instance.P // mesh_2d):
            if factor <= self.l3_P:
                self.l3_P = factor
                break

        self.einsums_2d = ["QK", "SLN", "SLD", "SLNV"]
        
        self.l3_P = min([self.get_max_l3_p_in_transformer(), self.l3_P])
    
    def get_max_l3_p_in_transformer(self):
        l3_P_factors = self.factors(self.instance.P // self.arch_config.mesh_2d)
        max_l3_P = 1
        for l3_P in l3_P_factors:
            l3_pe_factors = {
                "B": 1,
                "D": 1,
                "H": self.l3_H,
                "E": self.instance.E,
                "F": self.instance.F,
                "M": 1,
                "N": self.instance.N,
                "P": l3_P * self.arch_config.mesh_2d,
                "S": self.arch_config.mesh_2d
            }
            einsums_group = [QKV_EINSUMS, MHA_EINSUMS, ANORM_EINSUMS, load_ffn_einsums_for_model(self.model)]
            einsums_max_occpuy = max([
                Einsum.load_from_einsums(name, einsums).get_l3_occupy(l3_pe_factors)
                for einsums in einsums_group
                for name in einsums.names
            ])
            if einsums_max_occpuy < self.arch_config.l3_size * (2**20):
                if l3_P > max_l3_P:
                    max_l3_P = l3_P
        return max_l3_P

    def get_target_einsums(self) -> Einsums:
        return MHA_EINSUMS

    def factors(self, n):
        # Source:
        # https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
        factors = list(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
        factors.sort(reverse=True)
        return factors

    def build_factors(self, dimensions: List[str], pe: PE) -> Dict[str, Dict[str, int]]:
        instance = asdict(self.instance)
        rst = {}

        #  DRAM
        dram_factors = {}
        for dim in dimensions:
            if dim == "P":
                P1_size = self.l3_P * self.arch_config.mesh_2d
                if P1_size < instance["P"]:
                    dram_factors["P"] = instance["P"] // P1_size
                else:
                    dram_factors["P"] = 1
            elif dim == "H":
                dram_factors[dim] = instance["H"] // self.l3_H
            elif dim == "E" or dim == "F" or dim == "N":
                dram_factors[dim] = 1
            else:
                dram_factors[dim] = instance[dim]
        rst["DRAM"] = dram_factors

        #  L3 factors
        l3_factors = {}
        for dim in dimensions:
            if dim == "P":
                l3_factors["P"] = self.l3_P
            elif dim == "H":
                l3_factors["H"] = self.l3_H
            elif dim == "E" or dim == "F":
                l3_factors[dim] = instance[dim]
            else:
                l3_factors[dim] = 1
        rst["L3"] = l3_factors

        pe_factors = {}
        for dim in dimensions:
            if dim == "P":
                pe_factors["P"] = self.arch_config.mesh_2d
            else:
                pe_factors[dim] = 1
        rst["PE"] = pe_factors

        if pe == PE.TWO_D:
            pecol_factors = {}
            for dim in dimensions:
                if dim == "N":
                    pecol_factors["N"] = self.arch_config.mesh_2d
                else:
                    pecol_factors[dim] = 1
            rst["PE_col"] = pecol_factors
        elif pe == PE.ONE_D:
            if "N" in dimensions:
                rst["L3"]["N"] = self.instance.N

        reg_factors = {dim: 1 for dim in dimensions}
        rst["reg_file"] = reg_factors

        return rst

    def get_einsum_pe(self, einsum: Einsum) -> PE:
        return PE.TWO_D if einsum.name in self.einsums_2d else PE.ONE_D

    def get_total_latency(self, latency: Dict[str, float]) -> float:
        qk = latency["QK"]
        lm_rm = latency["LM"] + latency["RM"]
        sln_sld = latency["SLN"] + latency["SLD"]
        prm_spd = latency["PRM"] + latency["SPD"]
        slnv = latency["SLNV"]
        spnv = latency["SPNV"]
        rd_rnv_av = latency["RD"] + latency["RNV"] + latency["AV"]

        total_latency = max(qk, rd_rnv_av) + lm_rm + max(sln_sld, prm_spd) + max(slnv, spnv)
        return total_latency
