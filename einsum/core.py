from typing import Dict, List, Union
from dataclasses import dataclass
from einsum.qkv import QKV_EINSUMS
from einsum.mha import MHA_EINSUMS
from einsum.ffn import FFN_EINSUMS, load_ffn_einsums_for_model
from einsum.add_norm import ANORM_EINSUMS
from einsum.einsums import Einsums

PERMUTATION  = {
    "DRAM":     ["D", "F", "E", "N", "M", "P", "H", "B"],
    "L3":       ["D", "F", "E", "P", "N", "M", "H", "B"],
    "PE_col":   ["D", "F", "E", "N", "M", "P", "H", "B"],
    "PE":       ["D", "F", "E", "N", "M", "P", "H", "B"],
    "reg_file": ["D", "F", "E", "N", "M", "P", "H", "B"]
}

EINSUMS_GROUP: List[Einsums] = [
    ANORM_EINSUMS,
    FFN_EINSUMS,
    MHA_EINSUMS,
    QKV_EINSUMS
]

EINSUMS_MAP: Dict[str, Einsums] = {
    "QKV": QKV_EINSUMS,
    "FFN": FFN_EINSUMS,
    "ANORM": ANORM_EINSUMS,
    "MHA": MHA_EINSUMS
}

class Einsum:
    def __init__(self, name: str, einsums: Einsums):
        self.name = name.upper()
        self.einsums = einsums
        self.dimensions = None
        self.permutation = None
        self.inputs = None

    @classmethod
    def load_from_name(cls, name: str, model: Union[str, None] = None) -> "Einsum":
        for _einsums in EINSUMS_GROUP:
            if name in _einsums.names:
                if _einsums == FFN_EINSUMS:
                    _einsums = load_ffn_einsums_for_model(model)
                return cls(name, _einsums)
        raise Exception(f"Cannot find the names '{name}'.")

    @classmethod
    def load_from_einsums(cls, name: str, einsums: Einsums) -> "Einsum":
        return cls(name, einsums)

    @staticmethod
    def load_qkv_einsums() -> Einsums:
        return QKV_EINSUMS

    def get_keep_bypass_with_schedule(
        self,
        dram_keepin = [],
        l3_bypass = [],
        schedules = Dict[str, dict]
    ) -> Dict[str, Dict[str, str]]:
        inputs = self.get_inputs()
        _l3_bypass = set(l3_bypass)
        my_pe = schedules[self.name]["pe"]
        for inp in inputs:
            if inp in schedules and my_pe == schedules[inp]["pe"]:
                _l3_bypass.add(inp)

        if len(self.einsums.dependency[self.name]) != 0 and all([schedules[following]["pe"] == my_pe for following in self.einsums.dependency[self.name]]):
            _l3_bypass.add(self.name)

        return self.get_keep_bypass(dram_keepin, list(_l3_bypass))

    def get_keep_bypass(
        self,
        dram_keepin = [],
        l3_bypass = []
    ) -> Dict[str, Dict[str, str]]:
        inputs = self.get_inputs()
        _keepin_dram = set(dram_keepin)
        _keepin_dram.update(self.einsums.keep_in_dram)
        _l3_bypass = set(l3_bypass)

        rst = {
            "DRAM": {"keep": [], "bypass": []},
            "L3": {"keep": [], "bypass": []},
            "reg_file": {"keep": [], "bypass": []}
        }
        #  DRAM
        #  DRAM only store input and output.
        for inp in inputs:
            if inp in _keepin_dram:
                rst["DRAM"]["keep"].append(inp)
            else:
                rst["DRAM"]["bypass"].append(inp)
        if self.name in _keepin_dram:
            rst["DRAM"]["keep"].append(self.name)
        else:
            rst["DRAM"]["bypass"].append(self.name)

        #  L3
        for inp in inputs:
            if inp in _l3_bypass:
                rst["L3"]["bypass"].append(inp)
            else:
                rst["L3"]["keep"].append(inp)
        if self.name in _l3_bypass:
            rst["L3"]["bypass"].append(self.name)
        else:
            rst["L3"]["keep"].append(self.name)

        #  RegFile
        rst["reg_file"]["keep"] += inputs
        rst["reg_file"]["keep"].append(self.name)

        return rst

    def get_dimensions(self) -> List[str]:
        if self.dimensions:
            return self.dimensions

        dims = []
        dims += [d for d in self.einsums.dimensions[self.name] if d not in dims]
        for inp in self.einsums.inputs[self.name]:
            dims += [d for d in self.einsums.dimensions[inp] if d not in dims]
        self.dimensions = list(dims)
        return self.dimensions

    def get_permutation(self) -> Dict[str, List[str]]:
        if self.permutation:
            return self.permutation

        dims = self.get_dimensions()
        rst: Dict[str, List[str]] = {}
        for level, permu in PERMUTATION.items():  #  e.g., DRAM, ["B", "D", ...]
            rst[level] = []
            for p in permu:
                if p in dims:
                    rst[level].append(p)
        self.permutation = rst
        return self.permutation

    def get_compute_count(self, instance: Dict[str, int]) -> int:
        einsum = self.name
        if einsum not in self.einsums.names:
            raise Exception(f"Unavailable einsum {einsum}")
        out_ds = self.einsums.dimensions[einsum].copy()
        ins_ds = []
        for input in self.einsums.inputs[einsum]:
            ins_ds.append(self.einsums.dimensions[input])

        rst = 1
        for d in out_ds:
            rst *= instance[d]
        for in_ds in ins_ds:
            for d in in_ds:
                #  For AV, ignore M
                if einsum == "AV" and d == "M":
                    continue
                if d not in out_ds:
                    rst *= instance[d]
                    out_ds.append(d)
        return rst * sum(self.einsums.compute_cost[self.name].values())

    def get_inputs(self) -> List[str]:
        if self.inputs is not None:
            return self.inputs
        self.inputs = self.einsums.inputs[self.name].copy()
        return self.inputs

    def get_name(self) -> str:
        return self.name

    def get_l3_occupy(self, l3_pe_factors: Dict[str, int]) -> float:
        total_l3 = 0

        params = [self.get_name()]
        params.extend(self.get_inputs())

        for p in params:
            mem = 1
            for d in self.einsums.dimensions[p]:
                mem *= l3_pe_factors[d]
            total_l3 += mem
        return total_l3 * 2  #  16-bit