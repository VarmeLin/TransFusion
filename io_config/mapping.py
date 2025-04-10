from .yaml_input import YamlInput
from typing import Dict, List

class Mapping(YamlInput):
    def __init__(self):
        super().__init__()

    @classmethod
    def load_from_file(cls, file: str) -> "Mapping":
        mapping = cls()
        mapping.input = YamlInput.load_input_from_file(file)
        return mapping

    @classmethod
    def load_default(cls, include_pe_col = False) -> "Mapping":
        mapping = cls()
        if include_pe_col:
            mapping.input = {
                "mapping": [
                    {"target": "DRAM",     "type": "temporal",  "factors": [], "permutation": []},
                    {"target": "DRAM",     "type": "dataspace", "keep": [], "bypass": []},
                    {"target": "L3",       "type": "temporal",  "factors": [], "permutation": []},
                    {"target": "L3",       "type": "dataspace", "keep": [], "bypass": []},
                    {"target": "PE_col",   "type": "spatial",   "factors": [], "split": 5, "permutation": []},
                    {"target": "PE",       "type": "spatial",   "factors": [], "split": 0, "permutation": []},
                    {"target": "reg_file", "type": "temporal",  "factors": [], "permutation": []},
                    {"target": "reg_file", "type": "dataspace", "keep": [], "bypass": []}
                ]
            }
        else:
            mapping.input = {
                "mapping": [
                    {"target": "DRAM",     "type": "temporal",  "factors": [], "permutation": []},
                    {"target": "DRAM",     "type": "dataspace", "keep": [], "bypass": []},
                    {"target": "L3",       "type": "temporal",  "factors": [], "permutation": []},
                    {"target": "L3",       "type": "dataspace", "keep": [], "bypass": []},
                    {"target": "PE",       "type": "spatial",   "factors": [], "split": 6, "permutation": []},
                    {"target": "reg_file", "type": "temporal",  "factors": [], "permutation": []},
                    {"target": "reg_file", "type": "dataspace", "keep": [], "bypass": []}
                ]
            }
        return mapping

    def update_factors(self, parameters: Dict[str, Dict[str, int]]):
        """Update the factors of each storage or PEs.

        Args:
            parameters (Dict[str, Dict[str, int]]): The factors. For examples {"DRAM": {"B": 1, ...}, "L3": {...}}
        """
        #factors = self.__clear_factors(self.input["mapping"][-2]["factors"])
        for map in self.input["mapping"]:
            target = map["target"]
            if target in parameters and "factors" in map:
                #_factors = factors.copy()
                _factors = []
                _params = parameters[target]
                for rank, shape in _params.items():
                    _factors.append(f"{rank}={shape}")
                    #self.__update_factors(_factors, rank, shape)
                map["factors"] = _factors

    def update_keep_bypass(self, params: Dict[str, Dict[str, str]]):
        for map in self.input["mapping"]:
            target = map["target"]
            type = map["type"]
            if target in params and type == "dataspace":
                map["keep"] = params[target]["keep"]
                map["bypass"] = params[target]["bypass"]

    def update_permutation(self, params: Dict[str, List[str]]):
        """Update permutation for each levels.

        Args:
            params (Dict[str, Dict[str, List[str]]]): _description_
        """
        for map in self.input["mapping"]:
            target = map["target"]
            if target in params and "permutation" in map:
                map["permutation"] =  params[target].copy()

    def get_mappings(self):
        return self.get_input()

    def __update_factors(self, factors: List[str], rank: str, shape: int):
        i = factors.index(rank + "=1")
        factors[i] = rank + "=" + str(shape)

    def __clear_factors(self, factors: List[str]):
        new = []
        for factor in factors:
            new.append(factor.split("=")[0] + "=1")
        return new

    def __get_factor(self, factors: List[str], rank: str):
        for factor in factors:
            split = factor.split("=")
            if split[0] == rank:
                return int(split[1])
