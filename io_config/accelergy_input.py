from io_config.yaml_input import YamlInput
from .stats import StatsOutput
from typing import Union, Dict, List
from pathlib import Path
from ruamel.yaml import YAML
from functools import reduce

class AccelergyInput(YamlInput):
    def __init__(self, data: Dict[str, Dict[str, int]]):
        super().__init__()
        self.data: Dict[str, Dict[str, int]] = data

    @classmethod
    def load_from_stats_file(cls, file: Union[str, Path], compute_cost: Dict[str, int]) -> "AccelergyInput":
        """Load AccelergyInput object from stats file.

        Args:
            file (Union[str, Path]): The file path.
            compute_cost (Dict[str, int]): The compute cost of the einsums. For example:
                {"mac": 6, "add": 1}

        Returns:
            AccelergyInput: _description_
        """
        stats = StatsOutput.load_from_file(file)
        return AccelergyInput.load_from_stats(stats, compute_cost)

    @classmethod
    def load_from_tl_outdir(cls, outdir: Union[str, Path], compute_cost: Dict[str, int]) -> "AccelergyInput":
        stats = StatsOutput.load_from_outdir(outdir)
        return AccelergyInput.load_from_stats(stats, compute_cost)

    @classmethod
    def combine(cls, inps: List["AccelergyInput"]) -> "AccelergyInput":
        data = {"DRAM": {"read": 0, "write": 0, "update": 0, "leak": 0},
            "L3": {"read": 0, "write": 0, "update": 0, "leak": 0},
            "reg_file_2d": {"read": 0, "write": 0, "update": 0, "leak": 0},
            "reg_file_1d": {"read": 0, "write": 0, "update": 0, "leak": 0},
            "func_2d": {"mac": {"compute": 0, "leak": 0},
                    "max": {"compute": 0, "leak": 0},
                    "add": {"compute": 0, "leak": 0}},
            "func_1d": {"mac": {"compute": 0, "leak": 0},
                    "max": {"compute": 0, "leak": 0},
                    "exponentiatial": {"compute": 0, "leak": 0},
                    "add": {"compute": 0, "leak": 0},
                    "divide": {"compute": 0, "leak": 0}}
            }
        def recursive_merge(d1, d2):
            merged = {}
            for key in set(d1) | set(d2):
                v1, v2 = d1.get(key, 0), d2.get(key, 0)
                if isinstance(v1, dict) and isinstance(v2, dict):
                    merged[key] = recursive_merge(v1, v2)
                else:
                    merged[key] = v1 + v2
            return merged

        datas = [i.data for i in inps]

        data = reduce(recursive_merge, datas, data)
        return cls(data)


    @classmethod
    def load_from_stats(cls, stats: StatsOutput, compute_cost: Dict[str, int]) -> "AccelergyInput":
        data = {"DRAM": {"read": 0, "write": 0, "update": 0, "leak": 0},
            "L3": {"read": 0, "write": 0, "update": 0, "leak": 0},
            "reg_file_2d": {"read": 0, "write": 0, "update": 0, "leak": 0},
            "reg_file_1d": {"read": 0, "write": 0, "update": 0, "leak": 0},
            "func_2d": {"mac": {"compute": 0, "leak": 0},
                    "max": {"compute": 0, "leak": 0},
                    "add": {"compute": 0, "leak": 0}},
            "func_1d": {"mac": {"compute": 0, "leak": 0},
                    "max": {"compute": 0, "leak": 0},
                    "exponentiatial": {"compute": 0, "leak": 0},
                    "add": {"compute": 0, "leak": 0},
                    "divide": {"compute": 0, "leak": 0}}
            }

        def combine_stats(data_level, stats_level, width = 1):
            _read, _fills, _updates = stats.read_scalar_rfu(stats_level)
            data[data_level]["read"] += _read / width
            data[data_level]["write"] += (_fills + _updates) / width

        ##  DRAM
        combine_stats("DRAM", "DRAM", 4)

        ##  L3
        combine_stats("L3", "L3", 256)

        ##  Reg
        if stats.is_2d():
            combine_stats("reg_file_2d", "reg_file")
            combine_stats("reg_file_2d", "reg0")
            combine_stats("reg_file_2d", "reg1")
            combine_stats("reg_file_2d", "reg2")
        else:
            combine_stats("reg_file_1d", "reg_file_1d")

        ## Compute
        compute = stats.read_pint(["=== mac ===", "Computes (total)"])
        if stats.is_2d():
            for fn, cost in compute_cost.items():
                data["func_2d"][fn] = {"compute": compute * cost, "leak": 0}
        else:
            for fn, cost in compute_cost.items():
                data["func_1d"][fn] = {"compute": compute * cost, "leak": 0}

        return cls(data)

    def write_to_file(self, file_1d: Union[str, Path], file_2d: Union[str, Path]):
        if not isinstance(file_1d, Path):
            file_1d = Path(file_1d)
        if not isinstance(file_2d, Path):
            file_2d = Path(file_2d)

        file_1d.parent.mkdir(parents=True, exist_ok=True)
        file_2d.parent.mkdir(parents=True, exist_ok=True)

        def __count(name: str, num: int, data: Dict[str, int]):
            counts = []
            for key, val in data.items():
                counts.append({})
                counts[-1]["name"] = key
                counts[-1]["counts"] = val
            full_name = f"system_top_level.{name}[{num}]"
            return {"action_counts": counts, "name": full_name}

        ###################### 2D #####################
        counts_2d = {"action_counts": {"version": "0.4", "local": []}}
        local = counts_2d["action_counts"]["local"]

        local.append(__count("DRAM", 1, self.data["DRAM"]))
        local.append(__count("global_buffer", 1, self.data["L3"]))
        local.append(__count("reg_file", 1, self.data["reg_file_2d"]))
        for fu_name, fu in self.data["func_2d"].items():
            local.append(__count(fu_name, 1, fu))

        yaml = YAML(typ="safe")
        with open(file_2d, "w") as f:
            yaml.dump(counts_2d, f)

        ###################### 1D #####################
        counts_1d = {"action_counts": {"version": "0.4", "local": []}}
        local = counts_1d["action_counts"]["local"]

        local.append(__count("reg_file", 1, self.data["reg_file_1d"]))
        for fu_name, fu in self.data["func_1d"].items():
            local.append(__count(fu_name, 1, fu))

        with open(file_1d, "w") as f:
            yaml.dump(counts_1d, f)
