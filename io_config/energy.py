from typing import Union, List, Dict
from pathlib import Path
from ruamel.yaml import YAML

# def read_energy_from_outdir(outdir: Union[str, Path]) -> int:
#     pass

def read_energy_from_files(files: List[Union[str, Path]]) -> float:
    yaml = YAML(typ="safe")

    energy = 0
    for file in files:
        with open(file, "r") as f:
            energy += yaml.load(f)["energy_estimation"]["Total"]
    return energy

def read_energy_from_files_foreach(files: List[Union[str, Path]]) -> Dict[str, float]:
    yaml = YAML(typ="safe")

    dram_energy = 0
    l3_energy = 0
    regfile_energy = 0
    pe_energy = 0

    for file in files:
        with open(file, "r") as f:
            components = yaml.load(f)["energy_estimation"]["components"]

        for comp in components:
            if comp["name"] == "system_top_level.DRAM[1]":
                dram_energy += comp["energy"]
            elif comp["name"] == "system_top_level.global_buffer[1]":
                l3_energy += comp["energy"]
            elif comp["name"] == "system_top_level.reg_file[1]":
                regfile_energy += comp["energy"]
            elif comp["name"] == "system_top_level.mac[1]":
                pe_energy += comp["energy"]
            elif comp["name"] == "system_top_level.max[1]":
                pe_energy += comp["energy"]
            elif comp["name"] == "system_top_level.exponentiatial[1]":
                pe_energy += comp["energy"]
            elif comp["name"] == "system_top_level.add[1]":
                pe_energy += comp["energy"]
            elif comp["name"] == "system_top_level.divide[1]":
                pe_energy += comp["energy"]

    return {
        "DRAM": dram_energy,
        "L3": l3_energy,
        "reg_file": regfile_energy,
        "PE": pe_energy,
        "total": sum([dram_energy, l3_energy, regfile_energy, pe_energy])
    }
