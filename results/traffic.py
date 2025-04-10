from typing import Dict, List

def get_einsums_groups_level_traffic(rsts: List[Dict]):
    traffic = {
        "DRAM": 0, "L3": 0, "reg_file": 0
    }

    for rst in rsts:
        for tl_rst in rst["tl_rsts"].values():
            level_traffic = tl_rst["tl_rst"]["traffic_energy_foreach"]["traffic"]
            traffic["DRAM"] += sum(level_traffic["DRAM"].values()) * 4
            traffic["L3"] += sum(level_traffic["L3"].values()) * 256
            traffic["reg_file"] += sum(level_traffic["reg_file_1d"].values())
            traffic["reg_file"] += sum(level_traffic["reg_file_2d"].values())

    return traffic

def get_transfusion_level_traffic(rst: Dict):
    traffic = {
        "DRAM": 0, "L3": 0, "reg_file": 0
    }

    for einsums_data in rst["einsums_outputs"].values():
        for tl_rst in einsums_data["tl_rsts"].values():
            level_traffic = tl_rst["traffic_energy_foreach"]["traffic"]
            traffic["DRAM"] += sum(level_traffic["DRAM"].values()) * 4
            traffic["L3"] += sum(level_traffic["L3"].values()) * 256
            traffic["reg_file"] += sum(level_traffic["reg_file_1d"].values())
            traffic["reg_file"] += sum(level_traffic["reg_file_2d"].values())
    return traffic

def combine_baseline_energy(rsts: List[Dict]):
    data = {"DRAM": 0, "L3": 0, "reg_file": 0, "PE": 0}
    for rst in rsts:
        for level in data.keys():
            data[level] += rst["energy_rsts"][level]

    return data

def combine_transfusion_energy(rsts: List[Dict]):
    data = {"DRAM": 0, "L3": 0, "reg_file": 0, "PE": 0}
    for rst in rsts:
        for level in data.keys():
            data[level] += rst["energy_rst"][level]

    return data
