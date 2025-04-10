import matplotlib.pyplot as plt
import numpy as np
from config.arch import ARCH_CLOUD, ARCH_EDGE, ArchConfig
from results.model import get_results_from_model
from itertools import product
from pathlib import Path

models = ["BERT", "TrXL", "T5", "XLM", "Llama3"]
seq_lens = ["1K", "4K", "16K", "64K", "256K", "1M"]
arch_configs = [ARCH_EDGE, ARCH_CLOUD]
methods = ["FLAT", "FuseMax", "FuseMax+LayerFuse", "TransFusion"]

def fetch_results(arch_config: ArchConfig, metric: str, baseline = "Unfused"):
    data = {\
            model: {\
                seq_len: {\
                    method: 0 for method in methods} \
        for seq_len in seq_lens} for model in models}

    for model in models:
        for seq_len in seq_lens:
            rst = get_results_from_model(arch_config, model, seq_len)
            if baseline != None:
                _baseline = rst[baseline][metric]
            else:
                _baseline = 1
            for method in methods:
                data[model][seq_len][method] = rst[method][metric] / _baseline
                if metric == "latency":
                    data[model][seq_len][method] = 1/data[model][seq_len][method]
    return data

def fetch_latency_results(arch_config: ArchConfig):
    return fetch_results(arch_config, "latency")

def fetch_energy_results(arch_config: ArchConfig):
    return fetch_results(arch_config, "energy")

def get_average_metric(arch_config: ArchConfig, metric):
    data = fetch_results(arch_config, metric)
    rst = {method: 0 for method in methods}

    for method in methods:
        total = 0
        count = 0
        for model in models:
            for seq_len in seq_lens:
                total += data[model][seq_len]["TransFusion"] / \
                        data[model][seq_len][method]
                count += 1
        rst[method] = total / count
    return rst

def fetch_layers_latency(arch_config: ArchConfig, model: str):
    data = {seq_len: {method: None for method in methods} for seq_len in seq_lens}

    for seq_len in seq_lens:
        rst = get_results_from_model(arch_config, model, seq_len)
        baseline = sum(rst["Unfused"]["layers_latency"].values())
        for method in methods:
            speedup = baseline / sum(rst[method]["layers_latency"].values())
            layers_latency = rst[method]["layers_latency"]
            total = sum(layers_latency.values())
            layers_latency = {k: v*speedup/total for k, v in layers_latency.items()}
            data[seq_len][method] = layers_latency
    return data

def fetch_layers_speedup_contribution(arch_config: ArchConfig, model: str):
    data = {seq_len: {method: None for method in methods} for seq_len in seq_lens}

    for seq_len in seq_lens:
        rst = get_results_from_model(arch_config, model, seq_len)
        baseline = rst["Unfused"]["layers_latency"]

        for method in methods:
            layers_latency = rst[method]["layers_latency"]
            speedup = sum(baseline.values()) / sum(layers_latency.values())

            Speedup_layers = {layer: baseline[layer] / l_lat for layer, l_lat in layers_latency.items()}
            w_layers = {layer: baseline[layer]/sum(baseline.values()) for layer in layers_latency.keys()}
            speedup_contribution = {layer: w_layers[layer]*Speedup_layers[layer] for layer in layers_latency.keys()}
            speedup_contribution_norm = {layer: speedup_contribution[layer] / sum(speedup_contribution.values()) for layer in layers_latency.keys()}
            layers_latency = {layer: speedup*speedup_contribution_norm[layer] for layer in layers_latency.keys()}
            data[seq_len][method] = layers_latency
    return data

def fetch_layers_level_traffic(arch_config: ArchConfig, model: str):
    data = {seq_len: {method: None for method in methods} for seq_len in seq_lens}

    for seq_len in seq_lens:
        rst = get_results_from_model(arch_config, model, seq_len)
        baseline = rst["Unfused"]["level_traffic"]
        for method in methods:
            level_traffic = rst[method]["level_traffic"]
            total_traffic = sum(level_traffic.values())
            energy_norm = total_traffic / sum(baseline.values())
            level_norm = {level: level_traffic[level]/total_traffic*energy_norm \
                           for level in level_traffic.keys()}
            data[seq_len][method] = level_norm
    return data

def fetch_level_energy(arch_config: ArchConfig, model: str):
    data = {seq_len: {method: None for method in methods} for seq_len in seq_lens}

    for seq_len in seq_lens:
        rst = get_results_from_model(arch_config, model, seq_len)
        baseline = rst["Unfused"]["level_energy"]
        for method in methods:
            level_energy = rst[method]["level_energy"]
            total_traffic = sum(level_energy.values())
            energy_norm = total_traffic / sum(baseline.values())
            level_norm = {level: level_energy[level]/total_traffic*energy_norm \
                           for level in level_energy.keys()}
            data[seq_len][method] = level_norm
    return data

def fetch_layers_energy(arch_config: ArchConfig, model: str):
    data = {seq_len: {method: None for method in methods} for seq_len in seq_lens}

    for seq_len in seq_lens:
        rst = get_results_from_model(arch_config, model, seq_len)
        baseline = sum(rst["Unfused"]["layers_energy"].values())
        for method in methods:
            _energy = sum(rst[method]["layers_energy"].values()) / baseline
            layers_energy = rst[method]["layers_energy"]
            total = sum(layers_energy.values())
            layers_energy = {k: v*_energy/total for k, v in layers_energy.items()}
            data[seq_len][method] = layers_energy
    return data

# print(get_average_metric(ARCH_CLOUD, "latency"))
# print(get_average_metric(ARCH_EDGE, "latency"))
# print(get_average_metric(ARCH_CLOUD, "energy"))
# print(get_average_metric(ARCH_EDGE, "energy"))
#print(fetch_layers_latency(ARCH_CLOUD, "BERT"))