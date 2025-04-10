import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from results.result import fetch_energy_results, fetch_latency_results, \
    fetch_results, fetch_layers_latency, fetch_layers_speedup_contribution, \
        fetch_layers_level_traffic, fetch_level_energy, fetch_layers_energy
from config.arch import ARCH_CLOUD, ARCH_EDGE
from pathlib import Path

def plot_models_bar_charts(data: Dict[str, Dict[str, Dict[str, float]]], ylabel="", outfile=None):
    models_num = len(data.keys())
    fig, axes = plt.subplots(1, models_num, figsize=((5*models_num), 5), sharey=True)

    fontsize = 18

    seq_lens = list(next(iter(data.values())).keys())
    methods = list(next(iter(next(iter(data.values())).values())).keys())

    x = np.arange(6)
    bar_width = 0.8 / len(methods)
    offsets = np.linspace(-0.4 + bar_width/2, 0.4 - bar_width/2, len(methods))

    for idx, (model, model_data) in enumerate(data.items()):
        ax = axes[idx]
        for i, method in enumerate(methods):
            values = [model_data[seq_len][method] for seq_len in seq_lens]
            ax.bar(x+offsets[i], values, width=bar_width, label=method)

        ax.set_title(model, fontsize=fontsize+2, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(seq_lens, fontsize=fontsize, fontweight="bold")
        ax.tick_params(axis="y", labelsize=fontsize)
        ax.set_xlabel("Sequence Length", fontsize=fontsize, fontweight="bold")

        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=fontsize, fontweight="bold")
            ax.legend(fontsize=fontsize-5)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(str(Path(__file__).parent / outfile))
    plt.show()

def plot_breakdown_model(data: Dict[str, Dict[str, Dict[str, float]]], ylabel="", outfile=None):
    fontsize = 20

    group_keys = list(data.keys())
    bar_keys = list(next(iter(data.values())).keys())
    segment_keys = list(next(iter(next(iter(data.values())).values())).keys())

    num_groups = len(group_keys)
    num_bars = len(bar_keys)
    bar_width = 0.6
    x = np.arange(num_bars)

    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    #colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
    colors = ['#4C72B0', '#8172B3', '#C44E52', '#55A868']
    bar_name_map = {"FLAT": "FL", "FuseMax": "FM",\
                    "FuseMax+LayerFuse": "FM+", "TransFusion": "TF"}

    seq_lens = data.keys()
    fig, axes = plt.subplots(1, len(seq_lens), figsize=(5*len(seq_lens), 5), sharey=True)

    for g, group_key in enumerate(group_keys):
        ax = axes[g]
        for i, bar_key in enumerate(bar_keys):
            segments = data[group_key][bar_key]
            bottom = 0
            for s, seg_key in enumerate(segment_keys):
                value = segments.get(seg_key, 0)
                ax.bar(x[i], value, bottom=bottom, width=bar_width,
                       color=colors[s], label=seg_key if g == 0 and i == 0 else "")
                bottom += value
        ax.set_xticks(x)
        ax.set_xticklabels([bar_name_map[bar] for bar in bar_keys], fontsize=fontsize, fontweight="bold")
        ax.set_title(group_key, fontsize=fontsize, fontweight="bold")

        if g == 0:
            ax.set_ylabel(ylabel, fontsize=fontsize, fontweight="bold")
            ax.legend(fontsize=fontsize-3)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    if outfile != None:
        plt.savefig(Path(__file__).parent / outfile)
    plt.show()

def plot_speedup_energy_figs():
    cloud_latency = fetch_latency_results(ARCH_CLOUD)
    edge_latency = fetch_latency_results(ARCH_EDGE)
    cloud_energy = fetch_energy_results(ARCH_CLOUD)
    edge_energy = fetch_energy_results(ARCH_EDGE)

    # cloud_latency = {model: {seq_len: {method: m_data for method, m_data in s_data.items() if method != "FuseMax+LayerFuse"} for seq_len, s_data in model_data.items()} for model, model_data in cloud_latency.items()}
    # edge_latency = {model: {seq_len: {method: m_data for method, m_data in s_data.items() if method != "FuseMax+LayerFuse"} for seq_len, s_data in model_data.items()} for model, model_data in edge_latency.items()}

    plot_models_bar_charts(cloud_latency, "Speedup Over Unfused", "cloud_latency.png")
    plot_models_bar_charts(edge_latency, "Speedup Over Unfused", "edge_latency.png")
    plot_models_bar_charts(cloud_energy, "Energy Consumption Over Unfused", "cloud_energy.png")
    plot_models_bar_charts(edge_energy, "Energy Consumption Over Unfused", "edge_energy.png")

def plot_utilization_figs():
    edge_util_1d = fetch_results(ARCH_EDGE, "1d", None)
    edge_util_2d = fetch_results(ARCH_EDGE, "2d", None)
    cloud_util_1d = fetch_results(ARCH_CLOUD, "1d", None)
    cloud_util_2d = fetch_results(ARCH_CLOUD, "2d", None)

    plot_models_bar_charts(edge_util_1d, "Utilization 1D", "edge_util_1d.png")
    plot_models_bar_charts(edge_util_2d, "Utilization 2D", "edge_util_2d.png")
    plot_models_bar_charts(cloud_util_1d, "Utilization 1D", "cloud_util_1d.png")
    plot_models_bar_charts(cloud_util_2d, "Utilization 2D", "cloud_util_2d.png")


def plot_latency_contribution_figs():
    cloud_llama3 = fetch_layers_speedup_contribution(ARCH_CLOUD, "Llama3")
    edge_llama3 = fetch_layers_speedup_contribution(ARCH_EDGE, "Llama3")
    # cloud_llama3 = fetch_layers_latency(ARCH_CLOUD, "Llama3")
    # edge_llama3 = fetch_layers_latency(ARCH_CLOUD, "Llama3")

    plot_breakdown_model(cloud_llama3, "Speedup Contribution", "cloud_speedup_contribution.png")
    plot_breakdown_model(edge_llama3, "Speedup Contribution", "edge_speedup_contribution.png")

def plot_traffic_breakdown_figs():
    cloud_llama3 = fetch_layers_level_traffic(ARCH_CLOUD, "Llama3")
    edge_llama3 = fetch_layers_level_traffic(ARCH_EDGE, "Llama3")

    plot_breakdown_model(cloud_llama3, "Traffic Over Energy", "cloud_traffic_levels.png")
    plot_breakdown_model(edge_llama3, "Traffic Over Energy", "edge_traffic_levels.png")

def plot_energy_breakdown_figs():
    cloud_llama3 = fetch_level_energy(ARCH_CLOUD, "Llama3")
    edge_llama3 = fetch_level_energy(ARCH_EDGE, "Llama3")

    plot_breakdown_model(cloud_llama3, "Energy Uses Over Energy", "cloud_energy_levels.png")
    plot_breakdown_model(edge_llama3, "Energy Uses Over Energy", "edge_energy_levels.png")

def plot_layers_energy_breakdown_figs():

    cloud_llama3 = fetch_layers_energy(ARCH_CLOUD, "Llama3")
    edge_llama3 = fetch_layers_energy(ARCH_EDGE, "Llama3")

    plot_breakdown_model(cloud_llama3, "Energy Uses Over Energy", "cloud_energy_layers.png")
    plot_breakdown_model(edge_llama3, "Energy Uses Over Energy", "edge_energy_layers.png")


def plot_results():
    plot_speedup_energy_figs()
    plot_utilization_figs()
    plot_latency_contribution_figs()
    plot_energy_breakdown_figs()
    plot_layers_energy_breakdown_figs()
