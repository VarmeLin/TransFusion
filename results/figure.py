import matplotlib.pyplot as plt
import numpy as np
from config.arch import ARCH_CLOUD, ARCH_EDGE
from results.model import get_results_from_model
from itertools import product
from pathlib import Path

models = ["BERT", "TrXL", "T5", "XLM", "Llama3"]
seq_lens = ["1K", "4K", "16K", "64K", "256K", "1M"]
arch_configs = [ARCH_EDGE, ARCH_CLOUD]
methods = ["Unfused", "FLAT", "FuseMax", "TransFusion"]

def plot_figures():
    rsts = {}
    for arch_config, model, seq_len in product(arch_configs, models, seq_lens):
        if arch_config.name not in rsts:
            rsts[arch_config.name] = {}
        if model not in rsts[arch_config.name]:
            rsts[arch_config.name][model] = {}
        rsts[arch_config.name][model][seq_len] = get_results_from_model(arch_config, model, seq_len)

    fusemax_speedup = []
    fusemax_energy = []
    flat_speedup = []
    flat_energy = []

    rrrr = []

    # for idx, (metric, title) in enumerate(zip(["latency", "energy"], ["Latency", "Energy"])):
    #     for jdx, device in enumerate(["cloud", "edge"]):
    for metric, title in zip(["latency", "energy"], ["Speedup over Unfused", "Energy Consumption over Unfused"]):
        for arch in ["cloud", "edge"]:
            bar_width = 0.2
            x = np.arange(len(seq_lens))
            fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

            for idx, model in enumerate(models):
                ax = axes[idx]
                for i, method in enumerate(methods):
                    data = []
                    for j, seq in enumerate(seq_lens):
                        baseline = rsts[arch][model][seq]["Unfused"][metric]

                        xxx = rsts[arch][model][seq]["TransFusion"][metric]
                        if method in rsts[arch][model][seq] and rsts[arch][model][seq][method][metric] != 0:
                            if metric == "latency":
                                d = baseline / rsts[arch][model][seq][method][metric]
                            else:
                                d = rsts[arch][model][seq][method][metric] / baseline
                            data.append(d)
                            rrrr.append({
                                "arch": arch,
                                "metric": metric,
                                "method": method,
                                "model": model,
                                "data": xxx/rsts[arch][model][seq][method][metric]
                            })
                        else:
                            data.append(0)
                    if methods[i] != "Unfused":
                        ax.bar(x + i * bar_width, data, width=bar_width, label=methods[i])

                ax.set_title(model)
                ax.set_xticks(x + 1.5 * bar_width)
                ax.set_xticklabels(seq_lens)
                ax.set_xlabel("Sequence Length")
                if idx == 0:
                    ax.set_ylabel(title)
                    ax.legend(fontsize=8)
            #fig.suptitle(f"{arch} {title}", fontsize=16, fontweight="bold")
            plt.tight_layout()
            plt.savefig(str(Path(__file__).parent / f"{arch}_{metric}.png"))
            #plt.show()


    def average(lll):
        return sum(lll) / len(lll)

    # def max(lll):
    #     return max(lll)

    for metric in ["latency", "energy"]:
        for arch in ["cloud", "edge"]:
            for method in ["FuseMax", "FLAT", "Unfused"]:
                ddd = [r["data"] for r in rrrr if r["arch"] == arch and r["metric"] == metric and r["method"] == method]
                print(metric, arch, method, "Ave: ", average(ddd), 1/average(ddd), "max: ", max(ddd), "min: ", min(ddd))
#    cloud_fusemax_speedup = [r["data"] for r in rrrr if r["arch"] == "cloud" and r["metric"] == "latency" and r["method"] == "FuseMax"]

    #cloud_flat_speedup
    #print(1/average(cloud_fusemax_speedup))