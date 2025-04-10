from baseline import *
from results.results_io import write_results
from config.arch import ArchConfig, ARCH_CLOUD, ARCH_EDGE
from einsum.core import MHA_EINSUMS, ANORM_EINSUMS, QKV_EINSUMS, FFN_EINSUMS
from pathlib import Path
from itertools import product
from pipeline.search_factor2 import get_all_candidates
from pipeline.mcts import MCTS
from pipeline.instance import get_instance
from dataclasses import asdict
from pipeline.core import TransFusion, TransFusionOptions
from results.figure2 import plot_results

def run_baseline(baseline: Baseline, opts: BaselineOptions):
    if isinstance(opts, FusedBaselineOptions):
        print(f"Running {baseline.name}: {opts.arch_config.name}, {opts.model}, {opts.seq_len}, fused_{opts.fused} ...")
        name = f"{baseline.name}_{'Fused' if opts.fused else 'Unfused'}"
    else:
        print(f"Running {baseline.name}: {opts.arch_config.name}, {opts.model}, {opts.seq_len} ...")
        name = f"{baseline.name}"
    base_dir = Path(__file__).parent
    outp = baseline.eval_einsums(
        baseline.get_target_einsums(),
        outdir=base_dir/"outs"/opts.arch_config.name/opts.model/opts.seq_len/name
    )
    write_results(name, opts, outp)

def MCTS_search(arch_config: ArchConfig, model: str, seq_len: str):
    print(f"Running TransFusion: {arch_config.name}, {model}, {seq_len} ...")
    instances = asdict(get_instance(model, seq_len, arch_config.mesh_2d))
    candidates = get_all_candidates(arch_config, instances)
    mcts = MCTS(candidates, arch_config, model, seq_len)
    factors = mcts.search(20)

    inp = TransFusionOptions(
        model,
        seq_len,
        arch_config
    )

    transfusion = TransFusion(inp)

    base_dir = Path(__file__).parent
    outp = transfusion.eval_transformer(factors, base_dir/"outs"/arch_config.name/model/seq_len/"TransFusion")

    write_results("TransFusion", inp, outp)


def main():
    models = ["BERT", "TrXL", "T5", "XLM", "Llama3"]
    seq_lens = ["1K", "4K", "16K", "64K", "256K", "1M"]
    arch_configs = [ARCH_EDGE, ARCH_CLOUD]
    baseline_class = [Unfused]
    fused_baseline_class = [FuseMax, Flat, FFN, LayerNorm, QKV, Softmax]
    for arch_config, model, seq_len in product(arch_configs, models, seq_lens):
        for b_class in baseline_class:
            opts = BaselineOptions(model, seq_len, arch_config)
            baseline = b_class(opts)
            run_baseline(baseline, opts)

        for b_class, fused in product(fused_baseline_class, [True, False]):
            opts = FusedBaselineOptions(model, seq_len, arch_config, fused)
            baseline = b_class(opts)
            run_baseline(baseline, opts)

        MCTS_search(arch_config, model, seq_len)

    plot_results()

if __name__ == "__main__":
    main()