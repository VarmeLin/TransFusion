from config.arch import ArchConfig
from typing import Union, Tuple
from pathlib import Path
import json
from results.utilization import get_einsum_groups_active, get_transfusion_active
from results.traffic import get_einsums_groups_level_traffic, get_transfusion_level_traffic, combine_baseline_energy, combine_transfusion_energy

def read_results(name: str, arch_config: ArchConfig, model: str, seq_len: str, fused: Union[bool, None] = None) -> Tuple[float, float]:
    base_dir = Path(__file__).parent
    rst_file: str
    if fused is not None:
        name = f"{name}_{'Fused' if fused else 'Unfused'}"
    rst_file = base_dir / name / f"{arch_config.name}_{model}_{seq_len}.json"

    with open(rst_file, "r") as f:
        data = json.load(f)
    return data

def get_encoder_result(qkv, mha, add_norm, ffn):
    return qkv + mha + add_norm + ffn + add_norm

def get_decoder_result(qkv, mha, add_norm, ffn):
    return qkv + (mha + add_norm)*2 + (ffn + add_norm)

def get_encoder_decoder_result(qkv, mha, add_norm, ffn):
    return get_encoder_result(qkv, mha, add_norm, ffn) + \
        get_decoder_result(qkv, mha, add_norm, ffn)

MODEL_FUNC = {
    "BERT": get_encoder_result,
    "TrXL": get_decoder_result,
    "T5": get_encoder_decoder_result,
    "XLM": get_encoder_decoder_result,
    "Llama3": get_decoder_result
}

def get_results_from_model(arch_config: ArchConfig, model: str, seq_len: str):
    ffn_fused = read_results("FFN", arch_config, model, seq_len, True)
    ffn_unfused = read_results("FFN", arch_config, model, seq_len, False)
    flat_fused = read_results("Flat", arch_config, model, seq_len, True)
    flat_unfused = read_results("Flat", arch_config, model, seq_len, False)
    fusemax_fused = read_results("FuseMax", arch_config, model, seq_len, True)
    fusemax_unfused = read_results("FuseMax", arch_config, model, seq_len, False)
    layer_norm_fused = read_results("LayerNorm", arch_config, model, seq_len, True)
    layer_norm_unfused = read_results("LayerNorm", arch_config, model, seq_len, False)
    qkv_fused = read_results("QKV", arch_config, model, seq_len, True)
    qkv_unfused = read_results("QKV", arch_config, model, seq_len, False)
    softmax_fused = read_results("Softmax", arch_config, model, seq_len, True)
    softmax_unfused = read_results("Softmax", arch_config, model, seq_len, False)
    unfused = read_results("Unfused", arch_config, model, seq_len)
    transfusion = read_results("TransFusion", arch_config, model, seq_len)

    flat_fused = {"latency": flat_fused["latency"] + softmax_fused["latency"],
                  "energy": flat_fused["energy"] + softmax_fused["energy"],
                  "energy_rsts": {level: flat_fused["energy_rsts"][level] + softmax_fused["energy_rsts"][level] for level in flat_fused["energy_rsts"].keys()},
                  "tl_rsts": {**flat_fused["tl_rsts"], **softmax_fused["tl_rsts"]}}
    flat_unfused = {"latency": flat_unfused["latency"] + softmax_fused["latency"],
                  "energy": flat_unfused["energy"] + softmax_fused["energy"],
                  "energy_rsts": {level: flat_unfused["energy_rsts"][level] + softmax_fused["energy_rsts"][level] for level in flat_unfused["energy_rsts"].keys()},
                  "tl_rsts": {**flat_unfused["tl_rsts"], **softmax_fused["tl_rsts"]}}
    unfused  = {"latency": unfused["latency"] + softmax_unfused["latency"],
                "energy": unfused["energy"] + softmax_unfused["energy"],
                "energy_rsts": {level: unfused["energy_rsts"][level] + softmax_unfused["energy_rsts"][level] for level in unfused["energy_rsts"].keys()},
                "tl_rsts": {**unfused["tl_rsts"], **softmax_unfused["tl_rsts"]}}

    lat_func = lambda *rst: MODEL_FUNC[model](*[r["latency"] for r in rst])
    eng_func = lambda *rst: MODEL_FUNC[model](*[r["energy"] for r in rst])
    util_1d_func = lambda *rst: MODEL_FUNC[model](*[get_einsum_groups_active(r)["1d"] for r in rst])
    util_2d_func = lambda *rst: MODEL_FUNC[model](*[get_einsum_groups_active(r)["2d"] for r in rst])
    trans_util_1d_func = lambda *rst: MODEL_FUNC[model](*[get_transfusion_active(r)["1d"] for r in rst])
    trans_util_2d_func = lambda *rst: MODEL_FUNC[model](*[get_transfusion_active(r)["2d"] for r in rst])
    rst = {
        "FuseMax+LayerFuse": {
            "latency": lat_func(qkv_fused, fusemax_fused, layer_norm_fused, ffn_fused),
            "energy": eng_func(qkv_fused, fusemax_fused, layer_norm_fused, ffn_fused),
            "1d": util_1d_func(qkv_fused, fusemax_fused, layer_norm_fused, ffn_fused),
            "2d": util_2d_func(qkv_fused, fusemax_fused, layer_norm_fused, ffn_fused),
            "layers_latency": {
                "QKV": qkv_fused["latency"], "MHA": fusemax_fused["latency"],
                "Add & LayerNorm": layer_norm_fused["latency"], "FFN": ffn_fused["latency"]
            },
            "layers_energy": {
                "QKV": qkv_fused["energy"], "MHA": fusemax_fused["energy"],
                "Add & LayerNorm": layer_norm_fused["energy"], "FFN": ffn_fused["energy"]
            },
            "level_traffic": get_einsums_groups_level_traffic([qkv_fused, fusemax_fused, layer_norm_fused, ffn_fused]),
            "level_energy": combine_baseline_energy([qkv_fused, fusemax_fused, layer_norm_fused, ffn_fused])
        },
        "FuseMax": {
            "latency": lat_func(qkv_unfused, fusemax_unfused, layer_norm_unfused, ffn_unfused),
            "energy": eng_func(qkv_unfused, fusemax_unfused, layer_norm_unfused, ffn_unfused),
            "1d": util_1d_func(qkv_unfused, fusemax_unfused, layer_norm_unfused, ffn_unfused),
            "2d": util_2d_func(qkv_unfused, fusemax_unfused, layer_norm_unfused, ffn_unfused),
            "layers_latency": {
                "QKV": qkv_unfused["latency"], "MHA": fusemax_unfused["latency"],
                "Add & LayerNorm": layer_norm_unfused["latency"], "FFN": ffn_unfused["latency"]
            },
            "layers_energy": {
                "QKV": qkv_unfused["energy"], "MHA": fusemax_unfused["energy"],
                "Add & LayerNorm": layer_norm_unfused["energy"], "FFN": ffn_unfused["energy"]
            },
            "level_traffic": get_einsums_groups_level_traffic([qkv_unfused, fusemax_unfused, layer_norm_unfused, ffn_unfused]),
            "level_energy": combine_baseline_energy([qkv_unfused, fusemax_unfused, layer_norm_unfused, ffn_unfused])
        },
        "FLAT": {
            "latency": lat_func(qkv_unfused, flat_unfused, layer_norm_unfused, ffn_unfused),
            "energy": eng_func(qkv_unfused, flat_unfused, layer_norm_unfused, ffn_unfused),
            "1d": util_1d_func(qkv_unfused, flat_unfused, layer_norm_unfused, ffn_unfused),
            "2d": util_2d_func(qkv_unfused, flat_unfused, layer_norm_unfused, ffn_unfused),
            "layers_latency": {
                "QKV": qkv_unfused["latency"], "MHA": flat_unfused["latency"],
                "Add & LayerNorm": layer_norm_unfused["latency"], "FFN": ffn_unfused["latency"]
            },
            "layers_energy": {
                "QKV": qkv_unfused["energy"], "MHA": flat_unfused["energy"],
                "Add & LayerNorm": layer_norm_unfused["energy"], "FFN": ffn_unfused["energy"]
            },
            "level_traffic": get_einsums_groups_level_traffic([qkv_unfused, flat_unfused, layer_norm_unfused, ffn_unfused]),
            "level_energy": combine_baseline_energy([qkv_unfused, flat_unfused, layer_norm_unfused, ffn_unfused])
        },
        "Unfused": {
            "latency": lat_func(qkv_unfused, unfused, layer_norm_unfused, ffn_unfused),
            "energy": eng_func(qkv_unfused, unfused, layer_norm_unfused, ffn_unfused),
            "1d": util_1d_func(qkv_unfused, unfused, layer_norm_unfused, ffn_unfused),
            "2d": util_2d_func(qkv_unfused, unfused, layer_norm_unfused, ffn_unfused),
            "layers_latency": {
                "QKV": qkv_unfused["latency"], "MHA": unfused["latency"],
                "Add & LayerNorm": layer_norm_unfused["latency"], "FFN": ffn_unfused["latency"]
            },
            "layers_energy": {
                "QKV": qkv_unfused["energy"], "MHA": unfused["energy"],
                "Add & LayerNorm": layer_norm_unfused["energy"], "FFN": ffn_unfused["energy"]
            },
            "level_traffic": get_einsums_groups_level_traffic([qkv_unfused, unfused, layer_norm_unfused, ffn_unfused]),
            "level_energy": combine_baseline_energy([qkv_unfused, unfused, layer_norm_unfused, ffn_unfused])
        },
        "TransFusion": {
            "latency": transfusion["latency"],
            "energy": transfusion["energy"],
            "1d": trans_util_1d_func(transfusion["einsums_outputs"]["QKV"], transfusion["einsums_outputs"]["MHA"], transfusion["einsums_outputs"]["LayerNorm"], transfusion["einsums_outputs"]["FFN"]),
            "2d": trans_util_2d_func(transfusion["einsums_outputs"]["QKV"], transfusion["einsums_outputs"]["MHA"], transfusion["einsums_outputs"]["LayerNorm"], transfusion["einsums_outputs"]["FFN"]),
            "layers_latency": {
                "QKV": transfusion["einsums_outputs"]["QKV"]["latency"],
                "MHA": transfusion["einsums_outputs"]["MHA"]["latency"],
                "Add & LayerNorm": transfusion["einsums_outputs"]["LayerNorm"]["latency"],
                "FFN": transfusion["einsums_outputs"]["FFN"]["latency"]
            },
            "layers_energy": {
                "QKV": transfusion["einsums_outputs"]["QKV"]["energy"],
                "MHA": transfusion["einsums_outputs"]["MHA"]["energy"],
                "Add & LayerNorm": transfusion["einsums_outputs"]["LayerNorm"]["energy"],
                "FFN": transfusion["einsums_outputs"]["FFN"]["energy"]
            },
            "level_traffic": get_transfusion_level_traffic(transfusion),
            "level_energy": combine_transfusion_energy(transfusion["einsums_outputs"].values())
        }
    }

    for _, _rst in rst.items():
        _rst["1d"] = _rst["1d"] / _rst["latency"]
        _rst["2d"] = _rst["2d"] / _rst["latency"]

    return rst