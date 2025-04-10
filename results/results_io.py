from baseline.base import BaselineOptions, BaselineOutputs
from pathlib import Path
import json
from dataclasses import asdict, is_dataclass, fields
from typing import Tuple, Union, Any
from config.arch import ArchConfig, PE
import pandas as pd
from pipeline.core import TransFusionOptions, TransFusionTransformerOutputs

def recursive_asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = recursive_asdict(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return type(obj)(recursive_asdict(v) for v in obj)
    elif isinstance(obj, dict):
        return {k: recursive_asdict(v) for k, v in obj.items()}
    elif isinstance(obj, PE):
        return obj.value
    else:
        return obj

def write_results(
    name,
    inp: Union[TransFusionOptions, BaselineOptions],
    rst: Union[TransFusionTransformerOutputs, BaselineOutputs]
):
    base_dir = Path(__file__).parent / name
    base_dir.mkdir(parents=True, exist_ok=True)
    rst_file = base_dir / f"{inp.arch_config.name}_{inp.model}_{inp.seq_len}.json"

    with open(rst_file, "w") as f:
        json.dump(recursive_asdict(rst), f, indent=4)

def read_results(name: str, arch_config: ArchConfig, model: str, seq_len: str, fused: Union[bool, None] = None) -> Tuple[float, float]:
    base_dir = Path(__file__).parent
    rst_file: str
    if fused is not None:
        name = f"{name}_{'Fused' if fused else 'Unfused'}"
    rst_file = base_dir / name / f"{arch_config.name}_{model}_{seq_len}.json"

    with open(rst_file, "r") as f:
        data = json.load(f)
    return data["latency"], data["energy"]

