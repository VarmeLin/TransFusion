from dataclasses import dataclass
from config.arch import ArchConfig
from pipeline.instance import Instance
from einsum.core import Einsum
from pipeline.scheduler import Schedule
from pipeline.session import Session, SessionOutput, SessionInput
from pathlib import Path
import fcntl
import csv
from typing import Dict
from engine.timeloop import TimeloopResult
from dataclasses import asdict
import os
from pathlib import Path
from pipeline.core import TransFusion, TransFusionOptions, TransFusionTransformerOutputs

@dataclass(frozen=True)
class PregenerateInput:
    arch_config: ArchConfig
    model: str
    seq_len: str
    outdir: Path

class Pregenerate:
    def factors_to_field_values(self, factors: Dict[str, Dict[str, int]]):
        fields = []
        values = {}
        for level in factors.keys():
            for dim in factors[level].keys():
                key = f"{level}.{dim}"
                fields.append(key)
                values[key] = factors[level][dim]
        return fields, values

    def result_to_field_values(self, outp: TransFusionTransformerOutputs):
        fields = ["latency", "energy",
                  "QKV.latency", "QKV.energy",
                  "MHA.latency", "MHA.energy",
                  "LayerNorm.latency", "LayerNorm.energy",
                  "FFN.latency", "FFN.energy"]
        values = {"latency": outp.latency, "energy": outp.energy,
                  "QKV.latency": outp.einsums_outputs["QKV"].latency, "QKV.energy": outp.einsums_outputs["QKV"].energy,
                  "MHA.latency": outp.einsums_outputs["MHA"].latency, "MHA.energy": outp.einsums_outputs["MHA"].energy,
                  "LayerNorm.latency": outp.einsums_outputs["LayerNorm"].latency, "LayerNorm.energy": outp.einsums_outputs["LayerNorm"].energy,
                  "FFN.latency": outp.einsums_outputs["FFN"].latency, "FFN.energy": outp.einsums_outputs["FFN"].energy}

        return fields, values


    def write_to_file(self, inp: PregenerateInput, outp: TransFusionTransformerOutputs):
        file_path = Path(__file__).parent / "pregenerate"
        file_path.mkdir(parents=True, exist_ok=True)
        file_path /= f"{inp.arch_config.name}_{inp.model}_{inp.seq_len}.csv"
        with open(file_path, "a", newline="") as f:
            f1, v1 = self.factors_to_field_values(outp.factors)
            f2, v2 = self.result_to_field_values(outp)
            writer = csv.DictWriter(f, fieldnames=f1+f2)

            if os.stat(file_path).st_size == 0:
                writer.writeheader()

            row = {}
            row.update(v1)
            row.update(v2)
            writer.writerow(row)

    def read_from_file(
        self,
        arch_config: ArchConfig,
        model: str,
        seq_len: str,
        factors: Dict[str, Dict[str, int]]) -> Dict[str, float]:

        file_path = Path(__file__).parent / "pregenerate"
        file_path /= f"{arch_config.name}_{model}_{seq_len}.csv"

        if not file_path.exists():
            return None

        _, values = self.factors_to_field_values(factors)

        with open(file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if all(row.get(col) == str(val) for col, val in values.items()):
                    return {
                        "latency": float(row.get("latency")),
                        "energy": float(row.get("energy"))
                    }
        return None

    def pregenerate_factors(
        self,
        inp: PregenerateInput,
        factors: Dict[str, Dict[str, int]],
        write: bool
    ) -> Dict[str, float]:
        arch_config = inp.arch_config
        model = inp.model
        seq_len = inp.seq_len

        accel_core = TransFusion(TransFusionOptions(
            model=model,
            seq_len=seq_len,
            arch_config=arch_config
        ))

        outp: TransFusionTransformerOutputs = accel_core.eval_transformer(
            factors,
            inp.outdir / arch_config.name / model / seq_len
        )

        if write:
            self.write_to_file(inp, outp)

        return {
            "latency": outp.latency,
            "energy": outp.energy
        }

    def pregenerate_random(self, inp: PregenerateInput, write=True, iter_count = 1) -> TransFusionTransformerOutputs:
        arch_config = inp.arch_config
        model = inp.model
        seq_len = inp.seq_len

        accel_core = TransFusion(TransFusionOptions(
            model=model,
            seq_len=seq_len,
            arch_config=arch_config
        ))

        for i in range(iter_count):
            print(f"{arch_config.name}, {model}, {seq_len}, {i} ...")
            outp: TransFusionTransformerOutputs = accel_core.eval_transformer_with_random_factors(
                inp.outdir / arch_config.name / model / seq_len
            )

            if write:
                self.write_to_file(inp, outp)
