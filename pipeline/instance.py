from pathlib import Path
from dataclasses import dataclass
import pandas as pd

@dataclass
class Instance:
    B: int
    D: int
    E: int
    F: int
    H: int
    M: int
    N: int
    P: int
    S: int

    @classmethod
    def load_model_instance(cls, model: str, seq_len: str, mesh_2d: int = 256) -> "Instance":
        return get_instance(model, seq_len, mesh_2d)

def get_instance(model: str, seq_len: str, mesh_2d: int = 256) -> Instance:
    file = Path(__file__).parent / "instances" / "instances.csv"
    try:
        df = pd.read_csv(file)
        result = df[(df["models"] == model) & (df["seq_len"] == seq_len)]

        if not result.empty:
            row = result.iloc[0]
            return Instance(
                B = int(row["B"]),
                D = int(row["D"]),
                E = int(row["E"]),
                F = int(row["F"]),
                H = int(row["H"]),
                M = int(row["M"]) // mesh_2d,
                N = mesh_2d,
                P = int(row["P"]),
                S = int(row["S"])
            )
        else:
            raise Exception(f"Cannot find instance with model {model} and seq_len {seq_len}")
    except Exception as e:
        print(f"Read csv failed: {e}")
        raise
