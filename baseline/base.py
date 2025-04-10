from pipeline.instance import Instance
from pipeline.session import Session, SessionInput, SessionOutput
from einsum.core import *
from functools import reduce
from dataclasses import dataclass, asdict
from config.arch import ArchConfig, PE
from typing import List, Dict
from pathlib import Path
from io_config.accelergy_input import AccelergyInput
from abc import ABC, abstractmethod
from itertools import product

@dataclass(frozen=True)
class BaselineOptions:
    model: str
    seq_len: str
    arch_config: ArchConfig

@dataclass(frozen=True)
class FusedBaselineOptions(BaselineOptions):
    fused: bool

@dataclass(frozen=True)
class BaselineOutputs:
    latency: float
    energy: float
    energy_rsts: Dict[str, float]
    tl_rsts: Dict[str, SessionOutput]

class Baseline(ABC):
    def __init__(self, base_input: BaselineOptions):
        self.arch_config = base_input.arch_config
        self.instance = Instance.load_model_instance(
            base_input.model,
            base_input.seq_len,
            self.arch_config.mesh_2d
        )

        self.name = "Baseline"
        self.dram_keepin = []
        self.l3_bypass = []
        self.session = Session()

    @abstractmethod
    def get_target_einsums(self) -> Einsums:
        pass

    @abstractmethod
    def get_einsum_pe(self, einsum: Einsum) -> PE:
        pass

    @abstractmethod
    def build_factors(self, dimensions: List[str], pe: PE) -> Dict[str, Dict[str, int]]:
        pass

    def eval_einsum(self, einsum: Einsum, outdir: Path) -> SessionOutput:
        pe = self.get_einsum_pe(einsum)

        outp: SessionOutput = self.session.eval_einsum(SessionInput(
            instance=self.instance,
            einsum=einsum,
            arch_config=self.arch_config,
            pe=pe,
            factors=self.build_factors(einsum.get_dimensions(), pe),
            outdir=outdir,
            dram_keepin=self.dram_keepin,
            l3_bypass=self.l3_bypass
        ))

        return outp

    @abstractmethod
    def get_total_latency(self, latency: Dict[str, float]) -> float:
        pass

    def eval_einsums(self, einsums: Einsums, outdir: Path) -> BaselineOutputs:
        acc_inps = {}
        tl_rsts: Dict[str, SessionOutput] = {}
        latency = {}
        for einsum in einsums.names:
            print(f"Running {self.name} einsum {einsum} ...")
            einsum = Einsum.load_from_einsums(name=einsum, einsums=einsums)

            einsum_outdir = outdir / einsum.name
            tl_rst = self.eval_einsum(einsum, einsum_outdir)

            print(f"Finished {self.name} einsum {einsum.name}.")

            tl_rsts[einsum.name] = tl_rst
            latency[einsum.name] = max(tl_rst.tl_rst.comp_latency, tl_rst.tl_rst.mem_latency)

            acc_inp = AccelergyInput.load_from_tl_outdir(einsum_outdir, einsum.einsums.compute_cost[einsum.name])
            acc_inps[einsum.name] = acc_inp

        total_latency = self.get_total_latency(latency)

        acc_inp = AccelergyInput.combine(acc_inps.values())
        energy = self.session.eval_energy2(self.arch_config, acc_inp, outdir / "energy")

        return BaselineOutputs(
            tl_rsts=tl_rsts,
            latency=total_latency,
            energy=energy["total"],
            energy_rsts=energy
        )

    def assign_HEF(self, H, E, M) -> Dict[str, int]:
        factors_H = self.factors(H)
        factors_E = self.factors(E)
        factors_M = self.factors(M)

        rst_h = 1
        rst_e = 1
        rst_m = float("-inf")
        for fe, fh in product(factors_E, factors_H):
            he = fh*fe
            if he in factors_M:
                if he > rst_m:
                    rst_m = he
                    rst_h = fh
                    rst_e = fe
        return rst_h, rst_e