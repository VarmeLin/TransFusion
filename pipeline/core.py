from config.arch import PE as PE
from dataclasses import dataclass, asdict
from config.arch import ArchConfig
from typing import Dict, List
from pipeline.session import SessionOutput, Session, SessionInput, search_scheduler, TimeloopResult
from pipeline.instance import Instance
from einsum.core import Einsums, QKV_EINSUMS, MHA_EINSUMS, ANORM_EINSUMS, FFN_EINSUMS, load_ffn_einsums_for_model, Einsum
from pipeline.scheduler import ScheduleResult, run_scheduler, run_scheduler2, Schedule
from pathlib import Path
from functools import reduce
from itertools import product
from io_config.accelergy_input import AccelergyInput
from pipeline.search_factor2 import random_factors
from results.model import MODEL_FUNC

@dataclass(frozen=True)
class TransFusionOptions:
    model: str
    seq_len: str
    arch_config: ArchConfig

@dataclass(frozen=True)
class TransFusionOutputs:
    latency: float
    energy: float
    energy_rst: Dict[str, float]
    tl_rsts: Dict[str, Dict[str, SessionOutput]]
    sch_rst: ScheduleResult

@dataclass(frozen=True)
class TransFusionTransformerOutputs:
    latency: float
    energy: float
    factors: Dict[str, Dict[str, int]]
    einsums_outputs: Dict[str, TransFusionOutputs]

class TransFusion:
    def __init__(self, inps: TransFusionOptions):
        self.arch_config = inps.arch_config
        self.instance = Instance.load_model_instance(
            inps.model,
            inps.seq_len,
            inps.arch_config.mesh_2d
        )
        self.model = inps.model
        self.seq_len = inps.seq_len

        self.name = "TransFusion"
        self.session = Session()

        self.schedules = {}

    def factors(self, n):
        # Source:
        # https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
        factors = list(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
        factors.sort(reverse=True)
        return factors

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

    def build_QKV_factors(
        self,
        einsum: Einsum,
        pe: PE,
        factors: Dict[str, Dict[str, int]]
    ):
        dimensions = einsum.get_dimensions()
        dram_factors = {d: v for d, v in factors["DRAM"].items() if d in dimensions}
        l3_factors = {d: v for d, v in factors["L3"].items() if d in dimensions}
        pe_factors = {d: 1 for d in dimensions}
        pe_col_factors = {d: 1 for d in dimensions}

        if pe == PE.ONE_D:
            _pe_factor = 1
            if "P" in dimensions:
                pe_P = factors["PE"]["P"]
                pe_factors["P"] = pe_P
                _pe_factor *= pe_P
            elif "N" in dimensions:
                pe_factors["N"] = self.instance.N
                _pe_factor *= self.instance.N
            else:
                raise Exception("Neither 'P' or 'N' are in the QKV's dimensions.")

            if "E" in dimensions:
                pe_H, pe_E = self.assign_HEF(self.instance.H, self.instance.E, self.arch_config.mesh_1d // _pe_factor)
                pe_factors["H"] = pe_H
                pe_factors["E"] = pe_E
                l3_factors["H"] = self.instance.H // pe_H
                l3_factors["E"] = self.instance.E // pe_E
            elif "F" in dimensions:
                pe_H, pe_F = self.assign_HEF(self.instance.H, self.instance.F, self.arch_config.mesh_1d // _pe_factor)
                pe_factors["H"] = pe_H
                pe_factors["F"] = pe_F
                l3_factors["H"] = self.instance.H // pe_H
                l3_factors["F"] = self.instance.F // pe_F
            else:
                raise Exception("Neigher E nor F are in the QKV's dimensions")

            return {
                "DRAM": dram_factors,
                "L3": l3_factors,
                "PE": pe_factors
            }
        elif pe == PE.TWO_D:
            if "P" in dimensions:
                pe_factors["P"] = factors["PE"]["P"]
            elif "N" in dimensions:
                pe_factors["N"] = self.instance.N
            else:
                raise Exception("Neither 'P' or 'N' are in the QKV's dimensions.")

            if "E" in dimensions:
                pe_H, pe_E = self.assign_HEF(self.instance.H, self.instance.E, self.arch_config.mesh_2d)
                pe_col_factors["H"] = pe_H
                pe_col_factors["E"] = pe_E
                l3_factors["H"] = self.instance.H // pe_H
                l3_factors["E"] = self.instance.E // pe_E
            elif "F" in dimensions:
                pe_H, pe_F = self.assign_HEF(self.instance.H, self.instance.F, self.arch_config.mesh_2d)
                pe_col_factors["H"] = pe_H
                pe_col_factors["F"] = pe_F
                l3_factors["H"] = self.instance.H // pe_H
                l3_factors["F"] = self.instance.F // pe_F
            else:
                raise Exception("Neigher E nor F are in the QKV's dimensions")

            return {
                "DRAM": dram_factors,
                "L3": l3_factors,
                "PE": pe_factors,
                "PE_col": pe_col_factors
            }
        else:
            raise Exception(f"Invalid PE {pe}.")

    def build_MHA_factors(
        self,
        einsum: Einsum,
        pe: PE,
        factors: Dict[str, Dict[str, int]]
    ):
        dimensions = einsum.get_dimensions()
        dram_factors = {d: v for d, v in factors["DRAM"].items() if d in dimensions}
        l3_factors = {d: v for d, v in factors["L3"].items() if d in dimensions}
        pe_factors = {d: 1 for d in dimensions}
        pe_col_factors = {d: 1 for d in dimensions}

        if pe == PE.ONE_D:
            _remaining_pe = self.arch_config.mesh_1d
            pe_P = factors["PE"]["P"]
            pe_factors["P"] = pe_P
            _remaining_pe //= pe_P
            if "N" in dimensions:
                pe_factors["N"] = _remaining_pe
                l3_factors["N"] = self.instance.N // pe_factors["N"]
            else:
                #  Improve H
                pe_factors["H"] = min(_remaining_pe, l3_factors["H"], 4)
                l3_factors["H"] //= pe_factors["H"]
            return {
                "DRAM": dram_factors,
                "L3": l3_factors,
                "PE": pe_factors
            }
        elif pe == PE.TWO_D:
            pe_P = factors["PE"]["P"]
            pe_factors["P"] = pe_P

            if "N" in dimensions:
                pe_col_factors["N"] = self.arch_config.mesh_2d
            else:
                #  Improve H
                pe_col_factors["H"] = min(l3_factors["H"], self.arch_config.mesh_2d, 4)
                l3_factors["H"] //= pe_col_factors["H"]

            return {
                "DRAM": dram_factors,
                "L3": l3_factors,
                "PE": pe_factors,
                "PE_col": pe_col_factors
            }

        else:
            raise Exception(f"Invalid PE {pe}.")

    def build_ANORM_factors(
        self,
        einsum: Einsum,
        pe: PE,
        factors: Dict[str, Dict[str, int]]
    ):
        dimensions = einsum.get_dimensions()
        dram_factors = {d: v for d, v in factors["DRAM"].items() if d in dimensions}
        l3_factors = {d: v for d, v in factors["L3"].items() if d in dimensions}
        pe_factors = {d: 1 for d in dimensions}
        pe_col_factors = {d: 1 for d in dimensions}

        if pe == PE.ONE_D:
            pe_P = factors["PE"]["P"]
            pe_factors["P"] = pe_P

            if "H" in dimensions and "F" in dimensions:
                pe_H, pe_F = self.assign_HEF(self.instance.H, self.instance.F, self.arch_config.mesh_1d // pe_P)
                pe_factors["H"] = pe_H
                pe_factors["F"] = pe_F
                l3_factors["H"] = self.instance.H // pe_H
                l3_factors["F"] = self.instance.F // pe_F

            return {
                "DRAM": dram_factors,
                "L3": l3_factors,
                "PE": pe_factors
            }
        elif pe == PE.TWO_D:
            pe_P = factors["PE"]["P"]
            pe_factors["P"] = pe_P

            if "H" in dimensions and "F" in dimensions:
                pe_H, pe_F = self.assign_HEF(self.instance.H, self.instance.F, self.arch_config.mesh_2d)
                pe_col_factors["H"] = pe_H
                pe_col_factors["F"] = pe_F
                l3_factors["H"] = self.instance.H // pe_H
                l3_factors["F"] = self.instance.F // pe_F

            return {
                "DRAM": dram_factors,
                "L3": l3_factors,
                "PE": pe_factors,
                "PE_col": pe_col_factors
            }
        else:
            raise Exception(f"Invalid PE {pe}.")

    def build_FFN_factors(
        self,
        einsum: Einsum,
        pe: PE,
        factors: Dict[str, Dict[str, int]]
    ):
        dimensions = einsum.get_dimensions()
        dram_factors = {d: v for d, v in factors["DRAM"].items() if d in dimensions}
        l3_factors = {d: v for d, v in factors["L3"].items() if d in dimensions}
        pe_factors = {d: 1 for d in dimensions}
        pe_col_factors = {d: 1 for d in dimensions}

        if pe == PE.ONE_D:
            pe_factors["P"] = factors["PE"]["P"]
            remaining_pe = self.arch_config.mesh_1d // pe_factors["P"]
            pe_factors["S"] = remaining_pe
            l3_factors["S"] *= factors["PE"]["S"] // remaining_pe
            return {
                "DRAM": dram_factors,
                "L3": l3_factors,
                "PE": pe_factors
            }
        elif pe == PE.TWO_D:
            pe_factors["P"] = factors["PE"]["P"]
            pe_col_factors["S"] = factors["PE"]["S"]
            return {
                "DRAM": dram_factors,
                "L3": l3_factors,
                "PE": pe_factors,
                "PE_col": pe_col_factors
            }
        else:
            raise Exception(f"Invalid PE {pe}.")

    def build_factors(
        self,
        einsum: Einsum,
        pe: PE,
        factors: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[str, int]]:
        rst: Dict[str, Dict[str, int]]
        if einsum.einsums.name == "QKV":
            rst = self.build_QKV_factors(einsum, pe, factors)
        elif einsum.einsums.name == "MHA":
            rst = self.build_MHA_factors(einsum, pe, factors)
        elif einsum.einsums.name == "LayerNorm":
            rst = self.build_ANORM_factors(einsum, pe, factors)
        elif einsum.einsums.name == "FFN":
            rst = self.build_FFN_factors(einsum, pe, factors)
        else:
            raise Exception(f"Invalid einsums {einsum.einsums.name}")

        rst["reg_file"] = {d: 1 for d in einsum.get_dimensions()}
        return rst

    def eval_einsum(
        self,
        einsum: Einsum,
        pe: PE,
        factors: Dict[str, Dict[str, int]],
        outdir: Path,
        schedules: Dict[str, Schedule]
    ) -> TimeloopResult:
        outp: SessionOutput = self.session.eval_einsum(SessionInput(
            instance=self.instance,
            einsum=einsum,
            arch_config=self.arch_config,
            pe=pe,
            factors=self.build_factors(einsum, pe, factors),
            outdir=outdir,
            schedules=schedules
        ))
        return outp.tl_rst

    def eval_einsums(
        self,
        einsums: Einsums,
        factors: Dict[str, Dict[str, int]],
        outdir: Path
    ) -> TransFusionOutputs:
        sch_rst: ScheduleResult
        if einsums.name in self.schedules:
            sch_rst = self.schedules[einsums.name]
        else:
            print(f"Searching schedule for {einsums.name} ...")
            #  Improve H
            sch_rst = search_scheduler(self.model, self.seq_len, einsums, self.arch_config, MHA_Improve_H=min(self.arch_config.mesh_2d, self.instance.H, 4))
            self.schedules[einsums.name] = sch_rst
            print("Finished searching.")

        latency = {}
        tl_rsts = {}
        acc_inps = {}
        for einsum in einsums.names:
            print(f"Running {self.name} einsum {einsum} ...")
            einsum = Einsum.load_from_einsums(einsum, einsums)

            einsum_outdir = outdir / einsum.name
            pe = sch_rst.schedules[einsum.name].pe
            tl_rst = self.eval_einsum(einsum, pe, factors, einsum_outdir, sch_rst.schedules)
            print(f"Finished {self.name} einsum {einsum.name}.")

            tl_rsts[einsum.name] = tl_rst
            if pe == PE.ONE_D:
                latency[einsum.name] = [max(tl_rst.comp_latency, tl_rst.mem_latency), float("inf")]
            elif pe == PE.TWO_D:
                latency[einsum.name] = [float("inf"), max(tl_rst.comp_latency, tl_rst.mem_latency)]
            else:
                raise Exception(f"Invalid PE {pe}.")

            acc_inps[einsum.name] = AccelergyInput.load_from_tl_outdir(einsum_outdir, einsum.einsums.compute_cost[einsum.name])

        sch_rst = run_scheduler(latency, sch_rst.dag)
        total_latency = sch_rst.latency

        acc_inp = AccelergyInput.combine(acc_inps.values())

        print("Evaluating energy...")
        energy = self.session.eval_energy2(self.arch_config, acc_inp, outdir/"energy")
        print("Done!")

        return TransFusionOutputs(
            latency=total_latency,
            energy=energy["total"],
            energy_rst=energy,
            tl_rsts=tl_rsts,
            sch_rst=sch_rst
        )

    def eval_transformer(
        self,
        factors: Dict[str, Dict[str, int]],
        outdir: Path
    ) -> TransFusionTransformerOutputs:
        rst: Dict[str, TransFusionOutputs] = {}
        ffn_einsums = load_ffn_einsums_for_model(self.model)
        for einsums in [QKV_EINSUMS, MHA_EINSUMS, ANORM_EINSUMS, ffn_einsums]:
            acc_out = self.eval_einsums(einsums, factors, outdir/einsums.name)
            rst[einsums.name] = acc_out

        mod_func = MODEL_FUNC[self.model]

        qkv = (rst["QKV"].latency, rst["QKV"].energy)
        mha = (rst["MHA"].latency, rst["MHA"].energy)
        add_norm = (rst["LayerNorm"].latency, rst["LayerNorm"].energy)
        ffn = (rst["FFN"].latency, rst["FFN"].energy)

        return TransFusionTransformerOutputs(
            latency=mod_func(rst["QKV"].latency, rst["MHA"].latency, rst["LayerNorm"].latency, rst["FFN"].latency),
            energy=mod_func(rst["QKV"].energy, rst["MHA"].energy, rst["LayerNorm"].energy, rst["FFN"].energy),
            factors=factors,
            einsums_outputs=rst
        )

    def eval_transformer_with_random_factors(
        self,
        outdir: Path
    ) -> TransFusionTransformerOutputs:
        factors = random_factors(self.arch_config, asdict(self.instance), False)
        return self.eval_transformer(factors, outdir)

