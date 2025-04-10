from config.arch import PE
from io_config import Architecture, Mapping, Problem, AccelergyInput
from dataclasses import dataclass, asdict, field
from pipeline.instance import Instance
from config.arch import ArchConfig, ARCH_CLOUD, ARCH_EDGE
from pipeline.search_factor import random_factors, l3_cache_storage
from pathlib import Path
from engine.timeloop import TimeloopResult, eval_model
from engine.accelergy_model import eval_energy, eval_energy2
from typing import Dict, Tuple, List
from einsum.core import Einsum, Einsums
from pipeline.scheduler import Schedule, run_scheduler2, ScheduleResult

@dataclass
class SessionInput:
    instance: Instance
    einsum: Einsum
    arch_config: ArchConfig
    outdir: Path
    dram_keepin: List[str] = field(default_factory=list)
    l3_bypass: List[str] = field(default_factory=list)
    factors: Dict[str, Dict[str, int]] = None
    pe: PE = None
    schedules: Dict[str, Schedule] = None

@dataclass
class SessionOutput:
    tl_rst: TimeloopResult
    factors: Dict[str, Dict[str, int]]

class Session:
    def __init__(self):
        pass

    def call_timeloop(
        self,
        einsum: Einsum,
        arch: Architecture,
        mapping: Mapping,
        problem: Problem,
        arch_config: ArchConfig,
        outdir: Path
    ) -> TimeloopResult:
        arch_mapping_problem_path = outdir / "arch+mapping+problem.yaml"
        Mapping.write_all_to_file(
            arch_mapping_problem_path,
            [arch, mapping, problem])

        return eval_model(
            [arch_mapping_problem_path],
            outdir=outdir,
            arch_config=arch_config,
            compute_cost=einsum.einsums.compute_cost[einsum.name]
        )

    def load_uninit_arch(self, pe: PE) -> Architecture:
        base_dir = Path(__file__).parent
        arch = None
        if pe is PE.ONE_D:
            arch = Architecture.load_from_file(base_dir / "arch" / "arch-1d.yaml")
        elif pe is PE.TWO_D:
            arch = Architecture.load_from_file(base_dir / "arch" / "arch-2d.yaml")
        else:
            raise Exception(f"Invalid PE {pe}")
        return arch

    def load_init_arch(self, pe: PE, arch_config: ArchConfig) -> Architecture:
        arch = self.load_uninit_arch(pe)
        arch.update_archconfig(arch_config, pe)
        return arch

    def load_init_accelergy_arch(self, arch_config: ArchConfig) -> Tuple[Architecture, Architecture]:
        base_dir = Path(__file__).parent
        arch_1d = Architecture.load_from_file(base_dir / "accelergy" / "arch_1d.yaml")
        arch_2d = Architecture.load_from_file(base_dir / "accelergy" / "arch_2d.yaml")

        arch_1d.update_archconfig(arch_config, PE.ONE_D)
        arch_2d.update_archconfig(arch_config, PE.TWO_D)

        return (arch_1d, arch_2d)

    def load_init_problem(self, einsum: Einsum, instance: Instance) -> Problem:
        problem = Problem.load_default()
        name = einsum.name
        inputs = einsum.get_inputs()
        problem.update_dataspaces(
            {n: einsum.einsums.dimensions[n] for n in inputs},
            {name: einsum.einsums.dimensions[name]}
        )
        problem.update_instance(asdict(instance))
        return problem

    def load_init_mapping(
        self,
        einsum: Einsum,
        pe: PE,
        factors: Dict[str, Dict[str, int]],
        dram_keepin: List[str] = [],
        l3_bypass: List[str] = [],
        schedules: Dict[str, Schedule] = None
    ) -> Mapping:
        mapping = Mapping.load_default(include_pe_col=(pe == PE.TWO_D))
        mapping.update_factors(factors)

        keep_bypass: Dict[str, Dict[str, str]]
        if schedules == None:
            keep_bypass = einsum.get_keep_bypass(l3_bypass=l3_bypass, dram_keepin=dram_keepin)
        else:
            _schedules = {ein: asdict(sch) for ein, sch in schedules.items()}
            keep_bypass = einsum.get_keep_bypass_with_schedule(l3_bypass, dram_keepin, _schedules)

        mapping.update_keep_bypass(keep_bypass)
        mapping.update_permutation(einsum.get_permutation())
        return mapping

    def eval_einsum(self, session_input: SessionInput) -> SessionOutput:
        arch_config = session_input.arch_config
        instance = session_input.instance
        einsum = session_input.einsum
        #  Problem
        problem = self.load_init_problem(einsum, instance)

        #  Mapping
        pe = session_input.pe
        factors = session_input.factors
        mapping = self.load_init_mapping(
            einsum,
            pe,
            factors,
            dram_keepin=session_input.dram_keepin,
            l3_bypass=session_input.l3_bypass,
            schedules=session_input.schedules
        )


        # Arch
        arch: Architecture
        if pe == PE.ONE_D:
            arch = self.load_init_arch(PE.ONE_D, arch_config)
        elif pe == PE.TWO_D:
            arch = self.load_init_arch(PE.TWO_D, arch_config)
        else:
            raise Exception(f"Invalid PE {pe}")

        rst = self.call_timeloop(
            einsum=einsum,
            mapping=mapping,
            arch=arch,
            problem=problem,
            arch_config=arch_config,
            outdir=session_input.outdir
        )

        return SessionOutput(
            tl_rst=rst,
            factors=factors
        )

    def eval_energy2(self, arch_config: ArchConfig, acc_inp: AccelergyInput, outdir: Path) -> Dict[str, float]:
        arch_1d, arch_2d = self.load_init_accelergy_arch(arch_config)

        action_counts_files = [
            outdir / "acc_action_counts_1d.yaml", outdir / "acc_action_counts_2d.yaml"
        ]
        arch_files = [
            outdir / "p_acc_arch_1d.yaml",
            outdir / "p_acc_arch_2d.yaml"
        ]

        acc_inp.write_to_file(*action_counts_files)
        arch_1d.write_to_file(arch_files[0])
        arch_2d.write_to_file(arch_files[1])

        return eval_energy2(
            arch_files=arch_files,
            action_counts_files=action_counts_files,
            outdir=outdir
        )

    def eval_energy(self, arch_config: ArchConfig, acc_inp: AccelergyInput, outdir: Path) -> float:
        return self.eval_energy2(arch_config, acc_inp, outdir)["total"]

def search_scheduler(
    model: str,
    seq_len: str,
    einsums: Einsums,
    arch_config: ArchConfig,
    MHA_Improve_H: int = 1
) -> ScheduleResult:
    mesh_1d = arch_config.mesh_1d
    mesh_2d = arch_config.mesh_2d
    clock_feq = arch_config.global_clock
    instance = Instance.load_model_instance(model, seq_len, min(mesh_1d, mesh_2d))
    latency = {}
    for einsum in einsums.names:
        _einsum: Einsum = Einsum.load_from_name(einsum, model)
        compute_count = _einsum.get_compute_count(asdict(instance))

        if einsums.name == "MHA":
            if "N" in _einsum.get_dimensions():
                latency[einsum] = []
                latency[einsum].append(compute_count / mesh_1d / clock_feq)  #  1D
                latency[einsum].append(compute_count / (mesh_2d ** 2) / clock_feq)  #  2D
            else:
                latency[einsum] = []
                pe_P = min(mesh_1d, mesh_2d)
                _remaining_P_1d = mesh_1d // pe_P
                pe_H = min(_remaining_P_1d, MHA_Improve_H)
                latency[einsum].append(compute_count / (pe_P * pe_H) / clock_feq)  #  1D
                latency[einsum].append(compute_count / (mesh_2d * MHA_Improve_H) / clock_feq)  #  2D
        elif einsums.name == "LayerNorm":
            if "H" in _einsum.get_dimensions() and "F" in _einsum.get_dimensions():
                latency[einsum] = []
                latency[einsum].append(compute_count / mesh_1d / clock_feq)  #  1D
                latency[einsum].append(compute_count / (mesh_2d ** 2) / clock_feq)  #  2D
            else:
                latency[einsum] = []
                latency[einsum].append(compute_count / min(mesh_2d, mesh_1d) / clock_feq)  #  1D
                latency[einsum].append(compute_count / mesh_2d / clock_feq)  #  2D
        else:
            latency[einsum] = []
            latency[einsum].append(compute_count / mesh_1d / clock_feq)  #  1D
            latency[einsum].append(compute_count / (mesh_2d ** 2) / clock_feq)  #  2D

        if "divide" in einsums.compute_cost[einsum] or "max" in einsums.compute_cost[einsum]:
            latency[einsum][1] = float("inf")
    sch_rst = run_scheduler2(latency, einsums.dependency)
    return sch_rst