import timeloopfe.v4 as tl
from typing import Union, List, Dict
from pathlib import Path
from dataclasses import dataclass
from io_config.stats import StatsOutput
from config.arch import ArchConfig

@dataclass
class TimeloopResult:
    comp_latency: float
    mem_latency: float
    traffic: float
    energy: float
    traffic_energy: float
    traffic_energy_foreach: Dict
    mac_instances: int
    mac_utilized_instances: int

def eval_model(
    inputs: List[Union[str, Path]],
    outdir: Union[str, Path],
    arch_config: ArchConfig,
    compute_cost: Dict[str, int],
    check_llb: bool=False
) -> TimeloopResult:
    run_model(inputs, outdir)
    stats = StatsOutput.load_from_outdir(outdir=outdir)
    return TimeloopResult(
        comp_latency=stats.get_compute_latency(arch_config.global_clock) * sum(compute_cost.values()),
        mem_latency=stats.get_mem_latency(mem_bandwidth=arch_config.bandwidth, check_llb=check_llb),
        traffic=stats.get_mem_traffic(check_llb=check_llb),
        energy=stats.get_energy(),
        traffic_energy=stats.estimate_traffic_energy(arch_config),
        traffic_energy_foreach=stats.estimate_traffic_energy_eachlevel(arch_config),
        mac_instances=stats.read_mac_instances(),
        mac_utilized_instances=stats.read_mac_utilized_instances()
    )

def run_model(inputs: List[Union[str, Path]], outdir: Union[str, Path]):
    """Run timeloop.

    Args:
        inputs (List[Union[str, Path]]): The input files of model, mapping, mapper, architecture
        outdir (Union[str, Path]): The output folder
    """
    _inputs = []
    for i in inputs:
        _inputs.append(str(i))
    _inputs.append(str(Path(__file__).parent / "_components" / "*"))

    spec = tl.Specification.from_yaml_files(*_inputs)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tl.call_model(spec, str(outdir), log_to=str(outdir/"pipeline-run-model.log"))
    #tl.call_model(spec, str(outdir))
