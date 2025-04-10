import timeloopfe.v4 as tl
from pathlib import Path
from typing import Union, List, Dict
import subprocess
from io_config.energy import read_energy_from_files, read_energy_from_files_foreach

def eval_energy2(arch_files: List[Union[str, Path]], action_counts_files: List[Union[str, Path]], outdir: Union[str, Path]) -> Dict[str, float]:
    run_model(arch_files, action_counts_files, outdir)

    if not isinstance(outdir, Path):
        outdir = Path(outdir)

    outfiles = []
    for i in range(len(arch_files)):
        outfiles.append(outdir / f"{i}" / "energy_estimation.yaml")
    return read_energy_from_files_foreach(outfiles)

def eval_energy(arch_files: List[Union[str, Path]], action_counts_files: List[Union[str, Path]], outdir: Union[str, Path]) -> float:
    return eval_energy2(arch_files, action_counts_files, outdir)["total"]

def run_model(arch_files: List[Union[str, Path]], action_counts_files: List[Union[str, Path]], outdir: Union[str, Path]) -> float:
    if len(arch_files) != len(action_counts_files):
        raise Exception("len(arch_files) != len(action_counts_files)")

    if not isinstance(outdir, Path):
        outdir = Path(outdir)

    for i, (arch_file, action_counts_file) in enumerate(zip(arch_files, action_counts_files)):
        _outdir = outdir / f"{i}"
        run_accelergy_area(arch_file, _outdir)

        with open(_outdir / "accelergy_verbose.yaml", "w") as f:
            subprocess.run([
                "accelergy",
                action_counts_file,
                _outdir / "parsed-processed-input.yaml",
                "-o",
                _outdir
            ], stderr=f)

def run_accelergy_area(arch_file: Union[str, Path], outdir: Union[str, Path]):
    common = [
        str(Path(__file__).parent / "area_energy" / "common" / "*"),
        str(arch_file)
    ]

    spec = tl.Specification.from_yaml_files(*common)

    tl.call_accelergy_verbose(
        spec,
        output_dir=str(outdir),
        log_to=str(outdir / "accelergy_verbose.log")
    )
