from typing import Union, Tuple, List
from pathlib import Path
from config.arch import ArchConfig

class StatsOutput:

    def __init__(self, file: Union[str, Path]):
        self.file = file

    @staticmethod
    def load_from_outdir(outdir: Union[str, Path]) -> "StatsOutput":
        if not isinstance(outdir, Path):
            outdir = Path(outdir)
        return StatsOutput.load_from_file(outdir / "timeloop-model.stats.txt")

    @classmethod
    def load_from_file(cls, file: Union[str, Path]) -> "StatsOutput":
        return cls(file)

    def is_2d(self) -> bool:
        with open(self.file, "r") as f:
            for line in f:
                if "PE_col" in line:
                    return True
        return False

    def get_energy(self) -> float:
        energy = self.read_str(["=== mac ===", "Energy"])
        return float(energy.rstrip(" pJ"))

    def estimate_traffic_energy_eachlevel(self, arch: ArchConfig) -> float:
        data = {"DRAM": {"read": 0, "write": 0, "update": 0, "leak": 0},
            "L3": {"read": 0, "write": 0, "update": 0, "leak": 0},
            "reg_file_2d": {"read": 0, "write": 0, "update": 0, "leak": 0},
            "reg_file_1d": {"read": 0, "write": 0, "update": 0, "leak": 0}
        }
        def combine_stats(data_level, stats_level, width = 1):
            _read, _fills, _updates = self.read_scalar_rfu(stats_level)
            data[data_level]["read"] += _read / width
            data[data_level]["write"] += (_fills + _updates) / width

        ##  DRAM
        combine_stats("DRAM", "DRAM", 4)

        ##  L3
        combine_stats("L3", "L3", 256)

        ##  Reg
        if self.is_2d():
            combine_stats("reg_file_2d", "reg_file")
            combine_stats("reg_file_2d", "reg0")
            combine_stats("reg_file_2d", "reg1")
            combine_stats("reg_file_2d", "reg2")
        else:
            combine_stats("reg_file_1d", "reg_file_1d")

        dram_energy = data["DRAM"]["read"] * arch.dram_energy.read + data["DRAM"]["write"] * arch.dram_energy.write
        l3_energy = data["L3"]["read"] * arch.global_buffer_energy.read + data["L3"]["write"] * arch.global_buffer_energy.write
        regfile_energy = data["reg_file_1d"]["read"] * arch.reg_file_energy.read + data["reg_file_1d"]["write"] * arch.reg_file_energy.write
        regfile_energy += data["reg_file_2d"]["read"] * arch.reg_file_energy.read + data["reg_file_2d"]["write"] * arch.reg_file_energy.write
        return {"DRAM": dram_energy, "L3": l3_energy, "reg_file": regfile_energy, "traffic": data}

    def estimate_traffic_energy(self, arch: ArchConfig) -> float:
        rst = self.estimate_traffic_energy_eachlevel(arch)
        return sum([val for key, val in rst.items() if key != "traffic"])

    def get_mem_traffic(self, check_llb = True) -> float:
        buf_rd = 0
        buf_wr = 0
        if check_llb:
            buf_rd, _, buf_wr = self.read_scalar_rfu("L3")
        mem_rd, _, mem_wr = self.read_scalar_rfu("DRAM")

        #  2B (16 bites) values.
        return (buf_rd + mem_rd) * 2 + (buf_wr + mem_wr) * 2

    def get_mem_latency(self, mem_bandwidth: int = 30, check_llb = True) -> float:
        """_summary_

        Args:
            mem_bandwidth (int, optional): The memory bandwidth. Measured in GB/s.
                                           Defaults to 30 GB/s.
            check_llb (bool, optional): Whether to check the llb (last level
                                        buffer). In this context, the llb is L3
                                        Cache/Global Buffer. Defaults to True.

        Returns:
            float: traffic / bandwidth
        """
        mem_bandwidth *=  2**30
        return self.get_mem_traffic(check_llb) / mem_bandwidth

    def get_compute_latency(self, global_clock = 940e6) -> float:
        cycles = self.read_pint(["=== mac ===", "Cycles"])
        comp_latency = cycles / global_clock
        return comp_latency

    def read_pint(self, prefixes: List[str]) -> int:
        """Read a positive int.
        """
        rst = self.read_str(prefixes)
        if not rst.isdigit():
            raise Exception("Result is not a digit")
        return int(rst)

    def read_str(self, prefixes: List[str]) -> str:
        with open(self.file, "r") as f:
            line = f.readline().strip()
            for prefix in prefixes:
                while line[:len(prefix)] != prefix:
                    line = f.readline()
                    if line == "":
                        break
                    line = line.strip()
        i = line.index(":")
        return line[i+2:]

    def read_scalar_reads(self, name: str) -> int:
        return self.read_scalar(name, "reads")

    def read_scalar_updates(self, name: str) -> int:
        return self.read_scalar(name, "updates")

    def read_scalar_fills(self, name: str) -> int:
        return self.read_scalar(name, "fills")

    def read_scalar(self, name: str, scalar: str) -> int:
        STATE_WAIT = 0
        STATE_CATCH = 1

        state = STATE_WAIT
        name = f"=== {name} ==="
        scalar = f"Scalar {scalar}"
        rst = 0
        with open(self.file, "r") as f:
            for line in f:
                line = line.strip()
                if state == STATE_WAIT:
                    if line.startswith(name):
                        state = STATE_CATCH
                        continue
                if state == STATE_CATCH:
                    if line.startswith(scalar):
                        i = line.index(":")
                        rst += int(line[i+2:])
                    if line.startswith("==="):
                        break
        return rst

    def read_scalar_rfu (self, level: str) -> Tuple[int, int, int]:
        """Read scalar rfu (read, fills and updates) at the same time.

        Args:
            level (str): DRAM or L3.

        Returns:
            Tuple[int, int]: (scalar reads, scalar fills, scalar updates).
        """
        STATE_WAIT = 0
        STATE_CATCH = 1

        state = STATE_WAIT
        name = f"=== {level} ==="
        scalar_reads = 0
        scalar_fills = 0
        scalar_updates = 0
        with open(self.file, "r") as f:
            _reads = None
            _fills = None
            _updates = None
            for line in f:
                line = line.strip()
                if state == STATE_WAIT:
                    if line.startswith(name):
                        state = STATE_CATCH
                        continue
                if state == STATE_CATCH:
                    def _read_int():
                        i = line.index(":")
                        return int(line[i+2:])
                    if line.startswith("Scalar reads"):
                        _reads = _read_int()
                    if line.startswith("Scalar fills"):
                        _fills = _read_int()
                    if line.startswith("Scalar updates"):
                        _updates = _read_int()
                    if _reads is not None and _fills is not None and _updates is not None:
                        if _reads != _updates or level != "DRAM":
                            scalar_reads += _reads
                        scalar_fills += _fills
                        scalar_updates += _updates
                        _reads = None
                        _fills = None
                        _updates = None
                    if line.startswith("==="):
                        break
        return (scalar_reads, scalar_fills, scalar_updates)

    def read_mac_utilized_instances(self) -> int:
        return self.read_pint(["=== mac ===", "Utilized instances"])

    def read_mac_instances(self) -> int:
        instances = self.read_str(["=== mac ===", "Instances"])
        instances = instances.split("(")[0]
        return int(instances)
