from io_config.yaml_input import YamlInput
import math
from config.arch import ArchConfig, PE

class Architecture(YamlInput):
    def __init__(self):
        super().__init__()

    @classmethod
    def load_from_file(cls, file: str) -> "Architecture":
        arch = cls()
        arch.input = YamlInput.load_input_from_file(file)
        return arch

    def update_archconfig(self, arch_config: ArchConfig, pe: PE):
        self.update_bandwidth(arch_config.bandwidth)
        self.update_L3_size(arch_config.l3_size)
        if pe == PE.ONE_D:
            self.update_PE_mesh(arch_config.mesh_1d)
        elif pe == PE.TWO_D:
            self.update_PE_mesh(arch_config.mesh_2d)
        else:
            raise Exception(f"Invalid PE {pe}")
        self.update_rdwr_ports(arch_config.rdwr_ports)
        self.update_global_clock(arch_config.global_clock)

    def update_global_clock(self, clock: int):
        for node in self.input["architecture"]["nodes"]:
            if "name" in node and node["name"] == "system":
                node["attributes"]["global_cycle_seconds"] = 1.0/clock
                break

    def update_bandwidth(self, bandwidth: int):
        """ Update bandwidth.

        Args:
            bandwidth (int): The bandwidth (GB/s).
                             The equation of the bandwidth is
                             "bandwidth * 2^30 B / s * word / 2B * s / 10^9"
        """

        b_value = math.floor(bandwidth * (2**30) / 2 / 1e9)

        for node in self.input["architecture"]["nodes"]:
            if "name" in node and node["name"] == "DRAM":
                node["attributes"]["shared_bandwidth"] = b_value
                break

    def update_L3_size(self, size: int, width: int = 4096):
        """Update L3 size.

        Args:
            size (int): The size of L3/Global Buffer.
                        size = width * depth. Measured in MB (megabytes).
            width (int, optional): The data buse width of L3/Global Buffer.
                                   Mearured in bites.
                                   Defaults to 4096 (256 values * 16 bits/value).
        """
        B_width = width // 8  #  B/rows
        depth = size * (2**20) // B_width

        for node in self.input["architecture"]["nodes"]:
            if "name" in node and (node["name"] == "L3" or node["name"] == "global_buffer"):
                node["attributes"]["depth"] = depth
                node["attributes"]["width"] = width
                break

    def update_PE_mesh(self, mesh: int):
        for node in self.input["architecture"]["nodes"]:
            if "name" in node and (node["name"] == "PE" or node["name"] == "PE_col"):
                if "meshX" in node["spatial"]:
                    node["spatial"]["meshX"] = mesh
                if "meshY" in node["spatial"]:
                    node["spatial"]["meshY"] = mesh

    def update_rdwr_ports(self, ports: int):
        """Update read and write ports of global buffer

        Args:
            ports (int): The sum of RD and WR ports.
        """
        for node in self.input["architecture"]["nodes"]:
            if "name" in node and (node["name"] == "L3" or node["name"] == "global_buffer"):
                if "n_rdwr_ports" in node["attributes"]:
                    node["attributes"]["n_rdwr_ports"] = ports
                break