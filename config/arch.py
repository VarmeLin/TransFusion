from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class ArchEnergyFactor:
    read: float
    write: float

@dataclass(frozen=True)
class ArchConfig:
    name: str
    bandwidth: int
    l3_size: int
    mesh_2d: int
    mesh_1d: int
    rdwr_ports: int
    global_clock: int

    #  Energy
    dram_energy: ArchEnergyFactor
    global_buffer_energy: ArchEnergyFactor
    reg_file_energy: ArchEnergyFactor

ARCH_CLOUD = ArchConfig(
    name="cloud",
    bandwidth=400,       #  GB/s
    l3_size=16,          #  MB
    mesh_1d=256,
    mesh_2d=256,         #  256*256
    rdwr_ports=4,        #  for global buffer
    global_clock=940e6,  #  Fixed value

    dram_energy=ArchEnergyFactor(read=249.6, write=249.6),
    global_buffer_energy=ArchEnergyFactor(read=5859.926552, write=7881.676552),
    reg_file_energy=ArchEnergyFactor(read=2.190435, write=1.902435)
)

ARCH_EDGE = ArchConfig(
    name="edge",
    bandwidth=30,   #  GB/s
    l3_size=5,      #  MB
    mesh_1d=256,
    mesh_2d=16,     #  16*16
    rdwr_ports=4,   #  for global buffer
    global_clock=940e6, #  Fixed value

    dram_energy=ArchEnergyFactor(read=249.6, write=249.6),
    global_buffer_energy=ArchEnergyFactor(read=12512.760572, write=15751.460572),
    reg_file_energy=ArchEnergyFactor(read=2.190435, write=1.902435)
)

class PE(Enum):
    ONE_D = "1d"
    TWO_D = "2d"