import random
from typing import Dict, List
from config.arch import ArchConfig
from itertools import product
from einsum.core import *

def generate_arrays(instance: Dict[str, int], level: int) -> List[Dict[str, int]]:
    rst = [{} for _ in range(level)]
    for dimension, value in instance.items():
        factors = []
        remaining_product = value

        # Get all factors for one number.
        def get_factors(n):
            factors = []
            for i in range(1, n + 1):
                if n % i == 0:
                    factors.append(i)
            return factors
        for j in range(level-1):
            factor = random.choice(get_factors(remaining_product))
            factors.append(factor)
            remaining_product //= factor
        factors.append(remaining_product)
        random.shuffle(factors)

        for j in range(level):
            rst[j][dimension] = factors[j]

    return rst

def random_factors(
    arch_config: ArchConfig,
    instance: Dict[str,int],
    min_limit: bool = False
):
    pe_P = arch_config.mesh_2d
    pe_S = arch_config.mesh_2d
    pe_N = instance["N"]

    dram_l3_factors = instance.copy()
    assert dram_l3_factors["P"] % pe_P == 0
    dram_l3_factors["P"] = dram_l3_factors["P"] // pe_P

    assert dram_l3_factors["S"] % pe_S == 0
    dram_l3_factors["S"] = dram_l3_factors["S"] // pe_S

    assert dram_l3_factors["N"] % pe_N == 0
    dram_l3_factors["N"] = dram_l3_factors["N"] // pe_N

    l3_H = dram_l3_factors["H"]
    dram_l3_factors["H"] = 1

    l3_E = dram_l3_factors["E"]
    dram_l3_factors["E"] = 1

    l3_F = dram_l3_factors["F"]
    dram_l3_factors["F"] = 1

    while True:
        _factors = generate_arrays(dram_l3_factors, level=2)
        dram_factors = _factors[0]
        l3_factors = _factors[1]
        l3_factors["H"] = l3_H
        l3_factors["E"] = l3_E
        l3_factors["F"] = l3_F

        avail_l3 = arch_config.l3_size * (2**20)
        l3_occupy = get_l3_cache(l3_factors, pe_P, pe_S, pe_N)
        if avail_l3 < l3_occupy:
            continue
        if min_limit and avail_l3 - (2**20) > l3_occupy:
            continue

        pe_factors = {k: 1 for k in instance.keys()}
        pe_factors["P"] = pe_P
        pe_factors["N"] = pe_N
        pe_factors["S"] = pe_S
        return {
            "DRAM": dram_factors,
            "L3": l3_factors,
            "PE": pe_factors
        }

def split_dict_values(x, n):
    def generate_factors(value, n):
        if n == 1:
            return [(value,)]
        factors = []
        for i in range(1, abs(value) + 1):
            if value % i == 0:
                sub_factors = generate_factors(value // i, n - 1)
                factors.extend([(i,) + sf for sf in sub_factors])
        return factors

    # print(x)
    split_options = {key: generate_factors(val, n) for key, val in x.items()}
    # print(split_options)
    # print(split_options.values())
    all_split = list(product(*split_options.values()))

    return [[dict(zip(x.keys(), values)) for values in zip(*split)] for split in all_split]

def random_factors2(
    arch_config: ArchConfig,
    instance: Dict[str,int]
    # min_limit: bool = False
) -> Dict[str, Dict[str, int]]:
    return random.choice(get_all_candidates(arch_config, instance))

def get_all_candidates(
    arch_config: ArchConfig,
    instance: Dict[str,int],
    min_limit: bool = False
) -> List[Dict[str, Dict[str, int]]]:
    pe_P = arch_config.mesh_2d
    pe_S = arch_config.mesh_2d
    pe_N = instance["N"]

    dram_l3_factors = instance.copy()
    assert dram_l3_factors["P"] % pe_P == 0
    dram_l3_factors["P"] = dram_l3_factors["P"] // pe_P

    assert dram_l3_factors["S"] % pe_S == 0
    dram_l3_factors["S"] = dram_l3_factors["S"] // pe_S

    assert dram_l3_factors["N"] % pe_N == 0
    dram_l3_factors["N"] = dram_l3_factors["N"] // pe_N

    l3_H = dram_l3_factors["H"]
    dram_l3_factors["H"] = 1

    l3_E = dram_l3_factors["E"]
    dram_l3_factors["E"] = 1

    l3_F = dram_l3_factors["F"]
    dram_l3_factors["F"] = 1

    _all_factors = split_dict_values(dram_l3_factors, 2)
    _possible_factors = []
    for _fac in _all_factors:
        dram_factors = _fac[0].copy()
        l3_factors = _fac[1].copy()
        l3_factors["H"] = l3_H
        l3_factors["E"] = l3_E
        l3_factors["F"] = l3_F

        avail_l3 = arch_config.l3_size * (2**20)
        l3_occupy = get_l3_cache(l3_factors, pe_P, pe_S, pe_N)
        if avail_l3 < l3_occupy:
            continue
        if min_limit and avail_l3 - (2**20) > l3_occupy:
            continue

        pe_factors = {k: 1 for k in instance.keys()}
        pe_factors["P"] = pe_P
        pe_factors["N"] = pe_N
        pe_factors["S"] = pe_S
        _possible_factors.append({
            "DRAM": dram_factors,
            "L3": l3_factors,
            "PE": pe_factors
        })
    return _possible_factors

def get_l3_cache(l3_factors, pe_P, pe_S, pe_N):
    l3_pe_factors = l3_factors.copy()
    l3_pe_factors["N"] *= pe_N
    l3_pe_factors["P"] *= pe_P
    l3_pe_factors["S"] *= pe_S
    B = l3_pe_factors["B"]
    D = l3_pe_factors["D"]
    H = l3_pe_factors["H"]
    E = l3_pe_factors["E"]
    F = l3_pe_factors["F"]
    M = l3_pe_factors["M"]
    N = l3_pe_factors["N"]
    P = l3_pe_factors["P"]
    S = l3_pe_factors["S"]

    #  For Q, K, V.
    def get_QKV_cache(B, D, H, E, F, M, N, P, S, pe_P) -> float:
        Q = B*H*E*P
        BK = B*H*E*M*N
        BV = B*H*F*M*N
        INP1 = B*D*P
        INP2 = B*D*M*N
        WQ = D*H*E
        WK = D*H*E
        WV = D*H*F

        RM = B*H*P
        RNV = B*H*F*P
        RD = B*H*P
        AV = B*H*F*P
        return sum([Q, BK, BV, INP1, INP2, WQ, WK, WV, RM, RNV, RD, AV]) * 2

    #  For MHA
    def get_MHA_cache(B, D, H, E, F, M, N, P, S, pe_P) -> float:
        Q = B*H*E*P
        BK = B*H*E*M*N
        BV = B*H*F*M*N
        RM = B*H*P
        RNV = B*H*F*P
        RD = B*H*P
        AV = B*H*F*P

        #  Improve H
        block_cache = min(H, 4)*N*pe_P*2 + min(H, 4)*pe_P*8 * 2
        return (sum([Q, BK, BV, RM, RNV, RD, AV]) + block_cache) * 2

    def get_ANORM_cache(B, D, H, E, F, M, N, P, S, pe_P) -> float:
        AV = B*H*F*P
        INP = B*H*F*P
        NR = B*H*F*P

        block_cache = 2*2*B*pe_P * 2
        return (sum([AV, INP, NR]) + block_cache) * 2

    def get_FFN_cache(B, D, H, E, F, M, N, P, S, pe_P) -> float:
        NR = B*H*F*P
        WF = F*H*S
        WF2 = S*P
        BF = S*2
        FFNT = B*F*H*P

        block_cache = S*pe_P * 2
        return (sum([NR, WF, WF2, BF, FFNT]) + block_cache) * 2

    einsums_group: List[Einsums] = [QKV_EINSUMS, MHA_EINSUMS, ANORM_EINSUMS, FFN_EINSUMS]
    einsums_max_occpuy = max([
        Einsum.load_from_einsums(name, einsums).get_l3_occupy(l3_pe_factors)
        for einsums in einsums_group
        for name in einsums.names
    ])

    return max([
        get_QKV_cache(B, D, H, E, F, M, N, P, S, pe_P),
        get_MHA_cache(B, D, H, E, F, M, N, P, S, pe_P),
        get_ANORM_cache(B, D, H, E, F, M, N, P, S, pe_P),
        get_FFN_cache(B, D, H, E, F, M, N, P, S, pe_P),
        einsums_max_occpuy
    ])
