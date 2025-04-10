import random
import math
from typing import Dict, List
from einsum import *
from config.arch import PE
from itertools import chain
from config.arch import ArchConfig
from einsum.einsums import Einsums
from pipeline.scheduler import Schedule

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

def get_all_pe_possible_factors(
    mesh: int,
    dimensions: List[str],
    skip_factors = ["B"]
) -> List[List[int]]:

    def find_arrays(length, target, start=1, path=[], result=[]):
        if length == 0:
            if target == 1:
                result.append(path[:])
            return

        for i in range(start, target+1):
            if target % i == 0:
                path.append(i)
                find_arrays(length-1, target // i, 1, path, result)
                path.pop()
        return result

    arrs = find_arrays(len(dimensions), mesh)
    rst = []
    for arr in arrs:
        _fac = {dim: val for dim, val in zip(dimensions, arr)}

        #  HACK: Make the 'B' always '1' in PE/PE_col factors.
        # if "B" in _fac and _fac["B"] != 1:
        #     continue
        if any(s in _fac and _fac[s] != 1 for s in skip_factors):
            continue

        rst.append(_fac)
    return rst

def get_max_l3_cache_occupy(
    cache_storage: Dict[float, List[List[str]]],
    l3_pe_factors: Dict[str, int],
    einsums: Einsums
) -> int:  #  Measured in MB
    _factors = l3_pe_factors
    max_cache = float("-inf")
    for storage in cache_storage.values():
        total = 0
        for einsum in list(chain(*storage)):
            dimensions = einsums.dimensions[einsum]
            cache = math.prod([_factors[d] for d in dimensions])
            total += cache

        if max_cache < total:
            max_cache = total
    return max_cache * 2  #  16-bites

def random_factors_1d(
    dimensions: List[str],
    instance: Dict[str, int],
    arch_config: ArchConfig,
    cache_storage: Dict[float, List[List[str]]],
    einsums: Einsums
) -> Dict[str, Dict[str, int]]:
    all_possible_PE_factors = get_all_pe_possible_factors(
        arch_config.mesh_1d,
        dimensions,
        []
    )

    while True:
        _factors = generate_arrays(instance, 2)
        dram_factors = _factors[0]
        l3_pe_factors = _factors[1]

        l3_occupy = get_max_l3_cache_occupy(
            cache_storage=cache_storage,
            l3_pe_factors=l3_pe_factors,
            einsums=einsums
        )
        if l3_occupy > arch_config.l3_size * 1024 * 1024:
            continue

        if l3_occupy < (arch_config.l3_size-1) * 1024 * 1024:  #  Reduce the search space
            continue

        dram_factors = {d: dram_factors[d] for d in dimensions}
        l3_pe_factors = {d: l3_pe_factors[d] for d in dimensions}

        b = l3_pe_factors
        possible_factors = []
        for a in all_possible_PE_factors:
            if all(b[d] >= a[d] and b[d] % a[d] == 0 for d in dimensions):
                possible_factors.append({
                    "PE": a
                })

        if len(possible_factors) == 0:
            continue

        rst = random.choice(possible_factors)
        rst["DRAM"] = dram_factors
        rst["L3"] = {d: (l3_pe_factors[d] // rst["PE"][d]) for d in dimensions}
        return rst

def random_factors_2d(
    dimensions: List[str],
    instance: Dict[str, int],
    arch_config: ArchConfig,
    cache_storage: Dict[float, List[List[str]]],
    einsums: Einsums
) -> Dict[str, Dict[str, int]]:
    all_possible_PE_factors = get_all_pe_possible_factors(
        arch_config.mesh_2d,
        dimensions,
        []
    )
    all_possible_PE_col_factors = get_all_pe_possible_factors(
        arch_config.mesh_2d,
        dimensions,
        ["B"]  #  Hack: Avoid timeloop "Arithmetic Y exceeds" broken.
    )

    while True:
        _factors = generate_arrays(instance, 2)
        dram_factors = _factors[0]
        l3_pe_factors = _factors[1]

        l3_occupy = get_max_l3_cache_occupy(
            cache_storage=cache_storage,
            l3_pe_factors=l3_pe_factors,
            einsums=einsums
        )
        if l3_occupy > arch_config.l3_size * 1024 * 1024:
            continue

        dram_factors = {d: dram_factors[d] for d in dimensions}
        l3_pe_factors = {d: l3_pe_factors[d] for d in dimensions}

        c = l3_pe_factors
        possible_factors = []
        for a in all_possible_PE_factors:
            for b in all_possible_PE_col_factors:
                if all((x:=(a[d]*b[d])) <= c[d] and c[d] % x == 0 for d in dimensions):
                    possible_factors.append({
                        "PE": a,
                        "PE_col": b
                    })

        if len(possible_factors) == 0:
            continue

        rst = random.choice(possible_factors)
        rst["DRAM"] = dram_factors
        rst["L3"] = {d: (l3_pe_factors[d] // (rst["PE"][d]*rst["PE_col"][d])) for d in dimensions}
        return rst

def random_factors(
    dimensions: List[str],
    instance: Dict[str, int],
    pe: PE,
    arch_config: ArchConfig,
    cache_storage: Dict[float, List[List[str]]],
    einsums: Einsums
) -> Dict[str, Dict[str, int]]:

    factors: Dict[str, Dict[str, int]]
    if pe == PE.ONE_D:
        factors = random_factors_1d(dimensions, instance, arch_config, cache_storage, einsums)
    elif pe == PE.TWO_D:
        factors = random_factors_2d(dimensions, instance, arch_config, cache_storage, einsums)
    else:
        raise Exception(f"Invalid PE {pe}")

    _factors = {}
    for level, facs in factors.items():
        _factors[level] = {k: v for k, v in facs.items() if k in dimensions}

    _factors["reg_file"] = {d: 1 for d in dimensions}
    return _factors

def reverse_graph(graph):
        reversed_graph = {node: [] for node in graph}
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                if neighbor not in reversed_graph:
                    reversed_graph[neighbor] = []
                reversed_graph[neighbor].append(node)
        return reversed_graph

def l3_cache_storage(
    schedules: Dict[str, Schedule],
    einsums: Einsums,
    global_parameters = ["RM0", "RNV1", "RD1"]
) -> Dict[float, List[List[str]]]:
    time_points = []
    epoch_1 = []
    epoch_2 = []
    for einsum in einsums.names:
        sch = schedules[einsum]
        if sch.start_time not in time_points:
            time_points.append(sch.start_time)
        if sch.end_time not in time_points:
            time_points.append(sch.end_time)

        depends = reverse_graph(einsums.dependency)[einsum]
        for dep in depends:
            if dep in epoch_2:
                epoch_2.append(einsum)
                break
            dep_sch = schedules[dep]
            if dep_sch.end_time > sch.start_time:
                epoch_2.append(einsum)
                break
        if einsum not in epoch_2:
            epoch_1.append(einsum)

    time_points.sort()

    reversed_input = reverse_graph(einsums.inputs)

    rst = {}
    _cache_epoch1 = set()
    _cache_epoch2 = set()

    for time in time_points:
        doing = []
        done = []
        dont = []
        for einsum, sch in schedules.items():
            if sch.end_time <= time:
                done.append(einsum)
            elif sch.start_time <= time:
                doing.append(einsum)
            else:
                dont.append(einsum)

        for einsum in einsums.names:
            if einsum in doing:
                if einsum in epoch_1:
                    _cache_epoch1.add(einsum)
                    _cache_epoch1.update(einsums.inputs[einsum])
                if einsum in epoch_2:
                    _cache_epoch2.add(einsum)
                    _cache_epoch2.update(einsums.inputs[einsum])

            elif einsum in done:
                followings = reversed_input[einsum]

                for following in followings:
                    if einsum in epoch_1 and following in epoch_2:
                        _cache_epoch1.add(einsum)
                        break
                    elif following not in done:
                        if einsum in epoch_1:
                            _cache_epoch1.add(einsum)
                        if einsum in epoch_2:
                            _cache_epoch2.add(einsum)
                        break

            elif einsum in dont:
                if einsum in epoch_2:
                    inputs = einsums.inputs[einsum]
                    for inp in inputs:
                        if inp in epoch_1:
                            _cache_epoch2.add(inp)

        _global_parameters = [g for g in global_parameters if g not in _cache_epoch1 and g not in _cache_epoch2]
        rst[time] = [list(_cache_epoch1), list(_cache_epoch2), _global_parameters]
        _cache_epoch1.clear()
        _cache_epoch2.clear()
    return rst
