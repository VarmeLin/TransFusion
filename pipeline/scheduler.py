import random
import math
from itertools import chain, combinations
import networkx as nx
from typing import Dict, List, Tuple
from dataclasses import dataclass
from config.arch import PE

def split_dag(graph, fixed_nodes):
    _graph = graph
    graph = nx.DiGraph(_graph)
    start_nodes = [n for n in graph.nodes if graph.in_degree(n) == 0]
    end_nodes = [n for n in graph.nodes if graph.out_degree(n) == 0]

    if len(start_nodes) < 1 or len(end_nodes) < 1:
        raise ValueError("Graph must have at least one start and one end node.")

    nodes = set(graph.nodes)
    all_splits = []

    def all_subsets(iterable):
        """Returns all non-empty subsets of the iterable."""
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
    for subset in all_subsets(nodes - set(end_nodes)):
        subset = set(subset)
        if any(x not in subset for x in start_nodes):
            continue
        if fixed_nodes == None or fixed_nodes.issubset(subset): # or fixed_nodes.isdisjoint(subset):
            other_subset = nodes - subset
            subgraph1 = graph.subgraph(subset).copy()
            subgraph2 = graph.subgraph(other_subset).copy()

            def is_valid_subgraph(graph):
                return nx.is_directed_acyclic_graph(graph) and all(len(list(graph.successors(n)) + list(graph.predecessors(n))) > 0 for n in graph.nodes if n not in start_nodes)

            if is_valid_subgraph(subgraph1) and is_valid_subgraph(subgraph2):
                subgraph1_start_nodes = [n for n in subgraph1.nodes if subgraph1.in_degree(n) == 0]
                subgraph2_start_nodes = [n for n in subgraph2.nodes if subgraph2.in_degree(n) == 0]

                new_graph = {
                    "ROOT": subgraph1_start_nodes + subgraph2_start_nodes
                }

                subgraph1_dict = {node: list(subgraph1.successors(node)) for node in subgraph1.nodes}
                subgraph2_dict = {node: list(subgraph2.successors(node)) for node in subgraph2.nodes}

                new_graph.update(subgraph1_dict)
                new_graph.update(subgraph2_dict)

                all_splits.append(new_graph)

    new_graph = {"ROOT": start_nodes}
    new_graph.update(_graph)
    all_splits.append(new_graph)
    return all_splits

def split_dag2(graph):
    _graph = graph
    graph = nx.DiGraph(_graph)
    start_nodes = [n for n in graph.nodes if graph.in_degree(n) == 0]
    end_nodes = [n for n in graph.nodes if graph.out_degree(n) == 0]

    if len(start_nodes) < 1 or len(end_nodes) < 1:
        raise ValueError("Graph must have at least one start and one end node.")

    nodes = set(graph.nodes)
    all_splits = []

    def all_subsets(iterable):
        """Returns all non-empty subsets of the iterable."""
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    for subset in all_subsets(nodes - set(end_nodes)):
        subset = set(subset)
        if any(x not in subset for x in start_nodes):
            continue

        def is_dependency_available(subset, graph):
            for node in subset:
                for p_node in graph.predecessors(node):
                    if p_node not in subset:
                        return False
            return True

        if not is_dependency_available(subset, graph):
            continue

        other_subset = nodes - subset
        subgraph1 = graph.subgraph(subset).copy()
        subgraph2 = graph.subgraph(other_subset).copy()

        def controlable(start_nodes, subgraph):
            for node in start_nodes:
                reachable = nx.descendants(subgraph, node)
                if len(reachable) != len(subgraph.nodes) - len(start_nodes):
                    return False
            return True

        def is_valid_subgraph(graph):
            return nx.is_directed_acyclic_graph(graph) and nx.is_weakly_connected(graph)

        if is_valid_subgraph(subgraph1) and is_valid_subgraph(subgraph2):
            subgraph1_start_nodes = [n for n in subgraph1.nodes if subgraph1.in_degree(n) == 0]
            subgraph2_start_nodes = [n for n in subgraph2.nodes if subgraph2.in_degree(n) == 0]

            if not controlable(subgraph1_start_nodes, subgraph1):
                continue

            new_graph = {
                "ROOT": subgraph1_start_nodes + subgraph2_start_nodes
            }

            subgraph1_dict = {node: list(subgraph1.successors(node)) for node in subgraph1.nodes}
            subgraph2_dict = {node: list(subgraph2.successors(node)) for node in subgraph2.nodes}

            new_graph.update(subgraph1_dict)
            new_graph.update(subgraph2_dict)

            if "SPD" in subgraph1_dict.keys() and "RD" in subgraph2_dict.keys():
                new_graph["RD"].append("SPD")
            if "SPNV" in subgraph1_dict.keys() and "RNV" in subgraph2_dict.keys():
                new_graph["RNV"].append("SPNV")

            all_splits.append(new_graph)

    new_graph = {"ROOT": start_nodes}
    new_graph.update(_graph)
    all_splits.append(new_graph)
    return all_splits

def all_topological_sorts(graph):
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    result = []
    visited = set()
    def dfs(current_sort):
        if len(current_sort) == len(graph):
            result.append(current_sort[:])
            return
        for node in graph:
            if in_degree[node] == 0 and node not in visited:
                visited.add(node)
                current_sort.append(node)
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                dfs(current_sort)
                visited.remove(node)
                current_sort.pop()
                for neighbor in graph[node]:
                    in_degree[neighbor] += 1

    dfs([])
    return result

def reverse_graph(graph):
    reversed_graph = {node: [] for node in graph}

    for node, neighbors in graph.items():
        for neighbor in neighbors:
            reversed_graph[neighbor].append(node)

    return reversed_graph

def schedule_operators2(ops, times, dependencies):
    reversed_dep = reverse_graph(dependencies)
    dimensions = len(times)
    t = [0 for _ in range(dimensions)]
    end_time = {}

    timelines = []
    for op in ops:
        dep_max_end_time = 0
        for dep in reversed_dep[op]:
            if dep_max_end_time < end_time[dep]:
                dep_max_end_time = end_time[dep]
        start_time = [max(t[i], dep_max_end_time) for i in range(dimensions)]
        _end_time = [start_time[i] + times[i][op] for i in range(dimensions)]

        end_time[op] = min(_end_time)
        min_val = end_time[op]
        min_arg = max(i for i, v in enumerate(_end_time) if v == min_val) #_end_time.index(end_time[op])
        timelines.append((op, min_arg, start_time[min_arg], end_time[op]))
        t[min_arg] = end_time[op]

    return (timelines, max(t))

def schedule_operators(ops, time_1d, time_2d, dependencies):
    reversed_dep = reverse_graph(dependencies)
    t_1d = 0
    t_2d = 0
    end_time = {}

    timelines = []
    for op in ops:
        dep_max_end_time = 0
        for dep in reversed_dep[op]:
            if dep_max_end_time < end_time[dep]:
                dep_max_end_time = end_time[dep]
        start_time_1d = max(t_1d, dep_max_end_time)
        end_time_1d = start_time_1d + time_1d[op]
        start_time_2d = max(t_2d, dep_max_end_time)
        end_time_2d = start_time_2d + time_2d[op]

        end_time[op] = min(end_time_1d, end_time_2d)
        if end_time_2d < end_time_1d:
            timelines.append((op, "2d", start_time_2d, end_time_2d))
            t_2d = end_time_2d
        else:
            timelines.append((op, "1d", start_time_1d, end_time_1d))
            t_1d = end_time_1d
    return (timelines, max(t_1d, t_2d))

def schedule_pipeline(schedules, min_latency):
    #schedules, min_latency = schedule_pipeline2(ops, [time_1d, time_2d], dependencies=dependencies)
    def find_head_gap(schs):
        start_time = min([sch[2] for sch in schs])
        return start_time
    def get_operations_in_time_range(schs, time_range):
        return [sch[0] for sch in schs if (sch[2] >= time_range[0] and sch[3] <= time_range[1])]
    def find_tail_gap(schs, end_time):
        _end_time = max([sch[3] for sch in schs])
        return end_time - _end_time
    def shift_ops_to_tails(schs, ops):
        shifted_schs = []
        unshifted_schs = []
        end_time = max([sch[3] for sch in schs])
        for sch in schs:
            if sch[0] in ops:
                duration = sch[3] - sch[2] # end_time - start_time
                _end_time = end_time + duration
                shifted_schs.append((sch[0], sch[1], end_time, _end_time))
                end_time = _end_time
            else:
                unshifted_schs.append(sch)
        return unshifted_schs + shifted_schs
    def realign_schs(schs_list):
        start_time = min([sch[2] for schs in schs_list for sch in schs])
        end_time = float("-inf")
        if start_time == 0:
            end_time = max([sch[3] for schs in schs_list for sch in schs])
            return schs_list, end_time
        for schs in schs_list:
            for i in range(len(schs)):
                sch = schs[i]
                schs[i] = (
                    sch[0],
                    sch[1],
                    sch[2] - start_time,
                    sch[3] - start_time)
                if end_time < schs[i][3]:
                    end_time = schs[i][3]
        return schs_list, end_time

    schs_1d = [sch for sch in schedules if sch[1] == 0]
    schs_2d = [sch for sch in schedules if sch[1] == 1]

    head_gap_1d = find_head_gap(schs_1d)
    head_gap_2d = find_head_gap(schs_2d)

    end_time = min_latency
    tail_gap_1d = find_tail_gap(schs_1d, end_time)
    tail_gap_2d = find_tail_gap(schs_2d, end_time)

    if head_gap_1d != 0 and tail_gap_2d != 0:
        shifted_ops = get_operations_in_time_range(schs_2d, [0, head_gap_1d])
        schs_2d = shift_ops_to_tails(schs_2d, shifted_ops)
    elif head_gap_2d != 0 and tail_gap_1d != 0:
        shifted_ops = get_operations_in_time_range(schs_1d, [0, head_gap_2d])
        schs_1d = shift_ops_to_tails(schs_1d, shifted_ops)
    schs_list, min_latency = realign_schs([schs_1d, schs_2d])
    schedules = [sch for schs in schs_list for sch in schs]
    return schedules, min_latency

def simulated_annealing(candidates, schedule_func, valid_func, initial_temp=100, cooling_rate=0.95, max_iter=1000):
    def random_candidate():
        return random.choice(candidates)
    current_candidate = random_candidate()
    _schedule, _min_latency = schedule_func(current_candidate)
    best_schedule = _schedule
    best_latency = _min_latency
    temperature = initial_temp

    for _ in range(max_iter):
        new_candidate = current_candidate[:]
        i, j = random.sample(range(len(new_candidate)), 2)
        new_candidate[i], new_candidate[j] = new_candidate[j], new_candidate[i]
        if not valid_func(new_candidate):
            continue

        _schedule, _min_latency = schedule_func(current_candidate)
        delta = -(_min_latency - best_latency)

        if delta > 0 or random.random() < math.exp(delta/temperature):
            current_candidate = new_candidate
            if _min_latency < best_latency:
                best_schedule = _schedule
                best_latency = _min_latency

        temperature *= cooling_rate

    return best_schedule, best_latency

def traverse_candidates(candidates, schedule_func):
    best_schedule = None
    best_latency = float("inf")
    for candidate in candidates:
        _schedule, _min_latency = schedule_func(candidate)
        if _min_latency < best_latency:
            best_schedule = _schedule
            best_latency = _min_latency
    return best_schedule, best_latency

EINSUMS_DEPENDENCY = {
    "QK": ["LM"],
    "LM": ["RM"],
    "RM": ["SLN", "PRM"],
    "SLN": ["SLD", "SLNV"],
    "SLD": ["RD"],
    "SLNV": ["RNV"],
    "PRM": ["SPD", "SPNV"],
    "SPD": ["RD"],
    "RD": ["AV"],
    "SPNV": ["RNV"],
    "RNV": ["AV"],
    "AV": []}

@dataclass(frozen=True)
class Schedule:
    einsum: str
    pe: PE
    start_time: float
    end_time: float

    def __repr__(self):
        return f"Schedule(einsum='{self.einsum}', pe={self.pe}, start_time={self.start_time}, end_time={self.end_time})"

@dataclass(frozen=True)
class ScheduleResult:
    schedules: Dict[str, Schedule]
    latency: float
    dag: dict

def run_scheduler(
    einsums_latency: Dict[str, List[float]],
    dependencies: Dict[str, List[str]]=EINSUMS_DEPENDENCY
) -> ScheduleResult:
    einsums_latency = einsums_latency.copy()
    for einsum in dependencies.keys():
        if einsum not in einsums_latency:
            einsums_latency[einsum] = [0, 0]

    candidates = all_topological_sorts(dependencies)
    time_1d = {}
    time_2d = {}

    for op in einsums_latency.keys():
        time_1d[op] = einsums_latency[op][0]
        time_2d[op] = einsums_latency[op][1]

    def schedule_func(candidate):
        _schedules, _min_latency = schedule_operators2(
                candidate, [time_1d, time_2d], dependencies)

        return _schedules, _min_latency

    def valid_func(candidate):
        for i in range(len(candidate)):
            op = candidate[i]
            deps = dependencies[op]
            for dep in deps:
                if dep not in candidate[i:]:
                    return False
        return True
    # _schedule, _min_latency = simulated_annealing(
    #     candidates, schedule_func, valid_func, max_iter=10000)
    _schedule, _min_latency = traverse_candidates(candidates, schedule_func)

    _schedule = [item for item in _schedule if item[0] != "ROOT"]

    return ScheduleResult(
        schedules=deserialize(_schedule),
        latency=_min_latency,
        dag=dependencies
    )

def run_scheduler2(
    einsums_latency: Dict[str, List[float]],
    dependencies: Dict[str, List[str]],
    einsums_disable_break: List[str]=["RM", "RD"],
    dimensions_count=2) -> Tuple[Dict[str, Schedule], float]:
    def find_all_links(edges, targets = []):
        if targets == None:
            return None
        parents = set()
        visited = set()
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for parent, children in edges.items():
                if node in children and parent not in parents:
                    parents.add(parent)
                    dfs(parent)
        for target in targets:
            dfs(target)
            parents.add(target)
        return parents

    einsums_disable_break = [ein for ein in einsums_disable_break if ein in einsums_latency]
    if len(einsums_disable_break) == 0:
        einsums_disable_break = None

    latency = einsums_latency.copy()
    latency["ROOT"] = [0 for _ in range(dimensions_count)]

    disable_break = find_all_links(dependencies, einsums_disable_break)

    dags = split_dag2(dependencies)
    min_sch = None
    min_lat = float("inf")
    for dag in dags:
        rst = run_scheduler(latency, dag)

        if min_lat > rst.latency:
            min_sch = rst
            min_lat = rst.latency
    return min_sch

def deserialize(data: list) -> Dict[str, Schedule]:
    """Deserialize and change the type of list to map.
    """
    rst = {}
    for d in data:
        rst[d[0]] = Schedule(
            einsum = d[0],
            pe = PE.ONE_D if d[1] == 0 else PE.TWO_D,
            start_time = d[2],
            end_time = d[3]
        )
    return rst