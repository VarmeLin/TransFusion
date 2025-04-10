from einsum.einsums import Einsums

ANORM_EINSUMS = Einsums(
    name="LayerNorm",

    names=["IAV", "QAV", "SAV", "MAV", "SQAV", "MQAV",
    "DAV", "SR", "NR"],

    dimensions={
        "AV":   ["B", "F", "H", "P"],
        "IAV":  ["B", "F", "H", "P"],
        "INPI": ["B", "F", "H", "P"],
        "QAV":  ["B", "F", "H", "P"],
        "SAV":  ["B", "P"],
        "MAV":  ["B", "P"],
        "SQAV": ["B", "P"],
        "MQAV": ["B", "P"],
        "DAV":  ["B", "F", "H", "P"],
        "SR":   ["B", "P"],
        "NR":   ["B", "F", "H", "P"],
    },

    dependency={
        "IAV": ["QAV", "SAV"],
        "QAV": ["SQAV"],
        "SAV": ["MAV"],
        "MAV": ["DAV"],
        "SQAV": ["MQAV"],
        "MQAV": ["SR"],
        "DAV": ["NR"],
        "SR": ["NR"],
        "NR": [],
    },

    compute_cost={
        "IAV": {"add": 1},
        "QAV": {"mac": 1},
        "SAV": {"add": 1},
        "MAV": {"mac": 1},
        "SQAV": {"add": 1},
        "MQAV": {"mac": 1},
        "DAV": {"mac": 1},  #  Subtraction
        "SR": {"mac": 6},   #  SQRT_15: https://link.springer.com/article/10.1007/s11075-024-01932-7
        "NR": {"mac": 1},
    },

    inputs={
        "IAV": ["INPI", "AV"],
        "QAV": ["IAV"],
        "SAV": ["IAV"],
        "MAV": ["SAV"],
        "SQAV": ["QAV"],
        "MQAV": ["SQAV"],
        "DAV": ["IAV", "MAV"],
        "SR": ["MQAV"],
        "NR": ["SR", "DAV"],
    },

    block_inputs=["AV"],
    block_outputs=["NR"],
    keep_in_dram=[],
    global_parameters=[]
)
