from einsum.einsums import Einsums

ANORM_EINSUMS = Einsums(
    name="LayerNorm",

    names=["IAV", "SAV", "MAV", "DAV", "QAV", "SQAV", "MQAV", "SR", "NR"],

    dimensions={
        "AV":   ["B", "F", "H", "P"],
        "IAV":  ["B", "F", "H", "P"],
        "INPI": ["B", "F", "H", "P"],
        "SAV":  ["B", "P"],
        "MAV":  ["B", "P"],
        "DAV":  ["B", "F", "H", "P"],
        "QAV":  ["B", "F", "H", "P"],
        "SQAV": ["B", "P"],
        "MQAV": ["B", "P"],
        "SR":   ["B", "P"],
        "NR":   ["B", "F", "H", "P"],
    },

    dependency={
        "IAV": ["SAV"],
        "SAV": ["MAV"],
        "MAV": ["DAV"],
        "DAV": ["QAV"],
        "QAV": ["SQAV"],
        "SQAV": ["MQAV"],
        "MQAV": ["SR"],
        "SR": ["NR"],
        "NR": [],
    },

    compute_cost={
        "IAV": {"add": 1},
        "SAV": {"add": 1},
        "MAV": {"mac": 1},
        "DAV": {"add": 1},
        "QAV": {"mac": 1},
        "SQAV": {"add": 1},
        "MQAV": {"mac": 1},
        "SR": {"mac": 6},   #  SQRT_15: https://link.springer.com/article/10.1007/s11075-024-01932-7
        "NR": {"mac": 1},
    },

    inputs={
        "IAV": ["INPI", "AV"],
        "SAV": ["IAV"],
        "MAV": ["SAV"],
        "DAV": ["IAV", "MAV"],
        "QAV": ["DAV"],
        "SQAV": ["QAV"],
        "MQAV": ["SQAV"],
        "SR": ["MQAV"],
        "NR": ["DAV", "SR"]
    },

    block_inputs=["AV"],
    block_outputs=["NR"],
    keep_in_dram=[],
    global_parameters=[]
)
