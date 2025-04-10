from einsum.einsums import Einsums

QKV_EINSUMS = Einsums(
    name="QKV",

    names=["Q", "BK", "BV"],

    dimensions={
        "INPQ": ["B", "D", "P"],
        "WQ":   ["D", "E", "H"],
        "INPK": ["B", "D", "M", "N"],
        "WK":   ["D", "E", "H"],
        "INPV": ["B", "D", "M", "N"],
        "WV":   ["D", "H", "F"],
        "Q":    ["B", "E", "H", "P"],
        "BK":   ["B", "E", "H", "M", "N"],
        "BV":   ["B", "F", "H", "M", "N"]
    },

    dependency={
        "Q": [],
        "BK": [],
        "BV": []
    },

    compute_cost={
        "Q": {"mac": 1},
        "BK": {"mac": 1},
        "BV": {"mac": 1}
    },

    inputs={
        "Q":    ["INPQ", "WQ"],
        "BK":   ["INPK", "WK"],
        "BV":   ["INPV", "WV"]
    },

    block_inputs=["INPQ", "WQ", "INPK", "WK", "INPV", "WV"],
    block_outputs=["Q", "BK", "BV"],
    keep_in_dram=["INPQ", "WQ", "INPK", "WK", "WV", "BK", "BV"],
    global_parameters=[]
)
