from einsum.einsums import Einsums

MHA_EINSUMS = Einsums(
    name="MHA",

    names=["QK", "LM", "RM", "SLN", "SLD", "SLNV",
    "PRM", "SPD", "RD", "SPNV", "RNV", "AV",],

    dimensions={
        "Q":    ["B", "E", "H", "P"],
        "BK":   ["B", "E", "H", "M", "N"],
        "BV":   ["B", "F", "H", "M", "N"],
        "RNV1": ["B", "F", "H", "P"],
        "RD1":  ["B", "H", "P"],
        "AV":   ["B", "F", "H", "P"],
        "RM0":  ["B", "H", "M", "P"],
        "LM":   ["B", "H", "M", "P"],
        "PRM":  ["B", "H", "M", "P"],
        "QK":   ["B", "H", "M", "N", "P"],
        "RD":   ["B", "H", "M", "P"],
        "RM":   ["B", "H", "M", "P"],
        "RNV":  ["B", "F", "H", "M", "P"],
        "SLD":  ["B", "H", "M", "P"],
        "SLN":  ["B", "H", "M", "N", "P"],
        "SLNV": ["B", "F", "H", "M", "P"],
        "RD0":  ["B", "H", "M", "P"],
        "SPD":  ["B", "H", "M", "P"],
        "RNV0": ["B", "F", "H", "M", "P"],
        "SPNV": ["B", "F", "H", "M", "P"]
    },

    dependency={
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
        "AV": []
    },

    compute_cost={
        "QK": {"mac": 1},
        "LM": {"max": 1},
        "RM": {"max": 1},
        "PRM": {"mac": 6, "add": 1},
        "SPD": {"mac": 1},
        "SPNV": {"mac": 1},
        "SLN": {"mac": 6, "add": 1},
        "SLD": {"add": 1},
        "SLNV": {"mac": 1},
        "RD": {"add": 1},
        "RNV": {"add": 1},
        "AV": {"divide": 1},
    },

    inputs={
        "QK":   ["Q", "BK"],
        "LM":   ["QK"],
        "RM":   ["LM", "RM0"],
        "SLN":  ["QK", "RM"],
        "SLD":  ["SLN"],
        "SLNV": ["BV", "SLN"],
        "PRM":  ["RM", "RM0"],
        "SPD":  ["PRM", "RD0"],
        "RD":   ["SLD", "SPD"],
        "SPNV": ["PRM", "RNV0"],
        "RNV":  ["SLNV", "SPNV"],
        "AV":   ["RD1", "RNV1"]
    },

    block_inputs=["Q", "BK", "BV"],
    block_outputs=["AV"],
    keep_in_dram=["BK", "BV"],
    global_parameters=["RM0", "RNV1", "RD1"]
)