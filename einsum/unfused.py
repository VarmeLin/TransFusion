from einsum.einsums import Einsums

UNFUSED_EINSUMS = Einsums(
    name="Unfused",

    names=["UQK", "UAV"],

    dimensions={
        "UQK": ["B", "H", "M", "N", "P"],
        "UQ":  ["B", "H", "E", "P"],
        "UK":  ["B", "H", "E", "M", "N"],
        "UAV": ["B", "H", "F", "P"],
        "UA":  ["B", "H", "M", "N", "P"],
        "UV":  ["B", "H", "F", "M", "N"]
    },

    dependency={
        "UQK": [],
        "UAV": []
    },

    compute_cost={
        "UQK": {"mac": 1},
        "UAV": {"mac": 1}
    },

    inputs={
        "UQK":  ["UQ", "UK"],
        "UAV":  ["UA", "UV"]
    },

    block_inputs=["UQ", "UK", "UV"],
    block_outputs=["UAV"],
    keep_in_dram=["UQK", "UQ", "UK", "UAV", "UA", "UV"],
    global_parameters=[]
)