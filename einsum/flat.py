from einsum.einsums import Einsums

FLAT_EINSUMS = Einsums(
    name="FLAT",

    names=["FQK", "FAV"],

    dimensions={
        "FQK": ["B", "H", "M", "N", "P"],
        "FQ":  ["B", "H", "E", "P"],
        "FK":  ["B", "H", "E", "M", "N"],
        "FAV": ["B", "H", "F", "P"],
        "FA":  ["B", "H", "M", "N", "P"],
        "FV":  ["B", "H", "F", "M", "N"]
    },

    dependency={
        "FQK": [],
        "FAV": []
    },

    compute_cost={
        "FQK": {"mac": 1},
        "FAV": {"mac": 1}
    },

    inputs={
        "FQK":  ["FQ", "FK"],
        "FAV":  ["FA", "FV"]
    },

    block_inputs=["FQ", "FK", "FV"],
    block_outputs=["FAV"],
    keep_in_dram=[],
    global_parameters=[]
)