from einsum.einsums import Einsums

SOFTMAX_EINSUMS = Einsums(
    name="Softmax",

    names=["SM", "SN", "SD", "SA"],

    dimensions={
        "SM": ["B", "H", "P"],
        "SQK": ["B", "H", "M", "N", "P"],
        "SN": ["B", "H", "M", "N", "P"],
        "SD": ["B", "H", "P"],
        "SA": ["B", "H", "M", "N", "P"]
    },

    dependency={
        "SM": ["SN"],
        "SN": ["SD"],
        "SD": ["SA"],
        "SA": [],
    },

    compute_cost={
        "SM": {"max": 1},
        "SN": {"mac": 6, "add": 1},
        "SD": {"add": 1},
        "SA": {"divide": 1}
    },

    inputs={
        "SM":  ["SQK"],
        "SN":  ["SM", "SQK"],
        "SD": ["SN"],
        "SA": ["SN", "SD"]
    },

    block_inputs=["SQK"],
    block_outputs=["SA"],
    keep_in_dram=[],
    global_parameters=[]
)