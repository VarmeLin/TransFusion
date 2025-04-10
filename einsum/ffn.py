from einsum.einsums import Einsums
import copy

FFN_EINSUMS = Einsums(
    name="FFN",

    names=["FFN", "AR", "FFNT"],

    dimensions={
        "NR":   ["B", "F", "H", "P"],
        "FFN":  ["B", "S", "P"],
        "WF":   ["F", "H", "S"],
        "BF":   ["S"],
        "AR":   ["B", "S", "P"],
        "FFNT": ["B", "F", "H", "P"],
        "WFT":  ["F", "H", "S"],
        "BFT":  ["F", "H"],
    },

    dependency={
        "FFN": ["AR"],
        "AR": ["FFNT"],
        "FFNT": [],
    },

    compute_cost={
        "FFN": {"mac": 1},
        "AR": {"max": 1},
        "FFNT": {"mac": 1}
    },

    inputs={
        "FFN": ["WF","NR", "BF"],
        "AR": ["FFN"],
        "FFNT": ["WFT", "AR", "BFT"]
    },

    block_inputs=["WF", "NR", "BF", "WFT", "BFT"],
    block_outputs=["Q", "BK", "BV"],
    keep_in_dram=["WF", "BF", "WFT", "BFT"],
    global_parameters=[]
)

model_activation = {
    "BERT": "GELU",
    "TrXL": "ReLU",
    "T5": "ReLU",
    "XLM": "GELU",
    "Llama3": "SiLU"
}

activation_cost = {
    "ReLU": {"max": 1},
    "GELU": {"mac": 7, "add": 1, "divide": 1},  #  GELU: https://arxiv.org/abs/1606.08415
    "SiLU": {"mac": 7, "add": 1, "divide": 1}
}

activation_einsums = ["AR"]

def load_ffn_einsums_for_model(model: str) -> Einsums:
    if model in model_activation:
        einsums = copy.deepcopy(FFN_EINSUMS)
        for einsum in activation_einsums:
            einsums.compute_cost[einsum] = activation_cost[model_activation[model]]
        return einsums
    else:
        raise Exception(f"Cannot find model {model}")