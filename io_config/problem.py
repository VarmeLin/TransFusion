from .yaml_input import YamlInput
from typing import Dict, List

class Problem(YamlInput):
    def __init__(self):
        super().__init__()

    @classmethod
    def load_from_file(cls, file: str) -> "Problem":
        problem = cls()
        problem.input = YamlInput.load_input_from_file(file)
        return problem

    @classmethod
    def load_default(cls) -> "Problem":
        problem = cls()
        problem.input = {
            "problem": {
                "version": "0.4",
                "shape": {
                    "name": "", #"AV",
                    "dimensions": [
                        #"B", "F", "H", "P"
                    ],
                    "data_spaces": [
                        # {"name": "RNV", "projection": [[["B"]], [["F"]], [["H"]], [["P"]]]},
                        # {"name": "RD", "projection": [[["B"]], [["H"]], [["P"]]]},
                        # {"name": "AV", "projection": [[["B"]], [["F"]], [["H"]], [["P"]]], "read_write": True}
                    ]
                },
                "instance": {}  # {"B": 1, "F": 4, "H": 5, "M": 6, "P": 7}}}
            }
        }

        return problem

    def update_dataspaces(self, inputs: Dict[str, List[str]], output: Dict[str, List[str]]):
        """Update the dataspaces.

        Args:
            inputs (Dict[str, List[str]]): Key: The name of the einsum/value/parameter.
                                           Value: The dimensions/projections of the einsum.
                                           e.g., {"RNV": ["B", "F", "H", "P"]}
            output (Dict[str, List[str]]): Key: The name of the einsum/value/parameter.
                                           Value: The dimensions/projections of the einsum.
                                           MUST only contain one item.
        """
        if len(output.keys()) != 1:
            raise Exception(f"Invalid output: {output}")
        self.input["problem"]["shape"]["name"] = list(output.keys())[0]
        dimensions = []
        dataspaces = []
        for name, _dims in inputs.items():
            projection = [[[dim]] for dim in _dims]
            dimensions += [d for d in _dims if d not in dimensions]
            dataspaces.append({"name": name, "projection": projection})
        for name, _dims in output.items():
            projection = [[[dim]] for dim in _dims]
            dimensions += [d for d in _dims if d not in dimensions]
            dataspaces.append({"name": name, "projection": projection, "read_write": True})
        self.input["problem"]["shape"]["data_spaces"] = dataspaces
        self.input["problem"]["shape"]["dimensions"] = dimensions

    def update_instance(self, instance: dict):
        ins: dict = {}
        dimensions = self.input["problem"]["shape"]["dimensions"]
        for rank in dimensions:
            if rank not in instance:
                raise Exception("Cannot find rank in instance.")
            ins[rank] = instance[rank]
        self.input["problem"]["instance"] = ins

    def get_dimensions(self) -> List[str]:
        return self.input["problem"]["shape"]["dimensions"]