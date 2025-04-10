from ruamel.yaml import YAML, yaml_object
from pathlib import Path
from typing import Union, List
import fcntl

#yaml = YAML(typ="safe", pure=True)

class YamlConstructor:
    yaml_tag = None

    def __init__(self, **data):
        #self.data = data
        self.__dict__.update(data)

    def __repr__(self):
        data = self.__dict__
        return f"{self.__class__.__name__}: {data}"

    def __iter__(self):
        return iter(self.__dict__)

    def __getitem__(self, index):
        return self.__dict__[index]

    @classmethod
    def from_yaml(cls, loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        return cls(**mapping)

    @staticmethod
    def to_yaml(representer, data):
        _default_flow_style = representer.default_flow_style
        representer.default_flow_style = False

        node = representer.represent_mapping(
            data.__class__.yaml_tag,
            data.__dict__)

        representer.default_flow_style = _default_flow_style

        for key, value in node.value:
            if key.value == "attributes":
                for i in range(len(value.value)):
                    k, v  = value.value[i]
                    if v.tag == "tag:yaml.org,2002:str":
                        v.style = '"'
        return node

#@yaml_object(yaml)
class Container(YamlConstructor):
    yaml_tag = "!Container"
    def __init__(self, **data):
        super().__init__(**data)

#@yaml_object(yaml)
class Component(YamlConstructor):
    yaml_tag = "!Component"
    def __init__(self, **data):
        super().__init__(**data)


#@yaml_object(yaml)
class Parallel(YamlConstructor):
    yaml_tag = "!Parallel"
    def __init__(self, **data):
        super().__init__(**data)


class YamlInput:
    def __init__(self):
        self.input = {}

    @staticmethod
    def load_input_from_file(file: Union[str, Path]) -> dict:
        yaml = YAML(typ="safe", pure=True)
        yaml.register_class(Container)
        yaml.register_class(Component)
        yaml.register_class(Parallel)

        input = {}
        with open(file, "r") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                input.update(yaml.load(f))
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return input

    @staticmethod
    def write_all_to_file(file: Union[str, Path], inputs: List["YamlInput"]):
        yaml = YAML(typ="safe", pure=True)
        yaml.register_class(Container)
        yaml.register_class(Component)
        yaml.register_class(Parallel)

        rst = {}
        for inp in inputs:
            rst.update(inp.input)

        if isinstance(file, str):
            file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yaml.dump(rst, f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def write_to_file(self, file: Union[str, Path]):
        yaml = YAML(typ="safe", pure=True)
        yaml.register_class(Container)
        yaml.register_class(Component)
        yaml.register_class(Parallel)

        if isinstance(file, str):
            file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yaml.dump(self.input, f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def get_input(self):
        return self.input