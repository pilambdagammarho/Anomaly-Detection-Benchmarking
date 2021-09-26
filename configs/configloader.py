from typing import Dict
import yaml


def load_config(type: str) -> Dict:
    """
    This method is responsible for reading yaml config files and loading the configurations.
    :param type: String value indicating name of the file
    :return: Dict values containing key value pair of values
    """
    with open(f"{type}.yaml", "rb") as file:
        try:
            content = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return content

