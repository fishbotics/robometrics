from pathlib import Path


def get_module_path() -> Path:
    return Path(__file__).resolve().parent


def get_content_path() -> Path:
    return get_module_path() / "content"


def get_dataset_path() -> Path:
    return get_content_path() / "dataset"


def get_robot_path() -> Path:
    return get_content_path() / "robot"
