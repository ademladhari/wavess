from __future__ import annotations

import os
import warnings

from .constants import WATERMARK_METHODS


def _dataset_names() -> list[str]:
    return ["diffusiondb", "mscoco", "dalle3"]


def _default_dataset_name() -> str:
    dataset_name = os.environ.get("WAVES_DEFAULT_DATASET", "mscoco").strip().lower()
    if dataset_name not in _dataset_names():
        raise ValueError(f"WAVES_DEFAULT_DATASET must be one of {_dataset_names()}, found {dataset_name}")
    return dataset_name


def _valid_sources() -> list[str]:
    method_sources = list(WATERMARK_METHODS.keys())
    return ["real", *method_sources, *[f"real_{name}" for name in method_sources]]


def check_file_existence(path: str, name_pattern: str, limit: int) -> list[bool]:
    found_filenames = set(os.listdir(path))
    return [name_pattern.format(i) in found_filenames for i in range(limit)]


def existence_operation(existences1: list[bool], existences2: list[bool], op: str) -> list[bool]:
    if op == "difference":
        return [a and not b for a, b in zip(existences1, existences2)]
    if op == "union":
        return [a and b for a, b in zip(existences1, existences2)]
    raise ValueError(f"Invalid operation {op}, can either be 'difference' or 'union'")


def existence_to_indices(existences: list[bool], limit: int) -> list[int]:
    indices: list[int] = []
    for i in range(min(len(existences), limit)):
        if existences[i]:
            indices.append(i)
    return indices


def parse_image_dir_path(path: str, quiet: bool = True):
    data_dir = os.path.abspath(os.path.normpath(os.environ.get("DATA_DIR")))
    normalized_path = os.path.abspath(os.path.normpath(str(path)))
    if os.path.commonpath([data_dir, normalized_path]) != os.path.commonpath([data_dir]):
        raise ValueError(f"Image directory should be under the dataset directory {os.environ.get('DATA_DIR')}")

    path_parts = normalized_path.split(os.sep)
    try:
        mode, dataset_name, dirname = path_parts[-3:]
    except ValueError:  # pragma: no cover
        mode, dataset_name, dirname = "", "", ""

    if dataset_name not in _dataset_names():
        flat_dataset = _default_dataset_name()
        if normalized_path == data_dir:
            return flat_dataset, None, None, "real"
        if os.path.dirname(normalized_path) == data_dir:
            dirname = os.path.basename(normalized_path)
            if dirname in _valid_sources():
                return flat_dataset, None, None, dirname
            if len(dirname.split("-")) == 3:
                attack_name, attack_strength, source_name = dirname.split("-")
                try:
                    attack_strength = float(attack_strength)
                    if attack_strength <= 0:
                        raise ValueError("Attack strength must be positive")
                except ValueError:
                    raise ValueError("Attack strength must be a number")
                if source_name not in _valid_sources():
                    raise ValueError(f"Source name must be one of {_valid_sources()}")
                return flat_dataset, attack_name, attack_strength, source_name
        raise ValueError("Invalid image directory path, unable to parse")

    if mode == "attacked":
        if len(dirname.split("-")) != 3:
            raise ValueError(
                f"Attack directory name {dirname} is not in the format of 'attack_name-attack_strength-source_name'"
            )
        attack_name, attack_strength, source_name = dirname.split("-")
        try:
            attack_strength = float(attack_strength)
            if attack_strength <= 0:
                raise ValueError("Attack strength must be positive")
        except ValueError:
            raise ValueError("Attack strength must be a number")
        if source_name not in _valid_sources():
            raise ValueError(f"Source name must be one of {_valid_sources()}")
        if not quiet:
            print(" -- Dataset name:", dataset_name)
            print(" -- Attack name:", attack_name)
            print(" -- Attack strength:", attack_strength)
            print(" -- Source name:", source_name)
        return dataset_name, attack_name, attack_strength, source_name

    if mode == "main":
        if dirname not in ["real", *WATERMARK_METHODS.keys()]:
            raise ValueError(f"Source name must be one of {['real', *WATERMARK_METHODS.keys()]}")
        source_name = dirname
        if not quiet:
            print(" -- Dataset name:", dataset_name)
            print(" -- Attack name:", None)
            print(" -- Attack strength:", None)
            print(" -- Source name:", source_name)
        return dataset_name, None, None, source_name

    raise ValueError("Invalid image directory path, unable to parse")


def get_all_image_dir_paths(criteria=None):
    if criteria is not None and not callable(criteria):
        raise ValueError("criteria must be a callable function")
    data_dir = os.path.normpath(os.environ.get("DATA_DIR"))
    dir_paths: set[str] = set()

    for mode in ["main", "attacked"]:
        for dataset_name in _dataset_names():
            base = os.path.join(data_dir, mode, dataset_name)
            if not os.path.isdir(base):
                continue
            for dirname in os.listdir(base):
                path = os.path.join(base, dirname)
                if os.path.isdir(path):
                    dir_paths.add(path)

    if os.path.isdir(data_dir):
        dir_paths.add(data_dir)
        for dirname in os.listdir(data_dir):
            path = os.path.join(data_dir, dirname)
            if os.path.isdir(path):
                dir_paths.add(path)

    image_dir_dict = {}
    for path in sorted(dir_paths):
        try:
            key = parse_image_dir_path(path)
            if criteria is None or criteria(*key):
                image_dir_dict[key] = path
        except ValueError:
            warnings.warn(f"Found invalid image directory {path}, skipping")

    return image_dir_dict


def parse_json_path(path: str):
    result_root = os.environ.get("RESULT_DIR")
    if os.path.commonpath([result_root, str(path)]) != os.path.commonpath([result_root]):
        raise ValueError(f"JSON files should be under the result directory {os.environ.get('RESULT_DIR')}")
    if not str(path).endswith(".json"):
        raise ValueError("Invalid JSON file path, must end with .json")

    path_parts = os.path.normpath(str(path)).split(os.sep)
    dataset_name, filename = path_parts[-2:]
    if dataset_name not in _dataset_names():
        raise ValueError(f"Dataset name must be one of {_dataset_names()}, found {dataset_name}")

    if filename.count("-") == 1:
        attack_name, attack_strength, source_name, result_type = (
            None,
            None,
            *str(filename[:-5]).split("-"),
        )
    elif filename.count("-") == 3:
        attack_name, attack_strength, source_name, result_type = str(filename[:-5]).split("-")
        try:
            attack_strength = float(attack_strength)
            if attack_strength <= 0:
                raise ValueError("Attack strength must be positive")
        except ValueError:
            raise ValueError("Attack strength must be a number")
    else:
        raise ValueError(
            f"Invalid JSON file name {filename}, must be in the format of 'source_name-result_type.json' or 'attack_name-attack_strength-source_name-result_type.json'"
        )

    if result_type not in ["status", "reverse", "decode", "metric"]:
        raise ValueError("Invalid result type, must be one of ['status', 'reverse', 'decode', 'metric']")

    if source_name is not None and source_name not in _valid_sources():
        raise ValueError(f"Source name must be one of {_valid_sources()}")

    return dataset_name, attack_name, attack_strength, source_name, result_type


def get_all_json_paths(criteria=None):
    if criteria is not None and not callable(criteria):
        raise ValueError("criteria must be a callable function")

    json_paths: list[str] = []
    result_root = os.environ.get("RESULT_DIR")
    for dataset_name in _dataset_names():
        dataset_dir = os.path.join(result_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        for filename in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, filename)
            if os.path.isfile(path):
                json_paths.append(path)

    json_dict = {}
    for path in json_paths:
        try:
            key = parse_json_path(path)
            if criteria is None or criteria(*key):
                json_dict[key] = path
        except ValueError as e:
            if not path.endswith("prompts.json"):
                warnings.warn(f"Found invalid JSON file {path}, {e}, skipping")

    return json_dict
