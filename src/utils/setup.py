import json
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_classes():
    config = _load_config()
    classes = config.get("classes", [])
    if not isinstance(classes, list) or len(classes) == 0:
        raise ValueError(
            'Specify classes in src/config.json, e.g. {"classes": ["hello", "thank you"]}'
        )
    return classes


def get_colors():
    config = _load_config()
    classes = config.get("classes", [])
    colors = config.get("colors", [])
    if not isinstance(colors, list) or len(colors) == 0:
        raise ValueError(
            'Specify colors in src/config.json, e.g. {"colors": [[131, 193, 103], [240, 172, 95]]}'
        )
    if len(classes) != len(colors):
        raise ValueError(
            f"Please specify one color per class. Found {len(colors)} colors for {len(classes)} classes."
        )
    return colors


def get_dataset_root():
    config = _load_config()
    dataset_root = config.get("dataset_root", "data")
    if not isinstance(dataset_root, str) or len(dataset_root.strip()) == 0:
        raise ValueError('Specify "dataset_root" in src/config.json as a non-empty string, e.g. "data_10class".')
    return dataset_root


def get_train_split():
    config = _load_config()
    train_split = config.get("train_split", 0.85)
    if not isinstance(train_split, (float, int)) or not (0.0 < float(train_split) < 1.0):
        raise ValueError('Specify "train_split" in src/config.json as a float between 0 and 1, e.g. 0.85.')
    return float(train_split)


def get_split_seed():
    config = _load_config()
    split_seed = config.get("split_seed", 42)
    if not isinstance(split_seed, int):
        raise ValueError('Specify "split_seed" in src/config.json as an integer, e.g. 42.')
    return split_seed


if __name__ == "__main__":
    print(get_classes())
