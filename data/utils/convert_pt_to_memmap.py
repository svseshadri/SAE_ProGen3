import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


INPUT_DIR = Path("data/embeddings")
OUTPUT_DIR = Path("data/embeddings_memmap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": INPUT_DIR / "s1a70_train_layer6_raw.pt",
    "val": INPUT_DIR / "s1a70_val_layer6_raw.pt",
    "test": INPUT_DIR / "s1a70_test_layer6_raw.pt",
}

CHUNK_ROWS = 200_000  # tune if needed


def convert_one(split_name: str, pt_path: Path, output_dir: Path) -> None:
    print(f"\nLoading {split_name} from {pt_path}")
    obj = torch.load(pt_path, map_location="cpu")

    acts = obj["activations"]
    if not torch.is_tensor(acts):
        raise TypeError(f"'activations' in {pt_path} is not a torch.Tensor")

    n_rows, d_model = acts.shape
    print(f"{split_name}: shape={acts.shape}, dtype={acts.dtype}")

    if acts.dtype != torch.float16:
        print(f"Converting activations from {acts.dtype} to float16")
        acts = acts.to(torch.float16)

    bin_path = output_dir / f"s1a70_{split_name}_acts.f16.bin"
    meta_path = output_dir / f"s1a70_{split_name}_acts.meta.json"

    mm = np.memmap(
        bin_path,
        dtype=np.float16,
        mode="w+",
        shape=(n_rows, d_model),
    )

    for start in tqdm(range(0, n_rows, CHUNK_ROWS), desc=f"writing {split_name}"):
        end = min(start + CHUNK_ROWS, n_rows)
        chunk = acts[start:end].numpy()
        mm[start:end] = chunk

    mm.flush()
    del mm

    meta = {
        "split": split_name,
        "shape": [int(n_rows), int(d_model)],
        "dtype": "float16",
        "order": "C",
        "source_pt": str(pt_path),
        "hidden_index": int(obj.get("hidden_index", -1)),
        "model_name": obj.get("model_name", None),
        "sequence_column": obj.get("sequence_column", None),
        "split_name": obj.get("split_name", split_name),
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved memmap to {bin_path}")
    print(f"Saved metadata to {meta_path}")


def main():
    for split_name, pt_path in FILES.items():
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing file: {pt_path}")
        convert_one(split_name, pt_path, OUTPUT_DIR)


if __name__ == "__main__":
    main()