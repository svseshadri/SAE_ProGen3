import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class ActivationMemmap:
    def __init__(self, bin_path: str | Path, meta_path: str | Path):
        self.bin_path = Path(bin_path)
        self.meta_path = Path(meta_path)

        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)

        self.shape = tuple(self.meta["shape"])
        self.dtype = np.float16 if self.meta["dtype"] == "float16" else np.float32

        self._arr = np.memmap(
            self.bin_path,
            dtype=self.dtype,
            mode="r",
            shape=self.shape,
            order="C",
        )

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def dim(self) -> int:
        return self.shape[1]

    def get_rows(self, indices: np.ndarray) -> np.ndarray:
        return self._arr[indices]

    def sample_torch_batch(
        self,
        batch_size: int,
        device: str = "cpu",
        generator: Optional[torch.Generator] = None,
        out_dtype: torch.dtype = torch.float32,
        pin_memory: bool = False,
    ) -> torch.Tensor:
        if generator is None:
            idx = torch.randint(0, len(self), (batch_size,))
        else:
            idx = torch.randint(0, len(self), (batch_size,), generator=generator)

        idx_np = idx.numpy()
        batch_np = self._arr[idx_np]

        batch = torch.from_numpy(np.asarray(batch_np))
        batch = batch.to(out_dtype)

        if pin_memory:
            batch = batch.pin_memory()

        return batch.to(device, non_blocking=True)