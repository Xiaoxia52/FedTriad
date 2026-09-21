import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pytest
import torch

torch.set_num_threads(2)


@pytest.fixture
def medical_file(tmp_path):
    def create(name="bloodmnist", channels=3, classes=8, size=28):
        rng = np.random.default_rng(19)
        arrays = {}
        for split, repeats in (("train", 12), ("val", 4), ("test", 4)):
            labels = np.tile(np.arange(classes), repeats).astype(np.uint8)
            shape = (len(labels), size, size) + ((channels,) if channels > 1 else ())
            images = rng.integers(0, 100, size=shape, dtype=np.uint8)
            # Distinct synthetic class signals make client distributions nonidentical.
            images += (labels * 10).reshape((-1,) + (1,) * (len(shape) - 1))
            arrays[split + "_images"] = images
            arrays[split + "_labels"] = labels[:, None]
        path = tmp_path / (name + ".npz")
        np.savez_compressed(path, **arrays)
        return path
    return create
