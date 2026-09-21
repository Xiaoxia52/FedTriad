"""Stream official NPZ arrays to a versioned disk cache; never merge official splits."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from .config import DATASETS
from .io import write_json

SPLITS = ("train", "val", "test")


def array_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    raise ValueError("Unsupported NPY header version: %s" % (version,))


def inspect_npz(path):
    path = Path(path).resolve()
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        for split in SPLITS:
            for suffix in ("images", "labels"):
                key = split + "_" + suffix
                item = archive.getinfo(key + ".npy")
                with archive.open(item) as stream:
                    shape, order, dtype = array_header(stream)
                if dtype.hasobject or order:
                    raise ValueError("Object/Fortran arrays are not supported")
                arrays[key] = {"shape": list(shape), "dtype": str(dtype),
                               "crc32": item.CRC, "uncompressed_bytes": item.file_size}
    return {"file": str(path), "file_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns, "arrays": arrays}


def locate(config):
    if config.data_file:
        path = Path(config.data_file)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path
    root = Path(config.data_root)
    name = config.dataset + ("" if config.image_size == 28 else "_" + str(config.image_size)) + ".npz"
    choices = [root / name, root / config.dataset / name]
    for path in choices:
        if path.is_file():
            return path
    raise FileNotFoundError("Missing %s under %s. Set data_file to the existing NPZ. "
                            "No automatic download or resolution fallback." % (name, root))


def prepare(config):
    source = locate(config)
    metadata = inspect_npz(source)
    channels, classes = DATASETS[config.dataset]
    for split in SPLITS:
        shape = metadata["arrays"][split + "_images"]["shape"]
        if len(shape) not in (3, 4) or shape[1] != shape[2]:
            raise ValueError("Expected square NHW/NHWC images")
        actual_channels = 1 if len(shape) == 3 else shape[3]
        if actual_channels != channels:
            raise ValueError("Channel count does not match dataset")
        if metadata["arrays"][split + "_images"]["dtype"] != "uint8":
            raise ValueError("Official uint8 image arrays required")
        if shape[1] != config.image_size and not config.allow_resize:
            raise ValueError("Source resolution differs. Use allow_resize=true explicitly; "
                             "this is NOT the native MedMNIST+ resolution protocol.")
    metadata.update(dataset=config.dataset, image_size=config.image_size,
                    smoke_samples=config.smoke_samples, cache_version=1,
                    resize="PIL bilinear" if any(metadata["arrays"][s + "_images"]["shape"][1]
                                                 != config.image_size for s in SPLITS) else "none")
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    cache = Path(config.cache_dir) / key[:24]
    cache.mkdir(parents=True, exist_ok=True)
    ready = cache / "manifest.json"
    if ready.exists():
        if json.loads(ready.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Cache identity mismatch")
        for split in SPLITS:
            for kind in ("images", "labels"):
                np.load(cache / (split + "_" + kind + ".npy"), mmap_mode="r", allow_pickle=False)
        return cache, metadata, key
    lock = cache / "prepare.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("Cache preparation is active or interrupted: %s. Check running processes before removing its lock." % lock)
    os.close(fd)
    try:
        with zipfile.ZipFile(source) as archive:
            for split in SPLITS:
                labels = np.load(archive.open(split + "_labels.npy"), allow_pickle=False)
                if labels.ndim not in (1, 2) or (labels.ndim == 2 and labels.shape[1] != 1):
                    raise ValueError("Only single-label multiclass tasks supported")
                labels = labels.reshape(-1)
                if labels.dtype.kind not in "ui" or labels.min() < 0 or labels.max() >= classes:
                    raise ValueError("Invalid class labels")
                shape = metadata["arrays"][split + "_images"]["shape"]
                if len(labels) != shape[0]:
                    raise ValueError("Image/label count mismatch")
                n = min(len(labels), config.smoke_samples) if config.smoke_samples else len(labels)
                np.save(cache / (split + "_labels.npy"), labels[:n].astype(np.int64), allow_pickle=False)
                target = cache / (split + "_images.npy")
                temp = cache / (split + "_images.partial.npy")
                output = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint8,
                                                  shape=(n, config.image_size, config.image_size, channels))
                print("Preparing %s: %s images -> %spx disk cache" % (split, n, config.image_size), flush=True)
                with archive.open(split + "_images.npy") as stream:
                    array_header(stream)
                    stride = int(np.prod(shape[1:]))
                    for start in range(0, n, 128):
                        count = min(128, n - start)
                        raw = stream.read(count * stride)
                        if len(raw) != count * stride:
                            raise ValueError("Truncated image array")
                        chunk = np.frombuffer(raw, dtype=np.uint8).reshape((count,) + tuple(shape[1:]))
                        if shape[1] == config.image_size:
                            output[start:start + count] = chunk.reshape(count, shape[1], shape[2], channels)
                        else:
                            for j, row in enumerate(chunk):
                                if channels == 1:
                                    row = row.reshape(shape[1], shape[2])
                                resized = Image.fromarray(row).resize((config.image_size, config.image_size),
                                                                     Image.Resampling.BILINEAR)
                                output[start + j] = np.asarray(resized).reshape(config.image_size, config.image_size, channels)
                output.flush()
                del output
                os.replace(str(temp), str(target))
        write_json(ready, metadata)
    finally:
        lock.unlink()
    return cache, metadata, key


class Images(Dataset):
    def __init__(self, path, indices, return_indices=False):
        self.path = str(path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.return_indices = bool(return_indices)
        self.labels = np.load(self.path.replace("_images.npy", "_labels.npy"), allow_pickle=False)
        self._images = None

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_images"] = None
        return state

    def __getitem__(self, index):
        if self._images is None:
            self._images = np.load(self.path, mmap_mode="r", allow_pickle=False)
        actual = self.indices[index]
        image = torch.from_numpy(np.array(self._images[actual], copy=True)).permute(2, 0, 1).float()
        sample = (image.div_(127.5).sub_(1.0), int(self.labels[actual]))
        return sample + (int(actual),) if self.return_indices else sample


def loader(cache, split, indices, config, shuffle=False, seed=0, return_indices=False):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(Images(Path(cache) / (split + "_images.npy"), indices, return_indices),
                      batch_size=config.batch_size, shuffle=shuffle, num_workers=config.workers,
                      generator=generator, drop_last=False,
                      pin_memory=config.device.startswith("cuda"))
