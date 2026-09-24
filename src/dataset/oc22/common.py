import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import lmdb
import qqtools as qt
import torch

# Real OC22 LMDB samples were verified against the mounted dataset under
# `download/oc22/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total/...`.
# After `pickle.loads(blob)`, each entry is a `torch_geometric.data.data.Data`
# object whose payload lives directly in `sample.__dict__`, not in `_store`.
# Observed keys on train/val/test samples include:
# `atomic_numbers`, `cell`, `edge_attr`, `edge_index`, `face`, `fid`,
# `fixed`, `id`, `nads`, `natoms`, `normal`, `oc22`, `pos`, `sid`, `tags`,
# `x`, `y`; train/val samples also carry `force`.
# In the mounted OC22 release, all observed splits use `y` as the energy field;
# there is no raw `energy` field. Public `test_*` samples still contain `y`,
# but do not contain `force`, so force-supervised code should only be used on
# splits that actually provide forces. This matches the official evaluation
# flow where test predictions are exported for benchmark submission.
DEFAULT_OC22_PBC = torch.tensor([True, True, True], dtype=torch.bool)


def load_raw_oc22_sample(blob: bytes) -> Dict:
    # Real OC22 LMDB payloads are pickled PyG Data objects whose sample fields
    # are stored directly on the instance dict.
    return pickle.loads(blob).__dict__


def to_tensor(value, *, dtype=None):
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.detach().clone()
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def scalar_from_sample(raw_sample: Dict, keys: Iterable[str], default=None):
    for key in keys:
        if key not in raw_sample:
            continue
        value = raw_sample[key]
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.item()
            continue
        return value
    return default


class OC22LmdbDataset(qt.qDictDataset):
    def __init__(self, root, lmdb_path, cache_in_memory=False):
        self.lmdb_path = Path(lmdb_path)
        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"Missing LMDB file: {self.lmdb_path}")

        self.cache_in_memory = cache_in_memory
        self._cache = {}
        self._env = None
        self._txn = None
        self._length = None
        super().__init__(root=root)
        self._open_env()

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return []

    def download(self):
        return None

    def process(self):
        return None

    def _open_env(self):
        if self._env is not None:
            return
        self._env = lmdb.open(
            str(self.lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        self._txn = self._env.begin(write=False)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._txn = None

    def __del__(self):
        self.close()

    def len(self):
        if self._length is not None:
            return self._length
        blob = self._txn.get(b"length")
        if blob is not None:
            self._length = int(pickle.loads(blob))
        else:
            self._length = self._infer_length_without_metadata()
        return self._length

    def get(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        blob = self._txn.get(str(idx).encode("ascii"))
        if blob is None:
            blob = self._txn.get(str(idx).encode("utf-8"))
        if blob is None:
            raise IndexError(f"Sample index out of range: {idx} for {self.lmdb_path}")

        sample = self.normalize_raw_sample(load_raw_oc22_sample(blob))
        sample = self.postprocess_sample(sample)
        if self.cache_in_memory:
            self._cache[idx] = sample
        return sample

    @staticmethod
    def normalize_raw_sample(raw_sample: Dict) -> Dict:
        raise NotImplementedError

    @staticmethod
    def postprocess_sample(sample: Dict) -> Dict:
        return sample

    def _infer_length_without_metadata(self) -> int:
        cursor = self._txn.cursor()
        count = 0
        for key, _ in cursor:
            if key.isdigit():
                count += 1
        return count


def normalize_common_oc22_sample(
    raw_sample: Dict,
    *,
    include_force: bool,
    energy_keys: Optional[Iterable[str]] = None,
) -> Dict:
    pos = to_tensor(raw_sample["pos"], dtype=torch.float32)
    cell = to_tensor(raw_sample.get("cell"), dtype=torch.float32)
    if cell is not None and cell.dim() == 3 and cell.shape[0] == 1:
        cell = cell[0]

    atomic_numbers = to_tensor(raw_sample.get("atomic_numbers", raw_sample.get("z")), dtype=torch.long)
    if atomic_numbers is None:
        raise KeyError("OC22 sample is missing `atomic_numbers` / `z`.")

    natoms = int(raw_sample.get("natoms", atomic_numbers.shape[0]))
    tags = to_tensor(raw_sample.get("tags"), dtype=torch.long)
    if tags is None:
        tags = torch.zeros(natoms, dtype=torch.long)

    fixed = to_tensor(raw_sample.get("fixed"), dtype=torch.bool)
    if fixed is None:
        fixed = torch.zeros(natoms, dtype=torch.bool)
    else:
        fixed = fixed.bool()

    pbc = to_tensor(raw_sample.get("pbc"), dtype=torch.bool)
    if pbc is None:
        pbc = DEFAULT_OC22_PBC.clone()
    elif pbc.dim() == 2 and pbc.shape[0] == 1:
        pbc = pbc[0]

    if energy_keys is None:
        energy_keys = ("energy", "y", "target")
    energy = scalar_from_sample(raw_sample, energy_keys)
    if energy is None:
        raise KeyError("OC22 sample is missing an energy target field.")
    energy = torch.tensor(float(energy), dtype=torch.float32)

    sample = {
        "num_nodes": natoms,
        "natoms": natoms,
        "pos": pos,
        "cell": cell if cell is not None else torch.zeros(3, 3, dtype=torch.float32),
        "z": atomic_numbers,
        "atomic_numbers": atomic_numbers,
        "y": energy,
        "energy": energy,
        "fixed": fixed,
        "nfreeatoms": int((~fixed).sum().item()),
        "tags": tags,
        "pbc": pbc,
        "sid": int(scalar_from_sample(raw_sample, ("sid",), default=-1)),
        "fid": int(scalar_from_sample(raw_sample, ("fid",), default=-1)),
    }

    if include_force:
        forces = to_tensor(raw_sample.get("forces", raw_sample.get("force")), dtype=torch.float32)
        if forces is None:
            raise KeyError("OC22 S2EF sample is missing `forces` / `force`.")
        sample["forces"] = forces
        sample["force"] = forces

    edge_index = to_tensor(raw_sample.get("edge_index"), dtype=torch.long)
    if edge_index is not None:
        sample["edge_index"] = edge_index

    cell_offsets = to_tensor(raw_sample.get("cell_offsets", raw_sample.get("unit_shifts")), dtype=torch.long)
    if cell_offsets is not None:
        sample["cell_offsets"] = cell_offsets

    edge_distance_vec = to_tensor(raw_sample.get("edge_distance_vec"), dtype=torch.float32)
    if edge_distance_vec is not None:
        sample["edge_distance_vec"] = edge_distance_vec

    ref_energy = scalar_from_sample(raw_sample, ("ref_energy",), default=None)
    if ref_energy is not None:
        sample["ref_energy"] = torch.tensor(float(ref_energy), dtype=torch.float32)

    y_init = scalar_from_sample(raw_sample, ("y_init",), default=None)
    if y_init is not None:
        sample["y_init"] = torch.tensor(float(y_init), dtype=torch.float32)

    y_relaxed = scalar_from_sample(raw_sample, ("y_relaxed",), default=None)
    if y_relaxed is not None:
        sample["y_relaxed"] = torch.tensor(float(y_relaxed), dtype=torch.float32)

    pos_relaxed = to_tensor(raw_sample.get("pos_relaxed"), dtype=torch.float32)
    if pos_relaxed is not None:
        sample["pos_relaxed"] = pos_relaxed

    return sample
