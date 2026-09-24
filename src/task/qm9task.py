from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import qqtools as qt
import torch
from qqtools import qdist
from qqtools.plugins import qpipeline as qpp
from torch import Tensor

from src.dataset.qm9 import QM9

proj_root = qt.find_root(__file__, False)

__all__ = ["QM9Task"]

QM9_TARGETS = [
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
    "U0_atom",
    "U_atom",
    "H_atom",
    "G_atom",
    "A",
    "B",
    "C",
]

QM9_TARGETS_LOWER = [t.lower() for t in QM9_TARGETS]


def get_target_idx(target: str) -> str:
    """convert to index number"""
    if isinstance(target, int):
        return target
    elif isinstance(target, str):
        target = target.lower()
        assert target in QM9_TARGETS_LOWER, f"unrecognized target: {target}. Available targets: {QM9_TARGETS}"
        return QM9_TARGETS_LOWER.index(target)


ATOMIC_MASSES = [
    1.0,  # 0-index for placeholder
    1.008,  # H
    4.002602,
    6.94,
    9.0121831,
    10.81,
    12.011,  # C
    14.007,
    15.999,  # O
    18.99840316,
    20.1797,
    22.98976928,
    24.305,
    26.9815385,
    28.085,
    30.973762,
    32.06,
    35.45,
    39.948,
    39.0983,
    40.078,
    44.955908,
    47.867,
    50.9415,
    51.9961,
    54.938044,
    55.845,
    58.933194,
    58.6934,
    63.546,
    65.38,
    69.723,
    72.63,
    74.921595,
    78.971,
    79.904,
    83.798,
    85.4678,
    87.62,
    88.90584,
    91.224,
    92.90637,
    95.95,
    97.90721,
    101.07,
    102.9055,
    106.42,
    107.8682,
    112.414,
    114.818,
    118.71,
    121.76,
    127.6,
    126.90447,
    131.293,
    132.90545196,
    137.327,
    138.90547,
    140.116,
    140.90766,
    144.242,
    144.91276,
    150.36,
    151.964,
    157.25,
    158.92535,
    162.5,
    164.93033,
    167.259,
    168.93422,
    173.054,
    174.9668,
    178.49,
    180.94788,
    183.84,
    186.207,
    190.23,
    192.217,
    195.084,
    196.966569,
    200.592,
    204.38,
    207.2,
    208.9804,
    208.98243,
    209.98715,
    222.01758,
    223.01974,
    226.02541,
    227.02775,
    232.0377,
    231.03588,
    238.02891,
    237.04817,
    244.06421,
    243.06138,
    247.07035,
    247.07031,
    251.07959,
    252.083,
    257.09511,
    258.09843,
    259.101,
    262.11,
    267.122,
    268.126,
    271.134,
    270.133,
    269.1338,
    278.156,
    281.165,
    281.166,
    285.177,
    286.182,
    289.19,
    289.194,
    293.204,
    293.208,
    294.214,
]


class QM9Task(qpp.qTaskBase):
    root_dir = Path(proj_root, "./download/qm9/")
    atomic_masses = torch.tensor(ATOMIC_MASSES, dtype=torch.float32)

    def __init__(self, args):
        self.args = args.copy()
        super().__init__()

        batch_size = args.task.dataloader.batch_size
        eval_batch_size = args.task.dataloader.eval_batch_size or batch_size
        num_workers = args.task.dataloader.num_workers
        pin_memory = args.task.dataloader.pin_memory
        distributed = args.distributed
        target = args.task.target
        standarize = args.task.standarize

        loader_meta = {
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "distributed": distributed,
            "collate_fn": qt.qDictDataset.collate_graph_samples,
        }
        print(f"[qm9task] loader_meta: {loader_meta}")
        tr_dataset, val_dataset, te_dataset = self.prepare_dataset(self.root_dir)
        self.init_loader(tr_dataset, val_dataset, te_dataset, loader_meta)

        self.target = get_target_idx(target)
        self.standarize = standarize
        self.meta = {"target": self.target, "standarize": self.standarize, "norm_factor": self.norm_factor}
        print("[QM9Task]norm_factor:", self.norm_factor)

    @property
    def norm_factor(self):
        return self.train_loader.dataset.get_norm_factor(self.target)

    @staticmethod
    @qdist.ddp_safe
    def prepare_dataset(root_dir):
        feature_type = "one_hot"
        tr_dataset = QM9(root=root_dir, split="train", feature_type=feature_type)
        val_dataset = QM9(root=root_dir, split="valid", feature_type=feature_type)
        te_dataset = QM9(root=root_dir, split="test", feature_type=feature_type)

        print(f"Dataset splits:")
        print(f"{'Train:':<10}{len(tr_dataset)}")
        print(f"{'Val:':<10}{len(val_dataset)}")
        print(f"{'Test:':<10}{len(te_dataset)}")
        return tr_dataset, val_dataset, te_dataset

    def init_loader(self, tr_dataset, val_dataset, te_dataset, loader_meta):
        train_loader, val_loader, test_loader = qpp.prepare_dataloder(
            tr_dataset, val_dataset, te_dataset, **loader_meta
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def batch_forward(self, model, batch_data) -> Dict[str, Tensor]:
        z, pos, batch = batch_data["z"], batch_data["pos"], batch_data["batch"]
        out = model(qt.qData(z=z, pos=pos, batch=batch))
        if self.target == 5:  # r2 special
            pred = self.process_r2(out["pred_atomwise"], z, pos, batch)
        elif self.target == 0:  # mu special
            pred = self.process_mu(out["pred_atomwise"], z, pos, batch)
        else:
            pred = out["pred"]  # (bz,)
        return {"pred": pred}

    def batch_metric(self, out, batch_data) -> Dict[str, Tuple[Tensor, int]]:
        standarize = self.meta["standarize"]
        if standarize:
            mean, std = self.meta["norm_factor"]
            prd = out["pred"] * std + mean
        else:
            prd = out["pred"]
        gt = batch_data.y[:, self.target]
        with torch.no_grad():
            l1_metric = torch.nn.functional.l1_loss(prd, gt)
            l2_metric = torch.nn.functional.mse_loss(prd, gt)
        ret = {"mae": (l1_metric, gt.shape[0]), "mse": (l2_metric, gt.shape[0])}
        return ret

    def batch_loss(self, out, batch_data, loss_fn) -> Dict[str, Tuple[Tensor, int]]:
        standarize = self.meta["standarize"]
        if standarize:
            mean, std = self.meta["norm_factor"]
            prd = out["pred"] * std + mean
        else:
            prd = out["pred"]
        gt = batch_data.y[:, self.target]
        loss = loss_fn(prd, gt)
        return {"loss": (loss, gt.shape[0])}

    def post_metrics_to_value(self, result) -> float:
        return qt.ensure_scala(result["mae"])

    @staticmethod
    def bspm_collate(data_list):
        raise NotImplementedError()

    def pipe_middle_ware(self, pipe: qpp.qPipeline):
        extra_ckp_caches = {
            "norm_factor": self.norm_factor,
            "standarize": self.standarize,
            "target": self.target,
        }
        pipe.regist_extra_ckp_caches(extra_ckp_caches)

    def process_r2(self, atomic_charges, z, pos, batch):
        bz = torch.add(torch.max(batch).to(torch.int64), 1)
        mass = self.atomic_masses[z].view(-1, 1)  # (nA,1)
        mc = qt.scatter(mass * pos, batch, dim=0, dim_size=bz) / qt.scatter(mass, batch, dim=0, dim_size=bz)  # (bz, 3)
        r2 = torch.pow(pos - mc[batch], 2).sum(dim=1, keepdim=True)  # (nA,1)
        r2_expect = r2 * atomic_charges.view(-1, 1)  # (nA, 1)
        # r2_sqrt = torch.norm(pos - mc[batch], dim=1, keepdim=True)  # (nA,1)
        # r2_expect = (r2_sqrt**2) * atomic_charges.view(-1, 1)  # (nA,1)
        assert r2_expect.shape[1] == 1
        r2_expect = qt.scatter(r2_expect, batch, dim=0, dim_size=bz)  # (bz,)
        return r2_expect.view(-1)

    @classmethod
    def process_mu(cls, atomic_charges, z, pos, batch):
        bz = torch.add(torch.max(batch).to(torch.int64), 1)
        dipoe = pos * atomic_charges.view(-1, 1)  # (nA, 3)
        dipoe = qt.scatter(dipoe, batch, dim=0, dim_size=bz)  # (bz,3)
        mu = torch.norm(dipoe, dim=1)  # (bz,)
        return mu

    def to(self, device):
        self.atomic_masses = self.atomic_masses.to(device)
