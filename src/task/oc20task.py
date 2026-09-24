from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import qqtools as qt
import torch
from qqtools.plugins import qpipeline as qpp
from qqtools.plugins.qpipeline.task.qtask import qTaskBase as _qTaskBase
from torch import Tensor
from torch.utils.data import ConcatDataset

from src.dataset.oc20 import OC20IS2REDataset, OC20S2EFDataset

if not hasattr(qt, "qTaskBase"):
    qt.qTaskBase = _qTaskBase

proj_root = qt.find_root(__file__, False)

S2EF_TRAIN_SUBDIRS = {
    "200k": "raw/s2ef/200k/train",
    "2m": "raw/s2ef/2M/train",
    "20m": "raw/s2ef/20M/train",
    "all": "raw/s2ef/all/train",
}

S2EF_VAL_SUBDIRS = {
    "val_id": "raw/s2ef/all/val_id",
    "val_cat": "raw/s2ef/all/val_ood_cat",
    "val_ads": "raw/s2ef/all/val_ood_ads",
    "val_both": "raw/s2ef/all/val_ood_both",
    "val_ood_cat": "raw/s2ef/all/val_ood_cat",
    "val_ood_ads": "raw/s2ef/all/val_ood_ads",
    "val_ood_both": "raw/s2ef/all/val_ood_both",
}

IS2RE_SPLIT_ALIASES = {
    "train": "train",
    "val": "val_id",
    "valid": "val_id",
    "val_id": "val_id",
    "val_cat": "val_ood_cat",
    "val_ads": "val_ood_ads",
    "val_both": "val_ood_both",
    "val_ood_cat": "val_ood_cat",
    "val_ood_ads": "val_ood_ads",
    "val_ood_both": "val_ood_both",
    "test": "test_id",
    "test_id": "test_id",
}


def _expand_glob_patterns(root_dir: Path, patterns: Iterable[str]) -> List[Path]:
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(sorted(root_dir.glob(pattern)))
    uniq = []
    seen = set()
    for path in matches:
        if path.suffix != ".lmdb":
            continue
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            uniq.append(path)
    return uniq


class OC20Task(qt.qTaskBase):
    root_dir = Path(proj_root, "./download/oc20")

    @staticmethod
    def _parse_loss_weight(weight_cfg) -> float:
        if isinstance(weight_cfg, (list, tuple)) and len(weight_cfg) == 2:
            return float(weight_cfg[1])
        return 1.0

    def _resolve_ef_weight(self, task_args):
        fallback = task_args.get("ef_weight", [1.0, 1.0])
        if isinstance(fallback, (list, tuple)) and len(fallback) == 2:
            fallback = (float(fallback[0]), float(fallback[1]))
        else:
            fallback = (1.0, 1.0)

        optim_args = self.args.get("optim", {})
        loss_name = optim_args.get("loss", None)
        loss_params = optim_args.get("loss_params", None)
        if not (
            isinstance(loss_name, str)
            and loss_name.lower() in ["comboloss", "combo_loss", "composite", "combination"]
            and isinstance(loss_params, dict)
        ):
            return fallback

        energy_cfg = loss_params.get("energy", None)
        force_cfg = loss_params.get("force", None)
        if energy_cfg is None or force_cfg is None:
            return fallback
        return (self._parse_loss_weight(energy_cfg), self._parse_loss_weight(force_cfg))

    def __init__(self, args):
        self.args = args.copy()
        super().__init__()

        task_args = args.task
        batch_size = task_args.dataloader.batch_size
        eval_batch_size = task_args.dataloader.eval_batch_size or batch_size
        self.loader_meta = {
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "num_workers": task_args.dataloader.num_workers,
            "pin_memory": task_args.dataloader.pin_memory,
            "distributed": args.distributed,
            "collate_fn": qt.qDictDataset.collate_graph_samples,
        }

        self.oc20_mode = task_args.get("oc20_mode", "s2ef").lower()
        self.with_force = bool(task_args.get("with_force", self.oc20_mode == "s2ef"))
        self.train_on_free_atoms = bool(task_args.get("train_on_free_atoms", True))
        self.eval_on_free_atoms = bool(task_args.get("eval_on_free_atoms", True))
        self.standarize = bool(task_args.get("standarize", False))
        self.primary_metric = task_args.get("primary_metric", "f_mae" if self.with_force else "e_mae")
        self.ef_weight = self._resolve_ef_weight(task_args)

        root_override = task_args.get("root_dir", None)
        self.dataset_root = Path(root_override) if root_override is not None else self.root_dir

        print(f"[oc20task] loader_meta: {self.loader_meta}")
        tr_dataset, val_dataset, te_dataset = self.prepare_dataset()
        self.init_loader(tr_dataset, val_dataset, te_dataset)

        if self.standarize:
            configured = task_args.get("norm_factor", None)
            if configured is not None:
                self._norm_factor = (float(configured[0]), float(configured[1]))
            elif self.oc20_mode == "s2ef":
                self._norm_factor = OC20S2EFDataset.lmdb_norm_factor
            elif self.oc20_mode == "is2re":
                self._norm_factor = OC20IS2REDataset.lmdb_norm_factor
            else:
                self._norm_factor = (0.0, 1.0)
        else:
            self._norm_factor = (0.0, 1.0)

        self.meta = {
            "oc20_mode": self.oc20_mode,
            "with_force": self.with_force,
            "train_on_free_atoms": self.train_on_free_atoms,
            "eval_on_free_atoms": self.eval_on_free_atoms,
            "standarize": self.standarize,
            "norm_factor": self._norm_factor,
            "ef_weight": self.ef_weight,
            "primary_metric": self.primary_metric,
        }
        print(
            f"[OC20Task] mode={self.oc20_mode} with_force={self.with_force} "
            f"train_on_free_atoms={self.train_on_free_atoms} "
            f"eval_on_free_atoms={self.eval_on_free_atoms} norm_factor={self._norm_factor}"
        )

    @property
    def norm_factor(self):
        return self._norm_factor

    def _build_s2ef_split(self, split: str, size: str):
        size = size.lower()
        split = split.lower()
        if split == "train":
            subdir = S2EF_TRAIN_SUBDIRS[size]
        else:
            subdir = S2EF_VAL_SUBDIRS[split]
        base_dir = self.dataset_root / subdir
        lmdb_paths = sorted(base_dir.glob("*.lmdb"))
        if not lmdb_paths:
            raise FileNotFoundError(f"No S2EF LMDB files found under: {base_dir}")
        datasets = [OC20S2EFDataset(root=self.dataset_root, lmdb_path=path) for path in lmdb_paths]
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    def _build_is2re_split(self, split: str):
        split = split.lower()
        task_args = self.args.task
        split = IS2RE_SPLIT_ALIASES.get(split, split)
        train_size = str(task_args.get("train_size", "all")).lower()
        pattern_key = f"{split}_glob"
        if pattern_key in task_args and task_args.get(pattern_key):
            lmdb_paths = _expand_glob_patterns(self.dataset_root, [task_args.get(pattern_key)])
        else:
            candidates = [
                f"raw/is2re/{train_size}/{split}/data.lmdb",
                f"raw/is2re/{train_size}/{split}/*.lmdb",
                f"raw/is2re/all/{split}/data.lmdb",
                f"raw/is2re/all/{split}/*.lmdb",
                f"raw/is2re/{split}/data.lmdb",
                f"raw/is2re/{split}/*.lmdb",
            ]
            lmdb_paths = _expand_glob_patterns(self.dataset_root, candidates)
        if not lmdb_paths:
            raise FileNotFoundError(
                f"No IS2RE LMDB files found for split `{split}` under {self.dataset_root / 'raw' / 'is2re'}"
            )
        datasets = [OC20IS2REDataset(root=self.dataset_root, lmdb_path=path) for path in lmdb_paths]
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    @qt.qdist.ddp_safe
    def prepare_dataset(self):
        task_args = self.args.task
        if self.oc20_mode == "s2ef":
            train_size = task_args.get("train_size", "200k")
            val_split = task_args.get("val_split", "val_id")
            test_split = task_args.get("test_split", None)
            tr_dataset = self._build_s2ef_split("train", train_size)
            val_dataset = self._build_s2ef_split(val_split, train_size)
            te_dataset = self._build_s2ef_split(test_split, train_size) if test_split else None
            return tr_dataset, val_dataset, te_dataset

        if self.oc20_mode == "is2re":
            train_split = task_args.get("train_split", "train")
            val_split = task_args.get("val_split", "val")
            test_split = task_args.get("test_split", None)
            tr_dataset = self._build_is2re_split(train_split)
            val_dataset = self._build_is2re_split(val_split)
            te_dataset = self._build_is2re_split(test_split) if test_split else None
            return tr_dataset, val_dataset, te_dataset

        raise ValueError(f"Unsupported OC20 mode: {self.oc20_mode}")

    def init_loader(self, tr_dataset, val_dataset, te_dataset):
        meta = {
            "num_workers": self.loader_meta["num_workers"],
            "pin_memory": self.loader_meta["pin_memory"],
            "collate_fn": self.loader_meta["collate_fn"],
        }
        train_loader = qpp.build_loader(
            tr_dataset,
            distributed=self.loader_meta["distributed"],
            batch_size=self.loader_meta["batch_size"],
            shuffle=True,
            drop_last=True,
            **meta,
        )
        val_loader = qpp.build_loader(
            val_dataset,
            distributed=self.loader_meta["distributed"],
            batch_size=self.loader_meta["eval_batch_size"],
            shuffle=False,
            drop_last=False,
            **meta,
        )
        test_loader = None
        if te_dataset is not None:
            test_loader = qpp.build_loader(
                te_dataset,
                distributed=self.loader_meta["distributed"],
                batch_size=self.loader_meta["eval_batch_size"],
                shuffle=False,
                drop_last=False,
                **meta,
            )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def _prepare_model_input(self, batch_data, pos):
        payload = {
            "z": batch_data["z"],
            "pos": pos,
            "batch": batch_data["batch"],
        }
        for key in ("cell", "pbc", "edge_index", "cell_offsets"):
            if key in batch_data:
                payload[key] = batch_data[key]
        return qt.qData(**payload)

    @staticmethod
    def _resolve_loss_fn(loss_fn, key):
        if isinstance(loss_fn, dict):
            return loss_fn[key]
        return loss_fn

    @staticmethod
    def _free_atom_mask(batch_data):
        if "fixed" not in batch_data:
            return None
        return ~batch_data["fixed"].view(-1).bool()

    def _force_mask(self, batch_data, *, for_training: bool):
        enabled = self.train_on_free_atoms if for_training else self.eval_on_free_atoms
        if not enabled:
            return None
        return self._free_atom_mask(batch_data)

    @staticmethod
    def _model_outputs_force(model) -> bool:
        return bool(getattr(model, "produces_force", False))

    def batch_forward(self, model, batch_data) -> Dict[str, Tensor]:
        pos = batch_data["pos"]
        needs_grad_force = self.with_force and not self._model_outputs_force(model)
        if needs_grad_force:
            pos = pos.clone().detach().requires_grad_(True)

        model_out = model(self._prepare_model_input(batch_data, pos))
        energy = model_out["pred"].view(-1)
        out_force: Optional[Tensor] = model_out.get("force", None)

        if needs_grad_force and out_force is None:
            out_force = -torch.autograd.grad(
                energy.sum(),
                pos,
                create_graph=model.training,
                retain_graph=model.training,
            )[0]

        out = {"energy": energy}
        if out_force is not None:
            out["force"] = out_force
        return out

    def batch_metric(self, out, batch_data) -> Dict[str, Tuple[Tensor, int]]:
        mean, std = self.norm_factor
        e_pred = out["energy"] * std + mean if self.standarize else out["energy"]
        e_target = batch_data["energy"].view(-1)
        with torch.no_grad():
            e_mae = torch.nn.functional.l1_loss(e_pred, e_target)
        metrics = {"e_mae": (e_mae, e_target.shape[0])}

        if self.with_force and "force" in out:
            mask = self._force_mask(batch_data, for_training=False)
            f_pred = out["force"] * std if self.standarize else out["force"]
            f_target = batch_data["force"] if "force" in batch_data else batch_data["forces"]
            if mask is not None:
                f_pred = f_pred[mask]
                f_target = f_target[mask]
            with torch.no_grad():
                f_mae = torch.nn.functional.l1_loss(f_pred.view(-1, 3), f_target.view(-1, 3))
            metrics["f_mae"] = (f_mae, int(f_target.numel()))

        return metrics

    def batch_loss(self, out, batch_data, loss_fn) -> Dict[str, Tuple[Tensor, int]]:
        mean, std = self.norm_factor
        e_target = batch_data["energy"].view(-1)
        e_pred = out["energy"]
        if self.standarize:
            e_target = (e_target - mean) / std
        e_loss_fn = self._resolve_loss_fn(loss_fn, "energy")
        e_loss = e_loss_fn(e_pred, e_target)
        losses = {"e_loss": (e_loss, e_target.shape[0])}

        if self.with_force and "force" in out:
            f_pred = out["force"].view(-1, 3)
            f_target = (batch_data["force"] if "force" in batch_data else batch_data["forces"]).view(-1, 3)
            if self.standarize:
                f_target = f_target / std
            mask = self._force_mask(batch_data, for_training=True)
            if mask is not None:
                f_pred = f_pred[mask]
                f_target = f_target[mask]
            f_loss_fn = self._resolve_loss_fn(loss_fn, "force")
            f_loss = f_loss_fn(f_pred, f_target)
            losses["f_loss"] = (f_loss, int(f_target.numel()))
            ew, fw = self.ef_weight
            losses["loss"] = (ew * e_loss + fw * f_loss, e_target.shape[0])
        else:
            losses["loss"] = losses["e_loss"]

        return losses

    def post_metrics_to_value(self, result) -> float:
        if self.primary_metric == "f_mae" and "f_mae" in result:
            return qt.ensure_scala(result["f_mae"])
        if self.primary_metric == "ef_mae" and "f_mae" in result:
            return qt.ensure_scala(0.8 * result["e_mae"] + 0.2 * result["f_mae"])
        return qt.ensure_scala(result["e_mae"])

    def pipe_middle_ware(self, pipe: qpp.qPipeline):
        pipe.regist_extra_ckp_caches(
            {
                "oc20_mode": self.oc20_mode,
                "with_force": self.with_force,
                "train_on_free_atoms": self.train_on_free_atoms,
                "eval_on_free_atoms": self.eval_on_free_atoms,
                "norm_factor": self.norm_factor,
                "ef_weight": self.ef_weight,
                "primary_metric": self.primary_metric,
            }
        )
