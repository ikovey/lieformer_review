from pathlib import Path
from typing import Dict, Tuple

import qqtools as qt
from qqtools.plugins import qpipeline as qpp
from qqtools.plugins.qpipeline.entry_utils.loss import ComboLoss
from qqtools.plugins.qpipeline.task.qtask import qTaskBase as _qTaskBase
from qqtools.torch.ddp import BalancedDistributedSampler
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader

import torch

from src.dataset.oc22 import OC22S2EFDataset, load_profile_cache

if not hasattr(qt, "qTaskBase"):
    qt.qTaskBase = _qTaskBase

proj_root = qt.find_root(__file__, False)
repo_root = Path(__file__).resolve().parents[2]


def _find_lmdbs(base_dir: Path):
    lmdb_paths = sorted(base_dir.glob("*.lmdb"))
    if not lmdb_paths:
        raise FileNotFoundError(f"No LMDB files found under: {base_dir}")
    return lmdb_paths


class OC22Task(qt.qTaskBase):
    root_dir = Path(proj_root, "./download/oc22")
    s2ef_rel_dir = Path("s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total")

    @staticmethod
    def _bytes_to_gib(num_bytes: int) -> float:
        return num_bytes / (1024 ** 3)

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
        self.balance_ddp = bool(task_args.get("balance_ddp", True))
        self.sampler_seed = int(task_args.get("sampler_seed", 777))
        # Keep the qqtools planner explicit in resolved run configs.  The LPT
        # default constructs cost-balanced batches before assigning them to
        # ranks, avoiding the heavy-batch peaks of the legacy V3 ordering.
        self.balance_strategy = str(task_args.get("balance_strategy", "lpt"))
        self.data_profile_alias = task_args.get("data_profile", None)
        self.validation_profile_alias = task_args.get("validation_profile", None)
        self.profile_cache = None
        self.validation_profile_cache = None

        self.with_force = bool(task_args.get("with_force", True))
        self.train_on_free_atoms = bool(task_args.get("train_on_free_atoms", True))
        self.eval_on_free_atoms = bool(task_args.get("eval_on_free_atoms", True))
        self.standarize = bool(task_args.get("standarize", False))
        self.primary_metric = task_args.get("primary_metric", "f_mae" if self.with_force else "e_mae")
        self.ef_weight = self._resolve_ef_weight(task_args)

        root_override = task_args.get("root_dir", None)
        self.dataset_root = Path(root_override) if root_override else self.root_dir
        s2ef_root_override = task_args.get("s2ef_root_dir", None)
        self.s2ef_root = Path(s2ef_root_override) if s2ef_root_override else self.dataset_root / self.s2ef_rel_dir

        tr_dataset, val_dataset, te_dataset = self.prepare_dataset()
        self.init_loader(tr_dataset, val_dataset, te_dataset)

        if self.standarize:
            configured = task_args.get("norm_factor", None)
            if configured is not None:
                self._energy_norm = (float(configured[0]), float(configured[1]))
            else:
                self._energy_norm = OC22S2EFDataset.lmdb_norm_factor

            configured_f = task_args.get("force_norm_factor", None)
            if configured_f is not None:
                self._force_norm = (float(configured_f[0]), float(configured_f[1]))
            else:
                self._force_norm = OC22S2EFDataset.force_norm_factor
        else:
            self._energy_norm = (0.0, 1.0)
            self._force_norm = (0.0, 1.0)

        self.meta = {
            "with_force": self.with_force,
            "train_on_free_atoms": self.train_on_free_atoms,
            "eval_on_free_atoms": self.eval_on_free_atoms,
            "standarize": self.standarize,
            "energy_norm": self._energy_norm,
            "force_norm": self._force_norm,
            "ef_weight": self.ef_weight,
            "primary_metric": self.primary_metric,
        }
        if self.profile_cache is not None:
            self.meta.update(
                {
                    "data_profile_alias": str(self.data_profile_alias),
                    "data_profile_id": self.profile_cache.profile_id,
                    "validation_profile_alias": str(self.validation_profile_alias or self.data_profile_alias),
                    "validation_profile_id": (
                        self.validation_profile_cache.profile_id
                        if self.validation_profile_cache is not None
                        else self.profile_cache.profile_id
                    ),
                    "validation_protocol_id": (
                        self.validation_profile_cache.validation_id
                        if self.validation_profile_cache is not None
                        else self.profile_cache.validation_id
                    ),
                    "profile_cache": str(self.profile_cache.cache_dir),
                    "profile_cache_manifest": self.profile_cache.manifest,
                }
            )
        model_args = args.get("model", {})
        self.profile_memory = bool(model_args.get("profile_memory", False))
        self.profile_memory_freq = max(int(model_args.get("profile_memory_freq", 1)), 1)
        self._profile_step = 0
        print(
            f"[OC22Task] with_force={self.with_force} "
            f"train_on_free_atoms={self.train_on_free_atoms} "
            f"eval_on_free_atoms={self.eval_on_free_atoms} "
            f"energy_norm={self._energy_norm} force_norm={self._force_norm}"
        )
        if self.profile_cache is not None:
            print(
                f"[OC22Task] data_profile={self.data_profile_alias} "
                f"profile_id={self.profile_cache.profile_id} "
                f"validation_profile={self.validation_profile_alias or self.data_profile_alias} "
                f"validation_id={self.validation_profile_cache.validation_id if self.validation_profile_cache else self.profile_cache.validation_id} "
                f"train_samples={len(self.profile_cache.train_natoms)} "
                f"validation_samples={len(self.validation_profile_cache.validation_natoms if self.validation_profile_cache else self.profile_cache.validation_natoms)}"
            )

    def _memory_snapshot(self, device: torch.device):
        return {
            "alloc_gib": self._bytes_to_gib(torch.cuda.memory_allocated(device)),
            "reserved_gib": self._bytes_to_gib(torch.cuda.memory_reserved(device)),
            "peak_gib": self._bytes_to_gib(torch.cuda.max_memory_allocated(device)),
        }

    def _maybe_log_profile(self, model, batch_data, stage: str):
        if not self.profile_memory:
            return
        pos = batch_data["pos"]
        if pos.device.type != "cuda":
            return

        graph_stats = getattr(model, "_last_profile_stats", None)
        mem = self._memory_snapshot(pos.device)
        prefix = (
            f"[profile][{stage}] "
            f"graphs={graph_stats['num_graphs']} real_nodes={graph_stats['num_real_nodes']} "
            f"total_nodes={graph_stats['num_total_nodes']} edges={graph_stats['num_edges']} "
            f"avg_neighbors={graph_stats['avg_neighbors']:.2f} max_neighbors={graph_stats['max_neighbors']} "
            f"dtype={graph_stats['dtype']} "
            f"graph_edge_index={graph_stats.get('graph_edge_index_gib', 0.0):.2f}GiB "
            f"graph_edge_vec={graph_stats.get('graph_edge_vec_gib', 0.0):.2f}GiB "
            f"embed_edge_dist={graph_stats.get('embed_edge_dist_gib', 0.0):.2f}GiB "
            f"embed_rbf={graph_stats.get('embed_rbf_basis_gib', 0.0):.2f}GiB "
            f"embed_edge_decay={graph_stats.get('embed_edge_decay_gib', 0.0):.2f}GiB "
            f"embed_a_lo={graph_stats.get('embed_a_lo_gib', 0.0):.2f}GiB "
            f"embed_e_lo={graph_stats.get('embed_e_lo_gib', 0.0):.2f}GiB "
            f"embed_a_hi={graph_stats.get('embed_a_hi_gib', 0.0):.2f}GiB "
            f"embed_geom={graph_stats.get('embed_node_geom_gib', 0.0):.2f}GiB "
            f"attn_edge_feat={graph_stats.get('attn_edge_feat_gib', 0.0):.2f}GiB "
            f"attn_alpha={graph_stats.get('attn_alpha_gib', 0.0):.2f}GiB "
            f"est_weighted_k={graph_stats['est_weighted_k_gib']:.2f}GiB "
            f"est_source_moment={graph_stats['est_source_moment_gib']:.2f}GiB "
            f"est_context_k={graph_stats['est_context_k_gib']:.2f}GiB "
            f"lieconv_edge_old={graph_stats.get('lieconv_est_edge_weighted_k_old_gib', 0.0):.2f}GiB "
            f"lieconv_edge_new={graph_stats.get('lieconv_est_edge_weighted_k_stream_gib', 0.0):.2f}GiB "
            f"lieconv_src_new={graph_stats.get('lieconv_est_source_moment_stream_gib', 0.0):.2f}GiB "
            f"lieconv_ctx_new={graph_stats.get('lieconv_est_context_stream_gib', 0.0):.2f}GiB "
            f"lieconv_dst_old={graph_stats.get('lieconv_est_dst_flat_context_old_gib', 0.0):.2f}GiB "
            f"lieconv_dst_new={graph_stats.get('lieconv_est_dst_stream_context_gib', 0.0):.2f}GiB "
            if graph_stats is not None
            else f"[profile][{stage}] "
        )
        print(
            prefix
            + f"cuda_alloc={mem['alloc_gib']:.2f}GiB "
            + f"cuda_reserved={mem['reserved_gib']:.2f}GiB "
            + f"cuda_peak={mem['peak_gib']:.2f}GiB"
        )

    def _build_split(self, split: str):
        base_dir = self.s2ef_root / split
        lmdb_paths = _find_lmdbs(base_dir)
        datasets = [OC22S2EFDataset(root=self.dataset_root, lmdb_path=p) for p in lmdb_paths]
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    def _build_profile_split(self, path: Path):
        return OC22S2EFDataset(root=self.dataset_root, lmdb_path=path)

    @qt.qdist.ddp_safe
    def prepare_dataset(self):
        task_args = self.args.task
        if self.data_profile_alias:
            registry_override = task_args.get("data_profile_registry", None)
            registry_path = (
                Path(registry_override)
                if registry_override
                else repo_root / "configs/data_profiles/oc22.yaml"
            )
            self.profile_cache = load_profile_cache(
                self.dataset_root, registry_path, str(self.data_profile_alias)
            )
            self.validation_profile_cache = (
                load_profile_cache(
                    self.dataset_root, registry_path, str(self.validation_profile_alias)
                )
                if self.validation_profile_alias
                else self.profile_cache
            )
            tr = self._build_profile_split(self.profile_cache.train_lmdb)
            val = self._build_profile_split(self.validation_profile_cache.validation_lmdb)
            return tr, val, None

        train_split = task_args.get("train_split", "train")
        val_split = task_args.get("val_split", "val_id")
        test_split = task_args.get("test_split", None)

        tr = self._build_split(train_split)
        val = self._build_split(val_split)
        te = self._build_split(test_split) if test_split else None
        return tr, val, te

    def init_loader(self, tr_dataset, val_dataset, te_dataset):
        meta = {
            "num_workers": self.loader_meta["num_workers"],
            "pin_memory": self.loader_meta["pin_memory"],
            "collate_fn": self.loader_meta["collate_fn"],
        }
        if self.profile_cache is not None and self.balance_ddp:
            validation_cache = self.validation_profile_cache or self.profile_cache
            sampler_rank = int(self.args.rank) if self.args.distributed else None
            sampler_world_size = int(self.args.world_size) if self.args.distributed else None
            train_sampler = BalancedDistributedSampler(
                self.profile_cache.train_natoms,
                batch_size=self.loader_meta["batch_size"],
                rank=sampler_rank,
                world_size=sampler_world_size,
                shuffle=True,
                drop_last=True,
                pad=False,
                seed=self.sampler_seed,
                strategy=self.balance_strategy,
            )
            val_sampler = BalancedDistributedSampler(
                validation_cache.validation_natoms,
                batch_size=self.loader_meta["eval_batch_size"],
                rank=sampler_rank,
                world_size=sampler_world_size,
                shuffle=False,
                drop_last=False,
                seed=self.sampler_seed,
                strategy=self.balance_strategy,
            )
            self.train_loader = DataLoader(
                tr_dataset,
                batch_size=self.loader_meta["batch_size"],
                sampler=train_sampler,
                **meta,
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.loader_meta["eval_batch_size"],
                sampler=val_sampler,
                **meta,
            )
        else:
            self.train_loader = qpp.build_loader(
                tr_dataset, distributed=self.loader_meta["distributed"],
                batch_size=self.loader_meta["batch_size"], shuffle=True, drop_last=True, **meta,
            )
            self.val_loader = qpp.build_loader(
                val_dataset, distributed=self.loader_meta["distributed"],
                batch_size=self.loader_meta["eval_batch_size"], shuffle=False, drop_last=False, **meta,
            )
        self.test_loader = None
        if te_dataset is not None:
            self.test_loader = qpp.build_loader(
                te_dataset, distributed=self.loader_meta["distributed"],
                batch_size=self.loader_meta["eval_batch_size"], shuffle=False, drop_last=False, **meta,
            )

    def _prepare_model_input(self, batch_data, pos):
        payload = {"z": batch_data["z"], "pos": pos, "batch": batch_data["batch"]}
        for key in ("cell", "pbc", "edge_index", "cell_offsets"):
            if key in batch_data:
                payload[key] = batch_data[key]
        return qt.qData(**payload)

    @staticmethod
    def _model_outputs_force(model) -> bool:
        return bool(getattr(model, "produces_force", False))

    def batch_forward(self, model, batch_data) -> Dict[str, Tensor]:
        pos = batch_data["pos"]
        requires_force_fallback = self.with_force and not self._model_outputs_force(model)
        if requires_force_fallback:
            pos = pos.clone().detach().requires_grad_(True)

        should_profile = (
            self.profile_memory
            and pos.device.type == "cuda"
            and self._profile_step % self.profile_memory_freq == 0
        )
        if should_profile:
            torch.cuda.reset_peak_memory_stats(pos.device)

        model_out = model(self._prepare_model_input(batch_data, pos))
        energy = model_out["pred"].view(-1)
        out: Dict[str, Tensor] = {"energy": energy}
        model_force = model_out.get("force", None)
        if model_force is not None:
            out["force"] = model_force
        if should_profile:
            self._maybe_log_profile(model, batch_data, stage="after_forward")

        if requires_force_fallback and "force" not in out:
            force = -torch.autograd.grad(
                energy.sum(), pos,
                create_graph=model.training, retain_graph=model.training,
            )[0]
            out["force"] = force
            if should_profile:
                self._maybe_log_profile(model, batch_data, stage="after_force_grad")
        self._profile_step += 1
        return out

    def batch_metric(self, out, batch_data) -> Dict[str, Tuple[Tensor, int]]:
        e_mean, e_std = self._energy_norm
        e_pred = out["energy"] * e_std + e_mean if self.standarize else out["energy"]
        e_target = batch_data["energy"].view(-1)
        with torch.no_grad():
            e_mae = torch.nn.functional.l1_loss(e_pred, e_target)
        metrics: Dict[str, Tuple[Tensor, int]] = {"e_mae": (e_mae, e_target.shape[0])}

        if self.with_force and "force" in out:
            f_mean, f_std = self._force_norm
            f_pred = out["force"] * f_std + f_mean if self.standarize else out["force"]
            f_target = batch_data.get("force", batch_data.get("forces")).view(-1, 3)
            mask = self._force_mask(batch_data, for_training=False)
            if mask is not None:
                f_pred, f_target = f_pred[mask], f_target[mask]
            with torch.no_grad():
                f_mae = torch.nn.functional.l1_loss(f_pred.view(-1, 3), f_target.view(-1, 3))
            metrics["f_mae"] = (f_mae, int(f_target.numel()))
        return metrics

    @staticmethod
    def _decompose_loss_fn(loss_fn):
        if isinstance(loss_fn, ComboLoss):
            loss_fns = {k: v for k, v in loss_fn.loss_fns.items()}
            weights = dict(loss_fn.loss_weights)
            return loss_fns, weights
        if isinstance(loss_fn, dict):
            return loss_fn, None
        return {"energy": loss_fn, "force": loss_fn}, None

    def batch_loss(self, out, batch_data, loss_fn) -> Dict[str, Tuple[Tensor, int]]:
        loss_fns, combo_weights = self._decompose_loss_fn(loss_fn)

        e_mean, e_std = self._energy_norm
        e_target = batch_data["energy"].view(-1)
        e_pred = out["energy"]
        if self.standarize:
            e_target = (e_target - e_mean) / e_std

        e_loss_fn = loss_fns["energy"]
        e_loss = e_loss_fn(e_pred, e_target)
        losses: Dict[str, Tuple[Tensor, int]] = {"e_loss": (e_loss, e_target.shape[0])}

        if self.with_force and "force" in out:
            f_mean, f_std = self._force_norm
            f_pred = out["force"].view(-1, 3)
            f_target = (batch_data.get("force", batch_data.get("forces"))).view(-1, 3)
            if self.standarize:
                f_target = (f_target - f_mean) / f_std
            mask = self._force_mask(batch_data, for_training=True)
            if mask is not None:
                f_pred, f_target = f_pred[mask], f_target[mask]
            f_loss_fn = loss_fns["force"]
            f_loss = f_loss_fn(f_pred, f_target)
            losses["f_loss"] = (f_loss, int(f_target.numel()))
            if combo_weights is not None:
                ew = combo_weights.get("energy", 1.0)
                fw = combo_weights.get("force", 1.0)
            else:
                ew, fw = self.ef_weight
            losses["loss"] = (ew * e_loss + fw * f_loss, e_target.shape[0])
        else:
            losses["loss"] = losses["e_loss"]
        return losses

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

    def post_metrics_to_value(self, result) -> float:
        if self.primary_metric == "f_mae" and "f_mae" in result:
            return qt.ensure_scala(result["f_mae"])
        if self.primary_metric == "ef_mae" and "f_mae" in result:
            return qt.ensure_scala(0.8 * result["e_mae"] + 0.2 * result["f_mae"])
        return qt.ensure_scala(result["e_mae"])

    def pipe_middle_ware(self, pipe: qpp.qPipeline):
        caches = {
            "with_force": self.with_force,
            "train_on_free_atoms": self.train_on_free_atoms,
            "eval_on_free_atoms": self.eval_on_free_atoms,
            "energy_norm": self._energy_norm,
            "force_norm": self._force_norm,
            "ef_weight": self.ef_weight,
            "primary_metric": self.primary_metric,
        }
        if self.profile_cache is not None:
            validation_cache = self.validation_profile_cache or self.profile_cache
            caches.update(
                {
                    "data_profile_alias": str(self.data_profile_alias),
                    "data_profile_id": self.profile_cache.profile_id,
                    "validation_profile_alias": str(self.validation_profile_alias or self.data_profile_alias),
                    "validation_profile_id": validation_cache.profile_id,
                    "validation_protocol_id": validation_cache.validation_id,
                    "profile_cache_manifest": self.profile_cache.manifest,
                }
            )
        pipe.regist_extra_ckp_caches(caches)
