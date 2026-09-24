import sys

import qqtools as qt
import torch

# for absolute import
proj_root = qt.find_root(__file__)
sys.path.insert(0, str(proj_root))

from qqtools.plugins.qpipeline import get_param_stats, prepare_cmd_args, qPipeline
from qqtools.plugins.qpipeline.qpipeline import prepare_logdir

from src.model.lieformer_shr_mh import SHRMultiHeadLieFormer
from src.task.oc20task import OC20Task
from src.task.oc22task import OC22Task
from src.task.qm9task import QM9Task


def prepare_model(args):
    model_name = getattr(args.model, "name", "lieformer_shr_mh").lower()
    if model_name != "lieformer_shr_mh":
        raise ValueError(f"Unsupported model: {model_name}")
    return SHRMultiHeadLieFormer.from_config(args.model)


def prepare_task(args):
    ds_name = args.task["dataset"].lower()
    if ds_name == "qm9":
        task = QM9Task(args)
    elif ds_name == "oc20":
        task = OC20Task(args)
    elif ds_name == "oc22":
        task = OC22Task(args)
    else:
        raise NotImplementedError
    return task


class MyPipeline(qPipeline):
    prepare_model = staticmethod(prepare_model)
    prepare_task = staticmethod(prepare_task)

    def _place_model(self, model):
        placed_model = super()._place_model(model)
        compile_mode = str(
            self.args.get(
                "compile_mode",
                self.args.model.get("compile_mode", "none"),
            )
        ).lower()
        if compile_mode in ("local", "submodules"):
            target_model = (
                placed_model.module
                if isinstance(placed_model, torch.nn.parallel.DistributedDataParallel)
                else placed_model
            )
            target_model.enable_compile(compile_mode)
            qt.qdist.main_print("Enabled local torch.compile for LieFormer submodules")
        return placed_model

    @staticmethod
    def prepare_env(args):
        if bool(args.get("tf32", False)):
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif str(args.model.get("name", "")).startswith("lieformer_shr"):
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        if args.ddp_detect:
            qt.qdist.init_distributed_mode(args)
        else:
            args.distributed = False
            args.rank = 0
            args.local_rank = 0

        qt.freeze_rand(args.seed)
        prepare_logdir(args)

        args.device = qt.parse_device(args.local_rank)
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            qt.qdist.main_print("Enabled TF32 matmul" if bool(args.get("tf32", False)) else "Using default matmul precision")
        qt.qdist.main_print(f"Set device to: {args.device}")


def train(args):

    if str(args.model.get("name", "")).startswith("lieformer_shr"):
        qt.freeze_rand(args.seed)
    model = prepare_model(args)
    print("model", get_param_stats(model))

    pipe = MyPipeline(args, mode="train", model=model)

    pipe.fit()


def infer(args):
    model = prepare_model(args)
    model.eval()

    task = prepare_task(args)

    pipe = MyPipeline(args, mode="test", model=model, task=task)

    dataloader = task.test_loader
    pipe.infer(dataloader)


if __name__ == "__main__":
    args = prepare_cmd_args()
    if args.test:
        infer(args)
    else:
        train(args)
