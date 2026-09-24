"""Standalone qqtools training entry for the SHR multi-head backend."""
import sys
import torch
import qqtools as qt

proj_root = qt.find_root(__file__)
sys.path.insert(0, str(proj_root))

from qqtools.plugins.qpipeline import get_param_stats, prepare_cmd_args
from src.model.lieformer_shr_mh import SHRMultiHeadLieFormer
from src.task.oc22task import OC22Task
from entry import MyPipeline


def prepare_model(args):
    model_name = str(args.model.get("name", "lieformer_shr_mh")).lower()
    if model_name != "lieformer_shr_mh":
        raise ValueError(f"Unsupported SHR model: {model_name}")
    return SHRMultiHeadLieFormer.from_config(args.model)


class SHRTrainPipeline(MyPipeline):
    prepare_model = staticmethod(prepare_model)
    prepare_task = staticmethod(lambda args: OC22Task(args))


def train(args):
    qt.freeze_rand(args.seed)
    model = prepare_model(args)
    qt.qdist.main_print(f"SHR model parameters: {get_param_stats(model)}")
    SHRTrainPipeline(args, mode="train", model=model).fit()


def infer(args):
    model = prepare_model(args)
    model.eval()
    task = OC22Task(args)
    SHRTrainPipeline(args, mode="test", model=model, task=task).infer(task.test_loader or task.val_loader)


if __name__ == "__main__":
    args = prepare_cmd_args()
    if args.test:
        infer(args)
    else:
        train(args)
