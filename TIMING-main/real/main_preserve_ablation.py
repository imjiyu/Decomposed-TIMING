import sys
from os import path
from pathlib import Path
print(path.dirname( path.dirname( path.abspath(__file__) ) ))
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))

import multiprocessing as mp
import numpy as np
import random
import torch as th
import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List

from datasets.PAM import PAM
from datasets.boiler import Boiler
from datasets.epilepsy import Epilepsy
from datasets.wafer import Wafer
from datasets.freezer import Freezer
from tint.metrics import (
    accuracy,
    comprehensiveness,
    cross_entropy,
    log_odds,
    sufficiency,
)
from real.cumulative_difference import cumulative_difference
from real.classifier import MimicClassifierNet
from torch.utils.data import DataLoader, TensorDataset


def run_eval(
    classifier, x_test, mask_test, timesteps, test_loader,
    attr, areas, top, testbs, device, output_file, fp, lock,
    seed, fold, lambda_1, lambda_2, lambda_3,
    largest, mode_tag,   # mode_tag: "CPD" or "CPP"
    save_dir,
):
    """largest=True → CPD, largest=False → CPP. mode_tag는 npy 저장 경로 구분용."""
    os.makedirs(save_dir, exist_ok=True)
    for baselines in [0.0]:
        for topk in areas:
            for k, v in attr.items():
                cum_diff, AUCC, cum_50_diff, pred_diff = cumulative_difference(
                    classifier,
                    x_test,
                    attributions=v.cpu(),
                    baselines=baselines,
                    topk=topk,
                    top=top,
                    testbs=testbs,
                    largest=largest,
                    additional_forward_args=(mask_test, timesteps, False),
                )
                np.save(f'{save_dir}/{k}_result_{fold}_{seed}.npy', pred_diff)

                total_acc = total_comp = total_ce = total_lodds = total_suff = 0.0
                total_samples = 0
                for batch_idx, batch in enumerate(test_loader):
                    x_batch = batch[0].to(device)
                    data_mask_batch = batch[1].to(device)
                    batch_size = x_batch.shape[0]
                    timesteps_batch = timesteps[batch_idx * batch_size: batch_idx * batch_size + batch_size]
                    baselines_batch = baselines
                    v_batch = v[batch_idx * batch_size: batch_idx * batch_size + batch_size].to(device)

                    total_acc   += accuracy(classifier, x_batch, attributions=v_batch, baselines=baselines_batch, topk=topk, additional_forward_args=(data_mask_batch, timesteps_batch, False)) * batch_size
                    total_comp  += comprehensiveness(classifier, x_batch, attributions=v_batch, baselines=baselines_batch, topk=topk, additional_forward_args=(data_mask_batch, timesteps_batch, False)) * batch_size
                    total_ce    += cross_entropy(classifier, x_batch, attributions=v_batch, baselines=baselines_batch, topk=topk, additional_forward_args=(data_mask_batch, timesteps_batch, False)) * batch_size
                    total_lodds += log_odds(classifier, x_batch, attributions=v_batch, baselines=baselines_batch, topk=topk, additional_forward_args=(data_mask_batch, timesteps_batch, False)) * batch_size
                    total_suff  += sufficiency(classifier, x_batch, attributions=v_batch, baselines=baselines_batch, topk=topk, additional_forward_args=(data_mask_batch, timesteps_batch, False)) * batch_size
                    total_samples += batch_size

                fp.write(f"{seed},{fold},zeros,{topk},{k}_{mode_tag},{lambda_1},{lambda_2},{lambda_3},"
                         f"{cum_50_diff:.4},{cum_diff:.4},{AUCC:.4},"
                         f"{total_acc/total_samples:.4},{total_comp/total_samples:.4},"
                         f"{total_ce/total_samples:.4},{total_lodds/total_samples:.4},"
                         f"{total_suff/total_samples:.4}\n")
                print(f"[{mode_tag}] {k} topk={topk} done")


def main(
    explainers: List[str],
    data: str,
    areas: list,
    device: str = "cpu",
    fold: int = 0,
    seed: int = 42,
    is_train: bool = False,
    deterministic: bool = False,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    lambda_3: float = 1.0,
    num_segments: int = 50,
    min_seg_len: int = 1,
    max_seg_len: int = 48,
    mask_lr: float = 0.1,
    output_file: str = "results_ablation.csv",
    model_type: str = "state",
    testbs: int = 30,
    top: int = 0,
    skip_train_motif: bool = True,
    skip_train_timex: bool = True,
    prob: float = 0.1,
):
    if deterministic:
        seed_everything(seed=seed, workers=True)

    accelerator = device.split(":")[0]
    device_id = [int(device.split(":")[1])] if len(device.split(":")) > 1 else 1
    lock = mp.Lock()

    # ── 데이터셋 & 모델 설정 ──────────────────────────────────────────────
    if data == "PAM":
        datamodule = PAM(fold=fold, seed=seed)
        classifier = MimicClassifierNet(feature_size=17, n_state=8, n_timesteps=600, hidden_size=200, regres=True, loss="cross_entropy", lr=0.0001, l2=1e-3, model_type=model_type)
    elif data == "boiler":
        datamodule = Boiler(fold=fold, seed=seed)
        classifier = MimicClassifierNet(feature_size=20, n_state=2, n_timesteps=36,  hidden_size=200, regres=True, loss="cross_entropy", lr=0.0001, l2=1e-3, model_type=model_type)
    elif data == "epilepsy":
        datamodule = Epilepsy(fold=fold, seed=seed)
        classifier = MimicClassifierNet(feature_size=1,  n_state=2, n_timesteps=178, hidden_size=200, regres=True, loss="cross_entropy", lr=0.0001, l2=1e-3, model_type=model_type)
    elif data == "freezer":
        datamodule = Freezer(n_folds=5, fold=fold, seed=seed)
        classifier = MimicClassifierNet(feature_size=1,  n_state=2, n_timesteps=301, hidden_size=200, regres=True, loss="cross_entropy", lr=0.0001, l2=1e-3, model_type=model_type)
    elif data == "wafer":
        datamodule = Wafer(n_folds=5, fold=fold, seed=seed)
        classifier = MimicClassifierNet(feature_size=1,  n_state=2, n_timesteps=152, hidden_size=200, regres=True, loss="cross_entropy", lr=0.0001, l2=1e-3, model_type=model_type)
    else:
        raise ValueError(f"Unknown data: {data}")

    trainer = Trainer(
        max_epochs=100, accelerator=accelerator, devices=device_id,
        deterministic=deterministic,
        logger=TensorBoardLogger(save_dir=".", version=random.getrandbits(128)),
    )
    if is_train:
        trainer.fit(classifier, datamodule=datamodule)
        os.makedirs(f"./model/{data}", exist_ok=True)
        th.save(classifier.state_dict(), f"./model/{data}/{model_type}_classifier_{fold}_{seed}_no_imputation")
    else:
        classifier.load_state_dict(th.load(f"./model/{data}/{model_type}_classifier_{fold}_{seed}_no_imputation"))

    with lock:
        x_train   = datamodule.preprocess(split="train")["x"].to(device)
        x_test    = datamodule.preprocess(split="test")["x"].to(device)
        mask_test = datamodule.preprocess(split="test")["mask"].to(device)

    classifier.eval().to(device)
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    data_len, t_len, _ = x_test.shape
    timesteps = th.linspace(0, 1, t_len, device=x_test.device).unsqueeze(0).repeat(data_len, 1)
    test_loader = DataLoader(TensorDataset(x_test, mask_test), batch_size=testbs, shuffle=False)

    # ── attribution 로드: combined(|T+R|) vs |T|+|R| ─────────────────────
    seg = f"kalman_seg{num_segments}_min{min_seg_len}_max{max_seg_len}"
    keys = {
        f"timing_td_combined_{seg}":   f"./results_our/{data}_{model_type}_timing_td_combined_{seg}_result_{fold}_{seed}.npy",
        f"timing_td_T_plus_R_{seg}":   f"./results_our/{data}_{model_type}_timing_td_T_plus_R_{seg}_result_{fold}_{seed}.npy",
    }
    attr = {}
    for k, fpath in keys.items():
        attr[k] = th.Tensor(np.load(fpath)).to(device)
        print(f"loaded {k}  shape={attr[k].shape}")

    # ── CPD + CPP 순차 실행, 결과는 method 열에 _CPD / _CPP 태그로 구분 ──
    os.makedirs(path.dirname(output_file) if path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "a") as fp, lock:
        # CPD (largest=True)
        run_eval(
            classifier, x_test, mask_test, timesteps, test_loader,
            attr, areas, top, testbs, device, output_file, fp, lock,
            seed, fold, lambda_1, lambda_2, lambda_3,
            largest=True, mode_tag="CPD",
            save_dir="./results_ablation/cpd",
        )
        # CPP (largest=False)
        run_eval(
            classifier, x_test, mask_test, timesteps, test_loader,
            attr, areas, top, testbs, device, output_file, fp, lock,
            seed, fold, lambda_1, lambda_2, lambda_3,
            largest=False, mode_tag="CPP",
            save_dir="./results_ablation/cpp",
        )

    print(f"[done] {data} fold={fold}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data",        type=str,   default="epilepsy")
    parser.add_argument("--areas",       type=float, nargs="+", default=[0.1])
    parser.add_argument("--device",      type=str,   default="cpu")
    parser.add_argument("--fold",        type=int,   default=0)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--train",       type=bool,  default=False)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--lambda-1",    type=float, default=0.001)
    parser.add_argument("--lambda-2",    type=float, default=0.01)
    parser.add_argument("--lambda-3",    type=float, default=0.01)
    parser.add_argument("--output-file", type=str,   default="results_ablation/results_ablation.csv")
    parser.add_argument("--model_type",  type=str,   default="state")
    parser.add_argument("--testbs",      type=int,   default=30)
    parser.add_argument("--top",         type=int,   default=0)
    parser.add_argument("--num_segments",  type=int, default=50)
    parser.add_argument("--min_seg_len",   type=int, default=1)
    parser.add_argument("--max_seg_len",   type=int, default=48)
    parser.add_argument("--mask_lr",     type=float, default=0.01)
    parser.add_argument("--prob",        type=float, default=0.1)
    parser.add_argument("--skip_train_motif",  action="store_true")
    parser.add_argument("--skip_train_timex",  action="store_true")
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(
        explainers=[],
        data=args.data,
        areas=args.areas,
        device=args.device,
        fold=args.fold,
        seed=args.seed,
        is_train=args.train,
        deterministic=args.deterministic,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_3=args.lambda_3,
        num_segments=args.num_segments,
        min_seg_len=args.min_seg_len,
        max_seg_len=args.max_seg_len,
        mask_lr=args.mask_lr,
        output_file=args.output_file,
        model_type=args.model_type,
        testbs=args.testbs,
        top=args.top,
        skip_train_motif=args.skip_train_motif,
        skip_train_timex=args.skip_train_timex,
        prob=args.prob,
    )