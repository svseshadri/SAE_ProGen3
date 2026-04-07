from __future__ import annotations

import sys
from pathlib import Path

import optuna
import torch

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from topk_sae.dataset.memmap_dataset import ActivationMemmap
from topk_sae.train_topk_sae import TopKSAE, train_sae


DATA_DIR = repo_root / "data" / "embeddings_memmap" / "layer6"
OUT_DIR = repo_root / "outputs" / "sae_sweeps" / "layer6"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def make_datasets():
    train_data = ActivationMemmap(
        DATA_DIR / "s1a70_train_acts.f16.bin",
        DATA_DIR / "s1a70_train_acts.meta.json",
    )
    val_data = ActivationMemmap(
        DATA_DIR / "s1a70_val_acts.f16.bin",
        DATA_DIR / "s1a70_val_acts.meta.json",
    )
    return train_data, val_data


def objective(trial: optuna.Trial) -> float:
    train_data, val_data = make_datasets()

    d_in = 384
    d_sae = trial.suggest_categorical("d_sae", [8192, 16384, 32768])
    k = trial.suggest_categorical("k", [16, 32, 64])
    lr = trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3])
    weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-5])
    batch_size = trial.suggest_categorical("batch_size", [4096, 8192, 16384])
    steps = trial.suggest_categorical("steps", [5000, 10000])
    normalize_inputs = trial.suggest_categorical("normalize_inputs", [True])

    model = TopKSAE(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        normalize_decoder=True,
    )

    trial_dir = OUT_DIR / f"trial_{trial.number:04d}"
    result = train_sae(
        model=model,
        train_data=train_data,
        val_data=val_data,
        device=DEVICE,
        output_dir=trial_dir,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        eval_every=1000,
        val_batches=32,
        normalize_inputs=normalize_inputs,
        stats_batches=64,
        seed=42 + trial.number,
    )

    best_metrics = result["history"][-1] if len(result["history"]) > 0 else {}
    dead_frac = best_metrics.get("dead_latent_fraction", 1.0)

    objective_value = result["best_val_nmse"] + 0.05 * dead_frac

    trial.set_user_attr("best_ckpt_path", result["best_ckpt_path"])
    trial.set_user_attr("dead_latent_fraction", dead_frac)

    return objective_value


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="layer6_topk_sae",
        direction="minimize",
        storage=f"sqlite:///{(OUT_DIR / 'optuna.db').as_posix()}",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    print(study.best_trial.number)
    print(study.best_trial.value)
    print(study.best_trial.params)


if __name__ == "__main__":
    main()