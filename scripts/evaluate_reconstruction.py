from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGEN3_SRC = REPO_ROOT / "progen3" / "src"
if str(PROGEN3_SRC) not in sys.path:
    sys.path.insert(0, str(PROGEN3_SRC))

from progen3.batch_preparer import ProGen3BatchPreparer
from progen3.modeling import ProGen3ForCausalLM
from topk_sae.models.train_topk_sae import TopKSAE


DEFAULT_MODEL_NAME = "Profluent-Bio/progen3-112m"
DEFAULT_SAE_CKPT = "results/topk_sae_layer6_d4096_k32_run1/best.pt"
DEFAULT_DATASET = "data/processed/s1a70/s1a70_test.csv"
DEFAULT_OUTPUT = "results/reconstruction_evaluation.csv"


def relative_nll_degradation_pct(base_nll: float, patched_nll: float) -> float:
    denominator = abs(base_nll)
    if abs(base_nll) < 1e-12:
        return 0.0 if abs(patched_nll) < 1e-12 else float("inf")
    return 100.0 * (patched_nll - base_nll) / denominator


def compute_mean_kl(base_logits: torch.Tensor, patched_logits: torch.Tensor) -> float:
    base_logp = F.log_softmax(base_logits.float(), dim=-1)
    patched_logp = F.log_softmax(patched_logits.float(), dim=-1)
    base_probs = torch.exp(base_logp)
    kl = torch.sum(base_probs * (base_logp - patched_logp), dim=-1).mean()
    return float(kl.detach().cpu().item())


def top1_agreement(base_logits: torch.Tensor, patched_logits: torch.Tensor) -> float:
    base_top1 = base_logits.argmax(dim=-1)
    patched_top1 = patched_logits.argmax(dim=-1)
    return float((base_top1 == patched_top1).float().mean().detach().cpu().item())


def compute_nmse(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    x = x.float()
    x_hat = x_hat.float()
    mse = ((x - x_hat) ** 2).mean()
    denom = (x ** 2).mean().clamp_min(1e-8)
    return float((mse / denom).detach().cpu().item())


def compute_explained_variance(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    x = x.float()
    x_hat = x_hat.float()
    var_x = x.var(dim=0, unbiased=False).mean().clamp_min(1e-8)
    var_resid = (x - x_hat).var(dim=0, unbiased=False).mean()
    ev = 1.0 - (var_resid / var_x)
    return float(ev.clamp(0.0, 1.0).detach().cpu().item())


def load_sae_checkpoint(ckpt_path: str | Path, device: str) -> tuple[TopKSAE, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    sae = TopKSAE(d_in=cfg.get("d_in", 384), d_sae=cfg.get("d_sae", 4096), k=cfg.get("k", 32))
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.to(device)
    sae.eval()
    meta = {
        "input_mean": ckpt.get("input_mean"),
        "input_std": ckpt.get("input_std"),
    }
    return sae, meta


def sae_reconstruct_hidden(sae: TopKSAE, hidden: torch.Tensor, input_mean: torch.Tensor | None, input_std: torch.Tensor | None) -> torch.Tensor:
    x = hidden.float()
    if input_mean is not None and input_std is not None:
        x = (x - input_mean.to(x.device)) / input_std.to(x.device)
    out = sae(x.reshape(-1, x.shape[-1]))
    x_hat = out["x_hat"].reshape_as(hidden)
    if input_mean is not None and input_std is not None:
        x_hat = x_hat * input_std.to(x_hat.device) + input_mean.to(x_hat.device)
    return x_hat


def sequence_nll(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int) -> float:
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    mask = flat_labels != pad_token_id
    if not mask.any():
        return 0.0
    nll = F.cross_entropy(flat_logits, flat_labels, reduction="none")[mask]
    return float(nll.sum().detach().cpu().item() / mask.sum().detach().cpu().item())


def forward_with_replaced_hidden_state(
    model: ProGen3ForCausalLM,
    batch: dict[str, torch.Tensor],
    target_layer: int,
    replacement_hidden: torch.Tensor,
) -> torch.Tensor:
    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]
    sequence_ids = batch["sequence_ids"]

    if torch.is_autocast_enabled():
        target_dtype = torch.get_autocast_gpu_dtype()
    elif hasattr(model.config, "_pre_quantization_dtype"):
        target_dtype = model.config._pre_quantization_dtype
    elif model.config.fused_attention_norm:
        target_dtype = model.model.layers[0].norm_attn_norm.self_attn.q_proj.weight.dtype
    else:
        target_dtype = model.model.layers[0].self_attn.q_proj.weight.dtype

    hidden_states = model.model.embed_tokens(input_ids) + model.model.embed_seq_id(sequence_ids)
    hidden_states = hidden_states.to(target_dtype)

    if replacement_hidden.dim() == 2:
        replacement_hidden = replacement_hidden.unsqueeze(0)
    if replacement_hidden.shape[-1] != hidden_states.shape[-1]:
        raise ValueError(f"Replacement hidden shape mismatch: {replacement_hidden.shape} vs {hidden_states.shape}")
    if replacement_hidden.shape[0] != hidden_states.shape[0]:
        replacement_hidden = replacement_hidden.expand(hidden_states.shape[0], *replacement_hidden.shape[1:])

    for layer_idx, decoder_layer in enumerate(model.model.layers):
        layer_outputs = decoder_layer(
            hidden_states,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            output_router_weights=False,
            use_cache=False,
        )
        hidden_states = layer_outputs[0]
        if layer_idx == target_layer:
            hidden_states = replacement_hidden.to(hidden_states.dtype)

    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states).float()
    return logits


def evaluate_sequence(model: ProGen3ForCausalLM, sae: TopKSAE, seq: str, batch_preparer: ProGen3BatchPreparer, layer_index: int, device: str, sae_meta: dict[str, Any]) -> dict[str, float]:
    batch = batch_preparer.get_batch_kwargs([seq], device=device, reverse=False)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch, return_dict=True, output_hidden_states=True, use_cache=False)
        base_hidden = outputs.hidden_states[layer_index]
        base_logits = outputs.logits
        base_nll = sequence_nll(base_logits[0], batch["labels"][0], model.config.pad_token_id)

        patched_hidden = sae_reconstruct_hidden(
            sae=sae,
            hidden=base_hidden,
            input_mean=sae_meta["input_mean"],
            input_std=sae_meta["input_std"],
        )
        patched_logits = forward_with_replaced_hidden_state(model, batch, layer_index, patched_hidden)
        patched_nll = sequence_nll(patched_logits[0], batch["labels"][0], model.config.pad_token_id)

        mean_kl = compute_mean_kl(base_logits[0], patched_logits[0])
        top1 = top1_agreement(base_logits[0], patched_logits[0])
        nmse = compute_nmse(base_hidden[0].reshape(-1, base_hidden.shape[-1]), patched_hidden[0].reshape(-1, patched_hidden.shape[-1]))
        ev = compute_explained_variance(base_hidden[0].reshape(-1, base_hidden.shape[-1]), patched_hidden[0].reshape(-1, patched_hidden.shape[-1]))

    return {
        "base_nll": base_nll,
        "patched_nll": patched_nll,
        "delta_nll": patched_nll - base_nll,
        "relative_nll_degradation_pct": relative_nll_degradation_pct(base_nll, patched_nll),
        "mean_kl": mean_kl,
        "top1_agreement": top1,
        "nmse": nmse,
        "explained_variance": ev,
    }


def summarize_group(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"n_sequences": 0, "mean_base_nll": float("nan"), "mean_patched_nll": float("nan"), "mean_delta_nll": float("nan"), "mean_relative_nll_degradation_pct": float("nan")}
    base = [r["base_nll"] for r in rows]
    patched = [r["patched_nll"] for r in rows]
    delta = [r["delta_nll"] for r in rows]
    rel = [r["relative_nll_degradation_pct"] for r in rows]
    return {
        "n_sequences": len(rows),
        "mean_base_nll": float(sum(base) / len(base)),
        "mean_patched_nll": float(sum(patched) / len(patched)),
        "mean_delta_nll": float(sum(delta) / len(delta)),
        "mean_relative_nll_degradation_pct": float(sum(rel) / len(rel)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure whether a layer-6 SAE reconstruction preserves ProGen3 behavior.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--sae-checkpoint", type=str, default=DEFAULT_SAE_CKPT)
    parser.add_argument("--layer-index", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sequences", type=int, default=None, help="Optional limit for smoke-testing.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = ProGen3ForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32).to(args.device)
    model.eval()

    sae, sae_meta = load_sae_checkpoint(args.sae_checkpoint, args.device)
    batch_preparer = ProGen3BatchPreparer()

    df = pd.read_csv(args.dataset)
    if args.max_sequences is not None:
        df = df.head(args.max_sequences)

    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        seq = str(row["sequence"])
        metrics = evaluate_sequence(model, sae, seq, batch_preparer, args.layer_index, args.device, sae_meta)
        record = {
            "sequence_id": int(row.get("sequence_id", idx)),
            "class_label": int(row.get("class_label", 0)),
            "length": int(row.get("length", len(seq))),
            "is_s1a": int(row.get("class_label", 0) == 1),
            "sequence": seq,
            **metrics,
        }
        rows.append(record)

    per_seq_df = pd.DataFrame(rows)
    output_cols = [
        "sequence_id",
        "class_label",
        "length",
        "base_nll",
        "patched_nll",
        "delta_nll",
        "relative_nll_degradation_pct",
        "mean_kl",
        "top1_agreement",
        "nmse",
        "explained_variance",
    ]
    per_seq_df[output_cols].to_csv(out_path, index=False)

    summary = {
        "global": summarize_group(rows),
        "positive": summarize_group([r for r in rows if r["is_s1a"] == 1]),
        "background": summarize_group([r for r in rows if r["is_s1a"] == 0]),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
