"""Prepare single-tab anchors and train PGT with optional torchrun DDP.

Run with ``python -m aet_wf.train prepare ...`` or ``... fit ...``.
Neither command accepts a test split.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset, DistributedSampler

from .data import AnchorDataset, MixtureDataset
from .evaluate import calibrate as evaluate_calibrate
from .models import AETWFModel, MultiScaleTrafficEncoder




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="train the shared encoder and build prototypes")
    prepare.add_argument("--anchor-train", type=Path, required=True)
    prepare.add_argument("--anchor-val", type=Path, required=True)
    prepare.add_argument("--feature-stats", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    prepare.add_argument("--epochs", type=int, default=20)
    prepare.add_argument("--batch-size", type=int, default=64)
    prepare.add_argument("--num-workers", type=int, default=4)
    prepare.add_argument("--target-length", type=int, default=10_000)
    prepare.add_argument("--pad-length", type=int, help="single-site container length; defaults to target-length")
    prepare.add_argument("--learning-rate", type=float, default=1e-4)
    prepare.add_argument("--weight-decay", type=float, default=0.01)
    prepare.add_argument("--dropout", type=float, default=0.1)
    prepare.add_argument("--tokens-per-class", type=int, default=2_048)
    prepare.add_argument("--class-prototypes", type=int, default=8)
    prepare.add_argument("--background-prototypes", type=int, default=16)
    prepare.add_argument("--max-train-samples", type=int)
    prepare.add_argument("--max-val-samples", type=int)
    prepare.add_argument("--seed", type=int, default=42)

    fit = commands.add_parser("fit", help="train one method using train/validation mixtures")
    fit.add_argument("--method", choices=("aet",), default="aet")
    fit.add_argument("--train-data", type=Path, required=True)
    fit.add_argument("--val-data", type=Path, required=True)
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--feature-stats", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--prepared", type=Path)
    fit.add_argument("--anchor-train", type=Path)
    fit.add_argument("--no-anchor", action="store_true", help="PGT ablation: random encoder/prototypes and no single-site supervision")
    fit.add_argument("--resume", type=Path)
    fit.add_argument("--epochs", type=int, default=100)
    fit.add_argument("--batch-size", type=int, default=64)
    fit.add_argument("--anchor-batch-size", type=int, default=64)
    fit.add_argument("--num-workers", type=int, default=4)
    fit.add_argument("--target-length", type=int, default=20_000)
    fit.add_argument("--anchor-target-length", type=int, default=10_000)
    fit.add_argument("--anchor-pad-length", type=int, help="anchor container length; defaults to anchor-target-length")
    fit.add_argument("--aggregation-mode", choices=("uot", "independent_local"), default="uot")
    fit.add_argument("--learning-rate", type=float, default=3e-4)
    fit.add_argument("--encoder-learning-rate", type=float, default=1e-4)
    fit.add_argument("--weight-decay", type=float, default=0.01)
    fit.add_argument("--pos-weight", default="auto", help="auto: per-class train negatives/positives; or explicit positive scalar")
    fit.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    fit.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    fit.add_argument("--stop-after-epoch", type=int, help="checkpoint-only interruption for infrastructure checks; does not change the scheduler budget")
    fit.add_argument("--anchor-pos-weight", type=float, default=89.0)
    fit.add_argument("--dropout", type=float, default=0.1)
    fit.add_argument("--anchor-weight", type=float, default=0.5)
    fit.add_argument("--detach-plan-epochs", type=int, default=2)
    fit.add_argument("--affinity-temperature", type=float, default=0.1)
    fit.add_argument("--uot-epsilon", type=float, default=0.05)
    fit.add_argument("--uot-rho-token", type=float, default=0.5)
    fit.add_argument("--uot-rho-class", type=float, default=0.03)
    fit.add_argument("--uot-iters", type=int, default=30)
    fit.add_argument("--background-prior", type=float, default=0.5)
    fit.add_argument("--max-train-samples", type=int)
    fit.add_argument("--max-val-samples", type=int)
    fit.add_argument("--max-anchor-samples", type=int)
    fit.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    for name in ("epochs", "batch_size", "num_workers", "target_length"):
        if hasattr(args, name) and getattr(args, name) < (0 if name == "num_workers" else 1):
            parser.error(f"--{name.replace('_', '-')} has an invalid value")
    if args.command == "prepare":
        args.pad_length = args.target_length if args.pad_length is None else args.pad_length
        if args.pad_length < args.target_length:
            parser.error("--pad-length must be at least --target-length")
    if args.command == "prepare" and min(
        args.tokens_per_class, args.class_prototypes, args.background_prototypes
    ) < 1:
        parser.error("prototype and token counts must be positive")
    if args.command == "fit" and args.method == "aet" and not args.no_anchor and (
        args.prepared is None or args.anchor_train is None
    ):
        parser.error("PGT requires --prepared and --anchor-train")
    if args.command == "fit":
        if args.no_anchor and (args.method != "aet" or args.prepared is not None or args.anchor_train is not None):
            parser.error("--no-anchor applies to PGT without --prepared or --anchor-train")
        args.anchor_pad_length = args.anchor_target_length if args.anchor_pad_length is None else args.anchor_pad_length
        if not 0 < args.anchor_target_length <= args.anchor_pad_length:
            parser.error("anchor lengths must satisfy 0 < target-length <= pad-length")
        if args.method != "aet" and args.aggregation_mode != "uot":
            parser.error("--aggregation-mode only applies to PGT")
        if args.pos_weight != "auto":
            try:
                args.pos_weight = float(args.pos_weight)
            except ValueError:
                parser.error("--pos-weight must be auto or a positive scalar")
            if not math.isfinite(args.pos_weight) or args.pos_weight <= 0:
                parser.error("positive-class weights must be finite and positive")
        if args.anchor_pos_weight <= 0 or args.anchor_batch_size < 1:
            parser.error("anchor weight and batch size must be positive")
    return args


def jsonable(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(jsonable(payload), sort_keys=True) + "\n")


def reconcile_history(path: Path, resumed: dict) -> list[dict]:
    """Repair only the checkpoint epoch's missing/truncated final append."""
    epoch = resumed["epoch"]
    saved = resumed["epoch_record"]
    if type(epoch) is not int or epoch < 1 or saved.get("epoch") != epoch:
        raise ValueError("checkpoint epoch_record does not match its epoch")
    if saved.get("global_step") != resumed["global_step"]:
        raise ValueError("checkpoint epoch_record does not match its global_step")
    lines = path.read_text().splitlines() if path.exists() else []
    records, truncated = [], False
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            if index != len(lines) - 1:
                raise ValueError("history contains a malformed interior line") from error
            truncated = True
            break
        if not isinstance(record, dict) or type(record.get("epoch")) is not int or record["epoch"] != len(records) + 1:
            raise ValueError("history must contain a complete consecutive epoch prefix starting at 1")
        records.append(record)
    if len(records) < epoch - 1 or (truncated and len(records) != epoch - 1):
        raise ValueError("history is missing earlier epochs; only the checkpoint epoch's final append can be repaired")
    if len(records) == epoch - 1:
        records.append(saved)
    elif records[epoch - 1] != saved:
        raise ValueError("history checkpoint epoch disagrees with checkpoint.epoch_record")
    # A deliberate resume from an older checkpoint may rewind a fully valid suffix.
    records = records[:epoch]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    os.replace(temporary, path)
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def select_device(device_type: str = "cuda", distributed: bool = False) -> torch.device:
    if device_type == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu or configure a CUDA-capable PyTorch environment")
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if distributed else 0
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise RuntimeError("LOCAL_RANK exceeds the number of visible CUDA devices")
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def distributed_info() -> tuple[int, int]:
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def sync_buffers(model: nn.Module) -> None:
    if dist.is_initialized():
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)


class LogitsOnly(nn.Module):
    """Use the common logits interface for both single-process and DDP training."""

    def __init__(self, model: nn.Module, method: str):
        super().__init__()
        self.model, self.method = model, method

    def forward(self, features: Tensor, lengths: Tensor):
        return model_logits(self.model, self.method, features, lengths)


def autocast(device: torch.device, precision: str):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16")


def positive_weights(dataset: Dataset, setting: str | float) -> tuple[Tensor, dict]:
    if isinstance(dataset, Subset):
        labels = dataset.dataset.labels_array()[dataset.indices]
    else:
        labels = dataset.labels_array()
    if not len(labels) or not np.isin(labels, (0, 1)).all():
        raise ValueError("training labels must be a nonempty binary matrix")
    counts = labels.sum(0, dtype=np.int64)
    if np.any(counts == 0) or np.any(counts == len(labels)):
        raise ValueError("each training class needs positive and negative examples")
    values = (len(labels) - counts) / counts if setting == "auto" else np.full(len(counts), float(setting))
    return torch.tensor(values, dtype=torch.float32), {
        "strategy": "per_class_train_counts" if setting == "auto" else "explicit_scalar",
        "fit_split": "train", "samples": len(labels), "class_counts": counts.tolist(),
        "values": values.tolist(),
    }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if isinstance(state["cuda"], list):
            torch.cuda.set_rng_state_all(state["cuda"])
        else:
            torch.cuda.set_rng_state(state["cuda"])


def load_feature_stats(path: Path) -> tuple[dict, str]:
    stats = read_json(path)
    if stats.get("fit_split") != "train":
        raise ValueError("feature statistics must be fitted on train only")
    return stats, sha256(path)


def limited(dataset: Dataset, maximum: int | None, seed: int) -> Dataset:
    if maximum is None or maximum >= len(dataset):
        return dataset
    indices = np.sort(np.random.default_rng(seed).choice(len(dataset), maximum, replace=False))
    return Subset(dataset, indices.tolist())


def loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
    sampler=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=False,
        num_workers=workers,
        multiprocessing_context="spawn" if workers else None,
        pin_memory=True,
        persistent_workers=False,  # epoch-boundary worker seeding is reproducible on resume
        generator=generator,
    )


def cpu_state(module: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


@torch.inference_mode()
def validate_anchor(
    encoder: MultiScaleTrafficEncoder,
    head: nn.Linear,
    batches: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    encoder.eval()
    head.eval()
    correct = count = 0
    loss_sum = 0.0
    for batch in batches:
        features = batch["features"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = head(encoder(features, lengths)["global_summary"])
            loss = F.cross_entropy(logits, labels)
        correct += int((logits.argmax(1) == labels).sum())
        count += len(labels)
        loss_sum += float(loss) * len(labels)
    return correct / max(count, 1), loss_sum / max(count, 1)


def collect_tokens(
    encoder: MultiScaleTrafficEncoder,
    batches: DataLoader,
    num_classes: int,
    capacity: int,
    seed: int,
    device: torch.device,
) -> list[np.ndarray]:
    """Keep a deterministic random-priority reservoir for each class."""
    encoder.eval()
    rng = np.random.default_rng(seed)
    buckets = [np.empty((0, encoder.output_dim), np.float32) for _ in range(num_classes)]
    priorities = [np.empty(0, np.float64) for _ in range(num_classes)]
    with torch.inference_mode():
        for batch in batches:
            features = batch["features"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                encoded = encoder(features, lengths)
            tokens = F.normalize(encoded["tokens"].float(), dim=-1).cpu().numpy()
            masks = encoded["token_mask"].cpu().numpy()
            for row, label in enumerate(batch["labels"].tolist()):
                label = int(label)
                candidates = tokens[row, masks[row]]
                keys = rng.random(len(candidates))
                joined = np.concatenate((buckets[label], candidates))
                joined_keys = np.concatenate((priorities[label], keys))
                if len(joined) > capacity:
                    keep = np.argpartition(joined_keys, capacity - 1)[:capacity]
                    joined, joined_keys = joined[keep], joined_keys[keep]
                buckets[label], priorities[label] = joined, joined_keys
    return buckets


def normalized_centers(tokens: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(tokens) < count:
        raise ValueError(f"need at least {count} tokens, got {len(tokens)}")
    model = MiniBatchKMeans(
        n_clusters=count,
        batch_size=min(1024, len(tokens)),
        n_init=3,
        random_state=seed,
    ).fit(tokens)
    centers = model.cluster_centers_.astype(np.float32, copy=False)
    return centers / np.maximum(np.linalg.norm(centers, axis=-1, keepdims=True), 1e-12)


def prepare(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    seed_all(args.seed)
    stats, stats_hash = load_feature_stats(args.feature_stats)
    train_manifest, val_manifest = read_json(args.anchor_train), read_json(args.anchor_val)
    if train_manifest.get("partition") != "train" or val_manifest.get("partition") != "val":
        raise ValueError("anchor manifests must be train and validation partitions")
    if train_manifest["num_classes"] != val_manifest["num_classes"]:
        raise ValueError("anchor class spaces differ")
    num_classes = int(train_manifest["num_classes"])
    output = args.output_dir.resolve()
    if (output / "prepared.pt").exists() or (output / "prepare_history.jsonl").exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)

    train_data = limited(
        AnchorDataset(args.anchor_train, args.feature_stats, target_length=args.target_length, pad_length=args.pad_length),
        args.max_train_samples,
        args.seed + 1,
    )
    val_data = limited(
        AnchorDataset(args.anchor_val, args.feature_stats, target_length=args.target_length, pad_length=args.pad_length),
        args.max_val_samples,
        args.seed + 2,
    )
    train_generator = torch.Generator().manual_seed(args.seed + 3)
    train_loader = loader(train_data, args.batch_size, args.num_workers, True, train_generator)
    val_loader = loader(val_data, args.batch_size, args.num_workers, False)

    prepare_runtime = {
        "world_size": 1, "physical_gpu_mapping": os.environ.get("CUDA_VISIBLE_DEVICES", "") if device.type == "cuda" else "cpu",
        "device": str(device), "per_rank_batch": args.batch_size, "global_batch": args.batch_size,
        "precision": "bfloat16" if device.type == "cuda" else "float32", "batch_norm": "local", "cudnn_deterministic": True,
        "worker_start_method": "spawn" if args.num_workers else None,
        "persistent_workers": False,
    }
    write_json(output / "prepare_config.json", {
        "args": vars(args), "runtime": prepare_runtime, "epochs": args.epochs,
        "train_samples": len(train_data), "val_samples": len(val_data),
        "feature_stats_sha256": stats_hash,
        "anchor_train_sha256": sha256(args.anchor_train),
        "anchor_val_sha256": sha256(args.anchor_val),
        "source_sha256": {name: sha256(Path(__file__).with_name(name))
                          for name in ("train.py", "models.py", "data.py")},
    })
    encoder = MultiScaleTrafficEncoder(dropout=args.dropout).to(device)
    head = nn.Linear(encoder.output_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *head.parameters()],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_accuracy = -math.inf
    best_epoch = -1
    best_encoder = None
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        head.train()
        loss_sum = 0.0
        seen = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = head(encoder(features, lengths)["global_summary"])
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *head.parameters()], 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(labels)
            seen += len(labels)
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_started
        train_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        validation_started = time.perf_counter()
        accuracy, val_loss = validate_anchor(encoder, head, val_loader, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        val_seconds = time.perf_counter() - validation_started
        val_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        record = {
            "epoch": epoch,
            "train_ce": loss_sum / max(seen, 1),
            "val_ce": val_loss,
            "val_accuracy": accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_samples": seen, "val_samples": len(val_data),
            "train_seconds": train_seconds, "validation_seconds": val_seconds,
            "train_samples_per_second": seen / max(train_seconds, 1e-9),
            "validation_samples_per_second": len(val_data) / max(val_seconds, 1e-9),
            "max_rank_train_peak_bytes": train_peak, "max_rank_validation_peak_bytes": val_peak,
        }
        append_jsonl(output / "prepare_history.jsonl", record)
        print(json.dumps(record), flush=True)
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            best_encoder = cpu_state(encoder)

    if best_encoder is None:
        raise RuntimeError("no valid encoder checkpoint")
    encoder.load_state_dict(best_encoder)
    token_loader = loader(train_data, args.batch_size, args.num_workers, False)
    buckets = collect_tokens(
        encoder, token_loader, num_classes, args.tokens_per_class, args.seed + 4, device
    )
    class_centers = np.stack(
        [normalized_centers(tokens, args.class_prototypes, args.seed + 100 + cls)
         for cls, tokens in enumerate(buckets)]
    )
    background_candidates = []
    for cls, tokens in enumerate(buckets):
        own_similarity = tokens @ class_centers[cls].T
        low_count = max(1, len(tokens) // 4)
        low = np.argpartition(own_similarity.max(1), low_count - 1)[:low_count]
        background_candidates.append(tokens[low])
    background_candidates = np.concatenate(background_candidates)
    background_centers = normalized_centers(
        background_candidates, args.background_prototypes, args.seed + 10_000
    )

    provenance = {
        "schema": "aet-prepared-v1",
        "created_unix": time.time(),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "token_counts": [len(tokens) for tokens in buckets],
        "background_candidate_count": len(background_candidates),
        "seed": args.seed,
        "target_length": args.target_length,
        "pad_length": args.pad_length,
        "epochs": args.epochs,
        "anchor_train_sha256": sha256(args.anchor_train),
        "anchor_val_sha256": sha256(args.anchor_val),
        "feature_stats_sha256": stats_hash,
    }
    artifact = {
        "schema": "aet-prepared-v1",
        "encoder_state": best_encoder,
        "prototype_state": {
            "class_values": torch.from_numpy(class_centers),
            "background_values": torch.from_numpy(background_centers),
        },
        "encoder_args": {"input_dim": 2, "dropout": args.dropout},
        "num_classes": num_classes,
        "num_class_prototypes": args.class_prototypes,
        "num_background_prototypes": args.background_prototypes,
        "feature_stats": stats,
        "provenance": provenance,
    }
    atomic_save(output / "prepared.pt", artifact)
    history = [json.loads(line) for line in (output / "prepare_history.jsonl").read_text().splitlines()]
    write_json(
        output / "prepare_summary.json",
        provenance | {
            "artifact": str(output / "prepared.pt"),
            "artifact_sha256": sha256(output / "prepared.pt"),
            "elapsed_seconds": time.time() - started,
            "runtime": prepare_runtime,
            "train_seconds_total": sum(row["train_seconds"] for row in history),
            "validation_seconds_total": sum(row["validation_seconds"] for row in history),
            "max_rank_train_peak_bytes": max(row["max_rank_train_peak_bytes"] for row in history),
            "max_rank_validation_peak_bytes": max(row["max_rank_validation_peak_bytes"] for row in history),
        },
    )


def model_logits(model: nn.Module, method: str, features: Tensor, lengths: Tensor) -> Tensor:
    if method != "aet":
        raise ValueError("this package supports PGT only")
    output = model(features, lengths)
    return output["logits"] if isinstance(output, dict) else output


def mixture_loss(output, labels: Tensor, method: str, pos_weight: Tensor | None = None) -> Tensor:
    """Weighted binary cross-entropy for PGT's website-presence scores."""
    if method != "aet":
        raise ValueError("this package supports PGT only")
    logits = output["logits"] if isinstance(output, dict) else output
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def objective_record(method: str, no_anchor: bool = False) -> dict:
    if method != "aet":
        raise ValueError("this package supports PGT only")
    return {"loss": "weighted_bce", "pos_weight_applied": True,
            "single_site_supervision": not no_anchor}


def macro_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    values = [
        average_precision_score(labels[:, cls], scores[:, cls])
        for cls in range(labels.shape[1])
        if np.any(labels[:, cls] == 1)
    ]
    return float(np.mean(values)) if values else float("nan")


@torch.inference_mode()
def validate_fit(
    model: nn.Module,
    method: str,
    batches: DataLoader,
    device: torch.device,
    precision: str = "bfloat16",
) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    # Use the underlying model: exact, unpadded validation shards may have unequal batch counts.
    model.eval()
    sync_buffers(model)
    logits, labels, groups = [], [], []
    for batch in batches:
        features = batch["features"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        with autocast(device, precision):
            prediction = model_logits(model, method, features, lengths)
        logits.append(prediction.float().cpu().numpy())
        labels.append(batch["labels"].numpy())
        groups.append(batch["groups"].numpy())
    rank, world = distributed_info()
    parts = [(logits, labels, groups)]
    if world > 1:
        parts = [None] * world if rank == 0 else None
        dist.gather_object((logits, labels, groups), parts, dst=0)
    metrics, logits_array, labels_array = None, None, None
    if rank == 0:
        logits_array = np.concatenate([array for part in parts for array in part[0]])
        labels_array = np.concatenate([array for part in parts for array in part[1]])
        groups_array = np.concatenate([array for part in parts for array in part[2]])
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits_array, -80, 80)))
        by_group = {
            str(int(group)): macro_ap(labels_array[groups_array == group], scores[groups_array == group])
            for group in np.unique(groups_array)
        }
        selection = macro_ap(labels_array, scores)
        metrics = {"map": selection, "map_by_group": by_group,
                   "selection_map": selection, "samples": len(labels_array)}
    if world > 1:
        objects = [metrics]
        dist.broadcast_object_list(objects, src=0)
        metrics = objects[0]
    return metrics, logits_array, labels_array


def build_model(
    args: argparse.Namespace,
    num_classes: int,
    stats_hash: str,
) -> tuple[nn.Module, dict, str | None]:
    if args.method == "aet":
        if getattr(args, "no_anchor", False):
            model_args = {
                "num_classes": num_classes, "num_class_prototypes": 8,
                "num_background_prototypes": 16, "dropout": args.dropout,
                "affinity_temperature": args.affinity_temperature,
                "uot_epsilon": args.uot_epsilon, "uot_rho_token": args.uot_rho_token,
                "uot_rho_class": args.uot_rho_class, "uot_iters": args.uot_iters,
                "background_prior": args.background_prior, "detach_plan": False,
                "aggregation_mode": args.aggregation_mode,
            }
            return AETWFModel(**model_args), model_args, None
        prepared_hash = sha256(args.prepared)
        prepared = torch.load(args.prepared, map_location="cpu", weights_only=False)
        if prepared["num_classes"] != num_classes:
            raise ValueError("prepared artifact and mixture class spaces differ")
        if prepared["provenance"]["feature_stats_sha256"] != stats_hash:
            raise ValueError("prepared artifact used different feature statistics")
        if prepared["provenance"]["anchor_train_sha256"] != sha256(args.anchor_train):
            raise ValueError("prepared artifact used a different train anchor manifest")
        provenance = prepared["provenance"]
        observed = provenance.get("target_length")
        padded = provenance.get("pad_length", observed)
        if type(observed) is not int or type(padded) is not int or not 0 < observed <= padded:
            raise ValueError("prepared artifact needs valid observed target_length and pad_length provenance")
        anchor_pad = getattr(args, "anchor_pad_length", None)
        anchor_pad = args.anchor_target_length if anchor_pad is None else anchor_pad
        if (observed, padded) != (args.anchor_target_length, anchor_pad):
            raise ValueError("prepared artifact and fit anchor observation/padding lengths differ")
        model_args = {
            "num_classes": num_classes,
            "num_class_prototypes": int(prepared["num_class_prototypes"]),
            "num_background_prototypes": int(prepared["num_background_prototypes"]),
            "dropout": args.dropout,
            "affinity_temperature": args.affinity_temperature,
            "uot_epsilon": args.uot_epsilon,
            "uot_rho_token": args.uot_rho_token,
            "uot_rho_class": args.uot_rho_class,
            "uot_iters": args.uot_iters,
            "background_prior": args.background_prior,
            "detach_plan": False,
            "aggregation_mode": getattr(args, "aggregation_mode", "uot"),
        }
        model = AETWFModel(**model_args)
        model.encoder.load_state_dict(prepared["encoder_state"])
        model.prototype_bank.load_state_dict(prepared["prototype_state"])
        return model, model_args, prepared_hash

    raise ValueError("this package supports PGT only")


def validate_resume(resumed: dict, expected: dict, args: argparse.Namespace) -> None:
    """Normalize only the two known legacy defaults; keep every other check exact."""
    for key, value in expected.items():
        saved = resumed.get(key)
        if key == "model_args" and args.method == "aet" and isinstance(saved, dict):
            saved = {"aggregation_mode": "uot", **saved}
        if saved != value:
            raise ValueError(f"resume mismatch: {key}")
    saved_args = dict(resumed["args"])
    if saved_args.get("anchor_pad_length") is None:
        saved_args["anchor_pad_length"] = saved_args.get("anchor_target_length")
    saved_args.setdefault("aggregation_mode", "uot")
    saved_args.setdefault("no_anchor", False)
    for key, value in jsonable(vars(args)).items():
        if key not in {"resume", "stop_after_epoch"} and saved_args.get(key) != value:
            raise ValueError(f"resume argument mismatch: {key}")


def aggregation_record(args: argparse.Namespace) -> dict | None:
    if args.method != "aet":
        return None
    return {"mode": args.aggregation_mode, "inactive_parameters":
            ["uot_rho_token", "uot_rho_class", "uot_iters"] if args.aggregation_mode == "independent_local" else []}


def checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int,
    args: argparse.Namespace,
    manifest_hash: str,
    feature_stats: dict,
    feature_stats_hash: str,
    model_args: dict,
    prepared_hash: str | None,
    generators: dict[str, torch.Generator],
    rank_states: list[dict] | None = None,
    runtime: dict | None = None,
) -> dict:
    return {
        "schema": "aet-training-checkpoint-v1",
        "method": args.method,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "rng_by_rank": rank_states,
        "runtime": runtime,
        "rng": capture_rng(),
        "loader_rng": {name: generator.get_state() for name, generator in generators.items()},
        "args": jsonable(vars(args)),
        "manifest_sha256": manifest_hash,
        "feature_stats": feature_stats,
        "feature_stats_sha256": feature_stats_hash,
        "model_args": model_args,
        "aggregation": aggregation_record(args),
        "prepared_artifact_sha256": prepared_hash,
        "selection_rule": "global validation class-mAP; composition validation must be Cross only",
        "calibration": None,
    }


def validate_manifest(path: Path, train_path: Path, val_path: Path) -> tuple[dict, str]:
    manifest = read_json(path)
    for split, expected in (("train", train_path), ("val", val_path)):
        recorded = Path(manifest["splits"][split]["path"])
        if recorded.resolve() != expected.resolve():
            raise ValueError(f"manifest {split} path does not match --{split}-data")
    return manifest, sha256(path)


def fit(args: argparse.Namespace) -> None:
    device = select_device(args.device, distributed=True)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank, world = distributed_info()
    seed_all(args.seed)
    feature_stats, stats_hash = load_feature_stats(args.feature_stats)
    _, manifest_hash = validate_manifest(args.manifest, args.train_data, args.val_data)
    output = args.output_dir.resolve()
    if args.resume is None and any(
        (output / name).exists() for name in ("best.pt", "last.pt", "history.jsonl")
    ):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)

    train_base = MixtureDataset(args.train_data, args.feature_stats, target_length=args.target_length, method=args.method)
    val_base = MixtureDataset(args.val_data, args.feature_stats, target_length=args.target_length, method=args.method)
    if train_base.num_classes != val_base.num_classes:
        raise ValueError("train/validation class spaces differ")
    # Reject old mixed reference/shift validation rather than silently tuning on OOD.
    if set(np.unique(val_base.groups_array())) not in ({0}, {1}):
        raise ValueError("validation must be Ordinary (0) or Cross-only (1)")
    num_classes = train_base.num_classes
    train_data = limited(train_base, args.max_train_samples, args.seed + 1)
    val_data = limited(val_base, args.max_val_samples, args.seed + 2)
    multi_pos_weight, weight_record = positive_weights(train_data, args.pos_weight)
    multi_pos_weight = multi_pos_weight.to(device)
    anchor_pos_weight = torch.full((num_classes,), args.anchor_pos_weight, device=device)
    generators = {name: torch.Generator().manual_seed(args.seed + offset + rank)
                  for name, offset in (("mixture", 3), ("validation", 7))}
    train_sampler = DistributedSampler(train_data, world, rank, seed=args.seed + 3) if world > 1 else None
    train_loader = loader(train_data, args.batch_size, args.num_workers, True,
                          generators["mixture"], train_sampler)
    val_shard = Subset(val_data, range(rank, len(val_data), world))
    val_loader = loader(val_shard, args.batch_size, args.num_workers, False, generators["validation"])

    model, model_args, prepared_hash = build_model(args, num_classes, stats_hash)
    if world > 1 and device.type == "cuda":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if args.method == "aet":
        encoder_parameters = list(model.encoder.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        optimizer = torch.optim.AdamW(
            [{"params": encoder_parameters, "lr": args.encoder_learning_rate},
             {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": args.learning_rate}],
            weight_decay=args.weight_decay,
        )
        anchor_loader, anchor_sampler = None, None
        if not getattr(args, "no_anchor", False):
            anchor_base = AnchorDataset(args.anchor_train, args.feature_stats, target_length=args.anchor_target_length,
                                        pad_length=args.anchor_pad_length)
            if anchor_base.num_classes != num_classes:
                raise ValueError("anchor and mixture class spaces differ")
            anchor_data = limited(anchor_base, args.max_anchor_samples, args.seed + 4)
            generators["anchor"] = torch.Generator().manual_seed(args.seed + 5 + rank)
            anchor_sampler = DistributedSampler(anchor_data, world, rank, seed=args.seed + 5) if world > 1 else None
            anchor_loader = loader(anchor_data, args.anchor_batch_size, args.num_workers, True,
                                   generators["anchor"], anchor_sampler)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        anchor_loader, anchor_sampler = None, None
    training = LogitsOnly(model, args.method)
    if world > 1:
        training = DDP(training, device_ids=[device.index] if device.type == "cuda" else None,
                       find_unused_parameters=args.method != "aet")
    seed_all(args.seed + rank)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # bfloat16 has float32 exponent range; scaling is unnecessary.
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    runtime = {
        "world_size": world, "per_rank_batch": args.batch_size,
        "global_batch": args.batch_size * world,
        "per_rank_anchor_batch": args.anchor_batch_size,
        "global_anchor_batch": args.anchor_batch_size * world,
        "physical_gpu_mapping": os.environ.get("CUDA_VISIBLE_DEVICES", "") if device.type == "cuda" else "cpu",
        "precision": args.precision,
        "cudnn_deterministic": True,
        "batch_norm": "synchronized" if world > 1 and device.type == "cuda" else "local",
        "train_sampler": "DistributedSampler padded full coverage" if world > 1 else "RandomSampler full coverage",
        "train_samples": len(train_data), "val_samples": len(val_data),
        "train_padding_repeats": (len(train_sampler) * world - len(train_data)) if train_sampler else 0,
        "validation_sampler": "exact strided shards without padding",
        "checkpoint_boundary": "end of epoch; same world_size, batch and recipe required",
    }
    start_epoch, global_step, best_metric, best_epoch = 1, 0, -math.inf, -1
    if args.resume is not None:
        if args.resume.resolve().parent != output:
            raise ValueError("resume must use its existing output directory so best/history remain coherent")
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        expected = {"method": args.method, "manifest_sha256": manifest_hash,
                    "feature_stats_sha256": stats_hash, "prepared_artifact_sha256": prepared_hash,
                    "model_args": model_args, "runtime": runtime}
        validate_resume(resumed, expected, args)
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        scaler.load_state_dict(resumed["scaler"])
        start_epoch, global_step = int(resumed["epoch"]) + 1, int(resumed["global_step"])
        best_metric, best_epoch = float(resumed["best_metric"]), int(resumed["best_epoch"])
        state = resumed["rng_by_rank"][rank]
        for name, generator in generators.items():
            generator.set_state(state["loader_rng"][name])
        restore_rng(state["rng"])
        if rank == 0:
            reconcile_history(output / "history.jsonl", resumed)
    if rank == 0:
        write_json(output / "pos_weight.json", weight_record | {"anchor_pos_weight": args.anchor_pos_weight,
                   "applied_to_mixture_loss": True})
        write_json(output / "config.json", {
            "args": vars(args), "device": str(device), "visible_gpu": runtime["physical_gpu_mapping"],
            "runtime": runtime, "pos_weight": weight_record,
            "manifest_sha256": manifest_hash, "feature_stats_sha256": stats_hash,
            "model_args": model_args, "prepared_artifact_sha256": prepared_hash,
            "aggregation": aggregation_record(args),
            "objective": objective_record(args.method, getattr(args, "no_anchor", False)),
            "input_contract": {"encoding": "direction_and_train_scaled_log_iat", "observation_unit": "packets"},
            "train_samples": len(train_data), "val_samples": len(val_data),
        })
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        training.train()
        if args.method == "aet":
            model.detach_plan = epoch <= args.detach_plan_epochs
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if anchor_sampler is not None:
            anchor_sampler.set_epoch(epoch * 1_000_000)
        anchor_cycle = 0
        anchor_iterator = iter(anchor_loader) if anchor_loader is not None else None
        mixture_loss_key = "multi_bce"
        totals = {"loss": 0.0, mixture_loss_key: 0.0, "anchor_bce": 0.0}
        seen = anchor_updates = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            features = batch["features"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            use_anchor = anchor_iterator is not None and (batch_index + 1) % 4 == 0
            # Accumulate mixture then anchor gradients; only the final backward reduces.
            with training.no_sync() if world > 1 and use_anchor else nullcontext():
                with autocast(device, args.precision):
                    logits = training(features, lengths)
                    multi_loss = mixture_loss(logits, labels, args.method, multi_pos_weight)
                scaler.scale(multi_loss).backward()
            anchor_loss = torch.zeros((), device=device)
            if use_anchor:
                try:
                    anchor = next(anchor_iterator)
                except StopIteration:
                    anchor_cycle += 1
                    if anchor_sampler is not None:
                        anchor_sampler.set_epoch(epoch * 1_000_000 + anchor_cycle)
                    anchor_iterator = iter(anchor_loader)
                    anchor = next(anchor_iterator)
                anchor_features = anchor["features"].to(device, non_blocking=True)
                anchor_lengths = anchor["lengths"].to(device, non_blocking=True)
                anchor_labels = anchor["labels"].to(device, non_blocking=True)
                anchor_targets = torch.zeros((len(anchor_labels), num_classes), device=device).scatter_(1, anchor_labels[:, None], 1)
                with autocast(device, args.precision):
                    anchor_logits = training(anchor_features, anchor_lengths)
                    anchor_loss = F.binary_cross_entropy_with_logits(anchor_logits, anchor_targets, pos_weight=anchor_pos_weight)
                scaler.scale(args.anchor_weight * anchor_loss).backward()
                anchor_updates += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            if args.method == "aet":
                model.prototype_bank.normalize_()
            global_step += 1
            count = len(labels)
            seen += count
            totals[mixture_loss_key] += float(multi_loss.detach()) * count
            totals["anchor_bce"] += float(anchor_loss.detach()) * count
            totals["loss"] += float(multi_loss.detach() + args.anchor_weight * anchor_loss.detach()) * count
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_started
        train_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        validation_started = time.perf_counter()
        metrics, _, _ = validate_fit(model, args.method, val_loader, device, args.precision)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        val_seconds = time.perf_counter() - validation_started
        val_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        counts = torch.tensor([seen, *totals.values()], dtype=torch.float64, device=device)
        resources = torch.tensor([train_seconds, val_seconds, train_peak, val_peak], dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(counts)
            dist.all_reduce(resources, op=dist.ReduceOp.MAX)
        total_seen, *total_losses = counts.tolist()
        train_seconds, val_seconds, train_peak, val_peak = resources.tolist()
        record = {
            "epoch": epoch, "global_step": global_step,
            "train": {key: value / max(total_seen, 1) for key, value in zip(totals, total_losses)},
            "anchor_updates": anchor_updates, "train_presentations": int(total_seen),
            "validation": metrics, "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "train_seconds": train_seconds, "validation_seconds": val_seconds,
            "train_samples_per_second": total_seen / max(train_seconds, 1e-9),
            "validation_samples_per_second": len(val_data) / max(val_seconds, 1e-9),
            "max_rank_train_peak_bytes": int(train_peak), "max_rank_validation_peak_bytes": int(val_peak),
        }
        score = metrics["selection_map"]
        if not math.isfinite(score):
            raise FloatingPointError("non-finite validation selection mAP")
        if score > best_metric:
            best_metric, best_epoch = score, epoch
            record["best"] = True
        state = {"rng": capture_rng(), "loader_rng": {name: g.get_state() for name, g in generators.items()}}
        rank_states = [state]
        if world > 1:
            rank_states = [None] * world if rank == 0 else None
            dist.gather_object(state, rank_states, dst=0)
        if rank == 0:
            payload = checkpoint(model, optimizer, scheduler, scaler, epoch, global_step,
                                 best_metric, best_epoch, args, manifest_hash, feature_stats,
                                 stats_hash, model_args, prepared_hash, generators, rank_states, runtime)
            payload["epoch_record"] = record
            payload["pos_weight"] = weight_record
            if record.get("best"):
                atomic_save(output / "best.pt", payload)
            atomic_save(output / "last.pt", payload)
            append_jsonl(output / "history.jsonl", record)
            print(json.dumps(record), flush=True)
        if world > 1:
            dist.barrier()
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            return

    if not (output / "best.pt").exists():
        raise RuntimeError("no validation-selected checkpoint exists")
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    validation, logits, labels = validate_fit(model, args.method, val_loader, device, args.precision)
    if rank == 0:
        calibration = evaluate_calibrate(logits, labels)
        best["calibration"] = calibration
        atomic_save(output / "best.pt", best)
        # last.pt retains its own weights and remains uncalibrated/resumable.
        history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
        write_json(output / "summary.json", {
            "method": args.method, "best_epoch": best_epoch, "best_selection_map": best_metric,
            "validation": validation, "calibration": calibration, "runtime": runtime,
            "best_checkpoint": str(output / "best.pt"), "last_checkpoint": str(output / "last.pt"),
            "session_elapsed_seconds": time.time() - started,
            "train_seconds_total": sum(record["train_seconds"] for record in history),
            "validation_seconds_total": sum(record["validation_seconds"] for record in history),
            "timing_scope": "session wall time since this invocation's training loop; totals sum all saved epoch train/validation phases, excluding downtime, checkpoint I/O and final calibration",
            "completed_epochs": args.epochs,
        })
    if world > 1:
        dist.barrier()


def main() -> None:
    args = parse_args()
    try:
        prepare(args) if args.command == "prepare" else fit(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
