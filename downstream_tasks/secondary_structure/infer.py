#!/usr/bin/env python3
"""MM-native RNA secondary-structure inference export without DanLing Runner."""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from chanfig import Config as ChanfigConfig
from danling import NestedTensor
from torch import Tensor, nn

from multimolecule.io import RnaSecondaryStructureRecord, write_dbn
from multimolecule.modules import MODELS
from multimolecule.runner import Config
from multimolecule.utils import contact_map_to_dot_bracket

LOGGER = logging.getLogger("unirna-ss-mm.infer")
ACCELERATOR_ERROR = getattr(torch, "AcceleratorError", RuntimeError)


class InferenceEngine:
    def __init__(self, *, train_script: Any, args: Any, state_dict: Mapping[str, Tensor]) -> None:
        self.train_script = train_script
        self.args = args
        self.state_dict = normalize_model_state(state_dict)
        self.tokenizer = self._build_tokenizer()
        self.model: nn.Module | None = None
        self.model_device: torch.device | None = None
        self.cuda_disabled = False

    def close(self) -> None:
        self.model = None
        self.model_device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def infer(self, sequence: str, *, prefer_cuda: bool) -> np.ndarray:
        device = self._select_device(prefer_cuda=prefer_cuda)
        try:
            return self._infer_on_device(sequence, device=device)
        except (torch.cuda.OutOfMemoryError, ACCELERATOR_ERROR, RuntimeError) as exc:
            if device.type == "cuda":
                LOGGER.exception("CUDA inference failed for length=%s; retrying on CPU", len(sequence))
                self.cuda_disabled = True
                with suppress(Exception):
                    torch.cuda.empty_cache()
                return self._infer_on_device(sequence, device=torch.device("cpu"))
            raise exc

    def _infer_on_device(self, sequence: str, *, device: torch.device) -> np.ndarray:
        model = self._model(device)
        input_ids = self._encode(sequence).to(device)
        with torch.inference_mode(), self._autocast(device):
            output, _ = model.backbone(input_ids)
            if model.neck is not None:
                output = model.neck(**output)
            prediction = model.heads["labels"](output, input_ids=input_ids).logits
        if isinstance(prediction, NestedTensor):
            prediction = prediction.tensor
        return prediction.detach().float().cpu().numpy()[0]

    def prepare(self) -> None:
        """Build and strictly load the saved run's model once before processing records."""
        device = self._select_device(prefer_cuda=True)
        LOGGER.info("Preparing inference model from saved run config on %s", device)
        self._model(device)

    def _model(self, device: torch.device) -> nn.Module:
        if self.model is None:
            model = self._build_model()
            model.load_state_dict(self.state_dict, strict=True)
            model.eval()
            self.model = model
            self.model_device = torch.device("cpu")
        if self.model_device != device:
            self.model.to(device)
            self.model_device = device
        return self.model

    def _run_config_path(self) -> Path:
        output_config = Path(self.args.output).resolve() / "train" / "metadata" / "config.full.yaml"
        if output_config.is_file():
            return output_config
        checkpoint = getattr(self.args, "ckpt", None)
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint).resolve()
            for parent in checkpoint_path.parents:
                if parent.name == "train":
                    candidate = parent / "metadata" / "config.full.yaml"
                    if candidate.is_file():
                        return candidate
        raise FileNotFoundError(
            "Saved effective training config not found beside the selected output or checkpoint; "
            "refusing to reconstruct a potentially different model head."
        )

    def _build_model(self) -> nn.Module:
        # This network subtree was persisted after the training runner inferred its task from
        # the training dataset. It includes the exact head task metadata and dimensions.
        run_config_path = self._run_config_path()
        run_config = ChanfigConfig.load(run_config_path).dict()
        network = run_config.get("network")
        if not isinstance(network, Mapping):
            raise ValueError(f"Saved effective config {run_config_path} has no network mapping.")
        heads = network.get("heads")
        label_head = heads.get("labels") if isinstance(heads, Mapping) else None
        if not isinstance(label_head, Mapping) or label_head.get("num_labels") is None:
            raise ValueError(f"Saved effective config {run_config_path} has no labels head num_labels.")
        saved_num_labels = int(label_head["num_labels"])
        weight_key = "heads.labels.decoder.prediction.weight"
        checkpoint_weight = self.state_dict.get(weight_key)
        if not isinstance(checkpoint_weight, Tensor) or checkpoint_weight.ndim != 2:
            raise ValueError(f"Checkpoint is missing the expected 2D tensor {weight_key}.")
        checkpoint_num_labels = int(checkpoint_weight.shape[0])
        if saved_num_labels != checkpoint_num_labels:
            raise ValueError(
                f"Saved run config {run_config_path} declares num_labels={saved_num_labels}, "
                f"but the checkpoint head has {checkpoint_num_labels} outputs."
            )
        LOGGER.info(
            "Building model from %s (labels.num_labels=%s, problem_type=%s)",
            run_config_path,
            saved_num_labels,
            label_head.get("problem_type"),
        )
        return MODELS.build(**dict(network))

    def _build_tokenizer(self) -> Any:
        config = Config(
            pretrained=self.args.pretrained,
            encoder=self.args.encoder,
            random_init_backbone=self.args.random_init_backbone,
            tokenizer_factory=self.args.tokenizer_factory,
            trust_remote_code=self.args.trust_remote_code,
            data={"max_seq_length": self.args.max_seq_length} if self.args.max_seq_length is not None else {},
            network={"backbone": {"sequence": self.train_script.sequence_config(self.args.pretrained, self.args)}},
        )
        config.boot()
        return self.train_script.build_tokenizer(config)

    def _encode(self, sequence: str) -> Tensor:
        encoded = self.tokenizer(sequence, return_tensors="pt", padding=False, truncation=False)
        input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if not torch.is_tensor(input_ids):
            input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        return input_ids.long()

    def _select_device(self, *, prefer_cuda: bool) -> torch.device:
        platform = None if self.args.platform is None else str(self.args.platform)
        if self.cuda_disabled or not prefer_cuda or platform == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        if platform and platform.startswith("cuda"):
            return torch.device(platform)
        return torch.device("cuda")

    def _autocast(self, device: torch.device):
        precision = self.train_script.normalize_precision(self.args.precision)
        if device.type == "cuda" and precision == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and precision in {"fp16", "float16"}:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return torch.autocast(device_type=device.type, enabled=False)


def main() -> None:
    train_script = resolve_train_script()
    args = train_script.parse_args()
    # print(args)
    # import ipdb;ipdb.set_trace()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    out_dir = args.output.resolve()
    checkpoint =  args.ckpt or find_best_checkpoint(out_dir)
    LOGGER.info("Loading model state from checkpoint: %s", checkpoint)
    state_dict = load_model_state(checkpoint)
    run_inference(train_script=train_script, args=args, state_dict=state_dict, out_dir=out_dir)


def resolve_train_script() -> Any:
    main_module = sys.modules.get("__main__")
    if main_module is not None and hasattr(main_module, "build_tokenizer"):
        return main_module
    import train as train_script

    return train_script


def run_inference(*, train_script: Any, args: Any, state_dict: Mapping[str, Tensor], out_dir: Path) -> dict[str, int]:
    engine = InferenceEngine(train_script=train_script, args=args, state_dict=state_dict)
    totals = {"input": 0, "predicted": 0, "failed": 0, "skipped": 0}
    failures_by_source: list[str] = []
    sources = train_script.expand_inference_sources(args.infer)
    try:
        # Fail once at startup if the saved model/config cannot be strictly loaded.
        engine.prepare()
        for source in sources:
            LOGGER.info("Preparing inference dataset: %s", source)
            frame = train_script.read_frame(source.source, source.split)
            frame = normalize_frame(train_script=train_script, args=args, source=source, frame=frame)
            prediction_dir = out_dir / "infer" / source.name
            counts = infer_frame(engine=engine, train_script=train_script, args=args, frame=frame, out_dir=prediction_dir)
            LOGGER.info(
                "Dataset summary: %s inputs=%s predictions=%s failures=%s skipped=%s",
                source.name,
                counts["input"],
                counts["predicted"],
                counts["failed"],
                counts["skipped"],
            )
            for key in totals:
                totals[key] += counts[key]
            if counts["failed"]:
                failures_by_source.append(f"{source.name}={counts['failed']}")
    finally:
        engine.close()

    LOGGER.info(
        "Inference summary: sources=%s inputs=%s predictions=%s failures=%s skipped=%s",
        len(sources),
        totals["input"],
        totals["predicted"],
        totals["failed"],
        totals["skipped"],
    )
    if totals["input"] != totals["predicted"] + totals["failed"] + totals["skipped"]:
        raise RuntimeError(f"Inference accounting mismatch: {totals}")
    if totals["failed"]:
        raise RuntimeError(
            f"Inference failed for {totals['failed']} of {totals['input']} records "
            f"({', '.join(failures_by_source)}); inspect the corresponding .error files under {out_dir / 'infer'}."
        )
    return totals


def normalize_frame(*, train_script: Any, args: Any, source: Any, frame: pd.DataFrame) -> pd.DataFrame:
    if args.sequence_col not in frame.columns and args.sequence_col == "sequence" and "seq" in frame.columns:
        frame = frame.rename(columns={"seq": args.sequence_col})
    if args.sequence_col not in frame.columns:
        raise ValueError(f"{source.source} does not contain sequence column {args.sequence_col!r}.")
    if args.id_col not in frame.columns:
        frame[args.id_col] = [f"{source.name}_{idx:06d}" for idx in range(len(frame))]
    frame[args.sequence_col] = frame[args.sequence_col].map(train_script.normalize_sequence)
    return frame


def infer_frame(*, engine: InferenceEngine, train_script: Any, args: Any, frame: pd.DataFrame, out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    file_counts: dict[str, int] = {}
    written = 0
    failed = 0
    skipped = 0
    max_length = int(args.max_seq_length) if args.max_seq_length is not None else None
    ordered = frame.assign(_sequence_length=frame[args.sequence_col].map(len)).sort_values(
        "_sequence_length", kind="mergesort"
    )
    for idx, (_, row) in enumerate(ordered.iterrows()):
        record_id = str(row[args.id_col])
        sequence = train_script.normalize_sequence(row[args.sequence_col])
        file_id = train_script.unique_file_stem(train_script.sanitize_name(record_id) or f"record_{idx:06d}", file_counts)
        # Honor the configured sequence limit by skipping instead of truncating.
        if max_length is not None and len(sequence) > max_length:
            skipped += 1
            continue
        try:
            prefer_cuda = len(sequence) <= int(args.inference_cutoff)
            prediction = engine.infer(sequence, prefer_cuda=prefer_cuda)
            write_prediction(
                pred=prediction,
                record_id=record_id,
                sequence=sequence,
                out_dir=out_dir,
                file_id=file_id,
                threshold=args.threshold,
                logits=args.logits,
            )
            written += 1
        except Exception as exc:
            failed += 1
            LOGGER.exception("Inference failed for id=%s length=%s", record_id, len(sequence))
            write_error(out_dir / f"{file_id}.error", exc)
        finally:
            if torch.cuda.is_available():
                with suppress(Exception):
                    torch.cuda.empty_cache()
    if skipped:
        LOGGER.info("Skipped %s sequence(s) longer than max_seq_length=%s (scored as misses)", skipped, max_length)
    return {"input": len(frame), "predicted": written, "failed": failed, "skipped": skipped}


def find_best_checkpoint(out_dir: Path) -> Path:
    checkpoint_dir = out_dir / "train" / "checkpoints"
    candidates = (checkpoint_dir / "best.pth", checkpoint_dir / "best")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No best checkpoint found under {checkpoint_dir}; pass --checkpoint explicitly.")


def load_model_state(checkpoint: Path) -> dict[str, Tensor]:
    payload: Any = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]
    if not isinstance(payload, dict) or not all(torch.is_tensor(value) for value in payload.values()):
        raise ValueError(f"Checkpoint {checkpoint} does not contain a plain model state dict.")
    return normalize_model_state(payload)


def normalize_model_state(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {normalize_state_key(key): value.detach().cpu() for key, value in state_dict.items()}


def normalize_state_key(key: str) -> str:
    key = key.removeprefix("_orig_mod.")
    return key.replace("._orig_mod", "")


def write_prediction(
    *,
    pred: Any,
    record_id: str,
    sequence: str,
    out_dir: Path,
    file_id: str,
    threshold: float,
    logits: str,
) -> None:
    score = prediction_to_probability(pred, logits=logits)
    score = score[: len(sequence), : len(sequence)]
    score = ((score + score.T) / 2.0).astype(np.float32)
    dbn = contact_map_to_dot_bracket(score, unsafe=True, threshold=threshold, matching="greedy")
    write_dbn(RnaSecondaryStructureRecord(sequence=sequence, dot_bracket=dbn, id=record_id), out_dir / f"{file_id}.dbn")
    np.savetxt(out_dir / f"{file_id}.prob", score, fmt="%.8g", delimiter="\t")


def prediction_to_probability(pred: Any, *, logits: str) -> np.ndarray:
    arr = np.asarray(pred, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    if arr.ndim == 2 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D contact map prediction, got shape {arr.shape}")
    if logits == "logits" or (logits == "auto" and (np.nanmin(arr) < 0 or np.nanmax(arr) > 1)):
        arr = 1.0 / (1.0 + np.exp(-arr))
    return arr


def write_error(path: Path, exc: Exception) -> None:
    path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
