#!/usr/bin/env python3
"""MM-native RNA secondary-structure fine-tuning and inference export."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import re
import sys
from collections.abc import Mapping
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import danling as dl
import datasets
import pandas as pd
import torch
from chanfig import Config as ChanfigConfig
from torch import nn

from multimolecule.io import FASTA, read_fasta_records
from multimolecule.modules.backbones.sequences import SEQUENCES
from multimolecule.runner import RUNNERS, Config
from multimolecule.runner.runner import Runner
from transformers import AutoTokenizer
from rnaret_impl.tokenization_rnaret import RNAretKmerTokenizer
from multimolecule.modules import MODELS
import heads  # noqa: F401  (registers contact heads on import)
import modules  # noqa: F401  (registers sequence adapters on import)

LOGGER = logging.getLogger("unirna-ss-mm")
TABLE_SUFFIXES = {".parquet", ".csv", ".tsv", ".json", ".jsonl"}
FASTA_SUFFIXES = {f".{suffix}" for suffix in FASTA}
DEFAULT_INFERENCE_CUTOFF = 1000


@RUNNERS.register("rna_secondary_structure")
class RnaSecondaryStructureRunner(Runner):
    """Runner variant that keeps MM training/inference and swaps tokenizer build."""

    def __init__(self, config: Config | Mapping[str, Any]) -> None:
        if not isinstance(config, Config):
            config = Config(config)
        config.boot()
        if not config.training:
            config.optim = None
            config.sched = None
        dl.Runner.__init__(self, config)

        self.tokenizer = build_tokenizer(config)
        self.datasets = self.build_datasets()
        self.model = self._build_model()
        self._build_ema()
        self.train_metrics = self.build_metrics(mode="stream")
        self.evaluate_metrics = self.build_metrics(mode="stream")

    def _build_model(self):
        return MODELS.build(**self.network)

    def materialize_model(self) -> None:
        if self.model is None:
            raise ValueError("cannot materialize model: model is not initialized")

        model = self.model.to(self.device)
        self.model = model
        if self.fp8_enabled:
            self.apply_fp8_module_policy_to_model_parts()
            model = self.model

        should_wrap_ddp = self.distributed and not isinstance(
            model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)
        )
        with self.compiler.ddp_optimizer() if should_wrap_ddp else nullcontext():
            sequence = getattr(getattr(model, "backbone", None), "sequence", None)
            if sequence is not None:
                model.backbone.sequence = self.compiler.compile(sequence)
        if should_wrap_ddp:
            model = nn.parallel.DistributedDataParallel(model)
        self.model = model

        if self.ema is not None:
            self.ema = self.ema.to(self.device)


@dataclass(frozen=True)
class InferenceSource:
    name: str
    source: str
    split: str | None = None


def main() -> None:
    args = parse_args()
    if args.random_init_backbone and args.skip_training and args.ckpt is None:
        raise ValueError("--skip-training with random-init UniRNA requires --ckpt from a trained run.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_runner = build_runner(
        pretrained=args.pretrained,
        data=training_data_config(args),
        output_dir=out_dir / "train",
        args=args,
        training=True,
    )

    if args.ckpt:
        LOGGER.info("Loading model state checkpoint: %s", args.ckpt)
        train_runner.model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    if not args.skip_training:
        train_runner.train()
        load_best_checkpoint_for_inference(train_runner)
    trained_state = cpu_model_state(train_runner.model)

    import infer

    infer.run_inference(train_script=sys.modules[__name__], args=args, state_dict=trained_state, out_dir=out_dir)


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, help="YAML config with defaults for this script.")
    pre_parser.add_argument("--model", help="Model key under config.models.")
    pre_args, unknown_args = pre_parser.parse_known_args()
    if "-h" in unknown_args or "--help" in unknown_args:
        defaults = {}
    else:
        try:
            defaults = load_config(pre_args.config, model=pre_args.model)
        except ValueError as exc:
            pre_parser.error(str(exc))

    parser = argparse.ArgumentParser(
        description="MM-native RNA secondary-structure fine-tuning + inference export.",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--pretrained",
        required="pretrained" not in defaults,
        default=defaults.get("pretrained"),
        help="MM/HF model id or local checkpoint dir.",
    )
    parser.add_argument(
        "--encoder",
        default=defaults.get("encoder", "auto"),
        help="Backbone family: auto, unirna, rnaret, onehot, or another MM SEQUENCES adapter.",
    )
    parser.add_argument(
        "--encoder-factory", default=defaults.get("encoder_factory"), help="Optional non-Transformers encoder loader."
    )
    parser.add_argument("--encoder-hidden-size", type=int, default=defaults.get("encoder_hidden_size"))
    parser.add_argument(
        "--random-init-backbone", action="store_true", default=bool(defaults.get("random_init_backbone", False))
    )
    parser.add_argument(
        "--tokenizer-factory",
        default=defaults.get("tokenizer_factory"),
        help="Optional tokenizer factory: module:callable.",
    )
    parser.add_argument("--rnaret-k", type=int, default=defaults.get("rnaret_k", 1))
    parser.add_argument(
        "--trust-remote-code", action="store_true", default=bool(defaults.get("trust_remote_code", False))
    )

    parser.add_argument("--train-root", required="train_root" not in defaults, default=defaults.get("train_root"))
    parser.add_argument("--train-split", default=defaults.get("train_split", "train"))
    parser.add_argument("--validation-split", default=defaults.get("validation_split", "validation"))
    parser.add_argument("--test-split", default=defaults.get("test_split"))
    parser.add_argument("--infer", required="infer" not in defaults, nargs="+", default=defaults.get("infer"))
    parser.add_argument("--output", default=defaults.get("output"), type=Path)
    parser.add_argument("--experiment", default=defaults.get("experiment"))
    parser.add_argument("--lineage", default=defaults.get("lineage"))
    parser.add_argument("--ckpt", type=Path, default=defaults.get("ckpt"))
    parser.add_argument("--skip-training", action="store_true", default=bool(defaults.get("skip_training", False)))

    parser.add_argument("--id-col", default=defaults.get("id_col", "id"))
    parser.add_argument("--sequence-col", default=defaults.get("sequence_col", "sequence"))
    parser.add_argument("--label-col", default=defaults.get("label_col", "secondary_structure"))
    parser.add_argument("--infer-label-col", default=defaults.get("infer_label_col"))

    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 10))
    # danling nests model-selection under `score`; fall back to the old flat keys for
    # configs that predate the migration.
    score_defaults = defaults.get("score") or {}
    parser.add_argument("--patience", type=int, default=score_defaults.get("patience", defaults.get("patience")))
    parser.add_argument("--score-split", default=score_defaults.get("split", defaults.get("score_split", "validation")))
    parser.add_argument("--score-name", default=score_defaults.get("metric", defaults.get("score_name", "loss")))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 1))
    parser.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 4))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 1e-4))
    parser.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 1e-2))
    parser.add_argument("--pretrained-ratio", type=float, default=defaults.get("pretrained_ratio", 1e-2))
    parser.add_argument("--max-seq-length", type=int, default=defaults.get("max_seq_length"))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 1016))
    parser.add_argument("--precision", default=defaults.get("precision", "bf16"))
    compile_defaults = defaults.get("compile", False)
    compile_enabled = (
        bool(compile_defaults.get("enabled", compile_defaults.get("enable", False)))
        if isinstance(compile_defaults, dict)
        else bool(compile_defaults)
    )
    parser.add_argument("--compile", action="store_true", default=compile_enabled)
    parser.add_argument("--platform", default=defaults.get("platform"))
    parser.add_argument("--contact-head", default=defaults.get("contact_head", "contact.logits"))
    head_defaults = defaults.get("head", {})
    if head_defaults is None:
        head_defaults = {}
    if not isinstance(head_defaults, dict):
        parser.error("Config field 'head' must be a mapping.")
    parser.set_defaults(head_config=head_defaults)
    parser.add_argument("--probing", action="store_true", default=bool(defaults.get("probing", False)))
    parser.add_argument(
        "--gradient-checkpoint", action="store_true", default=bool(defaults.get("gradient_checkpoint", False))
    )
    ema_defaults = defaults.get("ema", {})
    if not isinstance(ema_defaults, dict):
        ema_defaults = {"enabled": bool(ema_defaults)}
    parser.add_argument("--ema", action="store_true", default=bool(ema_defaults.get("enabled", False)))
    parser.add_argument("--ema-beta", type=float, default=ema_defaults.get("beta"))
    parser.add_argument("--ema-update-after-step", type=int, default=ema_defaults.get("update_after_step"))
    parser.add_argument("--ema-update-every", type=int, default=ema_defaults.get("update_every"))
    parser.add_argument("--logits", choices=("auto", "logits", "probabilities"), default=defaults.get("logits", "auto"))
    parser.add_argument("--threshold", type=float, default=defaults.get("threshold", 0.5))
    parser.add_argument(
        "--inference-cutoff",
        type=int,
        default=defaults.get("inference_cutoff", DEFAULT_INFERENCE_CUTOFF),
        help="Sequences longer than this are inferred on CPU.",
    )
    args = parser.parse_args()
    if args.random_init_backbone:
        if args.encoder.lower() != "unirna":
            parser.error("--random-init-backbone is only supported with --encoder unirna.")
        if args.encoder_factory is not None:
            parser.error("Random-init UniRNA requires the registered local UniRNA model classes.")
    args.ema_config = build_ema_config(args)
    args.compile_config = build_compile_config(args, compile_defaults)
    if isinstance(args.output, str):
        args.output = Path(args.output)
    if args.output is None:
        if args.model is None:
            args.model = defaults.get("model")
        if args.experiment is None or args.model is None:
            parser.error("--output is required unless config provides both experiment and model")

        args.output = Path("outputs") / str(args.lineage) / str(args.experiment) / str(args.model)
    if isinstance(args.ckpt, str):
        args.ckpt = Path(args.ckpt)
    if isinstance(args.infer, str):
        args.infer = [args.infer]
    return args


def load_config(path: Path | None, *, model: str | None = None) -> dict[str, Any]:
    if path is None:
        return {}
    config = ChanfigConfig.load(path).dict()
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must be a mapping.")
    if "models" in config:
        models = config["models"]
        if not isinstance(models, dict):
            raise ValueError(f"Config {path} field 'models' must be a mapping.")
        if model is None:
            raise ValueError(f"--model is required when using multi-model config {path}.")
        if model not in models:
            raise ValueError(f"Unknown model {model!r}; available models: {', '.join(sorted(models))}.")
        config = models[model]
        if isinstance(config, dict):
            config.setdefault("experiment", path.stem)
            config.setdefault("lineage", path.parent.stem)
    if not isinstance(config, dict):
        raise ValueError(f"Selected config from {path} must be a mapping.")
    if model is not None:
        config.setdefault("model", model)
    return coerce_config_defaults({key.replace("-", "_"): expand_config_value(value) for key, value in config.items()})


def coerce_config_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    list_keys = {"infer"}
    int_keys = {
        "encoder_hidden_size",
        "rnaret_k",
        "epochs",
        "patience",
        "batch_size",
        "num_workers",
        "max_seq_length",
        "seed",
        "inference_cutoff",
    }
    float_keys = {"lr", "weight_decay", "pretrained_ratio", "threshold"}
    bool_keys = {
        "trust_remote_code",
        "skip_training",
        "probing",
        "gradient_checkpoint",
        "random_init_backbone",
    }

    for key in list_keys:
        if isinstance(defaults.get(key), str):
            defaults[key] = [defaults[key]]
    for key in int_keys:
        if defaults.get(key) is not None:
            defaults[key] = int(defaults[key])
    for key in float_keys:
        if defaults.get(key) is not None:
            defaults[key] = float(defaults[key])
    for key in bool_keys:
        if isinstance(defaults.get(key), str):
            defaults[key] = defaults[key].lower() in {"1", "true", "yes", "on"}
        elif defaults.get(key) is not None:
            defaults[key] = bool(defaults[key])
    return defaults


def build_ema_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.ema:
        return None
    config: dict[str, Any] = {"enabled": True}
    if args.ema_beta is not None:
        config["beta"] = args.ema_beta
    if args.ema_update_after_step is not None:
        config["update_after_step"] = args.ema_update_after_step
    if args.ema_update_every is not None:
        config["update_every"] = args.ema_update_every
    return config


def build_compile_config(args: argparse.Namespace, defaults: Any) -> dict[str, Any] | None:
    if not args.compile:
        return None
    if isinstance(defaults, dict):
        config = {k: v for k, v in defaults.items() if k != "enable"}
        config["enabled"] = True
        return config
    return {"enabled": True}


def normalize_precision(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"", "none", "null", "fp32", "float32"}:
        return None
    return value


def expand_config_value(value: Any) -> Any:
    if isinstance(value, str):
        return expand_shell_default(value)
    if isinstance(value, list):
        return [expand_config_value(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_config_value(item) for key, item in value.items()}
    return value


def expand_shell_default(value: str) -> str:
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default or "")

    return pattern.sub(replace, value)


def training_data_config(args: argparse.Namespace) -> dict[str, Any]:
    data = {
        "root": args.train_root,
        "train": args.train_split,
        "validation": args.validation_split,
        "feature_cols": [args.sequence_col],
        "sequence_cols": [args.sequence_col],
        "label_cols": [args.label_col],
    }
    if args.max_seq_length is not None:
        data["truncation"] = True
        data["max_seq_length"] = args.max_seq_length
    if args.test_split:
        data["test"] = args.test_split
    return data


def build_runner(
    *,
    pretrained: str,
    data: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    training: bool,
):
    runner = RUNNERS.build(
        build_config(pretrained=pretrained, data=data, output_dir=output_dir, args=args, training=training)
    )
    if not isinstance(runner, RnaSecondaryStructureRunner):
        raise TypeError(
            "Expected RnaSecondaryStructureRunner, got "
            f"{type(runner).__module__}.{type(runner).__qualname__}. "
            "Check that this script's local RUNNERS registration is being used."
        )
    LOGGER.info(
        "Runner: %s.%s train_epoch=%s.%s",
        type(runner).__module__,
        type(runner).__qualname__,
        runner.train_epoch.__func__.__module__,
        runner.train_epoch.__func__.__qualname__,
    )
    return runner


def build_config(
    *,
    pretrained: str,
    data: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    training: bool,
) -> Config:
    config = Config(
        runner=runner_type(args),
        encoder=args.encoder,
        random_init_backbone=args.random_init_backbone,
        seed=args.seed,
        training=training,
        pretrained=pretrained,
        tokenizer_factory=args.tokenizer_factory,
        trust_remote_code=args.trust_remote_code,
        data=data,
        dataloader={"batch_size": args.batch_size, "num_workers": args.num_workers},
        network={
            "type": "poly",
            "probing": args.probing,
            "backbone": {"sequence": sequence_config(pretrained, args)},
            "heads": {"labels": build_head_config(args)},
        },
        optim={
            "type": "adamw",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pretrained_ratio": args.pretrained_ratio,
        },
        sched={"type": "cosine", "final_lr": 0.0},
        epochs=args.epochs,
        precision=normalize_precision(args.precision),
        platform=args.platform,
        dir=str(output_dir),
    )
    # The runner writes to `workspace.dir`; the top-level `dir` is no longer the run directory in
    # current danling (it defaults to `experiments/<lineage>/<id>`). Pin the workspace to the
    # intended `outputs/<lineage>/<experiment>/<model>/...` path so runs land under `outputs/`.
    config.workspace.dir = str(output_dir)
    # Mutate the typed sub-configs in place. Passing `score=`/`ema=`/`compile=` as `Config(...)`
    # kwargs (or reassigning them) makes chanfig re-wrap the value as the top Config class, whose
    # `post()` expects a top-level `network` and crashes when booted as a child. Updating the
    # existing default (ScoreConfig/EmaConfig/CompileConfig) keeps the correct type.
    config.score.split = args.score_split
    config.score.metric = args.score_name
    if args.patience is not None:
        config.score.patience = args.patience
    if args.ema_config is not None:
        config.ema.update(args.ema_config)
    if args.compile_config is not None:
        config.compile.update(args.compile_config)
    return config


def build_head_config(args: argparse.Namespace) -> dict[str, Any]:
    head = dict(args.head_config)
    head["type"] = args.contact_head
    return head


def load_best_checkpoint_for_inference(runner: Runner) -> None:
    checkpoint_manager = getattr(runner, "checkpoint_manager", None)
    if checkpoint_manager is not None and hasattr(checkpoint_manager, "maybe_wait_for_staging"):
        checkpoint_manager.maybe_wait_for_staging()

    checkpoint_dir = Path(runner.workspace.checkpoint_dir)
    for best_path in (checkpoint_dir / "best.pth", checkpoint_dir / "best"):
        if best_path.exists():
            LOGGER.info("Loading best checkpoint for inference: %s", best_path)
            runner.load_checkpoint(best_path)
            return
    raise FileNotFoundError(f"No best checkpoint found under {checkpoint_dir}; aborting inference.")


def cpu_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def runner_type(args: argparse.Namespace) -> str:
    return "rna_secondary_structure"


def sequence_config(pretrained: str, args: argparse.Namespace) -> dict[str, Any]:
    encoder = args.encoder.lower()
    config: dict[str, Any] = {
        "name": pretrained,
        "use_pretrained": not bool(args.random_init_backbone) and encoder != "onehot",
        "gradient_checkpoint": args.gradient_checkpoint,
    }
    if encoder != "auto":
        sequence_type = "rna_onehot" if encoder == "onehot" else encoder
        if sequence_type not in SEQUENCES:
            raise ValueError(f"Encoder {sequence_type!r} is not registered in MM SEQUENCES.")
        config["type"] = sequence_type
        config["name"] = sequence_type if encoder == "onehot" else pretrained
        config["pretrained_path"] = None if encoder == "onehot" else pretrained
    if args.encoder_factory is not None:
        config["factory"] = args.encoder_factory
    if args.encoder_hidden_size is not None:
        config["hidden_size"] = args.encoder_hidden_size
    if args.trust_remote_code:
        config["trust_remote_code"] = True
    if encoder == "rnaret":
        config["k"] = args.rnaret_k
    if args.random_init_backbone:
        config["random_init"] = True
    return config


def build_tokenizer(config: Config):
    encoder = str(config.get("encoder", "auto")).lower()
    max_seq_length = config.data.get("max_seq_length", None)
    tokenizer_kwargs = {}
    if max_seq_length is not None:
        tokenizer_kwargs["model_max_length"] = int(max_seq_length)
    tokenizer_factory = config.get("tokenizer_factory", None)
    if tokenizer_factory is not None:
        return load_factory(tokenizer_factory)(config.pretrained, **tokenizer_kwargs)
    if encoder == "auto":
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained,
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
        # Some tokenizers expose special-token attributes without adding those tokens.
        # Keep the dataset length offset aligned with the tokens actually present in input_ids.
        probe = "ACGU"
        with suppress(Exception):
            if len(tokenizer(probe, add_special_tokens=True)["input_ids"]) == len(probe):
                for attr in ("cls_token", "sep_token", "eos_token", "bos_token"):
                    with suppress(Exception):
                        setattr(tokenizer, attr, None)
        return tokenizer
    if encoder == "onehot":
        return modules.build_onehot_tokenizer(**tokenizer_kwargs)
    if encoder == "rnaret":
        tokenizer = RNAretKmerTokenizer(k=int(config.network.backbone.sequence.get("k", 1)))
        if max_seq_length is not None:
            tokenizer.model_max_length = int(max_seq_length)
        return tokenizer
    if encoder == "aido_rna":
        # genbio ships AIDO.RNA weights only (no vocab.txt on the Hub) and its `model_type`
        # collides with multimolecule's RnaBert, so AutoTokenizer would return the wrong
        # tokenizer. Use the vendored genbio RNABertTokenizer + its bundled vocab instead.
        from aido_rna_impl import load_aido_rna_tokenizer

        return load_aido_rna_tokenizer(**tokenizer_kwargs)
    if encoder == "unirna":
        importlib.import_module("unirna_tf")
        from unirna_tf import UniRNATokenizer

        if bool(config.get("random_init_backbone", False)):
            tokenizer_kwargs["local_files_only"] = True
        return UniRNATokenizer.from_pretrained(config.pretrained, **tokenizer_kwargs)
    raise ValueError(f"Unsupported encoder tokenizer family: {encoder!r}")


def expand_inference_sources(values: list[str]) -> list[InferenceSource]:
    specs: list[InferenceSource] = []
    for value in values:
        name, source = split_named_source(value)
        path = Path(source).expanduser()
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.suffix.lower() in TABLE_SUFFIXES | FASTA_SUFFIXES)
            specs.extend(
                InferenceSource(name=f"{name}_{p.stem}" if name != path.name else p.stem, source=str(p)) for p in files
            )
            continue
        split = None
        if not path.exists() and ":" in source:
            source, split = source.rsplit(":", 1)
        specs.append(InferenceSource(name=name or Path(source).stem.replace("/", "_"), source=source, split=split))
    if not specs:
        raise ValueError("No inference datasets found.")
    return specs


def read_frame(source: str, split: str | None) -> pd.DataFrame:
    path = Path(source).expanduser()
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix in {".json", ".jsonl"}:
            return pd.read_json(path, lines=suffix == ".jsonl")
        if suffix in FASTA_SUFFIXES:
            return read_fasta_frame(path)
        raise ValueError(f"Unsupported inference file type: {path}")

    if split is None:
        raise ValueError(f"HF dataset source {source!r} requires a split, e.g. name={source}:test")
    return datasets.load_dataset(source, split=split).to_pandas()


def read_fasta_frame(path: Path) -> pd.DataFrame:
    records = [
        {"id": record.id or f"{path.stem}_{idx:06d}", "sequence": normalize_sequence(record.sequence)}
        for idx, record in enumerate(read_fasta_records(path))
    ]
    return pd.DataFrame.from_records(records)


def split_named_source(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = Path(value).expanduser()
        return path.stem if path.suffix else path.name.replace("/", "_"), value
    name, source = value.split("=", 1)
    return sanitize_name(name), source


def normalize_sequence(seq: Any) -> str:
    return re.sub(r"\s+", "", str(seq)).upper().replace("T", "U")


def sanitize_name(name: str) -> str:
    name = name.strip().replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def unique_file_stem(stem: str, counts: dict[str, int]) -> str:
    count = counts.get(stem, 0)
    counts[stem] = count + 1
    if count == 0:
        return stem
    return f"{stem}.{count}"


def load_factory(spec: str):
    module_name, attr = spec.replace(":", ".").rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr)


if __name__ == "__main__":
    main()
