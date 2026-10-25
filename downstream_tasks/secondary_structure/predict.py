# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

from __future__ import annotations

import argparse
import logging
import re
from contextlib import suppress
from pathlib import Path

import torch
from transformers import pipeline

import multimolecule as mm  # noqa: F401
from multimolecule.io import RnaSecondaryStructureRecord, read_fasta_records, write_dbn

LOGGER = logging.getLogger("predict")
ACCELERATOR_ERROR = getattr(torch, "AcceleratorError", RuntimeError)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict RNA secondary structures with a MultiMolecule pipeline.")
    parser.add_argument("--input", nargs="+", type=Path, help="Input FASTA file(s).")
    parser.add_argument("-o", "--output", type=Path, default=Path("predictions"), help="Output directory for DBN files.")
    parser.add_argument("-m", "--model", default="ufold", help="Model name or local path.")
    parser.add_argument("--device", default="auto", help="Pipeline device: auto, cpu, cuda, or a CUDA device index.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing DBN files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    model = f"multimolecule/{args.model}" if "/" not in args.model else args.model
    engine = PipelineInferenceEngine(model=model, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    total = 0
    try:
        for fasta_path in args.input:
            records = sorted(
                enumerate(read_fasta_records(fasta_path), start=1),
                key=lambda item: len(item[1].sequence),
            )
            for index, record in records:
                record_id = record.id or f"{fasta_path.stem}_{index}"
                output_path = args.output / f"{_safe_name(record_id)}.dbn"
                if output_path.exists() and not args.overwrite:
                    continue

                try:
                    output = engine.predict(record.sequence)
                except Exception:
                    LOGGER.exception("Prediction failed for id=%s length=%s", record_id, len(record.sequence))
                    continue

                write_dbn(
                    RnaSecondaryStructureRecord(
                        id=record_id,
                        sequence=output["sequence"],
                        dot_bracket=output["secondary_structure"],
                    ),
                    output_path,
                )
                total += 1
    finally:
        engine.close()

    print(f"Wrote {total} DBN file(s) to {args.output}")


class PipelineInferenceEngine:
    def __init__(self, *, model: str, device: str) -> None:
        self.model = model
        self.device = device
        self.predictors = {}
        self.cuda_disabled = False

    def close(self) -> None:
        self.predictors.clear()
        if torch.cuda.is_available():
            with suppress(Exception):
                torch.cuda.empty_cache()

    def predict(self, sequence: str):
        device = self._select_device()
        try:
            return self._predictor(device)(sequence)
        except (torch.cuda.OutOfMemoryError, ACCELERATOR_ERROR, RuntimeError) as exc:
            if _is_cuda_device(device):
                LOGGER.exception("CUDA prediction failed for length=%s; retrying on CPU", len(sequence))
                self.cuda_disabled = True
                self.predictors.pop("cuda", None)
                if torch.cuda.is_available():
                    with suppress(Exception):
                        torch.cuda.empty_cache()
                return self._predictor(-1)(sequence)
            raise exc

    def _predictor(self, device: int | str):
        key = "cuda" if _is_cuda_device(device) else "cpu"
        if key not in self.predictors:
            self.predictors[key] = pipeline("rna-secondary-structure", model=self.model, device=device)
        return self.predictors[key]

    def _select_device(self) -> int | str:
        if self.device == "cpu" or self.cuda_disabled or not torch.cuda.is_available():
            return -1
        if self.device == "auto" or self.device == "cuda":
            return 0
        try:
            parsed = int(self.device)
        except ValueError:
            return self.device
        return parsed


def _is_cuda_device(device: int | str) -> bool:
    if isinstance(device, int):
        return device >= 0
    return str(device).startswith("cuda")


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "sequence"


if __name__ == "__main__":
    main()
