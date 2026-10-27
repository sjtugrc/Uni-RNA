import argparse
import os
import torch
from types import SimpleNamespace
from model import build_model


def parse_fasta(fasta_path):
    """Parse a FASTA file and return raw sequence strings (case and spaces preserved)."""
    sequences = []
    with open(fasta_path, 'r') as f:
        current_seq = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = ""
            else:
                current_seq += line
        if current_seq:
            sequences.append(current_seq)
    return sequences


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-sequence embeddings from a FASTA file using the Uni-RNA "
                    "encoder in model.py, saved as a {sequence: vector} dict via torch.save."
    )
    parser.add_argument(
        "--model", type=str, default="Uni-RNA", choices=["Uni-RNA"],
        help="encoder to use (this archive ships Uni-RNA features)"
    )
    parser.add_argument(
        "--features_mode", type=str, default="cls_token",
        choices=["cls_token", "mean_pooling"],
        help="feature extraction mode"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to run on"
    )
    parser.add_argument(
        "--fasta", type=str, required=True,
        help="input FASTA file"
    )
    parser.add_argument(
        "--output", type=str, default="features.pt",
        help="output .pt file (dict saved via torch.save)"
    )
    args = parser.parse_args()

    # args object for build_model (only the encoder is used)
    model_args = SimpleNamespace(
        base_model_name=args.model,
        base_model_path="",          # use the default location in model.py
        features_mode=args.features_mode,
        task_type="reg",             # placeholder; only the encoder is used
        random_init_base=False
    )

    print(f"Loading encoder: {args.model} (features_mode={args.features_mode}) ...")
    full_model = build_model(model_args, output_dim=1)
    encoder = full_model.encoder
    encoder.eval()
    encoder.to(args.device)

    sequences = parse_fasta(args.fasta)
    print(f"Found {len(sequences)} sequences from {args.fasta}")

    results = {}
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            # one sequence per call: sequence must be a list with a single string
            feat = encoder(sequence=[seq])
            feat = feat.squeeze(0).cpu()  # (output_dim,)
            results[seq] = feat
            if (i + 1) % 10 == 0 or (i + 1) == len(sequences):
                print(f"Processed {i + 1}/{len(sequences)} sequences")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(results, args.output)
    print(f"Saved {len(results)} entries to {args.output}")


if __name__ == "__main__":
    main()
