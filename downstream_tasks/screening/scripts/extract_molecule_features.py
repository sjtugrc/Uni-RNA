"""Extract UniMol embeddings for a list of SMILES (vendored unimol_lite, no external
package dependencies beyond torch/transformers/rdkit).

Outputs a .pt dict {smiles: 512-d vector}; SMILES that fail extraction map to None and
are skipped on save, mirroring the original pipeline's behavior.
"""
import os
import sys
import argparse
import multiprocessing as mp

_THIRD_PARTY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party")
sys.path.insert(0, _THIRD_PARTY)

DEFAULT_WEIGHTS = os.environ.get(
    "UR_UNIMOL_WEIGHTS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "weights", "base_unimol"))


def extract_chunk(smiles_list, model_path, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from unimol_lite.model import UniMolModel, UniMolConfig
    from unimol_lite.data import smiles2unimolitem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniMolModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    def move_to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [move_to_device(v) for v in obj]
        return obj

    embeddings = {}
    for smiles in smiles_list:
        try:
            items = move_to_device(smiles2unimolitem(smiles))
            with torch.no_grad():
                encoder_output, _ = model(**items)
            embeddings[smiles] = encoder_output.mean(dim=1).squeeze().cpu().numpy()
        except Exception:
            embeddings[smiles] = None
    return embeddings


def _worker(args):
    chunk, model_path, gpu = args
    return extract_chunk(chunk, model_path, gpu)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_txt", required=True,
                        help="text file with one SMILES per line")
    parser.add_argument("--output", required=True, help="output .pt path")
    parser.add_argument("--model_path", default=DEFAULT_WEIGHTS,
                        help="UniMol pretrained weights (overridable via UR_UNIMOL_WEIGHTS)")
    parser.add_argument("--n_workers", type=int, default=8)
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    smiles = [l.strip() for l in open(args.smiles_txt) if l.strip()]
    print(f"Total unique SMILES: {len(smiles)}")

    import torch as _torch
    n_gpus = _torch.cuda.device_count()
    n = min(args.n_workers, mp.cpu_count())
    chunks = [smiles[i::n] for i in range(n)]

    merged, failed = {}, 0
    with mp.Pool(n) as pool:
        for result in pool.imap_unordered(
                _worker, [(c, args.model_path, i % max(1, n_gpus)) for i, c in enumerate(chunks)]):
            for s, v in result.items():
                if v is not None:
                    merged[s] = v
                else:
                    failed += 1

    print(f"Extracted: {len(merged)}, failed: {failed}")
    import torch
    torch.save(merged, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
