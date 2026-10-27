import os
import argparse
import json
import pandas as pd
import numpy as np
import torch

from train import BottleneckModel, calc_metrics, combine_embeddings, split_combined, build_split


def main():
    parser = argparse.ArgumentParser(
        description='Reproduce test-set predictions from a checkpoint trained by train.py.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model.pt produced by train.py (default: configs/test.yaml)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help="directory containing test.csv")
    parser.add_argument('--config', type=str, default='configs/test.yaml',
                        help='YAML config with parser defaults (CLI arguments override it)')
    parser.add_argument('--rna_features', type=str, default='features/Uni-RNA.pt',
                        help='pre-extracted RNA embedding dictionary (.pt, keyed by sequence)')
    parser.add_argument('--mol_features', type=str, default='features/unimol_embeddings_dict.pt',
                        help='UniMol molecule embedding dictionary (.pt, keyed by SMILES)')
    parser.add_argument('--output_dir', type=str, default='',
                        help='default: inference_repro/')
    parser.add_argument('--device', type=str, default='')
    cmd_args, _ = parser.parse_known_args()
    if cmd_args.config and os.path.exists(cmd_args.config):
        import yaml
        with open(cmd_args.config) as fh:
            cfg = yaml.safe_load(fh)
        cfg.pop('output_dir', None)  # output_dir in the config is the training default
        parser.set_defaults(**cfg)
    args = parser.parse_args()

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    targs = ckpt['args']
    print(f"Checkpoint: {args.checkpoint}")
    print(f"norm_mode: {targs['norm_mode']}, device: {device}")

    # Load embeddings (identical preprocessing as train.py)
    rna_dict = torch.load(args.rna_features, map_location='cpu', weights_only=False)
    first_rna_emb = list(rna_dict.values())[0]
    rna_dim = first_rna_emb.shape[-1] if isinstance(first_rna_emb, torch.Tensor) else len(first_rna_emb)
    smiles_dict = torch.load(args.mol_features, map_location='cpu', weights_only=False)

    # Load test data and log every dropped row with its reason
    test_df, test_dropped = build_split(
        os.path.join(args.data_dir, 'test.csv'), rna_dict, smiles_dict)
    print(f"Test: kept {len(test_df)} "
          f"(pos={(test_df.target==1).sum()}, neg={(test_df.target==0).sum()}), "
          f"dropped {len(test_dropped)}")
    if len(test_dropped):
        print(test_dropped['drop_reason'].value_counts().to_string())

    # Rebuild model with the architecture recorded in the checkpoint
    model = BottleneckModel(
        rna_dim=rna_dim, mol_dim=512,
        bottleneck_dim=targs['bottleneck_dim'],
        hidden_dim=targs['head_hidden_dim'],
        dropout=targs['dropout'],
        norm_mode=targs['norm_mode'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_df['combined_embedding'] = test_df.apply(
        lambda row: combine_embeddings(row, rna_dict, smiles_dict), axis=1
    )
    X_test = np.stack(test_df['combined_embedding'].values)
    y_test = test_df['target'].values.astype(int)
    rna_test, mol_test = split_combined(X_test)

    with torch.no_grad():
        logits = model(rna_test.to(device), mol_test.to(device))
    test_prob = torch.sigmoid(logits).cpu().numpy()

    metrics = calc_metrics(y_test, test_prob)

    out_dir = args.output_dir if args.output_dir else 'inference_repro'
    os.makedirs(out_dir, exist_ok=True)
    if len(test_dropped):
        test_dropped.assign(split='test').to_csv(
            os.path.join(out_dir, 'dropped_samples.csv'), index=False)
    df_pred = pd.DataFrame({
        'id': test_df['id'].values,
        'smiles': test_df['ligand_smiles'].values,
        'label': y_test,
        'predict_prob': test_prob,
        'predict_label': (test_prob > 0.5).astype(int)
    })
    pred_path = os.path.join(out_dir, 'test_predictions.csv')
    df_pred.to_csv(pred_path, index=False)
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                             for k, v in metrics.items()},
        }, f, indent=2)
    print(f"Predictions saved to {pred_path}")


if __name__ == '__main__':
    main()
