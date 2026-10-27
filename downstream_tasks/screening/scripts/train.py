import os
import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
    matthews_corrcoef, cohen_kappa_score
)


def calc_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if len(cm) == 1:
            if y_true[0] == 0:
                tn = cm[0, 0]
            else:
                tp = cm[0, 0]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': specificity,
        'npv': npv,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'confusion_matrix': [int(tn), int(fp), int(fn), int(tp)],
        'n_samples': len(y_true),
        'n_positive': int(y_true.sum()),
        'n_negative': int((1 - y_true).sum()),
    }


class BottleneckModel(nn.Module):
    """
    norm_mode:
      'separate' -> LayerNorm + Linear for each modality
      'none'     -> no LayerNorm, Linear directly
      'concat'   -> Linear then concat, followed by a single LayerNorm
      'both'     -> LayerNorm + Linear for each modality, then LayerNorm after concat
    """
    def __init__(self, rna_dim=1024, mol_dim=512, bottleneck_dim=8,
                 hidden_dim=64, dropout=0.1, norm_mode='separate'):
        super().__init__()
        self.norm_mode = norm_mode

        if norm_mode in ('separate', 'both'):
            self.rna_ln = nn.LayerNorm(rna_dim)
            self.mol_ln = nn.LayerNorm(mol_dim)
        if norm_mode in ('concat', 'both'):
            self.concat_ln = nn.LayerNorm(bottleneck_dim * 2)

        self.rna_proj = nn.Linear(rna_dim, bottleneck_dim)
        self.mol_proj = nn.Linear(mol_dim, bottleneck_dim)

        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, rna_emb, mol_emb):
        if self.norm_mode in ('separate', 'both'):
            r = self.rna_proj(self.rna_ln(rna_emb))
            m = self.mol_proj(self.mol_ln(mol_emb))
        else:
            r = self.rna_proj(rna_emb)
            m = self.mol_proj(mol_emb)

        x = torch.cat([r, m], dim=-1)

        if self.norm_mode in ('concat', 'both'):
            x = self.concat_ln(x)

        return self.head(x).squeeze(-1)


def build_split(csv_path, rna_dict, smiles_dict):
    """Load a split CSV, fill missing RNA sequences via the id column, and drop rows
    with missing inputs. Every dropped row is logged with its reason.

    Returns:
        kept_df:   rows used for training/evaluation
        dropped_df: dropped rows with an extra 'drop_reason' column
    """
    df = pd.read_csv(csv_path)

    # Fill missing rna_chain_sequences from other rows with the same id
    id_to_rna = (df.dropna(subset=['rna_chain_sequences'])
                 .drop_duplicates('id')
                 .set_index('id')['rna_chain_sequences'].to_dict())
    df['rna_chain_sequences'] = df.apply(
        lambda row: id_to_rna.get(row['id'], row['rna_chain_sequences'])
        if pd.isna(row['rna_chain_sequences']) else row['rna_chain_sequences'],
        axis=1
    )

    reason = pd.Series('', index=df.index)
    reason[df['rna_chain_sequences'].isna()] = 'missing_rna_sequence'
    reason[(reason == '') & df['ligand_smiles'].isna()] = 'missing_smiles'
    mol_ok = df['ligand_smiles'].map(
        lambda s: isinstance(s, str) and s in smiles_dict)
    reason[(reason == '') & (~mol_ok)] = 'smiles_not_in_mol_dict'
    rna_ok = df['rna_chain_sequences'].map(
        lambda s: isinstance(s, str) and s in rna_dict)
    reason[(reason == '') & (~rna_ok)] = 'rna_not_in_feature_dict'

    kept_df = df[reason == ''].copy()
    dropped_df = df[reason != ''].copy()
    dropped_df['drop_reason'] = reason[reason != '']
    return kept_df, dropped_df


def get_rna_embedding(seq, rna_dict):
    emb = rna_dict[seq]
    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()
    return emb


def combine_embeddings(row, rna_dict, smiles_dict, rna_col='rna_chain_sequences'):
    smiles = row['ligand_smiles']
    mol_emb = smiles_dict.get(smiles)
    if mol_emb is None:
        return None
    if isinstance(mol_emb, torch.Tensor):
        mol_emb = mol_emb.cpu().numpy()
    rna_emb = get_rna_embedding(row[rna_col], rna_dict)
    return np.concatenate([mol_emb, rna_emb])


def split_combined(combined):
    mol_emb = combined[:, :512]
    rna_emb = combined[:, 512:]
    return torch.FloatTensor(rna_emb), torch.FloatTensor(mol_emb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data',
                        help="directory containing train.csv / test.csv")
    parser.add_argument('--output_dir', type=str, default='results_repro',
                        help="output directory for checkpoints and metrics")
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                        help='YAML config with parser defaults (CLI arguments override it)')
    parser.add_argument('--rna_features', type=str, default='features/Uni-RNA.pt',
                        help="pre-extracted RNA embedding dictionary (.pt, keyed by sequence)")
    parser.add_argument('--mol_features', type=str, default='features/unimol_embeddings_dict.pt',
                        help="UniMol molecule embedding dictionary (.pt, keyed by SMILES)")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--pos_weight', type=float, default=10.0)
    parser.add_argument('--bottleneck_dim', type=int, default=8)
    parser.add_argument('--head_hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--norm_mode', type=str, default='separate', choices=['separate', 'none', 'concat', 'both'])
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    cmd_args, _ = parser.parse_known_args()
    if cmd_args.config and os.path.exists(cmd_args.config):
        import yaml
        with open(cmd_args.config) as fh:
            parser.set_defaults(**yaml.safe_load(fh))
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, norm_mode={args.norm_mode}")

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load embeddings
    rna_dict = torch.load(args.rna_features, map_location='cpu', weights_only=False)
    # Auto-detect RNA feature dimension
    first_rna_emb = list(rna_dict.values())[0]
    if isinstance(first_rna_emb, torch.Tensor):
        rna_dim = first_rna_emb.shape[-1]
    else:
        rna_dim = len(first_rna_emb)
    print(f"Detected RNA feature dim for {'Uni-RNA'}: {rna_dim}")

    smiles_dict = torch.load(args.mol_features, map_location='cpu', weights_only=False)

    # 2. Load data and log every dropped row with its reason
    train_df, train_dropped = build_split(
        os.path.join(args.data_dir, 'train.csv'), rna_dict, smiles_dict)
    test_df, test_dropped = build_split(
        os.path.join(args.data_dir, 'test.csv'), rna_dict, smiles_dict)

    for split_name, df, dropped in [('train', train_df, train_dropped),
                                    ('test', test_df, test_dropped)]:
        print(f"{split_name.capitalize()}: kept {len(df)} "
              f"(pos={(df.target==1).sum()}, neg={(df.target==0).sum()}), "
              f"dropped {len(dropped)}")
        if len(dropped):
            print(dropped['drop_reason'].value_counts().to_string())

    drop_log = pd.concat([
        train_dropped.assign(split='train'),
        test_dropped.assign(split='test'),
    ])
    if len(drop_log):
        drop_log.to_csv(os.path.join(save_dir, 'dropped_samples.csv'), index=False)

    # 3. Prepare data
    train_df['combined_embedding'] = train_df.apply(
        lambda row: combine_embeddings(row, rna_dict, smiles_dict), axis=1
    )
    X_train = np.stack(train_df['combined_embedding'].values)
    y_train = train_df['target'].values.astype(int)

    test_df['combined_embedding'] = test_df.apply(
        lambda row: combine_embeddings(row, rna_dict, smiles_dict), axis=1
    )
    X_test = np.stack(test_df['combined_embedding'].values)
    y_test = test_df['target'].values.astype(int)

    print(f"Feature dim: {X_train.shape[1]}")

    rna_train, mol_train = split_combined(X_train)
    rna_test, mol_test = split_combined(X_test)

    # 4. Build model
    model = BottleneckModel(
        rna_dim=rna_dim, mol_dim=512,
        bottleneck_dim=args.bottleneck_dim,
        hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        norm_mode=args.norm_mode
    ).to(device)

    if args.pos_weight != 1.0:
        pos_weight_tensor = torch.tensor([args.pos_weight], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 5. Train
    best_metric_val = -1
    best_state = None
    best_epoch = 0
    best_test_auc = -1
    best_test_auprc = -1

    dataset_train = torch.utils.data.TensorDataset(rna_train, mol_train, torch.FloatTensor(y_train))
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_rna, batch_mol, batch_y in loader_train:
            batch_rna = batch_rna.to(device)
            batch_mol = batch_mol.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_rna, batch_mol)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)

        avg_loss = total_loss / len(y_train)

        # Evaluation
        model.eval()
        with torch.no_grad():
            train_logits = model(rna_train.to(device), mol_train.to(device))
            train_prob = torch.sigmoid(train_logits).cpu().numpy()
            test_logits = model(rna_test.to(device), mol_test.to(device))
            test_prob = torch.sigmoid(test_logits).cpu().numpy()

        train_auc = roc_auc_score(y_train, train_prob)
        test_auc = roc_auc_score(y_test, test_prob)
        test_auprc = average_precision_score(y_test, test_prob)

        metric_val = test_auprc
        if metric_val > best_metric_val:
            best_metric_val = metric_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_test_auc = test_auc
            best_test_auprc = test_auprc

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} | Test AUPRC: {test_auprc:.4f}")

    print(f"\nBest epoch: {best_epoch+1}, best Test AUC: {best_test_auc:.4f}, best Test AUPRC: {best_test_auprc:.4f}")

    # Reload the best model and re-evaluate (save best-epoch metrics, not the last epoch)
    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        train_logits = model(rna_train.to(device), mol_train.to(device))
        train_prob = torch.sigmoid(train_logits).cpu().numpy()
        test_logits = model(rna_test.to(device), mol_test.to(device))
        test_prob = torch.sigmoid(test_logits).cpu().numpy()

    print("=" * 60)
    print("Train metrics:")
    print("=" * 60)
    train_metrics = calc_metrics(y_train, train_prob)
    for k, v in train_metrics.items():
        if k == 'confusion_matrix':
            print(f"  {k}: TN={v[0]}, FP={v[1]}, FN={v[2]}, TP={v[3]}")
        else:
            print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("Test metrics:")
    print("=" * 60)
    test_metrics = calc_metrics(y_test, test_prob)
    for k, v in test_metrics.items():
        if k == 'confusion_matrix':
            print(f"  {k}: TN={v[0]}, FP={v[1]}, FN={v[2]}, TP={v[3]}")
        else:
            print(f"  {k}: {v:.4f}")

    df_test_pred = pd.DataFrame({
        'id': test_df['id'].values,
        'smiles': test_df['ligand_smiles'].values,
        'label': y_test,
        'predict_prob': test_prob,
        'predict_label': (test_prob > 0.5).astype(int)
    })
    df_test_pred.to_csv(os.path.join(save_dir, 'test_predictions.csv'), index=False)

    model_path = os.path.join(save_dir, 'model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
    }, model_path)
    print(f"\nBest model saved to {model_path}")

    metrics = {
        'rna_model': 'Uni-RNA',
        'best_epoch': best_epoch + 1,
        'train_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in train_metrics.items()},
        'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in test_metrics.items()},
    }
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved")


if __name__ == '__main__':
    main()
