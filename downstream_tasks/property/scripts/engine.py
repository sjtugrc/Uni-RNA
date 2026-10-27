import os
import logging
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, rankdata
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, r2_score

from data import read_data, scale_data, RNADataset, collate_fn
from model import build_model, load_ckpt
from utils import set_seed

def metrics_reg(y_true, y_pred, calc_method='feature_wise'):
    # feature_wise: average over features; sample_wise: average over samples (e.g. 3UTR_test_dis)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    axis = 0 if calc_method == 'feature_wise' else 1

    if calc_method == 'sample_wise':
        assert y_true.shape[1] > 1, f"Number of features must be greater than 1 for sample_wise calculation, got {y_true.shape[1]}"

    with np.errstate(divide='ignore', invalid='ignore'):
        x_mean = np.mean(y_true, axis=axis, keepdims=True)
        y_mean = np.mean(y_pred, axis=axis, keepdims=True)
        x_diff = y_true - x_mean
        y_diff = y_pred - y_mean
        cov = np.sum(x_diff * y_diff, axis=axis)
        std_x_sum = np.sqrt(np.sum(x_diff**2, axis=axis))
        std_y_sum = np.sqrt(np.sum(y_diff**2, axis=axis))

        valid = (std_x_sum > 0) & (std_y_sum > 0)
        pearson = np.zeros_like(cov)
        pearson[valid] = cov[valid] / (std_x_sum[valid] * std_y_sum[valid])

        x_rank = rankdata(y_true, axis=axis)
        y_rank = rankdata(y_pred, axis=axis)
        xr_mean = np.mean(x_rank, axis=axis, keepdims=True)
        yr_mean = np.mean(y_rank, axis=axis, keepdims=True)
        xr_diff = x_rank - xr_mean
        yr_diff = y_rank - yr_mean

        cov_r = np.sum(xr_diff * yr_diff, axis=axis)
        std_xr_sum = np.sqrt(np.sum(xr_diff**2, axis=axis))
        std_yr_sum = np.sqrt(np.sum(yr_diff**2, axis=axis))

        valid_r = (std_xr_sum > 0) & (std_yr_sum > 0)
        spearman = np.zeros_like(cov_r)
        spearman[valid_r] = cov_r[valid_r] / (std_xr_sum[valid_r] * std_yr_sum[valid_r])

        if calc_method == 'feature_wise':
            r2 = r2_score(y_true, y_pred)
        else:
            ss_res = np.sum((y_true - y_pred)**2, axis=axis)
            ss_tot = np.sum(x_diff**2, axis=axis)
            valid_r2 = ss_tot > 0
            r2_scores = np.zeros_like(ss_res)
            r2_scores[valid_r2] = 1 - ss_res[valid_r2] / ss_tot[valid_r2]
            r2 = np.mean(r2_scores)

    res = {'pearson': np.mean(pearson), 'spearman': np.mean(spearman), 'r2': r2}
    return {k: round(float(v), 4) for k, v in res.items()}

def metrics_cls(y_true, y_pred, y_prob):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.flatten()
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()

    if y_true.ndim == 2 and y_pred.ndim == 1:
        y_true = np.argmax(y_true, axis=1)

    res = {}
    res['acc'] = accuracy_score(y_true, y_pred)
    res['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    res['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    res['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

    try:
        # infer task type from model output
        is_multiclass = (y_prob.ndim == 2 and y_prob.shape[1] > 2)

        if is_multiclass:
            res['auroc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')

            # robust one-hot conversion
            y_true_bin = np.eye(y_prob.shape[1])[y_true]
            res['auprc'] = average_precision_score(y_true_bin, y_prob, average='macro')

        else:
            # fix binary shape
            if y_prob.ndim == 2:
                if y_prob.shape[1] == 2:
                    y_prob_binary = y_prob[:, 1]
                elif y_prob.shape[1] == 1:
                    y_prob_binary = y_prob[:, 0]
                else:
                    raise ValueError("Invalid y_prob shape")
            else:
                y_prob_binary = y_prob

            res['auroc'] = roc_auc_score(y_true, y_prob_binary)
            res['auprc'] = average_precision_score(y_true, y_prob_binary)

    except Exception:
        pass

    return {k: round(float(v), 4) for k, v in res.items()}

def train_epoch(model, train_loader, optimizer, device, is_cls, grad_acc_steps=1):
    model.train()
    total_loss = 0
    all_preds, all_probs, all_targets = [], [], []
    
    optimizer.zero_grad()
    for idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        inputs = {'sequence': batch['sequence']}
        labels = batch['label'].to(device)
        
        outputs = model(**inputs)
        loss = model.compute_loss(outputs, labels)
        
        # Scale loss to account for gradient accumulation
        loss = loss / grad_acc_steps
        loss.backward()
        
        if (idx + 1) % grad_acc_steps == 0 or (idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
        
        # Record the true loss
        total_loss += loss.item() * grad_acc_steps

        if is_cls:
            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
            else:
                probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                preds = np.argmax(probs, axis=1)
            all_probs.extend(probs)
        else:
            preds = outputs.detach().cpu().numpy()
            
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds.flatten()
            
        all_preds.extend(preds)
        labels_np = labels.detach().cpu().numpy()
        if labels_np.ndim == 2 and labels_np.shape[1] == 1:
            labels_np = labels_np.flatten()
        all_targets.extend(labels_np)
            
    epoch_loss = total_loss / len(train_loader)
    return epoch_loss, all_targets, all_preds, all_probs if is_cls else None


def eval_epoch(model, val_loader, device, is_cls):
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_targets = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs = {'sequence': batch['sequence']}
            labels = batch['label'].to(device)
            
            outputs = model(**inputs)
            loss = model.compute_loss(outputs, labels)
            total_loss += loss.item()

            if is_cls:
                if outputs.shape[1] == 1:
                    probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
                    preds = (probs > 0.5).astype(int)
                else:
                    probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                    preds = np.argmax(probs, axis=1)
                all_probs.extend(probs)
            else:
                preds = outputs.detach().cpu().numpy()

            if preds.ndim == 2 and preds.shape[1] == 1:
                preds = preds.flatten()

            all_preds.extend(preds)
            labels_np = labels.detach().cpu().numpy()
            if labels_np.ndim == 2 and labels_np.shape[1] == 1:
                labels_np = labels_np.flatten()
            all_targets.extend(labels_np)

    epoch_loss = total_loss / len(val_loader)
    return epoch_loss, all_targets, all_preds, all_probs if is_cls else None


def train(args):
    # Set random seeds
    set_seed(args.seed)

    device = getattr(args, 'device', torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"))
    
    # 1. Start data preparation
    logging.info("Reading data...")
    train_dict = read_data(args.train_csv_path, seq_col=args.seq_column_name, label_cols=args.label_column_name_list)
    if args.val_csv_path:
        val_dict = read_data(args.val_csv_path, seq_col=args.seq_column_name, label_cols=args.label_column_name_list)
    else:
        val_dict = None
    
    # Simple logic for determining classes
    if args.task_type == 'cls':
        assert len(args.label_column_name_list) == 1, "Classification tasks only support single target (label_column_name_list must have length 1)."
        unique_labels = len(set(train_dict['label']))
        output_dim = unique_labels if unique_labels > 2 else 1
    else:
        # output dim: label length for list labels (e.g. 3UTR_test_dis), else number of label columns
        output_dim = len(train_dict['label'][0]) if isinstance(train_dict['label'][0], list) else len(args.label_column_name_list)
    
    scaler = None
    if args.task_type == 'reg' and getattr(args, 'do_norm', False):
        # Apply scaling to regression target values
        scaled_train_labels, scaler = scale_data(train_dict, scaler=None)
        train_dict['label'] = scaled_train_labels
        if val_dict:
            scaled_val_labels, _ = scale_data(val_dict, scaler=scaler)
            val_dict['label'] = scaled_val_labels
            
    train_dataset = RNADataset(train_dict)
    val_dataset = RNADataset(val_dict) if val_dict else None
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if val_dataset else None

    # 2. Build model 
    logging.info("Building model...")
    model = build_model(args, output_dim=output_dim)
    model.to(device)
    
    # 3. Set train configurations
    # Separate Base LR and Head LR parameter groups
    base_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': args.base_lr},
        {'params': head_params, 'lr': args.head_lr}
    ], weight_decay=args.weight_decay)
    
    # Warmup: freeze the base model for the first warmup_epochs epochs
    if args.warmup_epochs > 0:
        logging.info(f"Freezing base model for the first {args.warmup_epochs} epochs.")
        model.freeze_base_model(do_freeze=True)

    is_cls = args.task_type == 'cls'
    best_val_metric = float('-inf')

    patience = getattr(args, 'patience', 0)
    if patience > 0:
        logging.info(f"Early stopping patience set to {patience} epochs.")
    epochs_no_improve = 0

    logging.info("Starting training loop...")
    for epoch in range(args.num_epochs):
        
        # Unfreeze base model after warm up
        if args.warmup_epochs > 0 and epoch == args.warmup_epochs:
            logging.info("Warmup finished. Unfreezing base model parameters.")
            model.freeze_base_model(do_freeze=False)

        logging.info(f"Epoch {epoch+1}/{args.num_epochs}")
        train_loss, train_targets, train_preds, train_probs = train_epoch(model, train_loader, optimizer, device, is_cls, getattr(args, 'grad_acc_steps', 1))
        
        if is_cls:
            train_metrics = metrics_cls(train_targets, train_preds, train_probs)
        else:
            train_metrics = metrics_reg(train_targets, train_preds, getattr(args, 'reg_calc_method', 'feature_wise'))
        
        logging.info(f"[Train] Loss: {train_loss:.4f} | Metrics: {train_metrics}")
        
        if val_loader:
            val_loss, val_targets, val_preds, val_probs = eval_epoch(model, val_loader, device, is_cls)
            if is_cls:
                val_metrics = metrics_cls(val_targets, val_preds, val_probs)
            else:
                val_metrics = metrics_reg(val_targets, val_preds, getattr(args, 'reg_calc_method', 'feature_wise'))
    
            logging.info(f"[Eval] Loss: {val_loss:.4f} | Metrics: {val_metrics}")
            
            # Save best model
            is_best = False
            if is_cls:
                current_metric = val_metrics.get('auroc', val_metrics.get('acc', 0))
            else:
                current_metric = val_metrics.get('pearson', 0.0)
                
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                is_best = True
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if is_best and args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                save_path = os.path.join(args.output_dir, "best_model.pt")
                ckpt_dict = {
                    'model_state_dict': model.state_dict(),
                    'scaler': scaler
                }
                torch.save(ckpt_dict, save_path)
                logging.info(f"Saved new best model with metric val={current_metric:.4f}")

            if patience > 0 and epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered. No improvement for {patience} epochs (best={best_val_metric:.4f}).")
                break
                
    return best_val_metric

def infer(args):
    # Set random seeds
    set_seed(args.seed)

    device = getattr(args, 'device', torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"))
    
    logging.info("Reading inference data...")
    assert hasattr(args, 'infer_csv_path') and args.infer_csv_path, "infer_csv_path must be provided for inference mode."
    infer_dict = read_data(args.infer_csv_path, seq_col=args.seq_column_name, label_cols=args.label_column_name_list)
    
    if getattr(args, 'task_type', 'infer') == 'cls':
        assert len(args.label_column_name_list) == 1, "Classification tasks only support single target (label_column_name_list must have length 1)."
        
    logging.info("Building model for inference...")
    
    # probe the checkpoint to determine output_dim
    temp_ckpt = torch.load(args.saved_model_path, weights_only=False, map_location='cpu')
    if isinstance(temp_ckpt, dict) and ('model' in temp_ckpt or 'model_state_dict' in temp_ckpt):
        temp_state = temp_ckpt.get('model', temp_ckpt.get('model_state_dict'))
    else:
        temp_state = temp_ckpt
        
    # collect 2D Linear weights under the head
    head_weight_keys = [k for k in temp_state.keys() if 'head.' in k and k.endswith('.weight') and len(temp_state[k].shape) == 2]
    # the last Linear layer's shape[0] is the output dim
    output_dim = temp_state[head_weight_keys[-1]].shape[0]

    model = build_model(args, output_dim=output_dim)
    _, scaler = load_ckpt(model, args.saved_model_path)
    model.to(device)
    
    if args.task_type == 'reg' and scaler is not None:
        scaled_infer_labels, _ = scale_data(infer_dict, scaler=scaler)
        infer_dict['label'] = scaled_infer_labels
        logging.info("Normalized inference labels using saved scaler.")
        
    infer_dataset = RNADataset(infer_dict)
    infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    is_cls = args.task_type == 'cls'
    val_loss, val_targets, val_preds, val_probs = eval_epoch(model, infer_loader, device, is_cls)
    
    if is_cls:
        val_metrics = metrics_cls(val_targets, val_preds, val_probs)
    else:
        # Before taking metrics for regression, optionally inverse_transform if we scaled targets
        val_targets_for_metrics = np.array(val_targets)
        val_preds_for_metrics = np.array(val_preds)
        
        if args.task_type == 'reg' and scaler is not None:
            if val_targets_for_metrics.ndim == 1:
                val_targets_for_metrics = val_targets_for_metrics.reshape(-1, 1)
            if val_preds_for_metrics.ndim == 1:
                val_preds_for_metrics = val_preds_for_metrics.reshape(-1, 1)
                
            val_targets_for_metrics = scaler.inverse_transform(val_targets_for_metrics)
            val_preds_for_metrics = scaler.inverse_transform(val_preds_for_metrics)
            
        val_metrics = metrics_reg(val_targets_for_metrics, val_preds_for_metrics, getattr(args, 'reg_calc_method', 'feature_wise'))
        
    logging.info(f"[Infer] Loss: {val_loss:.4f} | Metrics: {val_metrics}")
    
    # save predictions (pkl-aware: dis held_out is a pickle)
    if str(args.infer_csv_path).endswith(".pkl"):
        df = pd.read_pickle(args.infer_csv_path)
    else:
        df = pd.read_csv(args.infer_csv_path)
    # drop unnamed passthrough columns from the source table
    df = df.loc[:, ~df.columns.str.match(r'^Unnamed:')]
    
    if args.task_type == 'reg' and scaler is not None:
        # already inverse-transformed above for metrics
        val_preds = val_preds_for_metrics
    else:
        val_preds = np.array(val_preds)
        
    for i, label_name in enumerate(args.label_column_name_list):
        if val_preds.ndim > 1 and val_preds.shape[1] > 1:
            df['pred_' + label_name] = val_preds[:, i]
        else:
            df['pred_' + label_name] = val_preds
            
        if is_cls and val_probs is not None:
            vp = np.array(val_probs)
            if vp.ndim > 1 and vp.shape[1] > 1:
                df['pred_prob_' + label_name] = vp.tolist()
            else:
                df['pred_prob_' + label_name] = vp
                
    os.makedirs(args.output_dir, exist_ok=True)
    res_path = os.path.join(args.output_dir, "predicted_result.csv")
    df.to_csv(res_path, index=False)
    logging.info(f"Saved prediction results to {res_path}")

def dataframe_to_markdown_safe(df, index=False):
    """Render a DataFrame as markdown and gracefully fall back if tabulate is missing."""
    try:
        return df.to_markdown(index=index)
    except ImportError:
        logging.warning("tabulate is not installed, fallback to plain table string output.")
        return df.to_string(index=index)

def param_search_evaluate(args, train_func=train):
    from copy import deepcopy
    from itertools import product
    import pandas as pd
    
    param_search = getattr(args, 'param_search', {})
    if not param_search:
        logging.warning("No parameter search space provided via --param_search.")
        return {}
        
    logging.info("#" * 50)
    logging.info(f"Starting parameter search with parameters: {param_search}")

    # parse combinations
    keys, values = zip(*param_search.items())
    values = list(values)
    for i, key in enumerate(keys):
        if not hasattr(args, key):
            raise ValueError(f"Parameter search key {key} is not a valid argument.")
        attr_type = type(getattr(args, key)) if getattr(args, key) is not None else type(values[i][0])
        values[i] = [attr_type(v) if v is not None else None for v in values[i]]
        
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    results = []
    best_overall_val = float('-inf')
    best_params = None
    
    for idx, param_set in enumerate(param_combinations):
        args_copy = deepcopy(args)
        for key, value in param_set.items():
            setattr(args_copy, key, value)
            
        logging.info(f"\n[{idx+1}/{len(param_combinations)}] Training with params: {param_set}")
        
        # per-combination subdirectory to avoid overwriting best models
        args_copy.output_dir = os.path.join(args.output_dir, f"search_{idx}")
        
        best_val_metric = train_func(args_copy)
        
        result_entry = {"params": param_set, "best_val": best_val_metric}
        results.append(result_entry)
        
    # select global best
        if best_val_metric > best_overall_val:
            best_overall_val = best_val_metric
            best_params = param_set
            
    # render results as a Markdown table
    records = []
    for res in results:
        record = {**res["params"]}
        record["best_val_metric"] = res["best_val"]
        records.append(record)
        
    df_results = pd.DataFrame.from_records(records)
    df_results = df_results.sort_values(by="best_val_metric", ascending=False)
    
    table_text = dataframe_to_markdown_safe(df_results, index=False)
    logging.info(f"Parameter search results:\n{table_text}\n")
    logging.info(f"Parameter search complete. Best parameters: {best_params}, Metric: {best_overall_val:.4f}")
    logging.info("#" * 50)
    
    return {"best_params": best_params, "best_metric": best_overall_val}
