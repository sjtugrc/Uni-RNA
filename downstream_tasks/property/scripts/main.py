import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import logging
from torch import nn
import yaml

from model import *
from utils import set_seed, setup_logging, enforce_arg_types
from engine import param_search_evaluate, train, infer

def parse_args():
    parser = argparse.ArgumentParser(description='RNA Foundation Model Benchmark Training and Inference Script.')

    # mode selection
    parser.add_argument('--mode', type=str, default="infer", choices=['train', 'post_train', 'infer'],
                        help='Execution mode: \n'
                             '  train: fine-tune the pretrained base with a freshly initialized head;\n'
                             '  post_train: continue training from an existing checkpoint;\n'
                             '  infer: perform inference using a trained model.')
    parser.add_argument('--task_type', type=str, default='reg', choices=['reg', 'cls'],
                        help='Downstream task type: regression ("reg") or classification ("cls").')
    
    # base settings: gpu, seed, log dir
    parser.add_argument("--config", type=str,
                        default="",
                        help="Path to the YAML config file. Command line arguments will override the config file settings if provided.")
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use for training/inference.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to ensure reproducibility.')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory to save logs and trained model checkpoints. Defaults to outputs/<mode>-<timestamp>.')
    
    # pre-trained model arguments
    parser.add_argument("--base_model_name", type=str,
                        default="Uni-RNA",
                        help="The name of the pre-trained RNA foundation model to test.")
    parser.add_argument("--base_model_path", type=str,
                        default="",
                        help="Path to the pre-trained model weights (tokenizer and backbone included)")
    parser.add_argument("--random_init_base", action='store_true',
                        help="If set, initialize the model with random weights instead of pre-trained weights (baseline).")
    parser.add_argument('--saved_model_path', type=str, default="",
                        help='Path to the fine-tuned model checkpoint to be loaded for "post_train" or "infer" mode.')
    parser.add_argument('--features_mode', type=str, default='cls_token', choices=['cls_token', 'mean_pooling'],
                        help='How to extract features from the base model: "cls_token" or "mean_pooling".')
    
    # data arguments
    parser.add_argument("--train_csv_path", type=str, default="",
                        help="Path to the training dataset CSV file.")
    parser.add_argument("--val_csv_path", type=str, default="",
                        help="Path to the validation dataset CSV file used during training.")
    parser.add_argument("--infer_csv_path", type=str, default="",
                        help="Path to the test/inference dataset CSV file.")
    parser.add_argument('--seq_column_name', type=str, default='sequence',
                        help='Column name for the RNA sequences in the CSV files.')
    parser.add_argument('--label_column_name_list', nargs='+', default=[],
                        help='List of label column names in the CSV files. The model output dimension will automatically match the number of labels (supports multi-label prediction).')
    parser.add_argument('--do_norm', action='store_true',
                        help='If set, normalize the labels to have zero mean and unit variance before training (recommended for regression).')

    # training arguments
    parser.add_argument('--grad_acc_steps', type=int, default=1,
                        help='Number of gradient accumulation steps.')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Total number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size used for training and inference.')
    parser.add_argument('--base_lr', type=float, default=1e-5,
                        help='Learning rate for fine-tuning the base pre-trained model.')
    parser.add_argument('--head_lr', type=float, default=1e-3,
                        help='Learning rate for the task-specific prediction head.')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay factor for the optimizer.')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of initial warmup epochs where only the prediction head is trained while freezing the base model.')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early stopping patience in epochs (0 = disabled).')
    parser.add_argument('--reg_calc_method', type=str, default='feature_wise', choices=['feature_wise', 'sample_wise'],
                        help="Regression metric aggregation: 'feature_wise' (default) or 'sample_wise'.")
    
    # parse only arguments actually passed on the command line (incl. --config)
    cmd_args, _ = parser.parse_known_args()
    
    config = {}
    if cmd_args.config:
        with open(cmd_args.config, 'r') as f:
            config = yaml.safe_load(f)
        # use config values as parser defaults
        for key, value in config.items():
            parser.set_defaults(**{key: value})

    # CLI arguments override config and defaults
    args = parser.parse_args()
    # enforce argument types
    args = enforce_arg_types(args, parser)
    return args

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device
    set_seed(args.seed)
    # default log directory: mode-timestamp if not given
    if args.output_dir == '':
        args.output_dir = f"outputs/{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    setup_logging(args.output_dir, args.mode)

    logging.info(f"Arguments: {args}")
    logging.info(f"Using device: {device}")
    logging.info(f"Random seed set to: {args.seed}")
    
    # run mode
    if args.mode in ['train', 'post_train']:
        # parameter search
        if hasattr(args, 'param_search') and args.param_search:
            # parameter search
            best_result = param_search_evaluate(args, train_func=train)

            # write best params back to args
            logging.info("Setting best parameters back to args for final training...")
            for key, value in best_result['params'].items():
                setattr(args, key, value)

        # training
        train(args)

    elif args.mode == 'infer':
        infer(args)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
