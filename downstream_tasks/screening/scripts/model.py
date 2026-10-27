import os
import sys
import torch
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModel
import logging

_THIRD_PARTY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party")
# Uni-RNA implementation: the repository-level copy (../../ from this package);
# override with UR_UNIRNA_PATH.
UNIRNA_PATH = os.environ.get("UR_UNIRNA_PATH", os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..")))
UNIRNA_MODEL_PATH = os.environ.get(
    "UR_UNIRNA_WEIGHTS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "weights", "unirna_L16"),
)


def load_ckpt(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")

    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')

    assert isinstance(ckpt, dict) and ('model' in ckpt or 'model_state_dict' in ckpt)
    load_state_dict = ckpt.get('model', ckpt.get('model_state_dict'))
    scaler = ckpt.get('scaler', None)

    res = model.load_state_dict(load_state_dict, strict=True)
    logging.info(f"Loaded checkpoint from {ckpt_path} with result: {res}")

    return res, scaler


class UniRNAEncoder(nn.Module):
    def __init__(self, base_model_path, features_mode='cls_token'):
        super().__init__()
        logging.info(f"Initializing UniRNAEncoder with base model path: {base_model_path}")
        if UNIRNA_PATH not in sys.path:
            sys.path.insert(0, UNIRNA_PATH)
        import unirna_tf
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base_model = AutoModel.from_pretrained(base_model_path, output_attentions=False)
        self.output_dim = self.base_model.config.hidden_size
        self.features_mode = features_mode
        logging.info(f"UniRNAEncoder initialized with output_dim={self.output_dim} and features_mode={self.features_mode}")

    def forward(self, **inputs):
        assert 'sequence' in inputs, "UniRNAEncoder requires 'sequence' list in inputs."
        device = next(self.parameters()).device

        # Tokenize raw sequences on the fly; sequences are normalized to the DNA alphabet
        # (U -> T), which makes the tokenizer's uracil handling a no-op on any revision.
        encoded = self.tokenizer(
            [seq.upper().replace("U", "T") for seq in inputs['sequence']],
            padding=True,
            return_tensors='pt'
        ).to(device)

        outputs = self.base_model(encoded['input_ids'], encoded['attention_mask'], output_attentions=False)
        last_hidden_state = outputs.last_hidden_state

        if self.features_mode == 'cls_token':
            features = last_hidden_state[:, 0, :]
        elif self.features_mode == 'mean_pooling':
            mask = encoded['attention_mask'].unsqueeze(-1).float()
            features = torch.sum(last_hidden_state * mask, dim=1) / torch.sum(mask, dim=1).clamp_min(1e-8)
        else:
            raise ValueError(f"Unknown features_mode: {self.features_mode}")
        return features


# ---------------------------------------------------------
# Downstream heads
# ---------------------------------------------------------

class MLPRegressionHead(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, seq_embedding, **kwargs):
        return self.mlp(seq_embedding)

    def compute_loss(self, outputs, labels):
        labels = labels.float()
        if outputs.shape != labels.shape:
            labels = labels.view_as(outputs)
        return self.loss_fn(outputs, labels)


class MLPClassificationHead(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
        if output_dim == 1:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, seq_embedding, **kwargs):
        return self.mlp(seq_embedding)

    def compute_loss(self, outputs, labels):
        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            labels = labels.float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
        else:
            labels = labels.long()
            if labels.dim() > 1 and labels.shape[1] == 1:
                labels = labels.squeeze(-1)
        return self.loss_fn(outputs, labels)


class FullModel(nn.Module):
    def __init__(self, encoder, head, random_init_base=False):
        super().__init__()
        self.encoder = encoder
        self.head = head

        if random_init_base:
            self._reinit_base_model()

    def _reinit_base_model(self):
        """Randomly initializes the entire base model for baseline comparison."""
        target_model = self.encoder.base_model if hasattr(self.encoder, 'base_model') else self.encoder
        for module in target_model.modules():
            # Respect Hugging Face model weight initializations if present
            if hasattr(module, 'init_weights'):
                try:
                    module.init_weights()
                    continue
                except Exception:
                    pass
            # Standard PyTorch layer initialization
            if hasattr(module, 'reset_parameters'):
                try:
                    module.reset_parameters()
                except Exception:
                    pass
            elif isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                if isinstance(module, nn.Embedding) and getattr(module, 'padding_idx', None) is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def freeze_base_model(self, do_freeze=True):
        requires_grad = not do_freeze
        if hasattr(self.encoder, 'base_model'):
            for param in self.encoder.base_model.parameters():
                param.requires_grad = requires_grad
        else:
            for param in self.encoder.parameters():
                param.requires_grad = requires_grad

    def forward(self, **inputs):
        features = self.encoder(**inputs)
        outputs = self.head(features)
        return outputs

    def compute_loss(self, outputs, labels):
        return self.head.compute_loss(outputs, labels)


# ---------------------------------------------------------
# Model assembly
# ---------------------------------------------------------
def build_model(args, output_dim=1):
    base_model_name = args.base_model_name
    assert base_model_name == "Uni-RNA", f"Unsupported model name: {base_model_name} (this archive ships Uni-RNA only)"
    features_mode = getattr(args, 'features_mode', 'cls_token')

    resolved_path = getattr(args, 'base_model_path', "") or UNIRNA_MODEL_PATH
    encoder = UniRNAEncoder(resolved_path, features_mode=features_mode)

    input_dim = encoder.output_dim
    if args.task_type == 'reg':
        head = MLPRegressionHead(input_dim=input_dim, output_dim=output_dim)
    elif args.task_type == 'cls':
        head = MLPClassificationHead(input_dim=input_dim, output_dim=output_dim)
    else:
        raise ValueError(f"Unknown task type: {args.task_type}")

    is_random_init_base = getattr(args, 'random_init_base', False)
    model = FullModel(encoder, head, random_init_base=is_random_init_base)
    return model
