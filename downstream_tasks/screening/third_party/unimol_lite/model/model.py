# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from .layers.encoders import TransformerEncoderWithPair
from .layers.heads import NonLinearHead, DistanceHead, MaskLMHead
from .layers.utils import GaussianLayer, init_bert_params
from .tokenizer import UniMolTokenizer
from .config import UniMolConfig


class UniMolModel(PreTrainedModel):
    config_class = UniMolConfig
    def __init__(self, configs: UniMolConfig):
        super().__init__(configs)
        self.configs = configs
        tokenizer = UniMolTokenizer(clean_up_tokenization_spaces=True)
        self.padding_idx = tokenizer.pad_token_id

        self.embed_tokens = nn.Embedding(len(tokenizer), configs.encoder_embed_dim, self.padding_idx)
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=configs.encoder_layers,
            embed_dim=configs.encoder_embed_dim,
            ffn_embed_dim=configs.encoder_ffn_embed_dim,
            attention_heads=configs.encoder_attention_heads,
            emb_dropout=configs.emb_dropout,
            dropout=configs.dropout,
            attention_dropout=configs.attention_dropout,
            activation_dropout=configs.activation_dropout,
            max_seq_len=configs.max_seq_len,
            activation_fn=configs.activation_fn,
            no_final_head_layer_norm=configs.delta_pair_repr_norm_loss < 0,
        )
        if configs.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=configs.encoder_embed_dim,
                output_dim=len(tokenizer),
                activation_fn=configs.activation_fn,
                weight=None,
            )

        K = 128
        n_edge_type = len(tokenizer) * len(tokenizer)
        self.gbf_proj = NonLinearHead(K, configs.encoder_attention_heads, configs.activation_fn)
        self.gbf = GaussianLayer(K, n_edge_type)

        if configs.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(configs.encoder_attention_heads, 1, configs.activation_fn)
        if configs.masked_dist_loss > 0:
            self.dist_head = DistanceHead(configs.encoder_attention_heads, configs.activation_fn)
        self.classification_heads = nn.ModuleDict()
        self.apply(init_bert_params)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True
        if src_tokens.dim() == 1:
            src_tokens = src_tokens.unsqueeze(0)
        else:
            assert src_tokens.dim() == 2, f"expected 2D tensor, got {src_tokens.dim()}"
        if src_distance.dim() == 2:
            src_distance = src_distance.unsqueeze(0)
        else:
            assert src_distance.dim() == 3, f"expected 3D tensor, got {src_distance.dim()}"
        if src_coord.dim() == 2:
            src_coord = src_coord.unsqueeze(0)
        else:
            assert src_coord.dim() == 3, f"expected 3D tensor, got {src_coord.dim()}"
        if src_edge_type.dim() == 2:
            src_edge_type = src_edge_type.unsqueeze(0)
        else:
            assert src_edge_type.dim() == 3, f"expected 3D tensor, got {src_edge_type.dim()}"
            
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        encoder_distance = None
        encoder_coord = None

        if not features_only:
            if self.configs.masked_token_loss > 0:
                logits = self.lm_head(encoder_rep, encoder_masked_tokens)
            if self.configs.masked_coord_loss > 0:
                coords_emb = src_coord
                if padding_mask is not None:
                    atom_num = torch.sum(1 - padding_mask.type_as(x), dim=1).view(
                        -1, 1, 1, 1
                    )  # consider BOS and EOS as part of the object
                else:
                    atom_num = src_coord.shape[1]
                delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
                coord_update = delta_pos / atom_num * attn_probs
                # Mask padding
                pair_coords_mask = (1 - padding_mask.float()).unsqueeze(-1) * (1 - padding_mask.float()).unsqueeze(1)
                coord_update = coord_update * pair_coords_mask.unsqueeze(-1)
                #
                coord_update = torch.sum(coord_update, dim=2)
                encoder_coord = coords_emb + coord_update
            if self.configs.masked_dist_loss > 0:
                encoder_distance = self.dist_head(encoder_pair_rep)

        if classification_head_name is not None:
            logits = self.classification_heads[classification_head_name](encoder_rep)
        if self.configs.mode == "infer":
            return encoder_rep, encoder_pair_rep
        else:
            return (
                logits,
                encoder_distance,
                encoder_coord,
                x_norm,
                delta_encoder_pair_rep_norm,
            )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

        self.train()

