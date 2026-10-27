from transformers.modeling_utils import PretrainedConfig



class UniMolConfig(PretrainedConfig):
    model_type = "unimol"

    def __init__(
        self,
        encoder_layers=15,
        encoder_embed_dim=512,
        encoder_ffn_embed_dim=2048,
        encoder_attention_heads=64,
        emb_dropout=0.1,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        pooler_dropout=0.0,
        max_seq_len=512,
        activation_fn="gelu",
        pooler_activation_fn="tanh",
        post_ln=False,
        mode="infer",
        delta_pair_repr_norm_loss=-1.0,
        masked_token_loss=-1.0,
        masked_coord_loss=-1.0,
        masked_dist_loss=-1.0,
        **kwargs
    ):
        self.encoder_layers = encoder_layers
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.emb_dropout = emb_dropout
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.pooler_dropout = pooler_dropout
        self.max_seq_len = max_seq_len
        self.activation_fn = activation_fn
        self.pooler_activation_fn = pooler_activation_fn
        self.post_ln = post_ln
        self.mode = mode
        self.delta_pair_repr_norm_loss = delta_pair_repr_norm_loss
        self.masked_token_loss = masked_token_loss
        self.masked_coord_loss = masked_coord_loss
        self.masked_dist_loss = masked_dist_loss
        super().__init__(**kwargs)
