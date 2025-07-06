import torch
import torch.nn as nn
import torch.nn.functional as F

from sub_models.attention_blocks import get_vector_mask
from sub_models.attention_blocks import (
    AttentionBlock,
    MHABlockKVCache,
    MLABlockKVCache,
    TEMAttentionBlockKVCache,
)
from sub_models.constants import DEVICE


class PositionalEncoding1D(nn.Module):
    def __init__(self, max_length: int, embed_dim: int):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim

        self.pos_emb = nn.Embedding(self.max_length, embed_dim)

    def forward(self, feat):
        pos_emb = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        # Add a batch dimention: [L, D] -> [B, L, D]
        pos_emb = pos_emb.unsqueeze(0).expand(
            feat.shape[0], pos_emb.shape[0], pos_emb.shape[1]
        )
        # Match postion embedding to the length of the input feature
        feat = feat + pos_emb[:, : feat.shape[1], :]
        return feat

    def forward_with_position(self, feat, position: int):
        """
        Add positional encoding to the feature at the given position.
        """
        assert feat.shape[1] == 1
        "The input feature should have a length of 1 at dim 1."

        pos_emb = self.pos_emb(torch.arange(self.max_length, device=feat.device))
        # Add a batch dimention: [L, D] -> [B, L, D]
        pos_emb = pos_emb.unsqueeze(0).expand(
            feat.shape[0], pos_emb.shape[0], pos_emb.shape[1]
        )
        # Get the position embedding at the particular position
        feat = feat + pos_emb[:, position : position + 1, :]
        return feat


class StochasticTransformer(nn.Module):
    def __init__(
        self,
        stoch_dim,
        action_dim,
        feat_dim,
        num_layers,
        num_heads,
        max_length,
        dropout,
    ):
        super().__init__()
        self.action_dim = action_dim

        # A network that takes [image_embedding + action] and projects it to a feature space
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
        )
        self.position_encoding = PositionalEncoding1D(
            max_length=max_length, embed_dim=feat_dim
        )
        self.layer_stack = nn.ModuleList(
            [
                AttentionBlock(
                    feat_dim=feat_dim,
                    hidden_dim=feat_dim * 2,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(
            feat_dim, eps=1e-6
        )  # TODO: check if this is necessary

        self.head = nn.Linear(feat_dim, stoch_dim)

    def forward(self, samples, action, mask):
        # One hot encode action
        action = F.one_hot(action.long(), self.action_dim).float()
        # Concatenate posterior samples and action
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for enc_layer in self.layer_stack:
            feats, attn = enc_layer(feats, mask)

        feat = self.head(feats)
        return feat


class StochasticTransformerKVCache(nn.Module):
    def __init__(
        self,
        stoch_dim,
        action_dim,
        feat_dim,
        num_layers,
        num_heads,
        max_length,
        dropout,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.kv_cache_list = []

        # A network that takes [image_embedding + action] and projects it to a feature space
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
        )
        self.position_encoding = PositionalEncoding1D(
            max_length=max_length, embed_dim=feat_dim
        )
        self.layer_stack = nn.ModuleList(
            [
                MHABlockKVCache(
                    feat_dim=feat_dim,
                    hidden_dim=feat_dim * 2,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(
            feat_dim, eps=1e-6
        )  # TODO: check if this is necessary

    def forward(self, samples, action, mask):
        """
        Normal forward pass
        """
        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats, attn = layer(feats, feats, feats, mask)

        return feats

    def reset_kv_cache_list(self, batch_size, dtype):
        """
        Reset self.kv_cache_list
        """
        self.kv_cache_list = []
        for layer in self.layer_stack:
            self.kv_cache_list.append(
                torch.zeros(
                    size=(batch_size, 0, self.feat_dim), dtype=dtype, device=DEVICE
                )
            )

    def forward_with_kv_cache(self, samples, action):
        """
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        """
        assert samples.shape[1] == 1
        mask = get_vector_mask(self.kv_cache_list[0].shape[1] + 1, samples.device)

        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding.forward_with_position(
            feats, position=self.kv_cache_list[0].shape[1]
        )
        feats = self.layer_norm(feats)

        for idx, layer in enumerate(self.layer_stack):
            self.kv_cache_list[idx] = torch.cat([self.kv_cache_list[idx], feats], dim=1)
            feats, attn = layer(
                feats, self.kv_cache_list[idx], self.kv_cache_list[idx], mask
            )

        return feats


class MLATransformerKVCache(nn.Module):
    """
    Transformer with Multi-head Latent Attention and kv cache
    """

    def __init__(
        self,
        stoch_dim,
        action_dim,
        feat_dim,
        num_layers,
        num_heads,
        max_length,
        dropout,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.kv_cache_list = []

        # A network that takes [image_embedding + action] and projects it to a feature space
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
        )
        self.position_encoding = PositionalEncoding1D(
            max_length=max_length, embed_dim=feat_dim
        )
        self.layer_stack = nn.ModuleList(
            [
                MLABlockKVCache(
                    feat_dim=feat_dim,
                    hidden_dim=feat_dim * 2,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.kv_proj_dim = self.layer_stack[0].slf_attn.kv_proj_dim
        self.layer_norm = nn.LayerNorm(
            feat_dim, eps=1e-6
        )  # TODO: check if this is necessary

    def forward(self, samples, action, mask):
        """
        Normal forward pass
        """
        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats, _, _ = layer(feats, None, mask)

        return feats

    def reset_kv_cache_list(self, batch_size, dtype):
        """
        Reset self.kv_cache_list
        """
        self.kv_cache_list = []
        # self.kv_cache_list.shape: [num_layers, B, 0, kv_proj_dim] at reset
        for layer in self.layer_stack:
            self.kv_cache_list.append(
                torch.zeros(
                    size=(batch_size, 0, self.kv_proj_dim),
                    dtype=dtype,
                    device=DEVICE,
                )
            )

    def forward_with_kv_cache(self, samples, action):
        """
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        """
        # Pass only one token at a time: samples.shape -> [B, 1, D_model]
        assert samples.shape[1] == 1, "Expecting only one token in the batch"
        # existing shape of the kv_cache_list: [num_layers, L_t, D_model]
        last_position = self.kv_cache_list[0].shape[1]
        mask = get_vector_mask(last_position + 1, samples.device)

        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        # Add the position encoding for particular token position
        feats = self.position_encoding.forward_with_position(
            feats, position=last_position
        )
        feats = self.layer_norm(feats)  # shape [B, 1, D_model]

        for idx, layer in enumerate(self.layer_stack):
            # Append in the time dimension, [B, 1, D_model],... [B, 2, D_model]
            feats, attn, compressed_kv = layer(feats, self.kv_cache_list[idx], mask)
            # update KV cache of the layer by the compressed_kv
            self.kv_cache_list[idx] = compressed_kv

        return feats


if __name__ == "__main__":
    B = 3
    L = 16
    D = 64
    action_dim = 5

    # Define the transformers with kv cache
    trans = StochasticTransformerKVCache(
        stoch_dim=D,
        action_dim=action_dim,
        feat_dim=512,
        num_layers=2,
        num_heads=8,
        max_length=L,
        dropout=0.1,
    ).to(device=DEVICE)

    def test_parameters():
        # Initialize the model with some parameters
        print(
            f"Number of parameters TEM Transformer: {sum(p.numel() for p in trans.parameters())}"
        )

    def test_forward():

        samples = torch.randn(B, L, D).to(device=DEVICE)
        action = torch.randint(0, 1, size=(B, L)).to(device=DEVICE)
        # action = F.one_hot(action.long(), action_dim).float()
        print(samples.shape, action.shape)
        temporal_mask = None  # get_subsequent_mask(latent)
        op_tem = trans.forward(samples, action, temporal_mask)

        print(f"Output shape TEM Transformer: {op_tem.shape}")
        assert op_tem.shape == (B, L, 512), "Output shape mismatch for TEM Transformer"

    def test_cache():
        samples = torch.randn(B, L, D).to(device=DEVICE)
        action = torch.randint(0, 1, size=(B, L)).to(device=DEVICE)
        # action = F.one_hot(action.long(), action_dim).float()
        print(samples.shape, action.shape)
        temporal_mask = None
        # REset the kv_cache_list
        trans.reset_kv_cache_list(B, samples.dtype)

        # Forward pass with kv_cache
        trans.forward_with_kv_cache(samples[:, 0:1], action[:, 0:1])
        print(f"forward call 1, TEM KV cache shape: {trans.kv_cache_list[0].shape}")

        print(f"\nInit KV kache shape/value")
        trans.forward_with_kv_cache(samples[:, 1:2], action[:, 1:2])
        print(f"forward call 2, TEM KV cache shape: {trans.kv_cache_list[0].shape}")

    print("\n\n!-----Testing Init-----!")
    test_parameters()
    print("\n\n!-----Testing forward pass-----!")
    test_forward()
    print("\n\n!-----Test Forwardpass with KV cache-----!")
    test_cache()

    # transformer = StochasticTransformer(
    #     stoch_dim=64 * 64,
    #     action_dim=10,
    #     feat_dim=512,
    #     num_layers=2,
    #     num_heads=8,
    #     max_length=64,
    #     dropout=0.1,
    # )
    # # print number of parameters and model architecture
    # # print(f"Number of parameters: {sum(p.numel() for p in transformer.parameters())}")
    # # # 8.7 M parameters
    # # print(transformer)
    # transformer_kv = StochasticTransformerKVCache(
    #     stoch_dim=64 * 64,
    #     action_dim=10,
    #     feat_dim=512,
    #     num_layers=2,
    #     num_heads=8,
    #     max_length=64,
    #     dropout=0.1,
    # )
    # # print number of parameters and model architecture
    # # print(
    # #     f"Number of parameters: {sum(p.numel() for p in transformer_kv.parameters())}"
    # # )
    # # # 6.6 M parameters
    # # print(transformer_kv)
