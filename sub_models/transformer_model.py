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


class RNNPositionalEncoding(nn.Module):
    """
    Positional encoding using RNN (GRU) to generate position encodings.
    """

    def __init__(
        self,
        max_length: int,
        embed_dim: int,
        hidden_dim: int = None,
        num_layers: int = 1,
    ):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        if self.hidden_dim != embed_dim:
            self.proj = nn.Linear(self.hidden_dim, embed_dim)
        else:
            self.proj = nn.Identity()

        # Position embeddings as input to RNN
        self.pre_embeddings = nn.Embedding(max_length, embed_dim)

    def get_position_encodings(
        self, batch_size: int, seq_len: int, device: torch.device
    ):
        """Generate position encodings for given batch size and sequence length"""
        # Create position indices [0, 1, ..., seq_len-1]
        positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )  # [B, L]
        # Get position embeddings [B, L, D]
        pre_emb = self.pre_embeddings(positions)
        # Process through GRU
        pos_encoding, _ = self.rnn(pre_emb)  # [B, L, hidden_dim]

        return pos_encoding

    def forward(self, feat):
        """Add positional encoding to the input features
        Args:
            feat: Input tensor of shape [B, L, D]
        Returns:
            Tensor of shape [B, L, D] with added positional encodings
        """
        B, L, _ = feat.shape
        pos_enc = self.get_position_encodings(B, L, feat.device)  # [B, L, D]
        if self.hidden_dim != self.embed_dim:
            # Project to original dimension if needed
            pos_enc = self.proj(pos_enc)  # [B, L, D]
        return pos_enc

    def forward_with_position(self, feat, position: int):
        """Add positional encoding at a specific position
        Args:
            feat: Input tensor of shape [B, 1, D]
            position: Position index to add encoding for
        Returns:
            Tensor of shape [B, 1, D] with added positional encoding
        """

        B, L, _ = feat.shape
        assert L == 1, "Input feature should have length 1 at dim 1"

        # Run the RNN with the full sequence up to this point
        rnn_output = self.get_position_encodings(
            B, position + 1, feat.device
        )  # [B, L, D]
        # Get only the last position's output
        pos_enc = rnn_output[:, -1:, :]  # [B, 1, hidden_dim]
        if self.hidden_dim != self.embed_dim:
            # Project to original dimension if needed
            pos_enc = self.proj(pos_enc)  # [B, 1, D]
        return pos_enc


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


class TEMTransformerKVCache(nn.Module):
    """
    Transformer with TEM modifications of attention and position encoding.
    Uses Multi-head attention with kv cache.
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
        self.position_encoding = RNNPositionalEncoding(
            max_length=max_length,
            embed_dim=feat_dim,
            hidden_dim=feat_dim,
        )
        self.layer_stack = nn.ModuleList(
            [
                TEMAttentionBlockKVCache(
                    feat_dim=feat_dim,
                    hidden_dim=feat_dim * 2,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # TODO: check if this is necessary
        self.layer_norm_x = nn.LayerNorm(feat_dim, eps=1e-6)
        self.layer_norm_e = nn.LayerNorm(feat_dim, eps=1e-6)

    def forward(self, samples, action, mask):
        """
        Normal forward pass
        """
        action = F.one_hot(action.long(), self.action_dim).float()
        x = self.stem(torch.cat([samples, action], dim=-1))
        e = self.position_encoding(x)
        x = self.layer_norm_x(x)
        e = self.layer_norm_e(e)

        for layer in self.layer_stack:
            x, attn = layer(e, x, mask)
            e = self.position_encoding(
                x
            )  # FIXME: Check this logic if it woks as intended

        return x

    def reset_kv_cache_list(self, batch_size, dtype):
        """
        Reset self.kv_cache_list
        """
        self.kv_cache_list = [
            torch.zeros(size=(batch_size, 0, self.feat_dim), dtype=dtype, device=DEVICE)
            for _ in range(len(self.layer_stack))
        ]

    def forward_with_kv_cache(self, samples, action):
        """
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        """
        assert samples.shape[1] == 1
        last_pos = self.kv_cache_list[0].shape[1]
        mask = get_vector_mask(last_pos + 1, samples.device)

        action = F.one_hot(action.long(), self.action_dim).float()
        x = self.stem(torch.cat([samples, action], dim=-1))
        e = self.position_encoding.forward_with_position(x, position=last_pos)
        x = self.layer_norm_x(x)
        e = self.layer_norm_e(e)

        for idx, layer in enumerate(self.layer_stack):
            self.kv_cache_list[idx] = torch.cat([self.kv_cache_list[idx], x], dim=1)
            x, attn = layer(e, self.kv_cache_list[idx], mask)
            e = self.position_encoding.forward_with_position(x, position=last_pos)

        return x


if __name__ == "__main__":

    transformer = StochasticTransformer(
        stoch_dim=64 * 64,
        action_dim=10,
        feat_dim=512,
        num_layers=2,
        num_heads=8,
        max_length=64,
        dropout=0.1,
    )
    # print number of parameters and model architecture
    # print(f"Number of parameters: {sum(p.numel() for p in transformer.parameters())}")
    # # 8.7 M parameters
    # print(transformer)
    transformer_kv = StochasticTransformerKVCache(
        stoch_dim=64 * 64,
        action_dim=10,
        feat_dim=512,
        num_layers=2,
        num_heads=8,
        max_length=64,
        dropout=0.1,
    )
    # print number of parameters and model architecture
    # print(
    #     f"Number of parameters: {sum(p.numel() for p in transformer_kv.parameters())}"
    # )
    # # 6.6 M parameters
    # print(transformer_kv)
