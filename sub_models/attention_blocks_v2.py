import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from einops import rearrange, repeat


def get_subsequent_mask_with_batch_length(batch_length: int, device: str):
    """
    For masking out the subsequent info.
    Example: where batch_length = 5
    tensor([[[ True, False, False, False, False],
         [ True,  True, False, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True]]], device='mps:0')
    """
    subsequent_mask = (
        1
        - torch.triu(
            torch.ones((1, batch_length, batch_length), device=device), diagonal=1
        )
    ).bool()
    return subsequent_mask


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""

    batch_length = seq.shape[1]
    subsequent_mask = get_subsequent_mask_with_batch_length(batch_length, seq.device)
    return subsequent_mask


def get_vector_mask(batch_length: int, device: str):
    mask = torch.ones((1, 1, batch_length), device=device).bool()
    # mask = torch.ones((1, batch_length, 1), device=device).bool()
    return mask


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            # Fill the masked part with -inf
            attn = attn.masked_fill(mask == 0, -6e4)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # Get the size of the batch
        batch_sz = q.size(0)
        # Get the len of the input sequences
        len_q, len_k, len_v = q.size(1), k.size(1), v.size(1)
        residual = q
        # Pass through the pre-attention projection: [B, L, N_head * D_v]
        # Separate different heads: [B, L, N_head, D_v]
        q = self.w_qs(q).reshape(batch_sz, len_q, self.n_head, self.d_k)
        k = self.w_ks(k).reshape(batch_sz, len_k, self.n_head, self.d_k)
        v = self.w_vs(v).reshape(batch_sz, len_v, self.n_head, self.d_v)

        # Transpose for attention dot product,
        # [B, L, N_head, D_v] -> [B, N_head, L, D_v]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head axis, broadcasting.
        # Apply self-attention
        feat, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: [B, L, N_head, D_v]
        # Combine the last two dimensions to concatenate all the heads together: [B, L, N_head * D_v]
        feat = feat.transpose(1, 2).contiguous().reshape(batch_sz, len_q, -1)
        feat = self.dropout(self.fc(feat))  # [B, L, N_head * D_v] -> [B, L, D_model]
        feat += residual
        feat = self.layer_norm(feat)
        return feat, attn


class MultiHeadLatentAttention(torch.nn.Module):
    """
    Multi-Head Latent Attention module.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        # Low rank projection dimensions
        self.q_proj_dim = d_model // 2
        self.kv_proj_dim = (2 * d_model) // 3
        # head dimension of q,k,v
        self.dh = d_model // n_heads

        # Define parameters for projections
        # Q projections
        self.W_dq = torch.nn.Parameter(0.01 * torch.randn((d_model, self.q_proj_dim)))
        self.W_uq = torch.nn.Parameter(0.01 * torch.randn((self.q_proj_dim, d_model)))
        self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)

        # KV projections
        self.W_dkv = torch.nn.Parameter(0.01 * torch.randn((d_model, self.kv_proj_dim)))
        self.W_ukv = torch.nn.Parameter(
            0.01 * torch.randn((self.kv_proj_dim, 2 * self.d_model))
        )
        self.kv_layernorm = torch.nn.LayerNorm(self.kv_proj_dim)

        self.attention = ScaledDotProductAttention(temperature=self.dh**0.5)
        # output projection
        # self.W_o = torch.nn.Parameter(0.01 * torch.randn((d_model, d_model)))
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, kv_cache=None, mask=None):
        # Get batchsize and sequence length
        B, L = x.size(0), x.size(1)
        # Residual connection
        residual = x
        # Q Projections
        compressed_q = x @ self.W_dq
        compressed_q = self.q_layernorm(compressed_q)
        Q = compressed_q @ self.W_uq

        # KV Projections
        if kv_cache is None:
            compressed_kv = x @ self.W_dkv
            compressed_kv = self.kv_layernorm(compressed_kv)
        else:
            # [B, 1, D_model] -> [B, 1, kv_dim]
            # Reduced multiplication dimensionality
            new_kv = x @ self.W_dkv
            new_kv = self.kv_layernorm(new_kv)
            # Append in the time dimension, [B, 0, D_model],... [B, 1, D_model]
            compressed_kv = torch.cat([kv_cache, new_kv], dim=1)

        KV = compressed_kv @ self.W_ukv
        K, V = torch.split(KV, self.d_model, dim=-1)

        # Split into multiple heads
        # [B, L, D_model(N_head * D_h)] -> [B, L, N_head, D_h] -> [B, N_head, L, D_h](transpose)
        q_heads = Q.view(B, -1, self.n_heads, self.dh).transpose(1, 2)
        k_heads = K.view(B, -1, self.n_heads, self.dh).transpose(1, 2)
        v_heads = V.view(B, -1, self.n_heads, self.dh).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head axis, broadcasting.

        # Apply self-attention
        feat, attn = self.attention(q_heads, k_heads, v_heads, mask=mask)

        # Transpose to move the head dimension back: [B, L, N_head, D_v]
        # Combine the last two dimensions to concatenate all the heads together: [B, L, N_head * D_h]
        feat = feat.transpose(1, 2).contiguous().reshape(B, L, -1)
        # Add linear projection
        feat = self.dropout(self.fc(feat))  # [B, L, N_head * D_h] -> [B, L, D_model]
        feat += residual  # TODO: check if this is necessary
        feat = self.layer_norm(feat)

        return x, attn, compressed_kv  # put this back in kv cache


class AttentionBlock(nn.Module):
    """
    Transformer block with multi-head attention and position-wise feed forward.
    """

    def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            num_heads,
            feat_dim,
            feat_dim // num_heads,
            feat_dim // num_heads,
            dropout=dropout,
        )
        self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class AttentionBlockKVCache(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            num_heads,
            feat_dim,
            feat_dim // num_heads,
            feat_dim // num_heads,
            dropout=dropout,
        )
        self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def forward(self, e, x, slf_attn_mask=None):
        output, attn = self.slf_attn(q, k, v, mask=slf_attn_mask)
        output = self.pos_ffn(output)
        return output, attn


class MLABlockKVCache(nn.Module):
    """
    Multi-Head Latent Attention block with position-wise feed forward and KV cache.
    """

    def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.slf_attn = MultiHeadLatentAttention(
            feat_dim,
            num_heads,
            dropout=dropout,
        )
        self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def forward(self, x, kv_cache=None, slf_attn_mask=None):
        x, attn, compressed_kv = self.slf_attn(x, kv_cache, mask=slf_attn_mask)
        output = self.pos_ffn(x)
        return output, attn, compressed_kv


class TEMScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, e, x, mask=None):
        attn = torch.matmul(e / self.temperature, e.transpose(2, 3))

        if mask is not None:
            # Fill the masked part with -inf
            attn = attn.masked_fill(mask == 0, -6e4)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, x)

        return output, attn


class TEMMultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_e, d_x, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_e = d_e
        self.d_x = d_x

        self.We = nn.Linear(d_model, n_head * d_e, bias=False)
        self.Wx = nn.Linear(d_model, n_head * d_x, bias=False)
        self.fc = nn.Linear(n_head * d_x, d_model, bias=False)

        self.attention = TEMScaledDotProductAttention(temperature=d_e**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, e, x, mask=None):
        # Get the size of the batch
        B = e.size(0)
        # Get the len of the input sequences
        len_e, len_x = e.size(1), x.size(1)
        residual = x
        # Pass through the pre-attention projection: [B, L, N_head * D_x]
        # Separate different heads: [B, L, N_head, D_x]
        e = self.We(e).reshape(B, len_e, self.n_head, self.d_e)
        x = self.Wx(x).reshape(B, len_x, self.n_head, self.d_x)

        # Transpose for attention dot product,
        # [B, L, N_head, D_x] -> [B, N_head, L, D_x]
        e, x = e.transpose(1, 2), x.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head axis, broadcasting.
        # Apply self-attention
        feat, attn = self.attention(e, x, mask=mask)

        # Transpose to move the head dimension back: [B, L, N_head, D_x]
        # Combine the last two dimensions to concatenate all the heads together: [B, L, N_head * D_x]
        feat = feat.transpose(1, 2).contiguous().reshape(B, len_e, -1)
        feat = self.dropout(self.fc(feat))  # [B, L, N_head * D_x] -> [B, L, D_model]
        feat += residual
        feat = self.layer_norm(feat)
        return feat, attn


class TEMAttentionBlockKVCache(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.slf_attn = TEMMultiHeadAttention(
            num_heads,
            feat_dim,
            feat_dim // num_heads,
            feat_dim // num_heads,
            dropout=dropout,
        )
        self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def forward(self, e, x, slf_attn_mask=None):
        output, attn = self.slf_attn(e, x, mask=slf_attn_mask)
        output = self.pos_ffn(output)
        return output, attn


if __name__ == "__main__":
    # Test PositionalEncoding1D
    # max_length = 5
    # embed_dim = 3
    # pos_encoder = PositionalEncoding1D(max_length, embed_dim)
    # feat = torch.randn(2, 5, 3)
    # print(feat)
    # print(pos_encoder(feat))

    # Test get_subsequent_mask
    # seq = torch.ones((1, 5, 5))
    # batch_size, batch_length = seq.shape[:2]
    # print(get_subsequent_mask(seq))

    # Test parameters
    # mha = AttentionBlockKVCache(feat_dim=512, hidden_dim=1024, num_heads=8, dropout=0.1)
    # mla = MLABlockKVCache(feat_dim=512, hidden_dim=1024, num_heads=8, dropout=0.1)
    # print(f"Number of parameters MHA:  {sum(p.numel() for p in mha.parameters())}")
    # print(f"Number of parameters MLA: {sum(p.numel() for p in mla.parameters())}")
    # # Number of parameters MHA:  2100736
    # # Number of parameters MLA: 2101418

    # # test amtrix multiplication in kv_cache scenario
    # x = torch.randn(8, 1, 512)
    # W_dkv = torch.nn.Parameter(0.01 * torch.randn((512, 2 * 512 // 3)))
    # new_kv = x @ W_dkv  # [B, 1, D_model] -> [B, 1, kv_dim]
    # print(x.shape, new_kv.shape)
    # Test RNNPositionalEncoding
    rnn_pos_encoder = RNNPositionalEncoding(max_length=5, embed_dim=8, hidden_dim=8)
    feat = torch.randn(2, 5, 8)
    print(f"RNN position encoder forward:", rnn_pos_encoder(feat))
    # Test forward_with_position
    feat = torch.randn(2, 1, 8)
    position = 2
    print(
        f"RNN position encoder forward_with_position:",
        rnn_pos_encoder.forward_with_position(feat, position),
    )
