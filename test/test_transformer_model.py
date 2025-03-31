import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from sub_models.transformer_model_v2 import (
    StochasticTransformerKVCache,
    MLATransformerKVCache,
)
from sub_models.constants import DEVICE

B = 4
L = 64
D = 64 * 64
action_dim = 5
mha_trans = StochasticTransformerKVCache(
    stoch_dim=D,
    action_dim=5,
    feat_dim=512,
    num_layers=2,
    num_heads=8,
    max_length=L,
    dropout=0.1,
).to(device=DEVICE)
mla_trans = MLATransformerKVCache(
    stoch_dim=D,
    action_dim=5,
    feat_dim=512,
    num_layers=2,
    num_heads=8,
    max_length=L,
    dropout=0.1,
).to(device=DEVICE)


def test_parameters():
    # Initialize the model with some parameters

    print(
        f"Number of parameters MHA Transformer: {sum(p.numel() for p in mha_trans.parameters())}"
    )
    print(
        f"Number of parameters MHA Transformer: {sum(p.numel() for p in mla_trans.parameters())}"
    )
    # Number of parameters MHA Transformer: 6599168
    # Number of parameters MHA Transformer: 6600532


def test_forward():

    samples = torch.randn(B, L, D)
    action = torch.randint(0, 1, size=(B, L))
    # action = F.one_hot(action.long(), action_dim).float()
    print(samples.shape, action.shape)
    temporal_mask = None  # get_subsequent_mask(latent)
    op_mha = mha_trans.forward(samples, action, temporal_mask)
    op_mla = mla_trans.forward(samples, action, temporal_mask)
    print(f"Output shape MHA Transformer: {op_mha.shape}")
    print(f"Output shape MLA Transformer: {op_mla.shape}")
    assert op_mha.shape == (B, L, 512), "Output shape mismatch for MHA Transformer"
    assert op_mla.shape == (B, L, 512), "Output shape mismatch for MLA Transformer"


def test_cache():
    samples = torch.randn(B, L, D).to(device=DEVICE)
    action = torch.randint(0, 1, size=(B, L)).to(device=DEVICE)
    # action = F.one_hot(action.long(), action_dim).float()
    print(samples.shape, action.shape)
    temporal_mask = None  # get_subsequent_mask(latent)\
    # REset the kv_cache_list
    mha_trans.reset_kv_cache_list(B, samples.dtype)
    mla_trans.reset_kv_cache_list(B, samples.dtype)
    print(f"Init KV kache shape of MHA Transformer: {mha_trans.kv_cache_list[0].shape}")
    print(f"Init KV kache shape of MLA Transformer: {mla_trans.kv_cache_list[0].shape}")

    # Forward pass with kv_cache
    mha_trans.forward_with_kv_cache(samples[:, 0:1], action[:, 0:1])
    print(f"forward call 1, MHAT KV cache shape: {mha_trans.kv_cache_list[0].shape}")
    mha_trans.forward_with_kv_cache(samples[:, 1:2], action[:, 1:2])
    print(f"forward call 2, MHAT KV cache shape: {mha_trans.kv_cache_list[0].shape}")

    mla_trans.forward_with_kv_cache(samples[:, 0:1], action[:, 0:1])
    print(f"forward call 1, MHAT KV cache shape: {mla_trans.kv_cache_list[0].shape}")
    mla_trans.forward_with_kv_cache(samples[:, 1:2], action[:, 1:2])
    print(f"forward call 2, MHAT KV cache shape: {mla_trans.kv_cache_list[0].shape}")
    # torch.Size([4, 64, 4096]) torch.Size([4, 64])
    # Init KV kache shape of MHA Transformer: torch.Size([4, 0, 512])
    # Init KV kache shape of MLA Transformer: torch.Size([4, 0, 341])
    # forward call 1, MHAT KV cache shape: torch.Size([4, 1, 512])
    # forward call 2, MHAT KV cache shape: torch.Size([4, 2, 512])
    # forward call 1, MHAT KV cache shape: torch.Size([4, 1, 341])
    # forward call 2, MHAT KV cache shape: torch.Size([4, 2, 341])


if __name__ == "__main__":
    # test_parameters()
    # test_forward()
    test_cache()
