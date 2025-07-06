import torch
import torch.nn as nn
import torch.nn.functional as F
from sub_models.constants import DEVICE


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


class TEMScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, e_q, e_k, x_v, mask=None):
        # [B, n_head, L, d_head]
        attn = torch.matmul(e_q / self.temperature, e_k.transpose(2, 3))

        if mask is not None:
            # Fill the masked part with -inf
            attn = attn.masked_fill(mask == 0, -6e4)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, x_v)

        return output, attn


class TEMMultiHeadAttention(nn.Module):
    """Multi-Head Attention module returns the updated version of the both streams"""

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
        self.layer_norm_x = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_e = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, e_q, e_k, x, mask=None):

        # Get the batch size and len of the input sequences
        B, len_e_q, len_e_k, len_x = e_q.size(0), e_q.size(1), e_k.size(1), x.size(1)

        residual_e = e_q
        # in the forward_pass() all elements has same seq_len: len_e_q = L; take all of x as residual
        # in the forward_with_kv_cache() the e_q and x will have different lengths
        # But feat has the same length as e_q, so to match the residual connection; len_e_q =1
        residual_x = x[:, -len_e_q:, :] #TODO
        #
        #  Pass through the pre-attention projection: [B, L, N_head * D_x]
        # Separate different heads: [B, L, N_head, D_x]
        # Transpose for attention dot product,
        # [B, L, N_head, D_x] -> [B, N_head, L, D_x]
        e_q = self.We(e_q).reshape(B, len_e_q, self.n_head, self.d_e).transpose(1, 2)
        e_k = self.We(e_k).reshape(B, len_e_k, self.n_head, self.d_e).transpose(1, 2)
        x = self.Wx(x).reshape(B, len_x, self.n_head, self.d_x).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head axis, broadcasting.
        # Apply self-attention
        # print(f"DEBUG: e_q shape:, {e_q.shape}, e_k shape: {e_k.shape}, x shape: {x.shape}")
        feat, attn = self.attention(e_q, e_k, x, mask=mask)
        # print(f"DEBUG: feat shape:, {feat.shape}")
        
        # Transpose to move the head dimension back: [B, L, N_head, D_x]
        # Combine the last two dimensions to concatenate all the heads together: [B, L, N_head * D_x]
        feat = feat.transpose(1, 2).contiguous().reshape(B, len_e_q, -1)
        feat = self.dropout(self.fc(feat))  # [B, L, N_head * D_x] -> [B, L, D_model]
        # print(f"DEBUG: feat after fc shape:, {feat.shape}, residual_x shape: {residual_x.shape}")
        feat_x = self.layer_norm_x(feat + residual_x)
        feat_e = self.layer_norm_e(feat + residual_e)
        return feat_e, feat_x


class TEMAttentionBlockKVCache(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.mha = TEMMultiHeadAttention(
            num_heads,
            feat_dim,
            feat_dim // num_heads,
            feat_dim // num_heads,
            dropout=dropout,
        )
        self.pos_ffn_e = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)
        self.pos_ffn_x = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def forward(self, e_q, e_k, x, slf_attn_mask=None):
        # get the updated e and x states
        e_updated, x_updated = self.mha(e_q, e_k, x, mask=slf_attn_mask)
        # Pass them through the position-wise feed-forward networks
        e_updated = self.pos_ffn_e(e_updated)
        x_updated = self.pos_ffn_x(x_updated)
        return e_updated, x_updated


class RNNPositionalEncoding(nn.Module):
    """
    Positional encoding using a GRU (Gated Recurrent Unit) to generate
    learned, dynamic position encodings, as described in papers relating
    transformers to hippocampal models.

    This module uses an RNN to process a sequence of zero vectors, and its
    hidden states are used as the positional encodings.
    """

    def __init__(
        self,
        max_length: int,
        embed_dim: int,
        hidden_dim: int = None,
        num_layers: int = 1,
    ):
        """
        Initializes the RNNPositionalEncoding module.

        Args:
            max_length (int): The maximum sequence length that this module
                              will be used for. Not used.
            embed_dim (int): The dimensionality of the input embeddings. The
                             positional encodings will also have this dimension.
            hidden_dim (int, optional): The dimensionality of the RNN's hidden
                                        state. If None, it defaults to embed_dim.
            num_layers (int, optional): The number of layers in the RNN.
                                        Defaults to 1.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.num_layers = num_layers

        # The core RNN (GRU) that learns to generate positional patterns.
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Crucial for [B, L, D] input shape
        )
        # self.init_input = nn.Parameter(torch.zeros(1, 1, self.embed_dim, device=DEVICE))
        # If the RNN's hidden dimension is different from the embedding dimension,
        # use linear layer to project it back to the correct size.
        if self.hidden_dim != embed_dim:
            self.proj = nn.Linear(self.hidden_dim, embed_dim)
        else:
            # If dimensions match, no projection is needed.
            self.proj = nn.Identity()

    def forward(self, feat: torch.Tensor):
        """
        Adds positional encoding to a complete input sequence.

        Args:
            feat: Input tensor of shape [B, L, D], where B is the batch size,
                  L is the sequence length, and D is the embedding dimension.

        Returns:
            A tensor of shape [B, L, D] with positional encodings added.
        """
        B, L = feat.shape[0], feat.shape[1]
        # Generate positional encodings for the entire sequence length.
        # initial hidden state, which is a learnable parameter
        # h_0 = self.init_input.expand(self.num_layers, B, self.hidden_dim)
        h_0 = (
            None  # TODO: change this parameterized hidden state default to zeros: None
        )
        # output shape: [B, L, hidden_dim]
        pos_enc, h_n = self.rnn(feat)
        # Project the encodings to the correct embedding dimension if necessary.
        pos_enc = self.proj(pos_enc)

        return pos_enc

    def forward_with_position(self, feat, position, last_hidden):
        """
        Adds positional encoding at a specific position. This is useful for
        autoregressive decoding where inputs are processed one at a time.

        Args:
            feat: Input tensor of shape [B, 1, D] for the single item.
            position: The position index (integer) to generate the encoding for.

        Returns:
            A tensor of shape [B, 1, D] with the specific positional encoding added.
        """
        B = feat.shape[0]
        # Generate encodings for all positions up to and including `position`.

        # TODO: change this parameterized hidden state default to zeros: None
        # if last_hidden is None:
        # last_hidden = self.init_input.expand(self.num_layers, B, self.hidden_dim)
        all_pos_enc, h_n = self.rnn(feat, last_hidden)
        # Select the encoding for the specific position we need.
        # The shape becomes [B, 1, D] to match the input `feat`.
        pos_enc_at_position = self.proj(all_pos_enc[:, position : position + 1, :])

        return pos_enc_at_position, h_n


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
        self.layer_norm_x = nn.LayerNorm(feat_dim, eps=1e-6)
        self.layer_norm_e = nn.LayerNorm(feat_dim, eps=1e-6)

        # self.reset_kv_cache_list()

    def reset_kv_cache_list(self, batch_size, dtype):
        """
        Reset two separate caches, e_cache and  x_cache.
        """
        # for each layer in the stack, initialize a list of zeros
        self.cache = {
            "e_k": [
                torch.zeros(
                    size=(batch_size, 0, self.feat_dim), dtype=dtype, device=DEVICE
                )
                for _ in self.layer_stack
            ],
            "x_v": [
                torch.zeros(
                    size=(batch_size, 0, self.feat_dim), dtype=dtype, device=DEVICE
                )
                for _ in self.layer_stack
            ],
            "h_n": None,
            "r_n": torch.empty(
                size=(batch_size, 0, self.feat_dim), dtype=dtype, device=DEVICE
            ),
        }

    def forward(self, samples, action, mask):
        """
        Normal forward pass
        """
        action = F.one_hot(action.long(), self.action_dim).float()
        raw_input = self.stem(torch.cat([samples, action], dim=-1))
        e = self.position_encoding(raw_input)
        x = self.layer_norm_x(raw_input)
        e = self.layer_norm_e(e)

        for layer in self.layer_stack:
            e, x = layer(e, e, x, slf_attn_mask=mask)

        return x

    def forward_with_kv_cache(self, samples, action):
        """
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        This takes a single sample and an action, i.e, L=1.
        But the RNN positional encoding expects a sequence of inputs.
        [0], [0,1], [0,1,2], etc. Thats why we need to keep the cache r_n and h_n
        """
        assert samples.shape[1] == 1
        action = F.one_hot(action.long(), self.action_dim).float()
        raw_input = self.stem(torch.cat([samples, action], dim=-1))  # [B, 1, D]
        
        # Update the cache with the new raw_input; concat in the sequence dimension
        # the first iteration it will be same as raw_input
        self.cache["r_n"] = torch.cat([self.cache["r_n"], raw_input], dim=1) # L=1, L=2..
        # get the last position in the cache (anyone would do)
        last_pos = self.cache["x_v"][0].shape[1]  # int

        # get position encodings; e_t shape: [B, 1, D]
        e_t, h_n = self.position_encoding.forward_with_position(
            self.cache["r_n"], last_pos, self.cache["h_n"]
        )
        # Update the cache with the next hidden state
        self.cache["h_n"] = h_n
        x = self.layer_norm_x(raw_input)
        e = self.layer_norm_e(e_t)

        # Generate the mask for the current position
        mask = get_vector_mask(last_pos + 1, samples.device)

        for idx, layer in enumerate(self.layer_stack):
            # update the cache with the new input
            # print(f"DEBUG, layer: {idx}, x shape: {x.shape}, e shape: {e.shape}")
            self.cache["e_k"][idx] = torch.cat([self.cache["e_k"][idx], e], dim=1)
            self.cache["x_v"][idx] = torch.cat([self.cache["x_v"][idx], x], dim=1)

            e, x = layer(e, self.cache["e_k"][idx], self.cache["x_v"][idx], mask)

        return e  # TODO: what should be the output? x or e?


if __name__ == "__main__":

    B = 3
    L = 16
    D = 64
    action_dim = 5

    # Define the transformers with kv cache
    tem_trans = TEMTransformerKVCache(
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
            f"Number of parameters TEM Transformer: {sum(p.numel() for p in tem_trans.parameters())}"
        )

    def test_forward():

        samples = torch.randn(B, L, D).to(device=DEVICE)
        action = torch.randint(0, 1, size=(B, L)).to(device=DEVICE)
        # action = F.one_hot(action.long(), action_dim).float()
        print(samples.shape, action.shape)
        temporal_mask = None  # get_subsequent_mask(latent)
        op_tem = tem_trans.forward(samples, action, temporal_mask)

        print(f"Output shape TEM Transformer: {op_tem.shape}")
        assert op_tem.shape == (B, L, 512), "Output shape mismatch for TEM Transformer"

    def test_broken_forward_with_kv(samples, action):
        print("\n\n!-----Testing Broken down forward pass with kv_cache-----!")
    
        temporal_mask = None  # get_subsequent_mask(latent)\
        # tem_trans.reset_kv_cache_list(B, samples.dtype)

        action = F.one_hot(action.long(), tem_trans.action_dim).float()
        raw_input = tem_trans.stem(torch.cat([samples, action], dim=-1))
        print(f"Raw Input shape: {raw_input.shape}")
        # Update the cache with the new raw_input; concat in the sequence dimension
        # the first iteration it will be same as raw_input
        tem_trans.cache["r_n"] = torch.cat([tem_trans.cache["r_n"], raw_input], dim=1) # L=1, L=2..
        
        # get the last position in the cache (anyone would do)
        last_pos = tem_trans.cache["x_v"][0].shape[1]
        print(f"Last position: {last_pos}")
        e_t, h_next = tem_trans.position_encoding.forward_with_position(
            tem_trans.cache["r_n"], last_pos, tem_trans.cache["h_n"]
        )
        print(f"e_t shape: {e_t.shape}, h_next shape: {h_next.shape}")
        
        # Update the cache with the next hidden state
        tem_trans.cache["h_n"] = h_next
        x = tem_trans.layer_norm_x(raw_input)
        e = tem_trans.layer_norm_e(e_t)

        # Generate the mask for the current position
        mask = get_vector_mask(last_pos + 1, samples.device)

        for idx, layer in enumerate(tem_trans.layer_stack):
            print(f"layer: {idx}, x shape: {x.shape}, e shape: {e.shape}")
            # update the cache with the new input
            tem_trans.cache["x_v"][idx] = torch.cat(
                [tem_trans.cache["x_v"][idx], x], dim=1
            )
            tem_trans.cache["e_k"][idx] = torch.cat(
                [tem_trans.cache["e_k"][idx], e], dim=1
            )

            e, x = layer(
                e, tem_trans.cache["e_k"][idx], tem_trans.cache["x_v"][idx], mask
            )
        
    def test_cache():
        samples = torch.randn(B, L, D).to(device=DEVICE)
        action = torch.randint(0, 1, size=(B, L)).to(device=DEVICE)
        # action = F.one_hot(action.long(), action_dim).float()
        print(samples.shape, action.shape)
        temporal_mask = None
        # REset the kv_cache_list
        tem_trans.reset_kv_cache_list(B, samples.dtype)
        print(f"\nInit KV kache shape/value")
        print(f"    x_v: {tem_trans.cache["x_v"][0].shape}")
        print(f"    e_k: {tem_trans.cache["e_k"][0].shape}")
        print(f"    h_n: {tem_trans.cache["h_n"]}")
        print(f"    r_n: {tem_trans.cache["r_n"].shape}\n")

        # Forward pass with kv_cache
        tem_trans.forward_with_kv_cache(samples[:, 0:1], action[:, 0:1])
        # test_broken_forward_with_kv(samples[:, 0:1], action[:, 0:1])
        print(f"forward call 1, TEM KV cache shape: {tem_trans.cache["x_v"][0].shape}")

        print(f"\nInit KV kache shape/value")
        print(f"    x_v: {tem_trans.cache["x_v"][0].shape}")
        print(f"    e_k: {tem_trans.cache["e_k"][0].shape}")
        print(f"    h_n: {tem_trans.cache["h_n"].shape}")
        print(f"    r_n: {tem_trans.cache["r_n"].shape}\n")
        tem_trans.forward_with_kv_cache(samples[:, 1:2], action[:, 1:2])
        # test_broken_forward_with_kv(samples[:, 1:2], action[:, 1:2])
        print(f"forward call 2, TEM KV cache shape: {tem_trans.cache["x_v"][0].shape}")

    print("\n\n!-----Testing Init-----!")
    test_parameters()
    print("\n\n!-----Testing forward pass-----!")
    test_forward()
    print("\n\n!-----Test Forwardpass with KV cache-----!")
    test_cache()
