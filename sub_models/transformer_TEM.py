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
        # self.layer_norm_x = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_e = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, e, x, mask=None):
        # Get the size of the batch
        B = e.size(0)
        # Get the len of the input sequences
        len_e, len_x = e.size(1), x.size(1)
        residual_e = e
        # residual_x = x
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
        # feat_x = feat + residual_x
        feat = feat + residual_e
        # feat_x = self.layer_norm_x(feat_x)
        feat = self.layer_norm_e(feat)
        return feat


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
        # self.pos_ffn_e = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)
        self.pos_ffn_x = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def forward(self, e, x, slf_attn_mask=None):
        """
        Information flow:
        1. Attention updates the memory stream `e`.
        2. The FFN updates the content stream `x` based on the new memory.
        """
        # The attention sub-layer updates the memory stream 'e'.
        # It uses the current memory 'e' to query the content 'x'.
        e_updated = self.slf_attn(e, x, mask=slf_attn_mask)

        # The feed-forward sub-layer updates the content stream 'x'.
        # It processes the *newly updated* memory state to generate the next content representation.
        # feat_e = self.pos_ffn_e(feat_e)
        x_updated = self.pos_ffn_x(e_updated)
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
        self.rnn_init = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # If the RNN's hidden dimension is different from the embedding dimension,
        # use linear layer to project it back to the correct size.
        if self.hidden_dim != embed_dim:
            self.proj = nn.Linear(self.hidden_dim, embed_dim)
        else:
            # If dimensions match, no projection is needed.
            self.proj = nn.Identity()

    def _generate_encodings(self, batch_size: int, seq_len: int, device: torch.device):
        """
        Internal helper function to generate the positional encodings.
        """
        # Create a dummy input tensor of zeros.
        # Shape: [B, L, embed_dim]
        # Use a trainable parameter as the input for each position (shared across positions)
        # dummy_input = torch.zeros(batch_size, seq_len, self.embed_dim, device=device)
        dummy_input = self.rnn_init.expand(batch_size, seq_len, self.embed_dim)

        # The GRU returns the output (hidden states for each time step) and
        # the final hidden state. We only need the former.
        # output shape: [B, L, hidden_dim]
        pos_enc, _ = self.rnn(dummy_input)

        # Project the encodings to the correct embedding dimension if necessary.
        # projected_pos_enc shape: [B, L, embed_dim]
        projected_pos_enc = self.proj(pos_enc)

        return projected_pos_enc

    def forward(self, feat: torch.Tensor):
        """
        Adds positional encoding to a complete input sequence.

        Args:
            feat: Input tensor of shape [B, L, D], where B is the batch size,
                  L is the sequence length, and D is the embedding dimension.

        Returns:
            A tensor of shape [B, L, D] with positional encodings added.
        """
        (
            B,
            L,
        ) = (
            feat.shape[0],
            feat.shape[1],
        )
        # Generate positional encodings for the entire sequence length.
        pos_enc = self._generate_encodings(B, L, feat.device)

        return pos_enc

    def forward_with_position(self, feat: torch.Tensor, position: int):
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
        # We need to run the RNN sequentially to get the correct hidden state.
        # Shape: [B, position + 1, D]
        all_pos_enc = self._generate_encodings(B, position + 1, feat.device)

        # Select the encoding for the specific position we need.
        # The shape becomes [B, 1, D] to match the input `feat`.
        pos_enc_at_position = all_pos_enc[:, position : position + 1, :]

        return pos_enc_at_position


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
            e, x = layer(e, x, slf_attn_mask=mask)

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
            e, x = layer(e, self.kv_cache_list[idx], mask)
            # e = self.position_encoding.forward_with_position(x, position=last_pos)

        return x


class RNNPositionalEncodingOld(nn.Module):
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


if __name__ == "__main__":

    from sub_models.constants import DEVICE

    B = 4
    L = 64
    D = 64 * 64
    action_dim = 5

    # Define the transformers with kv cache
    tem_trans = TEMTransformerKVCache(
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


def test_cache():
    samples = torch.randn(B, L, D).to(device=DEVICE)
    action = torch.randint(0, 1, size=(B, L)).to(device=DEVICE)
    # action = F.one_hot(action.long(), action_dim).float()
    print(samples.shape, action.shape)
    temporal_mask = None  # get_subsequent_mask(latent)\
    # REset the kv_cache_list
    tem_trans.reset_kv_cache_list(B, samples.dtype)
    print(f"Init KV kache shape of TEM Transformer: {tem_trans.kv_cache_list[0].shape}")

    # Forward pass with kv_cache
    tem_trans.forward_with_kv_cache(samples[:, 0:1], action[:, 0:1])
    print(f"forward call 1, TEM KV cache shape: {tem_trans.kv_cache_list[0].shape}")

    tem_trans.forward_with_kv_cache(samples[:, 1:2], action[:, 1:2])
    print(f"forward call 2, TEM KV cache shape: {tem_trans.kv_cache_list[0].shape}")


if __name__ == "__main__":
    test_parameters()
    print("/n/n Testing forward pass")
    test_forward()
    print("/n/n Testing forward pass with kv_cache")
    test_cache()
