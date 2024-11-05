import pytest
from unittest.mock import MagicMock, patch
import torch
from llama.model import Transformer, ModelArgs, RMSNorm, precompute_freqs_cis, reshape_for_broadcast, apply_rotary_emb, repeat_kv, Attention, FeedForward, TransformerBlock

@pytest.fixture
def mock_torch_cuda_is_available():
    with patch('torch.cuda.is_available', return_value=False):
        yield

@pytest.fixture
def model_args():
    return ModelArgs(dim=512, n_layers=2, n_heads=8, vocab_size=10000, max_batch_size=1, max_seq_len=50)

@pytest.fixture
def transformer_model(model_args):
    return Transformer(params=model_args)

def test_rmsnorm_forward_zero_input():
    """RMSNorm should not return nan for zero input tensor."""
    dim = 512
    input_tensor = torch.zeros(10, 20, dim)
    rms_norm = RMSNorm(dim)
    output = rms_norm(input_tensor)
    assert not torch.isnan(output).any(), "RMSNorm returned NaN on zero input."

def test_precompute_freqs_cis_negative_dim():
    """precompute_freqs_cis should raise an exception for negative dimensions."""
    dim = -512
    end = 100
    with pytest.raises(ValueError):
        precompute_freqs_cis(dim=dim, end=end)

def test_reshape_for_broadcast_incompatible_shapes():
    """reshape_for_broadcast should raise an exception for incompatible shapes."""
    freqs_cis = torch.randn(100, 256)
    x = torch.randn(10, 101, 512)  # Incompatible seq_len
    with pytest.raises(AssertionError):
        reshape_for_broadcast(freqs_cis, x)

def test_apply_rotary_emb_dtype_consistency(mock_torch_cuda_is_available):
    """apply_rotary_emb should maintain dtype consistency between input and output."""
    xq = torch.randn(10, 100, 512).float()
    xk = torch.randn(10, 100, 512).float()
    freqs_cis = torch.randn(100, 256).float()
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
    assert xq_out.dtype == xq.dtype, "apply_rotary_emb changes dtype of xq."
    assert xk_out.dtype == xk.dtype, "apply_rotary_emb changes dtype of xk."

def test_repeat_kv_negative_repetition():
    """repeat_kv should raise an exception for negative repetition values."""
    x = torch.randn(10, 100, 8, 64)
    n_rep = -2
    with pytest.raises(ValueError):
        repeat_kv(x, n_rep)

@pytest.mark.parametrize("invalid_start_pos", [-1, 51])
def test_attention_invalid_start_pos(transformer_model, invalid_start_pos):
    """Attention should raise an exception for invalid start positions."""
    attention = Attention(transformer_model.params)
    x = torch.randn(1, 50, transformer_model.params.dim)
    freqs_cis = torch.randn(50, transformer_model.params.dim // transformer_model.params.n_heads)
    mask = None
    with pytest.raises(IndexError):
        attention(x, invalid_start_pos, freqs_cis, mask)

def test_feedforward_forward_pass_negative_input():
    """FeedForward should handle negative inputs correctly."""
    dim = 512
    hidden_dim = 2048
    multiple_of = 256
    ffn_dim_multiplier = None
    feedforward = FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)
    x = torch.randn(10, 100, dim) * -1  # Negative input
    output = feedforward(x)
    assert not torch.isnan(output).any(), "FeedForward returned NaN on negative input."

def test_transformer_block_invalid_layer_id(transformer_model):
    """TransformerBlock should raise an exception for invalid layer IDs."""
    with pytest.raises(ValueError):
        TransformerBlock(layer_id=-1, args=transformer_model.params)

def test_transformer_input_out_of_vocab_range(transformer_model):
    """Transformer should handle input tokens out of vocab range gracefully."""
    tokens = torch.tensor([[transformer_model.vocab_size + 1]])  # Out of vocab range token
    start_pos = 0
    with pytest.raises(IndexError):
        transformer_model(tokens, start_pos)

@pytest.mark.parametrize("empty_input", [torch.empty(0, 50, dtype=torch.long), torch.empty(1, 0, dtype=torch.long)])
def test_transformer_empty_input(transformer_model, empty_input):
    """Transformer should handle empty input sequences gracefully."""
    start_pos = 0
    with pytest.raises(ValueError):
        transformer_model(empty_input, start_pos)