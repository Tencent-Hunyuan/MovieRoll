"""Contract tests for `KVCacheManager` bookkeeping.

The manager is the piece every layer shares, and the two things that silently go wrong are (a) the frame -> token
arithmetic used to size the cache and (b) the batch broadcast that lets one cached copy serve several CFG branches
or videos. Neither needs weights: the manager works on plain CPU tensors.

CPU only, no checkpoints.
"""

import pytest
import torch

from movieroll.commons.streaming_config import StreamingConfig
from movieroll.models.transformers.modules.kv_cache import KVCacheManager

HEIGHT, WIDTH = 30, 40        # latent grid of a 480x640 video at the VAE's /16 downsample
LAYERS = ["double_blocks.0", "double_blocks.1"]
CHUNK_FRAMES = 2


def _manager():
    return KVCacheManager(StreamingConfig())


def _merged(heads=2, head_dim=8, tokens=5, batch=1):
    """A fake cached layer: k / v shaped [batch, tokens, heads, head_dim]."""
    k = torch.arange(batch * tokens * heads * head_dim, dtype=torch.float32).reshape(batch, tokens, heads, head_dim)
    return k, k.clone()


# ----------------------------------------------------------------------------------------------
# lifecycle
# ----------------------------------------------------------------------------------------------

def test_manager_starts_uninitialised():
    manager = _manager()
    assert manager.is_init is False
    assert manager.first_layer_name is None


def test_init_registers_every_layer_and_the_geometry():
    manager = _manager()
    manager.init_kv_cache(LAYERS, (HEIGHT, WIDTH))
    assert manager.is_init is True
    assert manager.first_layer_name == LAYERS[0]
    for layer in LAYERS:
        assert manager.get_kv_cache(layer) is None
        assert manager.get_kv_cache_frame_indices(layer) == []


def test_init_keeps_height_and_width_for_the_token_arithmetic():
    manager = _manager()
    manager.init_kv_cache(LAYERS, (HEIGHT, WIDTH))
    assert (manager.height, manager.width) == (HEIGHT, WIDTH)


# ----------------------------------------------------------------------------------------------
# frame -> token arithmetic
# ----------------------------------------------------------------------------------------------

def test_frame_range_maps_to_tokens_by_area():
    """A cached frame range costs (end - start) * H * W tokens per layer; this drives the memory budget."""
    manager = _manager()
    manager.init_kv_cache(LAYERS, (HEIGHT, WIDTH))
    assert manager._frame_indices_to_tokens((0, CHUNK_FRAMES)) == CHUNK_FRAMES * HEIGHT * WIDTH
    assert manager._frame_indices_to_tokens((2, 5)) == 3 * HEIGHT * WIDTH
    assert manager._frame_indices_to_tokens((4, 4)) == 0


def test_frame_range_arithmetic_follows_a_different_geometry():
    manager = _manager()
    manager.init_kv_cache(LAYERS, (16, 16))
    assert manager._frame_indices_to_tokens((0, 1)) == 16 * 16


# ----------------------------------------------------------------------------------------------
# batch broadcast
# ----------------------------------------------------------------------------------------------

def test_absent_layer_returns_none():
    manager = _manager()
    manager.init_kv_cache(LAYERS, (HEIGHT, WIDTH))
    assert manager.get_kv_cache("not_a_layer") is None


def test_single_cached_batch_serves_several_branches_zero_copy():
    manager = _manager()
    manager.init_kv_cache(LAYERS, (HEIGHT, WIDTH))
    manager._merged_cache[LAYERS[0]] = _merged(batch=1)

    k, v = manager.get_kv_cache(LAYERS[0], current_batch_size=4)
    assert k.shape[0] == 4 and v.shape[0] == 4
    # expand, not repeat: the storage is shared and every row equals row 0
    assert k.stride(0) == 0
    assert torch.equal(k[0], k[3])


def test_multi_batch_cache_is_repeated_when_divisible():
    manager = _manager()
    manager.init_kv_cache(LAYERS, (HEIGHT, WIDTH))
    manager._merged_cache[LAYERS[0]] = _merged(batch=2)

    k, _ = manager.get_kv_cache(LAYERS[0], current_batch_size=4)
    assert k.shape[0] == 4
    assert k.stride(0) != 0, "a non-unit cached batch falls back to repeat(), which allocates"
    assert torch.equal(k[0], k[2]) and torch.equal(k[1], k[3])


def test_indivisible_batch_is_refused():
    """Broadcasting 3 cached rows to 4 branches would duplicate one branch's history onto another."""
    manager = _manager()
    manager.init_kv_cache(LAYERS, (HEIGHT, WIDTH))
    manager._merged_cache[LAYERS[0]] = _merged(batch=3)

    with pytest.raises(ValueError) as excinfo:
        manager.get_kv_cache(LAYERS[0], current_batch_size=4)
    assert "divisible" in str(excinfo.value)


def test_no_broadcast_when_the_cache_is_already_wide_enough():
    manager = _manager()
    manager.init_kv_cache(LAYERS, (HEIGHT, WIDTH))
    manager._merged_cache[LAYERS[0]] = _merged(batch=4)

    k, _ = manager.get_kv_cache(LAYERS[0], current_batch_size=4)
    assert k.shape[0] == 4


# ----------------------------------------------------------------------------------------------
# frame index bookkeeping
# ----------------------------------------------------------------------------------------------

def test_frame_indices_default_to_the_first_layer():
    manager = _manager()
    manager.init_kv_cache(LAYERS, (HEIGHT, WIDTH))
    manager.kv_cache_frame_indices[LAYERS[0]] = [(0, 1), (1, 3)]
    assert manager.get_kv_cache_frame_indices() == [(0, 1), (1, 3)]
    assert manager.get_kv_cache_frame_indices(LAYERS[1]) == []
