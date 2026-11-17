# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""FBCA (Flash Block-wise Causal Attention) and the other streaming attention kernels, on FlashAttention-3.

FBCA is ``block_wise_causal_attention_with_prefix`` and ``block_wise_causal_mm_attn_with_prefix``. An attention
block is one query unit: it attends bidirectionally inside itself and causally to the blocks before it, plus an
optional shared prefix (it is not a transformer block, as in HieraKV's full-history / sink-recent blocks). The
caller decides what a block is -- one chunk in the clean-history branch and in KV recompute, one training window
of chunks in the noisy branch of window-wise teacher forcing. FBCA unfolds the K/V visible to each query block
into its own sequence and packs them for one varlen call; contiguous groups of blocks bound the packed KV length
of a call (``packed_kv_max_mem``).

Training (all windows of a sample in one forward):
  - ``full_history_parallel_attention``: clean cond chunks + noisy windows with full history.
  - ``sink_recent_parallel_attention``: each window sees only [sink, recent, self].
Inference (one chunk per call, history from the KV cache):
  - ``chunk_inference_attention``, plus ``block_wise_causal_attention_with_prefix`` for KV recompute.

Image queries see the clean history, while the text queries of a block see only that block's [img, txt]. Ragged KV
sets are packed with ``gather_concat`` and run through the varlen kernel; no explicit attention masks are
materialized. The kernel is FlashAttention-3, or SageAttention for inference (``set_attention_backend``).
"""

import functools

import torch
import torch.distributed as dist
import torch.nn.functional as F

# FlashAttention-3 is resolved on first use instead of at import time: it is built from `hopper/` and needs an
# sm90 GPU, so importing it eagerly made this module -- and every entry point that reaches it -- unusable on
# A100 / RTX 4090 / RTX 5090. The absence is reported at the call site, the same way
# `utils/flash_attn_no_pad.py` already treats its v3 wrapper.
_fa3_kernels = None


def _load_fa3():
    """Return (flash_attn_func, flash_attn_varlen_func) from FlashAttention-3, importing on first call."""
    global _fa3_kernels
    if _fa3_kernels is None:
        try:
            from flash_attn_interface import flash_attn_func as func
            from flash_attn_interface import flash_attn_varlen_func as varlen_func
        except Exception as e:
            raise ImportError(
                "attention backend 'fa3' needs FlashAttention-3 (flash_attn_interface), which requires a "
                "Hopper GPU (sm90); install it as described in the README"
            ) from e
        _fa3_kernels = (func, varlen_func)
    return _fa3_kernels

from movieroll.models.transformers.modules.padding import prefix_pack, prefix_unpack
from movieroll.utils.flash_attn_no_pad import fa3_output

# ══════════════════════════════════════════════════════════════
#  Attention kernel: FlashAttention-3 (default) or SageAttention
# ══════════════════════════════════════════════════════════════
#
# Every attention call of this module is non-causal and goes through `_attention` / `_attention_varlen`.
# SageAttention (INT8 Q/K, FP8 P/V on Hopper) is an inference-only speed-up: its kernels have no backward, so the
# training entry points never select it. `movieroll.generate` selects it with `--attention-backend sage`.
ATTENTION_BACKENDS = ("fa3", "sage")
_attention_backend = "fa3"


def set_attention_backend(name):
    """Route this module's attention through FlashAttention-3 (``"fa3"``) or SageAttention 2 (``"sage"``)."""
    global _attention_backend
    if name not in ATTENTION_BACKENDS:
        raise ValueError(f"attention backend must be one of {ATTENTION_BACKENDS}, got {name!r}")
    if name == "sage":
        try:
            import sageattention  # noqa: F401
        except ImportError as e:
            raise ImportError("--attention-backend sage needs the `sageattention` package (SageAttention 2)") from e
    # Resolve the FlashAttention-3 kernels now rather than at the first attention call, so a missing Hopper
    # kernel is reported while the arguments are being parsed instead of mid-rollout.
    if name == "fa3":
        _load_fa3()
    _attention_backend = name


def _attention(query, key, value):
    """Non-causal attention over ``[B, L, H, D]`` tensors."""
    if _attention_backend == "sage":
        from sageattention import sageattn
        return sageattn(query, key, value, tensor_layout="NHD", is_causal=False)
    flash_attn_func_v3, _ = _load_fa3()
    return fa3_output(flash_attn_func_v3(query, key, value, causal=False))


def _attention_varlen(query, key, value, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
    """Non-causal attention over packed ``[total_tokens, H, D]`` sequences delimited by ``cu_seqlens_*``."""
    if _attention_backend == "sage":
        # SageAttention's own varlen entry is a Triton kernel, no faster than FlashAttention-3 on Hopper; its dense
        # kernel is. Inference packs few sequences (one per prompt, or one per recomputed chunk): run them one by one.
        from sageattention import sageattn
        out = torch.empty_like(query)
        bounds_q, bounds_k = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
        for q0, q1, k0, k1 in zip(bounds_q, bounds_q[1:], bounds_k, bounds_k[1:]):
            if q1 > q0:
                out[q0:q1] = sageattn(
                    query[None, q0:q1], key[None, k0:k1], value[None, k0:k1], tensor_layout="NHD", is_causal=False,
                )[0]
        return out
    _, flash_attn_varlen_func_v3 = _load_fa3()
    return fa3_output(flash_attn_varlen_func_v3(
        query, key, value,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        causal=False,
    ))


# ══════════════════════════════════════════════════════════════
#  Packed KV memory budget & automatic grouping
# ══════════════════════════════════════════════════════════════
#
# Upper bound (bytes) on the packed KV of a single flash_attn call in block-wise causal attention: when the N blocks
# together exceed it they are split into groups processed sequentially, which bounds the kernel workspace and lets
# the gradient checkpoint around each transformer block free each group's saved tensors in backward.
# _packed_kv_mem_to_max_tokens() turns the byte budget into a token budget at run time.
_DEFAULT_PACKED_KV_MAX_MEM = 8 * (1 << 30)

_group_attn_logged = False


def _log_group_attn_triggered(caller, num_blocks, block_size, num_groups, packed_kv_total_tokens, packed_kv_max_tokens):
    """Log once (rank 0) the first time grouped attention is triggered."""
    global _group_attn_logged
    if _group_attn_logged:
        return
    _group_attn_logged = True
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            f"[FBCA] Grouped KV packing activated | caller={caller}, "
            f"num_blocks={num_blocks}, block_size={block_size}, "
            f"num_groups={num_groups}, packed_kv_total_tokens={packed_kv_total_tokens}, "
            f"packed_kv_max_tokens={packed_kv_max_tokens}"
        )


def _packed_kv_mem_to_max_tokens(packed_kv_max_mem, num_heads, head_dim, element_size, batch_size=1):
    """Per-batch-element cap on packed KV tokens for a byte budget; None (no limit) when the budget is None.

        packed_kv_max_tokens = packed_kv_max_mem / (batch_size × 4 × num_heads × head_dim × element_size)

    4 = K + V + dK + dV, alive at the same time at the backward peak; ``num_heads`` is the per-GPU head count
    after Ulysses SP, so higher SP allows more tokens under the same budget.
    """
    if packed_kv_max_mem is None:
        return None
    bytes_per_kv_token = 4 * num_heads * head_dim * element_size
    return packed_kv_max_mem // (batch_size * bytes_per_kv_token)

class _GatherConcat(torch.autograd.Function):
    """Differentiable "pre-allocate + fill" concatenation.

    The output is allocated once with ``torch.empty`` and filled slice by slice; plain slice assignment would cut
    the autograd graph, so forward / backward are written explicitly.

    ``plan`` is a list of ``(source_idx, index_tuple, segment_length)``: ``sources[source_idx][index_tuple]``
    (``segment_length`` rows) is appended to the output, ``index_tuple`` being used directly as a tensor index --
    ``(b,)``, ``(b, slice(0, n))``, ``(b, block_idx)``, or ``(slice(start, end),)`` for a packed source. A region
    may appear in the plan several times; its gradients accumulate.
    """

    @staticmethod
    def forward(ctx, total_len, plan, *sources):
        """sources share their last two dims ([H, D]); returns [total_len, H, D] in plan order."""
        trailing_shape = sources[0].shape[-2:]
        output = torch.empty(total_len, *trailing_shape,
                             dtype=sources[0].dtype, device=sources[0].device)

        dst = 0
        for src_idx, idx, seg_len in plan:
            output[dst:dst + seg_len] = sources[src_idx][idx]
            dst += seg_len

        # Only shapes are needed in backward, no save_for_backward.
        ctx.plan = plan
        ctx.source_shapes = tuple(s.shape for s in sources)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """Scatter-add ``grad_output`` back along the plan: ``grad_sources[k][idx] += grad_output[dst:dst+L]``.

        ``+=`` because a source region may be referenced several times. Every += on a bf16 buffer rounds back to
        bf16, so the error of a token referenced m times grows like sqrt(m): sources with overlapping references
        accumulate in an fp32 buffer and are cast back at the end, sources referenced once keep their dtype.
        """
        # More referenced tokens than the source holds ⇒ overlapping references ⇒ fp32 accumulation.
        ref_tokens = [0] * len(ctx.source_shapes)
        for src_idx, _, seg_len in ctx.plan:
            ref_tokens[src_idx] += seg_len
        grads = [
            torch.zeros(shape, device=grad_output.device,
                        dtype=torch.float32 if n_ref > shape[:-2].numel() else grad_output.dtype)
            for shape, n_ref in zip(ctx.source_shapes, ref_tokens)
        ]

        dst = 0
        for src_idx, idx, seg_len in ctx.plan:
            grads[src_idx][idx] += grad_output[dst:dst + seg_len]
            dst += seg_len

        # (grad_total_len, grad_plan, grad_source_0, grad_source_1, ...)
        return (None, None) + tuple(g.to(grad_output.dtype) for g in grads)


def gather_concat(plan, *sources):
    """Gather the plan's segments from ``sources`` into one [total_len, H, D] tensor (see ``_GatherConcat``)."""
    total_len = sum(seg_len for _, _, seg_len in plan)
    return _GatherConcat.apply(total_len, plan, *sources)


def _compute_total_tokens(num_blocks, block_size, full_prefix_len):
    """Total packed KV tokens (one batch element) of img-only block-wise causal attention.

    The KV of block_i has P + i*C + C tokens, so the total is N*(P+C) + C*N*(N-1)/2.
    """
    return num_blocks * (block_size + full_prefix_len) + block_size * num_blocks * (num_blocks - 1) // 2

@functools.lru_cache(maxsize=256)
def _get_group_block_offsets(num_blocks, block_size, full_prefix_len, packed_kv_max_tokens):
    """Group boundaries over blocks, e.g. (0, 10, 20, 33) for the groups [0,10), [10,20), [20,33).

    The packed KV length of block i, w_i = P + (i+1)*C, does not depend on the grouping, so grouping just cuts
    w_0..w_{N-1} into contiguous segments: a left-to-right greedy pass with capacity packed_kv_max_tokens gives the
    fewest groups G, then, with G fixed, a binary search for the smallest capacity still yielding G groups
    minimizes max_g T_g. The slack ends up in the rightmost group, the one backward processes first (while the
    packed KV saved by all other groups is still alive).
    """
    block_kv_tokens = [full_prefix_len + (i + 1) * block_size for i in range(num_blocks)]
    if block_kv_tokens[-1] > packed_kv_max_tokens:
        raise ValueError(
            f"packed_kv_max_tokens({packed_kv_max_tokens}) is too small: a single block "
            f"(kv_tokens={block_kv_tokens[-1]}) already exceeds it, increase packed_kv_max_mem"
        )

    # Requires cap >= max(block_kv_tokens) so every group holds at least one block.
    def greedy_offsets(cap):
        offsets = [0]
        group_kv_tokens = 0
        for block_idx, kv_tokens in enumerate(block_kv_tokens):
            if group_kv_tokens + kv_tokens > cap:
                offsets.append(block_idx)
                group_kv_tokens = 0
            group_kv_tokens += kv_tokens
        offsets.append(num_blocks)
        return offsets

    num_groups = len(greedy_offsets(packed_kv_max_tokens)) - 1

    # The greedy group count is non-increasing in cap → binary search for the smallest feasible cap.
    lo = max(block_kv_tokens[-1], -(-sum(block_kv_tokens) // num_groups))
    hi = packed_kv_max_tokens
    while lo < hi:
        mid = (lo + hi) // 2
        if len(greedy_offsets(mid)) - 1 <= num_groups:
            hi = mid
        else:
            lo = mid + 1
    return tuple(greedy_offsets(lo))

# ══════════════════════════════════════════════════════════════
#  Img-only block-wise causal attention (no text / encoder input)
# ══════════════════════════════════════════════════════════════

def _block_wise_causal_attention_with_prefix(
    query,                    # [B, num_blocks * block_size, num_heads, head_dim]
    key,                      # [B, num_blocks * block_size, num_heads, head_dim]
    value,                    # [B, num_blocks * block_size, num_heads, head_dim]
    causal_prefix_key=None,   # [B, (num_blocks-1)*block_size, num_heads, head_dim]
    causal_prefix_value=None, # [B, (num_blocks-1)*block_size, num_heads, head_dim]
    full_prefix_key=None,     # [B, full_prefix_len, num_heads, head_dim]
    full_prefix_value=None,   # [B, full_prefix_len, num_heads, head_dim]
    block_size=None,          # int, img tokens per block
):
    """Img-only block-wise causal attention over num_blocks equal blocks (internal).

    Every block must have ``block_size`` tokens; the caller merges a differently sized block 0 into full_prefix.
    For block i: Q_i = img_i, KV_i = [full_prefix?, causal_prefix[:i*C]?, img_i]. Without any prefix this is
    block-diagonal self-attention over independent blocks.

    Returns out: [B, num_blocks * block_size, num_heads, head_dim].
    """
    B, total_img_len, num_heads, head_dim = query.shape
    device = query.device
    num_blocks = total_img_len // block_size
    total_seqs = B * num_blocks

    has_full_prefix = full_prefix_key is not None and full_prefix_value is not None
    has_causal_prefix = causal_prefix_key is not None and causal_prefix_value is not None
    full_prefix_len = full_prefix_key.shape[1] if has_full_prefix else 0

    # split per block: [B, num_blocks, block_size, H, D]
    query_per_block = query.view(B, num_blocks, block_size, num_heads, head_dim)
    key_per_block = key.view(B, num_blocks, block_size, num_heads, head_dim)
    value_per_block = value.view(B, num_blocks, block_size, num_heads, head_dim)

    # Fast path: no prefix → block-diagonal batch attention
    if not has_causal_prefix and not has_full_prefix:
        query_flat = query_per_block.reshape(total_seqs, block_size, num_heads, head_dim)
        key_flat = key_per_block.reshape(total_seqs, block_size, num_heads, head_dim)
        value_flat = value_per_block.reshape(total_seqs, block_size, num_heads, head_dim)
        out = _attention(query_flat, key_flat, value_flat)
        return out.reshape(B, total_img_len, num_heads, head_dim)

    # full_prefix only: KV_i = [full_prefix, self_i] for every block, same length → batch attention
    if not has_causal_prefix:
        query_flat = query_per_block.reshape(total_seqs, block_size, num_heads, head_dim)
        # broadcast full_prefix to every block: [B, 1, P, H, D] → [B, num_blocks, P, H, D]
        fp_key_expanded = full_prefix_key.unsqueeze(1).expand(-1, num_blocks, -1, -1, -1)
        fp_value_expanded = full_prefix_value.unsqueeze(1).expand(-1, num_blocks, -1, -1, -1)
        key_with_prefix = torch.cat([fp_key_expanded, key_per_block], dim=2)    # [B, N, P+C, H, D]
        value_with_prefix = torch.cat([fp_value_expanded, value_per_block], dim=2)
        key_flat = key_with_prefix.reshape(total_seqs, full_prefix_len + block_size, num_heads, head_dim)
        value_flat = value_with_prefix.reshape(total_seqs, full_prefix_len + block_size, num_heads, head_dim)
        out = _attention(query_flat, key_flat, value_flat)
        return out.reshape(B, total_img_len, num_heads, head_dim)

    # General varlen path: KV_i = [full_prefix?, causal_prefix[:cp_lens[i]], self_i]

    # causal prefix length per block, in pure Python so there is no GPU→CPU sync
    causal_total = causal_prefix_key.shape[1]
    assert causal_total == (num_blocks - 1) * block_size, \
        f"causal_total({causal_total}) != (num_blocks-1)({num_blocks-1}) * block_size({block_size})"
    causal_prefix_lens = [i * block_size for i in range(num_blocks)]
    causal_prefix_lens_gpu = torch.tensor(causal_prefix_lens, device=device, dtype=torch.int32)

    # Q: all blocks have the same length, so reshape + arange gives cu_seqlens
    query_packed = query_per_block.reshape(-1, num_heads, head_dim)     # [total_seqs * C, H, D]
    cu_seqlens_query = torch.arange(
        0, (total_seqs + 1) * block_size, block_size,
        device=device, dtype=torch.int32)

    # ragged KV, assembled by gather_concat in batch × block order
    plan = []
    if has_full_prefix:
        IDX_FULL, IDX_CAUSAL, IDX_SELF = 0, 1, 2
        key_sources = (full_prefix_key, causal_prefix_key, key_per_block)
        value_sources = (full_prefix_value, causal_prefix_value, value_per_block)
    else:
        IDX_CAUSAL, IDX_SELF = 0, 1
        key_sources = (causal_prefix_key, key_per_block)
        value_sources = (causal_prefix_value, value_per_block)

    for batch_idx in range(B):
        for block_idx in range(num_blocks):
            if has_full_prefix:
                plan.append((IDX_FULL, (batch_idx,), full_prefix_len))
            if causal_prefix_lens[block_idx] > 0:
                cp_len = causal_prefix_lens[block_idx]
                plan.append((IDX_CAUSAL, (batch_idx, slice(0, cp_len)), cp_len))
            plan.append((IDX_SELF, (batch_idx, block_idx), block_size))

    key_flat = gather_concat(plan, *key_sources)     # [total_kv_tokens, H, D]
    value_flat = gather_concat(plan, *value_sources)

    prefix_len_per_block = causal_prefix_lens_gpu + full_prefix_len         # [num_blocks]
    kv_lens = (prefix_len_per_block + block_size).repeat(B)                  # [total_seqs]
    cu_seqlens_kv = F.pad(kv_lens.cumsum(0), (1, 0)).to(torch.int32)
    max_kv_len = full_prefix_len + causal_total + block_size

    out = _attention_varlen(
        query_packed, key_flat, value_flat,
        cu_seqlens_q=cu_seqlens_query, cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=block_size, max_seqlen_k=max_kv_len,
    )
    return out.reshape(B, total_img_len, num_heads, head_dim)


def block_wise_causal_attention_with_prefix(
    query,                    # [B, total_img_len, num_heads, head_dim]
    key,                      # [B, total_img_len, num_heads, head_dim]
    value,                    # [B, total_img_len, num_heads, head_dim]
    full_prefix_key=None,     # [B, full_prefix_len, num_heads, head_dim]
    full_prefix_value=None,   # [B, full_prefix_len, num_heads, head_dim]
    causal_prefix_key=None,   # [B, first_block_size + (num_blocks - 2) * regular_block_size, num_heads, head_dim]
    causal_prefix_value=None, # [B, first_block_size + (num_blocks - 2) * regular_block_size, num_heads, head_dim]
    block_sizes=None,         # List[int], img tokens per block
    packed_kv_max_mem=_DEFAULT_PACKED_KV_MAX_MEM,  # backward peak memory budget (bytes) of the packed KV; grouped above
):
    """Img-only block-wise causal attention (FBCA): Q_i = img_i, KV_i = [full_prefix?, causal_prefix[:i]?, img_i].

    A block is the unit the caller packs (see the module docstring). In the general case each block's visible K/V
    is unfolded into its own sequence and the sequences are packed into one varlen call, or into one call per
    group of blocks once the packed KV exceeds ``packed_kv_max_mem``.

    ``causal_prefix_key/value`` is the K/V of blocks 0..N-2 as seen by later blocks, of length F + (N-2)*C
    (F = block_sizes[0], C = the size of blocks 1..N-1); None gives block-diagonal self-attention over
    independent blocks. ``full_prefix_key/value`` is seen by every block. With F != C, block 0 runs its own
    self-attention and its K/V -- causal_prefix[:F] -- is folded into full_prefix, which turns blocks 1~N-1 into
    an F == C problem handed back to this entry (recursion depth 1).

    Returns:
        out: [B, total_img_len, num_heads, head_dim]
    """
    B, total_img_len, num_heads, head_dim = query.shape

    first_block_size = block_sizes[0]
    num_blocks = len(block_sizes)
    regular_block_size = block_sizes[-1]

    assert first_block_size + regular_block_size * (num_blocks - 1) == total_img_len, \
        f"first_block_size({first_block_size}) + regular_block_size({regular_block_size}) * ({num_blocks - 1}) != total_img_len({total_img_len})"
    assert num_blocks == 1 or len(set(block_sizes[1:])) <= 1, \
        f"block 1~N-1 should all have the same size, got {set(block_sizes[1:])}"

    has_full_prefix = (full_prefix_key is not None and full_prefix_value is not None
                       and full_prefix_key.shape[1] > 0)
    full_prefix_len = full_prefix_key.shape[1] if has_full_prefix else 0

    has_causal_prefix = (causal_prefix_key is not None and causal_prefix_value is not None
                         and causal_prefix_key.shape[1] > 0)
    causal_prefix_len = causal_prefix_key.shape[1] if has_causal_prefix else 0

    if has_causal_prefix:
        assert num_blocks > 1, \
            f"num_blocks({num_blocks}) must be > 1 when causal_prefix is provided"
        assert first_block_size + regular_block_size * (num_blocks - 2) == causal_prefix_len, \
            f"first_block_size({first_block_size}) + regular_block_size({regular_block_size}) * ({num_blocks - 2}) != causal_prefix_len({causal_prefix_len})"

    packed_kv_max_tokens = _packed_kv_mem_to_max_tokens(
        packed_kv_max_mem, num_heads, head_dim, query.element_size(), batch_size=B)

    # Single block (N == 1): self-attention + optional full_prefix
    if num_blocks == 1:
        if has_full_prefix:
            key_with_prefix = torch.cat([full_prefix_key, key], dim=1)      # [B, P+F, H, D]
            value_with_prefix = torch.cat([full_prefix_value, value], dim=1)
            return _attention(query, key_with_prefix, value_with_prefix)
        return _attention(query, key, value)

    # F == C, N > 1: all blocks equal. When the packed KV exceeds the budget the N blocks are split into G groups
    # run in sequence; group [gs, ge) takes blocks gs..ge-2 as its causal_prefix and the original full_prefix plus
    # global blocks 0..gs-1 as its full_prefix (both sliced from the external causal_prefix), so global block i
    # sees exactly the KV it sees without grouping.
    if first_block_size == regular_block_size:
        # no causal prefix: independent blocks (block-diagonal pattern)
        if not has_causal_prefix:
            return _block_wise_causal_attention_with_prefix(
                query, key, value,
                causal_prefix_key=None,
                causal_prefix_value=None,
                full_prefix_key=full_prefix_key,
                full_prefix_value=full_prefix_value,
                block_size=regular_block_size,
            )

        packed_kv_total_tokens = _compute_total_tokens(num_blocks, regular_block_size, full_prefix_len)

        # grouped path: the packed KV exceeds the memory budget
        if packed_kv_max_tokens is not None and packed_kv_total_tokens > packed_kv_max_tokens:
            block_offsets = _get_group_block_offsets(num_blocks, regular_block_size, full_prefix_len, packed_kv_max_tokens)
            _log_group_attn_triggered(
                "img_only/F==C", num_blocks, regular_block_size,
                len(block_offsets) - 1, packed_kv_total_tokens, packed_kv_max_tokens)
            group_outputs = []
            for group_idx in range(len(block_offsets) - 1):
                group_start = block_offsets[group_idx]
                group_end = block_offsets[group_idx + 1]
                group_size = group_end - group_start

                # QKV of this group (views, no copy)
                query_group = query[:, group_start * regular_block_size:group_end * regular_block_size]
                key_group = key[:, group_start * regular_block_size:group_end * regular_block_size]
                value_group = value[:, group_start * regular_block_size:group_end * regular_block_size]

                # in-group causal prefix: the first group_size-1 blocks of the group
                if group_size > 1:
                    group_causal_prefix_key = causal_prefix_key[:, group_start * regular_block_size:(group_end - 1) * regular_block_size]
                    group_causal_prefix_value = causal_prefix_value[:, group_start * regular_block_size:(group_end - 1) * regular_block_size]
                else:
                    group_causal_prefix_key = None
                    group_causal_prefix_value = None

                # group full_prefix: original full_prefix + every block before the group
                if group_start > 0:
                    pre_group_key = causal_prefix_key[:, :group_start * regular_block_size]
                    pre_group_value = causal_prefix_value[:, :group_start * regular_block_size]
                    if has_full_prefix:
                        group_full_prefix_key = torch.cat([full_prefix_key, pre_group_key], dim=1)
                        group_full_prefix_value = torch.cat([full_prefix_value, pre_group_value], dim=1)
                    else:
                        group_full_prefix_key = pre_group_key
                        group_full_prefix_value = pre_group_value
                else:
                    group_full_prefix_key = full_prefix_key
                    group_full_prefix_value = full_prefix_value

                group_outputs.append(_block_wise_causal_attention_with_prefix(
                    query_group, key_group, value_group,
                    causal_prefix_key=group_causal_prefix_key,
                    causal_prefix_value=group_causal_prefix_value,
                    full_prefix_key=group_full_prefix_key,
                    full_prefix_value=group_full_prefix_value,
                    block_size=regular_block_size,
                ))
            return torch.cat(group_outputs, dim=1)

        # ungrouped path: pass causal_prefix through
        return _block_wise_causal_attention_with_prefix(
            query, key, value,
            causal_prefix_key=causal_prefix_key,
            causal_prefix_value=causal_prefix_value,
            full_prefix_key=full_prefix_key,
            full_prefix_value=full_prefix_value,
            block_size=regular_block_size,
        )

    # F != C, N > 1: block 0 (F tokens) runs on its own, then blocks 1~N-1 run causal or block-diagonal. With a
    # causal prefix, block 0's K/V as seen by the rest (= causal_prefix[:F]) is merged into rest_full_prefix,
    # making blocks 1~N-1 an F == C problem; without one the remaining blocks are independent.

    # split the img tensors into [F, (N-1)*C]
    query_first, query_rest = query.split_with_sizes(
        [first_block_size, total_img_len - first_block_size], dim=1)
    key_first, key_rest = key.split_with_sizes(
        [first_block_size, total_img_len - first_block_size], dim=1)
    value_first, value_rest = value.split_with_sizes(
        [first_block_size, total_img_len - first_block_size], dim=1)

    # block 0: self-attention, Q = img_0, KV = [full_prefix?, img_0]
    if has_full_prefix:
        key_block0 = torch.cat([full_prefix_key, key_first], dim=1)         # [B, P+F, H, D]
        value_block0 = torch.cat([full_prefix_value, value_first], dim=1)
        out_first = _attention(query_first, key_block0, value_block0)
    else:
        out_first = _attention(query_first, key_first, value_first)

    # blocks 1~N-1, independent without a causal prefix: each sees only [full_prefix?, self_i]
    if not has_causal_prefix:
        out_rest = _block_wise_causal_attention_with_prefix(
            query_rest, key_rest, value_rest,
            causal_prefix_key=None,
            causal_prefix_value=None,
            full_prefix_key=full_prefix_key,
            full_prefix_value=full_prefix_value,
            block_size=regular_block_size,
        )
        return torch.cat([out_first, out_rest], dim=1)

    # With a causal prefix, block 0's K/V seen by the rest is causal_prefix[:F]. Merging it into the full prefix
    # leaves blocks 1~N-1 all of size C, i.e. an F == C problem, handed back to this same entry.
    first_causal_prefix_key, rest_causal_prefix_key = causal_prefix_key.split_with_sizes(
        [first_block_size, causal_prefix_len - first_block_size], dim=1)
    first_causal_prefix_value, rest_causal_prefix_value = causal_prefix_value.split_with_sizes(
        [first_block_size, causal_prefix_len - first_block_size], dim=1)

    if has_full_prefix:
        rest_full_prefix_key = torch.cat([full_prefix_key, first_causal_prefix_key], dim=1)
        rest_full_prefix_value = torch.cat([full_prefix_value, first_causal_prefix_value], dim=1)
    else:
        rest_full_prefix_key = first_causal_prefix_key
        rest_full_prefix_value = first_causal_prefix_value

    # Blocks 1~N-1 all have `regular_block_size` tokens (asserted at the top), so this recursive call can only
    # land in the F == C branch or, when N == 2, in the N == 1 early return: recursion depth is 1. N == 2
    # therefore skips the packed-KV budget entirely, because the N == 1 branch never consults it. The budget
    # is forwarded explicitly so that a caller's smaller budget reaches the recursive frame too.
    out_rest = block_wise_causal_attention_with_prefix(
        query_rest, key_rest, value_rest,
        full_prefix_key=rest_full_prefix_key,
        full_prefix_value=rest_full_prefix_value,
        causal_prefix_key=rest_causal_prefix_key,
        causal_prefix_value=rest_causal_prefix_value,
        block_sizes=block_sizes[1:],
        packed_kv_max_mem=packed_kv_max_mem,
    )

    return torch.cat([out_first, out_rest], dim=1)


# ══════════════════════════════════════════════════════════════
#  MM attention where text does not see the cond image
#
#  img_Q_i -> KV = [cond_img_all?, causal_prefix?, img_i, txt_i]
#  txt_Q_i -> KV = [img_i, txt_i]  (no prefix at all)
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
#  Helper: txt_Q sees only [img, txt] of its own block (batched over blocks)
# ══════════════════════════════════════════════════════════════

def _txt_self_attn_per_block(
    encoder_query,     # [B, num_blocks, text_len, num_heads, head_dim]
    self_kv_key,       # [B, num_blocks, block_size + text_len, num_heads, head_dim]
    self_kv_value,     # [B, num_blocks, block_size + text_len, num_heads, head_dim]
    text_mask=None,    # [B, num_blocks, text_len] bool, prefix layout
    block_size=None,   # int
):
    """txt_Q attends only to [img, txt] of its own block, never to a prefix: txt_Q_i → KV = [img_i, valid_txt_i].

    encoder_query [B, N, S, H, D], self_kv_key/value [B, N, C+S, H, D] (prebuilt [img, txt] per block),
    text_mask [B, N, S] bool in prefix layout or None. Returns txt_out [B, N, S, H, D].
    """
    B, num_blocks, text_len, num_heads, head_dim = encoder_query.shape
    total_seqs = B * num_blocks
    block_seq_len = block_size + text_len

    # no text_mask: Q and KV have uniform lengths → batch mode
    if text_mask is None:
        q_flat = encoder_query.reshape(total_seqs, text_len, num_heads, head_dim)
        k_flat = self_kv_key.reshape(total_seqs, block_seq_len, num_heads, head_dim)
        v_flat = self_kv_value.reshape(total_seqs, block_seq_len, num_heads, head_dim)
        out = _attention(q_flat, k_flat, v_flat)
        return out.view(B, num_blocks, text_len, num_heads, head_dim)

    # text_mask: variable Q/KV lengths → prefix_pack + varlen. Valid Q length is text_valid[b,c], valid KV length
    # block_size + text_valid[b,c]; the KV layout is [img (all valid), txt (valid prefix, then padding)], so
    # keeping the first kv_valid tokens cuts exactly the txt padding.
    text_valid = text_mask.sum(dim=-1, dtype=torch.int32)           # [B, N]
    q_valid = text_valid.reshape(total_seqs)                        # [B*N]
    kv_valid = (block_size + text_valid).reshape(total_seqs)        # [B*N]

    cu_q = F.pad(q_valid.cumsum(0, dtype=torch.int32), (1, 0))
    cu_kv = F.pad(kv_valid.cumsum(0, dtype=torch.int32), (1, 0))

    # one tolist(); kv_valid_list is derived in Python so there is no second GPU→CPU sync
    q_valid_list  = q_valid.tolist()
    kv_valid_list = [block_size + vl for vl in q_valid_list]
    max_valid_len_q = max(q_valid_list)
    max_valid_len_kv = max(kv_valid_list)

    q_flat = prefix_pack(
        encoder_query.reshape(total_seqs, text_len, num_heads, head_dim), q_valid_list)
    k_flat = prefix_pack(
        self_kv_key.reshape(total_seqs, block_seq_len, num_heads, head_dim), kv_valid_list)
    v_flat = prefix_pack(
        self_kv_value.reshape(total_seqs, block_seq_len, num_heads, head_dim), kv_valid_list)

    out = _attention_varlen(
        q_flat, k_flat, v_flat,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_kv,
        max_seqlen_q=max_valid_len_q, max_seqlen_k=max_valid_len_kv,
    )
    return prefix_unpack(out, q_valid_list, total_seqs, text_len).view(
        B, num_blocks, text_len, num_heads, head_dim)


# ══════════════════════════════════════════════════════════════
#  Helper: split attention of a single block
# ══════════════════════════════════════════════════════════════

def _self_mm_attn_split(
    query,                    # [B, img_len, H, D]
    key,                      # [B, img_len, H, D]
    value,                    # [B, img_len, H, D]
    encoder_query,            # [B, text_len, H, D]
    encoder_key,              # [B, text_len, H, D]
    encoder_value,            # [B, text_len, H, D]
    full_prefix_key=None,     # [B, P, H, D] or None
    full_prefix_value=None,   # [B, P, H, D] or None
    text_mask=None,           # [B, text_len] bool, prefix layout
):
    """Split attention of a single block: img_Q → KV = [prefix, img, valid_txt], txt_Q → KV = [img, valid_txt].

    Without a prefix the two share KV = [img, valid_txt] and run as a single attention. ``text_mask`` is a
    [B, text_len] prefix bool mask (None = all valid). Returns (img_out, txt_out).
    """
    B, img_len, num_heads, head_dim = query.shape
    device = query.device
    text_len = encoder_query.shape[1]
    seqlen = img_len + text_len
    has_prefix = full_prefix_key is not None and full_prefix_value is not None

    # Valid text lengths, needed by both branches. tolist() runs once here; every derived valid_lens list below is
    # computed in Python, avoiding further GPU→CPU syncs.
    txt_valid = None
    txt_valid_list = None
    if text_mask is not None:
        txt_valid = text_mask.sum(dim=-1, dtype=torch.int32)      # [B]
        txt_valid_list = txt_valid.tolist()                       # the only sync

    # No prefix: img/txt share the KV → one attention
    if not has_prefix:
        q_cat = torch.cat([query, encoder_query], dim=1)          # [B, img+txt, H, D]
        k_cat = torch.cat([key, encoder_key], dim=1)
        v_cat = torch.cat([value, encoder_value], dim=1)

        if text_mask is None:
            out = _attention(q_cat, k_cat, v_cat)
        else:
            valid_lens = img_len + txt_valid                      # [B]  (GPU tensor for cumsum)
            cu = F.pad(valid_lens.cumsum(0, dtype=torch.int32), (1, 0))
            # derived from txt_valid_list, no extra sync
            valid_lens_list = [img_len + vl for vl in txt_valid_list]
            max_valid_len = max(valid_lens_list)
            out = _attention_varlen(
                prefix_pack(q_cat, valid_lens_list),
                prefix_pack(k_cat, valid_lens_list),
                prefix_pack(v_cat, valid_lens_list),
                cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=max_valid_len, max_seqlen_k=max_valid_len,
            )
            out = prefix_unpack(out, valid_lens_list, B, seqlen)

        return out[:, :img_len], out[:, img_len:]

    # Prefix: img_Q and txt_Q see different KV → two attentions
    prefix_len = full_prefix_key.shape[1]

    # txt_Q: KV = [img, valid_txt], no prefix. This concat is reused below with the prefix prepended, so
    # key / encoder_key are concatenated once instead of once per attention.
    k_self = torch.cat([key, encoder_key], dim=1)                 # [B, img+txt, H, D]
    v_self = torch.cat([value, encoder_value], dim=1)

    if text_mask is None:
        txt_out = _attention(encoder_query, k_self, v_self)
    else:
        txt_kv_valid = img_len + txt_valid                         # [B]  (GPU tensor for cumsum)
        cu_q = F.pad(txt_valid.cumsum(0, dtype=torch.int32), (1, 0))
        cu_kv = F.pad(txt_kv_valid.cumsum(0, dtype=torch.int32), (1, 0))
        # derived from txt_valid_list computed above, no extra sync
        txt_kv_valid_list = [img_len + vl for vl in txt_valid_list]
        max_valid_len_q = max(txt_valid_list)
        max_valid_len_kv = max(txt_kv_valid_list)

        txt_out = _attention_varlen(
            prefix_pack(encoder_query, txt_valid_list),
            prefix_pack(k_self, txt_kv_valid_list),
            prefix_pack(v_self, txt_kv_valid_list),
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_kv,
            max_seqlen_q=max_valid_len_q, max_seqlen_k=max_valid_len_kv,
        )
        txt_out = prefix_unpack(txt_out, txt_valid_list, B, text_len)

    # img_Q: KV = [prefix, img, valid_txt], reusing the k_self / v_self concat built above
    k_with_prefix = torch.cat([full_prefix_key, k_self], dim=1)
    v_with_prefix = torch.cat([full_prefix_value, v_self], dim=1)

    if text_mask is None:
        # uniform Q / KV lengths → batch mode (flash_attn allows len(Q) != len(KV))
        img_out = _attention(query, k_with_prefix, v_with_prefix)
    else:
        # Q all valid (img_len), valid KV = prefix_len + img_len + txt_valid
        img_kv_valid = prefix_len + img_len + txt_valid           # [B]  (GPU tensor for cumsum)
        cu_q = torch.arange(
            0, (B + 1) * img_len, img_len, device=device, dtype=torch.int32)
        cu_kv = F.pad(img_kv_valid.cumsum(0, dtype=torch.int32), (1, 0))
        # derived from txt_valid_list, no extra sync
        img_kv_valid_list = [prefix_len + img_len + vl for vl in txt_valid_list]
        max_valid_len_kv = max(img_kv_valid_list)

        img_out = _attention_varlen(
            query.reshape(-1, num_heads, head_dim),
            prefix_pack(k_with_prefix, img_kv_valid_list),
            prefix_pack(v_with_prefix, img_kv_valid_list),
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_kv,
            max_seqlen_q=img_len, max_seqlen_k=max_valid_len_kv,
        )
        img_out = img_out.reshape(B, img_len, num_heads, head_dim)

    return img_out, txt_out


# ══════════════════════════════════════════════════════════════
#  Core: multi-block internal function (equal blocks, split img_Q / txt_Q)
# ══════════════════════════════════════════════════════════════

def _block_wise_causal_mm_attn(
    query,                    # [B, num_blocks * block_size, num_heads, head_dim]
    key,                      # [B, num_blocks * block_size, num_heads, head_dim]
    value,                    # [B, num_blocks * block_size, num_heads, head_dim]
    encoder_query,            # [B, num_blocks * text_len, num_heads, head_dim]
    encoder_key,              # [B, num_blocks * text_len, num_heads, head_dim]
    encoder_value,            # [B, num_blocks * text_len, num_heads, head_dim]
    causal_prefix_key=None,   # [B, (num_blocks-1)*block_size, num_heads, head_dim]
    causal_prefix_value=None, # [B, (num_blocks-1)*block_size, num_heads, head_dim]
    full_prefix_key=None,     # [B, full_prefix_len, num_heads, head_dim]
    full_prefix_value=None,   # [B, full_prefix_len, num_heads, head_dim]
    text_mask=None,           # [B, num_blocks * text_len]
    block_size=None,          # int, img tokens per block
):
    """Split MM block-wise causal attention over num_blocks equal blocks (internal).

    Every block must have ``block_size`` tokens; the caller merges a differently sized block 0 into full_prefix.

      img_Q_i → KV = [full_prefix?, causal_prefix[:i*C]?, img_i, valid_txt_i]
      txt_Q_i → KV = [img_i, valid_txt_i]

    Without any prefix this degenerates to block-diagonal self-attention.

    Returns:
        img_out: [B, num_blocks * block_size, num_heads, head_dim]
        txt_out: [B, num_blocks * text_len, num_heads, head_dim]
    """
    B, total_img_len, num_heads, head_dim = query.shape
    device = query.device

    assert total_img_len % block_size == 0, \
        f"total_img_len({total_img_len}) must be divisible by block_size({block_size})"
    num_blocks = total_img_len // block_size
    text_len = encoder_query.shape[1] // num_blocks
    block_seq_len = block_size + text_len

    has_full_prefix = full_prefix_key is not None and full_prefix_value is not None
    has_causal_prefix = causal_prefix_key is not None and causal_prefix_value is not None
    full_prefix_len = full_prefix_key.shape[1] if has_full_prefix else 0

    total_seqs = B * num_blocks

    # split per block: [B, num_blocks, *, H, D]
    query_pc = query.view(B, num_blocks, block_size, num_heads, head_dim)
    key_pc = key.view(B, num_blocks, block_size, num_heads, head_dim)
    value_pc = value.view(B, num_blocks, block_size, num_heads, head_dim)
    enc_q_pc = encoder_query.view(B, num_blocks, text_len, num_heads, head_dim)
    enc_k_pc = encoder_key.view(B, num_blocks, text_len, num_heads, head_dim)
    enc_v_pc = encoder_value.view(B, num_blocks, text_len, num_heads, head_dim)

    text_mask_pc = text_mask.view(B, num_blocks, text_len) if text_mask is not None else None

    # block_valid_lens_list is tolist()'d once here and every derived list is computed with Python arithmetic; the
    # with-prefix path adds one more sync inside _txt_self_attn_per_block.
    if text_mask is not None:
        text_valid_pc = text_mask_pc.sum(dim=-1, dtype=torch.int32)       # [B, N]
        block_valid_lens = (block_size + text_valid_pc).reshape(total_seqs)
        block_valid_lens_list = block_valid_lens.tolist()                 # the only sync
    else:
        text_valid_pc = None
        block_valid_lens = None
        block_valid_lens_list = None

    # Fast path: no prefix → img/txt see the same KV, one merged attention
    if not has_causal_prefix and not has_full_prefix:
        combined_q = torch.cat([query_pc, enc_q_pc], dim=2)      # [B, N, C+S, H, D]
        combined_k = torch.cat([key_pc, enc_k_pc], dim=2)
        combined_v = torch.cat([value_pc, enc_v_pc], dim=2)

        if text_mask is None:
            out = _attention(
                combined_q.reshape(total_seqs, block_seq_len, num_heads, head_dim),
                combined_k.reshape(total_seqs, block_seq_len, num_heads, head_dim),
                combined_v.reshape(total_seqs, block_seq_len, num_heads, head_dim),
            )
        else:
            cu = F.pad(block_valid_lens.cumsum(0, dtype=torch.int32), (1, 0))
            max_valid_len = max(block_valid_lens_list)
            out = _attention_varlen(
                prefix_pack(combined_q.reshape(total_seqs, block_seq_len, num_heads, head_dim),
                            block_valid_lens_list),
                prefix_pack(combined_k.reshape(total_seqs, block_seq_len, num_heads, head_dim),
                            block_valid_lens_list),
                prefix_pack(combined_v.reshape(total_seqs, block_seq_len, num_heads, head_dim),
                            block_valid_lens_list),
                cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=max_valid_len, max_seqlen_k=max_valid_len,
            )
            out = prefix_unpack(out, block_valid_lens_list, total_seqs, block_seq_len)

        out = out.view(B, num_blocks, block_seq_len, num_heads, head_dim)
        img_out = out[:, :, :block_size].reshape(B, -1, num_heads, head_dim)
        txt_out = out[:, :, block_size:].reshape(B, -1, num_heads, head_dim)
        return img_out, txt_out

    # With prefix: split img_Q and txt_Q.
    # self KV = [img, txt] per block — shared by txt and img, built once
    self_kv_key = torch.cat([key_pc, enc_k_pc], dim=2)            # [B, N, C+S, H, D]
    self_kv_value = torch.cat([value_pc, enc_v_pc], dim=2)

    # Part 1: txt_Q, no prefix, current block only
    txt_out = _txt_self_attn_per_block(
        enc_q_pc, self_kv_key, self_kv_value,
        text_mask=text_mask_pc, block_size=block_size,
    )
    txt_out = txt_out.reshape(B, -1, num_heads, head_dim)

    # Part 2: img_Q, KV = [full_prefix?, causal_prefix?, self_img, self_txt]

    # full_prefix only, no causal: KV = [full_prefix, img, txt] per block, same length P + C + S for all
    if not has_causal_prefix:
        fp_k = full_prefix_key.unsqueeze(1).expand(-1, num_blocks, -1, -1, -1)
        fp_v = full_prefix_value.unsqueeze(1).expand(-1, num_blocks, -1, -1, -1)
        kv_key = torch.cat([fp_k, self_kv_key], dim=2)            # [B, N, P+C+S, H, D]
        kv_value = torch.cat([fp_v, self_kv_value], dim=2)
        kv_len = full_prefix_len + block_seq_len

        if text_mask is None:
            img_out = _attention(
                query_pc.reshape(total_seqs, block_size, num_heads, head_dim),
                kv_key.reshape(total_seqs, kv_len, num_heads, head_dim),
                kv_value.reshape(total_seqs, kv_len, num_heads, head_dim),
            )
        else:
            kv_valid = full_prefix_len + block_valid_lens          # [B*N]  (GPU tensor for cumsum)
            cu_q = torch.arange(
                0, (total_seqs + 1) * block_size, block_size,
                device=device, dtype=torch.int32)
            cu_kv = F.pad(kv_valid.cumsum(0, dtype=torch.int32), (1, 0))
            # derived from block_valid_lens_list, no extra sync
            kv_valid_list = [full_prefix_len + vl for vl in block_valid_lens_list]
            max_valid_len_kv = max(kv_valid_list)

            img_out = _attention_varlen(
                query_pc.reshape(-1, num_heads, head_dim),
                prefix_pack(kv_key.reshape(total_seqs, kv_len, num_heads, head_dim), kv_valid_list),
                prefix_pack(kv_value.reshape(total_seqs, kv_len, num_heads, head_dim), kv_valid_list),
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_kv,
                max_seqlen_q=block_size, max_seqlen_k=max_valid_len_kv,
            )

        img_out = img_out.reshape(B, -1, num_heads, head_dim)
        return img_out, txt_out

    # General varlen path: img_Q KV of block_i is ragged, KV_i = [full_prefix?, causal_prefix[:i*C], img_i,
    # valid_txt_i] against a uniform Q_i = img_i. gather_concat packs every block's KV into one dense sequence
    # and cu_seqlens marks the boundaries.

    # causal prefix length per block: block_0=0, block_1=C, block_2=2C, ...
    causal_total = causal_prefix_key.shape[1]
    assert causal_total == (num_blocks - 1) * block_size, \
        f"causal_total({causal_total}) != (num_blocks-1)({num_blocks-1}) * block_size({block_size})"
    causal_prefix_lens = [i * block_size for i in range(num_blocks)]
    causal_prefix_lens_gpu = torch.tensor(causal_prefix_lens, device=device, dtype=torch.int32)

    # Q packed: all blocks have block_size queries → reshape + arange gives cu_seqlens
    query_packed = query_pc.reshape(-1, num_heads, head_dim)      # [B*N*C, H, D]
    cu_seqlens_query = torch.arange(
        0, (total_seqs + 1) * block_size, block_size,
        device=device, dtype=torch.int32)

    # With text_mask the self KV is prefix_pack'd to drop txt padding before it enters the ragged KV.
    if text_mask is not None:
        self_kv_key_packed = prefix_pack(
            self_kv_key.reshape(total_seqs, block_seq_len, num_heads, head_dim),
            block_valid_lens_list)
        self_kv_value_packed = prefix_pack(
            self_kv_value.reshape(total_seqs, block_seq_len, num_heads, head_dim),
            block_valid_lens_list)
        # offset table into packed_self, to locate a sequence by index
        packed_self_offsets = [0]
        for valid_len in block_valid_lens_list:
            packed_self_offsets.append(packed_self_offsets[-1] + valid_len)

    # gather_concat plan over the sources IDX_FULL (full_prefix), IDX_CAUSAL (causal_prefix), IDX_SELF (self kv)
    plan = []
    key_sources = []
    value_sources = []
    next_idx = 0

    if has_full_prefix:
        IDX_FULL = next_idx
        next_idx += 1
        key_sources.append(full_prefix_key)
        value_sources.append(full_prefix_value)

    IDX_CAUSAL = next_idx
    next_idx += 1
    key_sources.append(causal_prefix_key)
    value_sources.append(causal_prefix_value)

    IDX_SELF = next_idx
    if text_mask is None:
        key_sources.append(self_kv_key)       # [B, N, C+S, H, D]
        value_sources.append(self_kv_value)
    else:
        key_sources.append(self_kv_key_packed)   # packed 1D tensor
        value_sources.append(self_kv_value_packed)

    # In batch × block order, every sequence appends [full?, causal?, self]; cu_seqlens below assumes this order.
    for batch_idx in range(B):
        for block_idx in range(num_blocks):
            if has_full_prefix:
                plan.append((IDX_FULL, (batch_idx,), full_prefix_len))
            if causal_prefix_lens[block_idx] > 0:
                cp_len = causal_prefix_lens[block_idx]
                plan.append((IDX_CAUSAL, (batch_idx, slice(0, cp_len)), cp_len))
            if text_mask is None:
                plan.append((IDX_SELF, (batch_idx, block_idx), block_seq_len))
            else:
                # self KV is already packed under text_mask, locate it through the offset table
                seq_idx = batch_idx * num_blocks + block_idx
                start = packed_self_offsets[seq_idx]
                end = packed_self_offsets[seq_idx + 1]
                plan.append((IDX_SELF, (slice(start, end),), end - start))

    key_flat = gather_concat(plan, *key_sources)     # [total_kv_tokens, H, D]
    value_flat = gather_concat(plan, *value_sources)

    # total KV length per sequence, for cu_seqlens_kv
    prefix_per_block = causal_prefix_lens_gpu + full_prefix_len   # [N]
    if text_mask is None:
        kv_lens = (prefix_per_block + block_seq_len).repeat(B)    # [B*N]
    else:
        kv_lens = prefix_per_block.repeat(B) + block_valid_lens   # [B*N]
    cu_seqlens_kv = F.pad(kv_lens.cumsum(0), (1, 0)).to(torch.int32)

    # max_seqlen_kv: upper bound (the last block has the longest KV), sizes the flash_attn workspace
    max_kv_len = full_prefix_len + causal_total + block_seq_len
    if text_mask is None:
        max_valid_len_kv = max_kv_len
    else:
        max_valid_len_kv = full_prefix_len + causal_total + max(block_valid_lens_list)

    img_out = _attention_varlen(
        query_packed, key_flat, value_flat,
        cu_seqlens_q=cu_seqlens_query, cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=block_size, max_seqlen_k=max_valid_len_kv,
    )
    img_out = img_out.reshape(B, -1, num_heads, head_dim)

    return img_out, txt_out


# ══════════════════════════════════════════════════════════════
#  Public entry: block-wise causal MM attention
# ══════════════════════════════════════════════════════════════

def block_wise_causal_mm_attn_with_prefix(
    query,                    # [B, total_img_len, num_heads, head_dim], x_t
    key,                      # [B, total_img_len, num_heads, head_dim], x_t
    value,                    # [B, total_img_len, num_heads, head_dim], x_t
    encoder_query,            # [B, num_blocks * text_len, num_heads, head_dim], text
    encoder_key,              # [B, num_blocks * text_len, num_heads, head_dim], text
    encoder_value,            # [B, num_blocks * text_len, num_heads, head_dim], text
    full_prefix_key=None,     # [B, full_prefix_len, num_heads, head_dim]
    full_prefix_value=None,   # [B, full_prefix_len, num_heads, head_dim]
    causal_prefix_key=None,   # [B, first_block_size + (num_blocks - 2) * regular_block_size, num_heads, head_dim]
    causal_prefix_value=None, # [B, first_block_size + (num_blocks - 2) * regular_block_size, num_heads, head_dim]
    text_mask=None,           # [B, num_blocks * text_len]
    block_sizes=None,         # List[int], img tokens per block
    packed_kv_max_mem=_DEFAULT_PACKED_KV_MAX_MEM,  # packed KV backward peak memory budget (bytes), grouping above it
):
    """Block-wise causal MM attention (FBCA) where text does not see any prefix (public entry).

    Blocks, unfolding and grouping are those of ``block_wise_causal_attention_with_prefix``.

      img_Q_i → KV = [prefix, self]        (img sees everything)
      txt_Q_i → KV = [self_img, self_txt]  (txt sees its own block only)

    ``causal_prefix_key/value`` is what blocks 1..N-1 see of the blocks before them, of length
    first_block_size + (N-2)*regular_block_size, None for block-diagonal; under window-wise teacher forcing the
    caller slices it off the clean cond tensors, not off these noisy blocks. With F != C block 0 runs standalone
    and blocks 1..N-1 come back through this entry with block 0's K/V folded into full_prefix (recursion depth 1).

    Returns:
        out: [B, total_img_len + num_blocks * text_len, H, D], img output first, text output after.
    """
    B, total_img_len, num_heads, head_dim = query.shape

    first_block_size = block_sizes[0]
    num_blocks = len(block_sizes)
    regular_block_size = block_sizes[-1]

    assert first_block_size + regular_block_size * (num_blocks - 1) == total_img_len, \
        f"first_block_size({first_block_size}) + regular_block_size({regular_block_size}) * ({num_blocks - 1}) != total_img_len({total_img_len})"
    assert num_blocks == 1 or len(set(block_sizes[1:])) <= 1, \
        f"block 1~N-1 should all have the same size, got {set(block_sizes[1:])}"

    total_text_len = encoder_query.shape[1]
    text_len = total_text_len // num_blocks

    has_full_prefix = (full_prefix_key is not None and full_prefix_value is not None
                       and full_prefix_key.shape[1] > 0)
    full_prefix_len = full_prefix_key.shape[1] if has_full_prefix else 0

    has_causal_prefix = (causal_prefix_key is not None and causal_prefix_value is not None
                         and causal_prefix_key.shape[1] > 0)
    causal_prefix_len = causal_prefix_key.shape[1] if has_causal_prefix else 0

    if has_causal_prefix:
        assert num_blocks > 1, \
            f"num_blocks({num_blocks}) must be > 1 when causal_prefix is provided"
        assert first_block_size + regular_block_size * (num_blocks - 2) == causal_prefix_len, \
            f"first_block_size({first_block_size}) + regular_block_size({regular_block_size}) * ({num_blocks - 2}) != causal_prefix_len({causal_prefix_len})"

    packed_kv_max_tokens = _packed_kv_mem_to_max_tokens(
        packed_kv_max_mem, num_heads, head_dim, query.element_size(), batch_size=B)

    # Single block (N == 1): split attention + optional full_prefix
    if num_blocks == 1:
        img_out, txt_out = _self_mm_attn_split(
            query, key, value,
            encoder_query, encoder_key, encoder_value,
            full_prefix_key=full_prefix_key,
            full_prefix_value=full_prefix_value,
            text_mask=text_mask,
        )
        return torch.cat([img_out, txt_out], dim=1)

    # F == C, N > 1: all blocks are equal. Same grouping strategy as the img-only version, except that text QKV
    # and text_mask are sliced on text_len boundaries, the prefixes only affect img_Q, and the internal function
    # returns an (img_out, txt_out) pair.
    if first_block_size == regular_block_size:
        # Without a causal prefix the blocks are independent (block-diagonal), so the packed KV does not grow
        # quadratically and no grouping is needed.
        if not has_causal_prefix:
            img_out, txt_out = _block_wise_causal_mm_attn(
                query, key, value,
                encoder_query, encoder_key, encoder_value,
                causal_prefix_key=None,
                causal_prefix_value=None,
                full_prefix_key=full_prefix_key,
                full_prefix_value=full_prefix_value,
                text_mask=text_mask,
                block_size=regular_block_size,
            )
            return torch.cat([img_out, txt_out], dim=1)

        # img-only token count: text_len is small and left out of the budget
        packed_kv_total_tokens = _compute_total_tokens(
            num_blocks, regular_block_size, full_prefix_len)

        # grouped path: the packed KV exceeds the memory budget
        if packed_kv_max_tokens is not None and packed_kv_total_tokens > packed_kv_max_tokens:
            block_offsets = _get_group_block_offsets(
                num_blocks, regular_block_size, full_prefix_len, packed_kv_max_tokens)
            _log_group_attn_triggered(
                "mm/F==C", num_blocks, regular_block_size,
                len(block_offsets) - 1, packed_kv_total_tokens, packed_kv_max_tokens)
            group_img_outputs = []
            group_txt_outputs = []
            for group_idx in range(len(block_offsets) - 1):
                group_start = block_offsets[group_idx]
                group_end = block_offsets[group_idx + 1]
                group_size = group_end - group_start

                # img QKV, sliced on block_size boundaries
                query_group = query[:, group_start * regular_block_size:group_end * regular_block_size]
                key_group = key[:, group_start * regular_block_size:group_end * regular_block_size]
                value_group = value[:, group_start * regular_block_size:group_end * regular_block_size]

                # text QKV and text_mask, sliced on text_len boundaries
                enc_q_group = encoder_query[:, group_start * text_len:group_end * text_len]
                enc_k_group = encoder_key[:, group_start * text_len:group_end * text_len]
                enc_v_group = encoder_value[:, group_start * text_len:group_end * text_len]

                text_mask_group = text_mask[:, group_start * text_len:group_end * text_len] if text_mask is not None else None

                # in-group causal prefix (img only): the causal_prefix rows of the first group_size-1 blocks
                if group_size > 1:
                    group_causal_prefix_key = causal_prefix_key[:, group_start * regular_block_size:(group_end - 1) * regular_block_size]
                    group_causal_prefix_value = causal_prefix_value[:, group_start * regular_block_size:(group_end - 1) * regular_block_size]
                else:
                    group_causal_prefix_key = None
                    group_causal_prefix_value = None

                # full_prefix of the group (img only): the original full_prefix + every block before the group
                if group_start > 0:
                    pre_group_key = causal_prefix_key[:, :group_start * regular_block_size]
                    pre_group_value = causal_prefix_value[:, :group_start * regular_block_size]
                    if has_full_prefix:
                        group_full_prefix_key = torch.cat([full_prefix_key, pre_group_key], dim=1)
                        group_full_prefix_value = torch.cat([full_prefix_value, pre_group_value], dim=1)
                    else:
                        group_full_prefix_key = pre_group_key
                        group_full_prefix_value = pre_group_value
                else:
                    group_full_prefix_key = full_prefix_key
                    group_full_prefix_value = full_prefix_value

                img_out_g, txt_out_g = _block_wise_causal_mm_attn(
                    query_group, key_group, value_group,
                    enc_q_group, enc_k_group, enc_v_group,
                    causal_prefix_key=group_causal_prefix_key,
                    causal_prefix_value=group_causal_prefix_value,
                    full_prefix_key=group_full_prefix_key,
                    full_prefix_value=group_full_prefix_value,
                    text_mask=text_mask_group,
                    block_size=regular_block_size,
                )
                group_img_outputs.append(img_out_g)
                group_txt_outputs.append(txt_out_g)

            img_out = torch.cat(group_img_outputs, dim=1)
            txt_out = torch.cat(group_txt_outputs, dim=1)
            return torch.cat([img_out, txt_out], dim=1)

        # ungrouped path: all blocks at once, causal_prefix passed through
        img_out, txt_out = _block_wise_causal_mm_attn(
            query, key, value,
            encoder_query, encoder_key, encoder_value,
            causal_prefix_key=causal_prefix_key,
            causal_prefix_value=causal_prefix_value,
            full_prefix_key=full_prefix_key,
            full_prefix_value=full_prefix_value,
            text_mask=text_mask,
            block_size=regular_block_size,
        )
        return torch.cat([img_out, txt_out], dim=1)

    # F != C, N > 1: block 0 (F tokens) goes through _self_mm_attn_split on its own, then blocks 1..N-1 run
    # causal or block-diagonal. With a causal prefix its first F rows are merged into full_prefix, making blocks
    # 1..N-1 an F == C problem; without one the blocks are fully independent and nothing is merged.

    # split the img/text tensors into block 0 and the rest
    query_first, query_rest = query.split_with_sizes(
        [first_block_size, total_img_len - first_block_size], dim=1)
    key_first, key_rest = key.split_with_sizes(
        [first_block_size, total_img_len - first_block_size], dim=1)
    value_first, value_rest = value.split_with_sizes(
        [first_block_size, total_img_len - first_block_size], dim=1)
    enc_q_first, enc_q_rest = encoder_query.split_with_sizes(
        [text_len, total_text_len - text_len], dim=1)
    enc_k_first, enc_k_rest = encoder_key.split_with_sizes(
        [text_len, total_text_len - text_len], dim=1)
    enc_v_first, enc_v_rest = encoder_value.split_with_sizes(
        [text_len, total_text_len - text_len], dim=1)

    text_mask_first = None
    text_mask_rest = None
    if text_mask is not None:
        text_mask_first, text_mask_rest = text_mask.split_with_sizes(
            [text_len, total_text_len - text_len], dim=1)

    # block 0: img sees full_prefix + self, txt sees self only
    img_out_first, txt_out_first = _self_mm_attn_split(
        query_first, key_first, value_first,
        enc_q_first, enc_k_first, enc_v_first,
        full_prefix_key=full_prefix_key,
        full_prefix_value=full_prefix_value,
        text_mask=text_mask_first,
    )

    # blocks 1..N-1, independent without a causal prefix: each sees only [full_prefix?, self_i]
    if not has_causal_prefix:
        img_out_rest, txt_out_rest = _block_wise_causal_mm_attn(
            query_rest, key_rest, value_rest,
            enc_q_rest, enc_k_rest, enc_v_rest,
            causal_prefix_key=None,
            causal_prefix_value=None,
            full_prefix_key=full_prefix_key,
            full_prefix_value=full_prefix_value,
            text_mask=text_mask_rest,
            block_size=regular_block_size,
        )

        img_out = torch.cat([img_out_first, img_out_rest], dim=1)
        txt_out = torch.cat([txt_out_first, txt_out_rest], dim=1)
        return torch.cat([img_out, txt_out], dim=1)

    # With a causal prefix, rest_full_prefix = [original full_prefix, causal_prefix[:, :F]], which turns the rest
    # into an equal-block problem handed back to this entry.
    first_causal_prefix_key, rest_causal_prefix_key = causal_prefix_key.split_with_sizes(
        [first_block_size, causal_prefix_len - first_block_size], dim=1)
    first_causal_prefix_value, rest_causal_prefix_value = causal_prefix_value.split_with_sizes(
        [first_block_size, causal_prefix_len - first_block_size], dim=1)

    if has_full_prefix:
        rest_full_prefix_key = torch.cat([full_prefix_key, first_causal_prefix_key], dim=1)
        rest_full_prefix_value = torch.cat([full_prefix_value, first_causal_prefix_value], dim=1)
    else:
        rest_full_prefix_key = first_causal_prefix_key
        rest_full_prefix_value = first_causal_prefix_value

    # Blocks 1..N-1 all have `regular_block_size` tokens (asserted at the top), so this recursive call can only
    # land in the F == C branch or, when N == 2, in the N == 1 early return: recursion depth is 1. N == 2
    # therefore skips the packed-KV budget entirely, because the N == 1 branch never consults it. The budget
    # is forwarded explicitly so that a caller's smaller budget reaches the recursive frame too.
    out_rest = block_wise_causal_mm_attn_with_prefix(
        query_rest, key_rest, value_rest,
        enc_q_rest, enc_k_rest, enc_v_rest,
        full_prefix_key=rest_full_prefix_key,
        full_prefix_value=rest_full_prefix_value,
        causal_prefix_key=rest_causal_prefix_key,
        causal_prefix_value=rest_causal_prefix_value,
        text_mask=text_mask_rest,
        block_sizes=block_sizes[1:],
        packed_kv_max_mem=packed_kv_max_mem,
    )
    # this entry returns [img | txt] concatenated, so split it back to prepend block 0's two halves
    img_out_rest, txt_out_rest = out_rest.split_with_sizes(
        [total_img_len - first_block_size, total_text_len - text_len], dim=1)

    img_out = torch.cat([img_out_first, img_out_rest], dim=1)
    txt_out = torch.cat([txt_out_first, txt_out_rest], dim=1)
    return torch.cat([img_out, txt_out], dim=1)


# ══════════════════════════════════════════════════════════════
#  Training entry: window-wise teacher forcing, full-history blocks
# ══════════════════════════════════════════════════════════════

def full_history_parallel_attention(
    query,                    # [B, total_img_len, num_heads, head_dim]
    key,                      # [B, total_img_len, num_heads, head_dim]
    value,                    # [B, total_img_len, num_heads, head_dim]
    encoder_query,            # [B, num_blocks * text_len, num_heads, head_dim]
    encoder_key,              # [B, num_blocks * text_len, num_heads, head_dim]
    encoder_value,            # [B, num_blocks * text_len, num_heads, head_dim]
    cond_query=None,          # [B, cond_total_img_len, num_heads, head_dim]
    cond_key=None,            # [B, cond_total_img_len, num_heads, head_dim]
    cond_value=None,          # [B, cond_total_img_len, num_heads, head_dim]
    text_mask=None,           # [B, num_blocks * text_len]
    cond_chunk_sizes=None,    # List[int], img tokens per cond (x_0) chunk
    noisy_chunk_sizes=None,   # List[int], img tokens per noisy (x_t) block, i.e. per training window
):
    """Window-wise teacher forcing attention for parallel training (text does not see cond_img).

    Two parts are computed independently and concatenated, both with FBCA (``block_wise_causal_*``):
      cond (x_0):       img-only, one block per cond chunk: chunk_i sees cond chunks 0..i.
      noisy (x_t, txt): one block per training window: img_Q sees [independent cond, overlap clean tail, self],
                        txt_Q only its own [img, txt].

    cond_key/cond_value is laid out as [independent cond, overlap clean tail]: the noisy full_prefix is the
    independent cond, while the noisy causal_prefix is sliced off the tail of cond_key/cond_value with the length
    of noisy's first N-1 blocks (first_block_size + (N-2)*regular_block_size), not taken from noisy's own
    key/value. That substitutes clean tokens for noisy's history and keeps noisy blocks independent of each other.

    Returns cat([cond_out?, noisy_img_out, text_out], dim=1) -- the cond part only when cond is present.
    """
    has_cond = cond_query is not None and cond_key is not None and cond_value is not None

    # cond (x_0): chunk_i sees cond chunks 0..i-1 + self_i, so the causal_prefix is cond's own first N-1 chunks
    # of K/V.
    if has_cond:
        cond_num_chunks = len(cond_chunk_sizes)
        cond_regular_chunk_size = cond_chunk_sizes[-1]
        if cond_num_chunks > 1:
            cond_causal_prefix_key = cond_key[:, :-cond_regular_chunk_size]
            cond_causal_prefix_value = cond_value[:, :-cond_regular_chunk_size]
        else:
            cond_causal_prefix_key = None
            cond_causal_prefix_value = None

        cond_out = block_wise_causal_attention_with_prefix(
            cond_query, cond_key, cond_value,
            causal_prefix_key=cond_causal_prefix_key,
            causal_prefix_value=cond_causal_prefix_value,
            block_sizes=cond_chunk_sizes,
        )

    # noisy (x_t): full_prefix is the independent cond, causal_prefix is the tail of cond_key/cond_value of length
    # first_block_size + (noisy_num_blocks - 2) * regular_block_size, i.e. noisy's first N-1 blocks.
    noisy_first_block_size = noisy_chunk_sizes[0]
    noisy_num_blocks = len(noisy_chunk_sizes)
    noisy_regular_block_size = noisy_chunk_sizes[-1]
    if noisy_num_blocks > 1 and has_cond:
        noisy_causal_prefix_len = noisy_first_block_size + (noisy_num_blocks - 2) * noisy_regular_block_size
        cond_total_len = cond_key.shape[1]
        assert cond_total_len >= noisy_causal_prefix_len, (
            f"cond_total_len({cond_total_len}) < noisy_causal_prefix_len({noisy_causal_prefix_len}); "
            f"cond must be long enough to provide the causal_prefix of noisy"
        )
        noisy_full_prefix_key, noisy_causal_prefix_key = cond_key.split_with_sizes(
            [cond_total_len - noisy_causal_prefix_len, noisy_causal_prefix_len],
            dim=1
        )
        noisy_full_prefix_value, noisy_causal_prefix_value = cond_value.split_with_sizes(
            [cond_total_len - noisy_causal_prefix_len, noisy_causal_prefix_len],
            dim=1
        )
    else:
        noisy_full_prefix_key = cond_key
        noisy_full_prefix_value = cond_value
        noisy_causal_prefix_key = None
        noisy_causal_prefix_value = None

    out = block_wise_causal_mm_attn_with_prefix(
        query, key, value,
        encoder_query, encoder_key, encoder_value,
        full_prefix_key=noisy_full_prefix_key,
        full_prefix_value=noisy_full_prefix_value,
        causal_prefix_key=noisy_causal_prefix_key,
        causal_prefix_value=noisy_causal_prefix_value,
        text_mask=text_mask,
        block_sizes=noisy_chunk_sizes,
    )

    if has_cond:
        out = torch.cat([cond_out, out], dim=1)
    return out


# ══════════════════════════════════════════════════════════════
#  Training entry: HieraKV sink-recent blocks
#
#  Every noisy chunk i sees [sink, recent_i, self]; every recent chunk sees [sink, self]; sink is self-attention.
#  RoPE is applied by the caller (parallel_attention) in slot form (sink@slot0, recent@slot1, noisy@slot2), so
#  this function is RoPE-agnostic and only does the packing + attention.
# ══════════════════════════════════════════════════════════════

def sink_recent_parallel_attention(
    query,                    # [B, total_noisy_img_len, H, D]   (noisy = num_window windows concatenated)
    key,                      # [B, total_noisy_img_len, H, D]
    value,                    # [B, total_noisy_img_len, H, D]
    encoder_query,            # [B, num_window * text_len, H, D]
    encoder_key,              # [B, num_window * text_len, H, D]
    encoder_value,            # [B, num_window * text_len, H, D]
    cond_query=None,          # [B, cond_total_img_len, H, D]  (kept cond = [sink?, recent×K])
    cond_key=None,            # [B, cond_total_img_len, H, D]
    cond_value=None,          # [B, cond_total_img_len, H, D]
    text_mask=None,           # [B, num_window * text_len]
    cond_chunk_sizes=None,    # List[int], tokens per kept-cond physical chunk: [sink?, recent×K]
    noisy_chunk_sizes=None,   # List[int], tokens per window (variable length)
    sink_chunks=1,            # int, fixed 1
    window_has_sink=None,     # Optional[List[bool]], per window: does this window see the sink
    window_has_recent=None,   # Optional[List[bool]], per window: does this window see its recent
):
    """Parallel training attention of the HieraKV sink-recent blocks, strictly per window.

    The attention unit is a window (num_window = len(noisy_chunk_sizes)); window w takes its view from
    window_has_sink[w] / window_has_recent[w], matching the inference rollout step by step:

      cond.sink    → KV = [sink];  cond.recent_i → KV = [sink, self]  (block-diagonal over recents)
      window_i.img → KV = [sink?, recent_w?, self_img_i, self_txt_valid_i]
      window_i.txt → KV = [self_img_i, self_txt_valid_i]

    cond is [sink?, recents of the windows that have one, in window order], described by
    cond_chunk_sizes = [sink_tokens?, recent_tokens × K] with K = sum(window_has_recent); window w's recent sits
    at index #{w' < w : window_has_recent[w']}. The sink is kept only when any(window_has_sink). Windows are
    variable length, so img_Q / txt_Q both go through flash-attn varlen packed by gather_concat.

    Returns cat([cond_out?, noisy_img_out, txt_out], dim=1) (cond_out only exists when there is a sink).
    """
    assert noisy_chunk_sizes is not None
    assert encoder_query is not None, "sink_recent_parallel_attention requires text (encoder_*)"

    B, total_noisy_img_len, num_heads, head_dim = query.shape
    device = query.device

    num_window = len(noisy_chunk_sizes)
    text_len = encoder_query.shape[1] // num_window

    # per-window cond visibility; the default (all True) makes every window see sink + recent
    if window_has_sink is None:
        window_has_sink = [True] * num_window
    if window_has_recent is None:
        window_has_recent = [True] * num_window
    assert len(window_has_sink) == num_window and len(window_has_recent) == num_window
    # A recent is newer than the sink, so a window with a recent always has the sink and the cond sequence always
    # contains the sink.
    assert all((not r) or s for s, r in zip(window_has_sink, window_has_recent)), (
        f"window_has_recent ⟹ window_has_sink violated: "
        f"sink={window_has_sink}, recent={window_has_recent}"
    )

    # index of each window's recent inside the recents sequence (-1 when it has none)
    recent_idx_of_window = [-1] * num_window
    num_recents = 0
    for w in range(num_window):
        if window_has_recent[w]:
            recent_idx_of_window[w] = num_recents
            num_recents += 1

    has_cond = (
        cond_query is not None and cond_key is not None and cond_value is not None
        and cond_chunk_sizes is not None and len(cond_chunk_sizes) > 0
    )
    # A non-empty kept cond always starts with the sink (it is kept whenever any(window_has_sink), and
    # recent ⟹ sink), so "the cond sequence has a sink" is equivalent to has_cond.
    has_sink = has_cond
    has_recent = has_cond and num_recents > 0                        # has_recent ⟹ has_sink (invariant above)
    sink_tokens = 0
    recent_tokens_list = []
    if has_cond:
        assert len(cond_chunk_sizes) == sink_chunks + num_recents, (
            f"cond chunk count({len(cond_chunk_sizes)}) should == sink({sink_chunks}) + "
            f"recents({num_recents}); cond_chunk_sizes={cond_chunk_sizes}"
        )
        sink_tokens = sum(cond_chunk_sizes[:sink_chunks])
        recent_tokens_list = cond_chunk_sizes[sink_chunks:]          # K entries, K = num_recents

    if has_sink:
        sink_query = cond_query[:, :sink_tokens]
        sink_key = cond_key[:, :sink_tokens]
        sink_value = cond_value[:, :sink_tokens]
    if has_recent:
        recent_query_all = cond_query[:, sink_tokens:]                # [B, sum(recent_tokens), H, D]
        recent_key_all = cond_key[:, sink_tokens:]
        recent_value_all = cond_value[:, sink_tokens:]

    # cond self-attention
    cond_out = None
    if has_sink:
        sink_out = _attention(sink_query, sink_key, sink_value)
        if has_recent:
            # each recent physical chunk sees [sink, self], block-diagonal over recents
            recent_out = block_wise_causal_attention_with_prefix(
                recent_query_all, recent_key_all, recent_value_all,
                full_prefix_key=sink_key,
                full_prefix_value=sink_value,
                causal_prefix_key=None,
                causal_prefix_value=None,
                block_sizes=recent_tokens_list,
            )
            cond_out = torch.cat([sink_out, recent_out], dim=1)
        else:
            cond_out = sink_out

    # token offset of each window / recent in its sequence
    noisy_offsets = [0]
    for s in noisy_chunk_sizes:
        noisy_offsets.append(noisy_offsets[-1] + s)
    if has_recent:
        recent_offsets = [0]
        for s in recent_tokens_list:
            recent_offsets.append(recent_offsets[-1] + s)

    # valid text length of each (batch, window)
    if text_mask is not None:
        txt_valid_pc = text_mask.view(B, num_window, text_len).sum(dim=-1, dtype=torch.int32)
        txt_valid_list = txt_valid_pc.reshape(B * num_window).tolist()
    else:
        txt_valid_list = [text_len] * (B * num_window)

    # noisy img_Q: KV = [sink?, recent_i?, self_img_i, self_txt_valid_i].
    # Sources are built dynamically: sink only if present, recent only if present, img / txt always.
    key_sources = []
    value_sources = []
    if has_sink:
        IDX_SINK = len(key_sources)
        key_sources.append(sink_key)
        value_sources.append(sink_value)
    if has_recent:
        IDX_RECENT = len(key_sources)
        key_sources.append(recent_key_all)
        value_sources.append(recent_value_all)
    IDX_IMG = len(key_sources)
    key_sources.append(key)
    value_sources.append(value)
    IDX_TXT = len(key_sources)
    key_sources.append(encoder_key)
    value_sources.append(encoder_value)

    img_plan = []
    q_lens = []
    kv_lens = []
    for batch_idx in range(B):
        for w in range(num_window):
            win = noisy_chunk_sizes[w]
            tv = txt_valid_list[batch_idx * num_window + w]
            n_off = noisy_offsets[w]
            t_off = w * text_len
            kv_len = win + tv
            w_has_sink = has_sink and window_has_sink[w]
            w_has_recent = has_recent and window_has_recent[w]
            if w_has_sink:
                img_plan.append((IDX_SINK, (batch_idx,), sink_tokens))
                kv_len += sink_tokens
            if w_has_recent:
                ridx = recent_idx_of_window[w]                      # index into recent_tokens_list
                rec_tok = recent_tokens_list[ridx]
                r_off = recent_offsets[ridx]
                img_plan.append((IDX_RECENT, (batch_idx, slice(r_off, r_off + rec_tok)), rec_tok))
                kv_len += rec_tok
            img_plan.append((IDX_IMG, (batch_idx, slice(n_off, n_off + win)), win))
            img_plan.append((IDX_TXT, (batch_idx, slice(t_off, t_off + tv)), tv))
            q_lens.append(win)
            kv_lens.append(kv_len)

    key_img_flat = gather_concat(img_plan, *key_sources)
    value_img_flat = gather_concat(img_plan, *value_sources)
    # The windows of query[b] are contiguous and in order, so the reshape already matches the (b, window) packed
    # order of img_plan.
    query_img_flat = query.reshape(B * total_noisy_img_len, num_heads, head_dim)

    cu_seqlens_q = F.pad(
        torch.tensor(q_lens, device=device, dtype=torch.int32).cumsum(0), (1, 0)
    ).to(torch.int32)
    cu_seqlens_kv = F.pad(
        torch.tensor(kv_lens, device=device, dtype=torch.int32).cumsum(0), (1, 0)
    ).to(torch.int32)

    img_out = _attention_varlen(
        query_img_flat, key_img_flat, value_img_flat,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
    )
    img_out = img_out.reshape(B, total_noisy_img_len, num_heads, head_dim)

    # noisy txt_Q: KV = [self_img_i, self_txt_valid_i]
    IMG_SRC, TXT_SRC = 0, 1
    txt_kv_key_sources = (key, encoder_key)
    txt_kv_value_sources = (value, encoder_value)

    txt_q_plan = []
    txt_kv_plan = []
    txt_q_lens = []
    txt_kv_lens = []
    for batch_idx in range(B):
        for w in range(num_window):
            win = noisy_chunk_sizes[w]
            tv = txt_valid_list[batch_idx * num_window + w]
            n_off = noisy_offsets[w]
            t_off = w * text_len
            txt_q_plan.append((0, (batch_idx, slice(t_off, t_off + tv)), tv))
            txt_kv_plan.append((IMG_SRC, (batch_idx, slice(n_off, n_off + win)), win))
            txt_kv_plan.append((TXT_SRC, (batch_idx, slice(t_off, t_off + tv)), tv))
            txt_q_lens.append(tv)
            txt_kv_lens.append(win + tv)

    txt_q_flat = gather_concat(txt_q_plan, encoder_query)
    txt_kv_key_flat = gather_concat(txt_kv_plan, *txt_kv_key_sources)
    txt_kv_value_flat = gather_concat(txt_kv_plan, *txt_kv_value_sources)

    cu_seqlens_txt_q = F.pad(
        torch.tensor(txt_q_lens, device=device, dtype=torch.int32).cumsum(0), (1, 0)
    ).to(torch.int32)
    cu_seqlens_txt_kv = F.pad(
        torch.tensor(txt_kv_lens, device=device, dtype=torch.int32).cumsum(0), (1, 0)
    ).to(torch.int32)

    txt_out_packed = _attention_varlen(
        txt_q_flat, txt_kv_key_flat, txt_kv_value_flat,
        cu_seqlens_q=cu_seqlens_txt_q, cu_seqlens_k=cu_seqlens_txt_kv,
        max_seqlen_q=max(txt_q_lens), max_seqlen_k=max(txt_kv_lens),
    )
    # [sum(tv), H, D] → [B*num_window, text_len, H, D] → [B, num_window*text_len, H, D]
    txt_out = prefix_unpack(txt_out_packed, txt_valid_list, B * num_window, text_len)
    txt_out = txt_out.reshape(B, num_window * text_len, num_heads, head_dim)

    # cond_out only exists when there is a sink; otherwise the output is [noisy_img, txt]
    if cond_out is not None:
        return torch.cat([cond_out, img_out, txt_out], dim=1)
    return torch.cat([img_out, txt_out], dim=1)


# ══════════════════════════════════════════════════════════════
#  Inference entry: single chunk, history from the KV cache
# ══════════════════════════════════════════════════════════════

def chunk_inference_attention(
    query,            # [B, chunk_size, H, D]          — noisy image (x_t) of the current chunk
    key,              # [B, chunk_size, H, D]          — or [B, cond_len+chunk_size, H, D] (cond pre-concatenated)
    value,            # [B, chunk_size, H, D]          — same as key
    encoder_query=None,    # [B, text_len, H, D]       — text of the current chunk (optional)
    encoder_key=None,      # [B, text_len, H, D]
    encoder_value=None,    # [B, text_len, H, D]
    text_mask=None,   # [B, text_len]                   — prefix style: k leading True, then False
):
    """Inference attention over a single chunk, where text does not see cond.

      img_Q → KV = [cond?, img, txt]    (img sees cond, keeping the visuals coherent)
      txt_Q → KV = [img, txt]           (txt does not see cond, keeping the prompt independent)

    Without text this is a single attention with Q = img, KV = [cond?, img]. cond is pre-concatenated in front of
    key/value by the caller, so cond_len = key.shape[1] - chunk_size.

    Returns:
      with text: [B, chunk_size + text_len, H, D]  (img first, text after)
      no text:   [B, chunk_size, H, D]
    """
    B, chunk_size, num_heads, head_dim = query.shape
    device = query.device

    has_text = encoder_query is not None and encoder_key is not None and encoder_value is not None

    # No text: Q = img, KV = [cond?, img]
    if not has_text:
        return _attention(query, key, value)

    # With text: whether to split depends on whether there is a cond
    text_len = encoder_query.shape[1]
    seqlen = chunk_size + text_len

    # cond length and the img-only key/value of the current chunk
    cond_len = key.shape[1] - chunk_size
    img_only_key = key[:, cond_len:]                              # [B, chunk_size, H, D]
    img_only_value = value[:, cond_len:]

    # no cond: img/txt see the same KV → one merged attention
    if cond_len == 0:
        q_cat = torch.cat([query, encoder_query], dim=1)          # [B, C+S, H, D]
        k_cat = torch.cat([img_only_key, encoder_key], dim=1)
        v_cat = torch.cat([img_only_value, encoder_value], dim=1)

        if text_mask is None:
            return _attention(q_cat, k_cat, v_cat)

        txt_valid = text_mask.sum(dim=-1, dtype=torch.int32)
        txt_valid_list = txt_valid.tolist()         # the only GPU->CPU sync on this branch (branches are exclusive)
        valid_lens = chunk_size + txt_valid
        cu = F.pad(valid_lens.cumsum(0, dtype=torch.int32), (1, 0))
        valid_lens_list = [chunk_size + vl for vl in txt_valid_list]
        max_valid_len = max(valid_lens_list)

        out = _attention_varlen(
            prefix_pack(q_cat, valid_lens_list),
            prefix_pack(k_cat, valid_lens_list),
            prefix_pack(v_cat, valid_lens_list),
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_valid_len, max_seqlen_k=max_valid_len,
        )
        return prefix_unpack(out, valid_lens_list, B, seqlen)

    # With cond, img_Q and txt_Q have different KV and take two attention calls: k_img is built by a single cat
    # and k_txt is a contiguous copy of its suffix.
    k_img = torch.cat([key, encoder_key], dim=1)                  # [B, (cond+C)+S, H, D]
    v_img = torch.cat([value, encoder_value], dim=1)
    k_txt = k_img[:, cond_len:].contiguous()                      # [B, C+S, H, D]
    v_txt = v_img[:, cond_len:].contiguous()

    if text_mask is None:
        img_out = _attention(query, k_img, v_img)
        txt_out = _attention(encoder_query, k_txt, v_txt)
    else:
        # with mask: 2× prefix_pack + varlen
        txt_valid = text_mask.sum(dim=-1, dtype=torch.int32)      # [B]
        txt_valid_list = txt_valid.tolist()         # the only GPU->CPU sync on this branch (branches are exclusive)

        kv_valid_img = cond_len + chunk_size + txt_valid           # [B]  (GPU tensor for cumsum)
        cu_q_img = torch.arange(
            0, (B + 1) * chunk_size, chunk_size,
            device=device, dtype=torch.int32)
        cu_kv_img = F.pad(kv_valid_img.cumsum(0, dtype=torch.int32), (1, 0))
        kv_valid_img_list = [cond_len + chunk_size + vl for vl in txt_valid_list]
        max_valid_len_img_kv = max(kv_valid_img_list)

        img_out = _attention_varlen(
            query.reshape(-1, num_heads, head_dim),
            prefix_pack(k_img, kv_valid_img_list),
            prefix_pack(v_img, kv_valid_img_list),
            cu_seqlens_q=cu_q_img, cu_seqlens_k=cu_kv_img,
            max_seqlen_q=chunk_size, max_seqlen_k=max_valid_len_img_kv,
        )
        img_out = img_out.reshape(B, chunk_size, num_heads, head_dim)

        kv_valid_txt = chunk_size + txt_valid                     # [B]  (GPU tensor for cumsum)
        cu_q_txt = F.pad(txt_valid.cumsum(0, dtype=torch.int32), (1, 0))
        cu_kv_txt = F.pad(kv_valid_txt.cumsum(0, dtype=torch.int32), (1, 0))
        # derived from txt_valid_list, no extra sync
        kv_valid_txt_list = [chunk_size + vl for vl in txt_valid_list]
        max_valid_len_txt_q = max(txt_valid_list)
        max_valid_len_txt_kv = max(kv_valid_txt_list)

        txt_out = _attention_varlen(
            prefix_pack(encoder_query, txt_valid_list),
            prefix_pack(k_txt, kv_valid_txt_list),
            prefix_pack(v_txt, kv_valid_txt_list),
            cu_seqlens_q=cu_q_txt, cu_seqlens_k=cu_kv_txt,
            max_seqlen_q=max_valid_len_txt_q, max_seqlen_k=max_valid_len_txt_kv,
        )
        txt_out = prefix_unpack(txt_out, txt_valid_list, B, text_len)

    return torch.cat([img_out, txt_out], dim=1)
