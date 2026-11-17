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

"""FlashAttention wrappers (FA2 and, when installed, FA3) for padded batches.

With batch size 1 the valid tokens are assumed to be a prefix of the sequence: attention runs on that prefix only and
the padded rows are filled with the query values. Larger batches are unpadded / repadded (padded rows are zero).
"""

import torch
from einops import rearrange

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_func
    from flash_attn.bert_padding import pad_input, unpad_input
except Exception:
    # Resolved at the call site like the v3 wrapper below, so that importing this module (and everything that
    # reaches `fa3_output`, which needs no kernel at all) does not require FlashAttention-2 to be installed.
    flash_attn_varlen_qkvpacked_func = None
    flash_attn_func = None
    pad_input = None
    unpad_input = None


def _require_fa2():
    if flash_attn_func is None:
        raise ImportError(
            "this code path needs FlashAttention-2 (`pip install flash-attn --no-build-isolation`); "
            "it is separate from the FlashAttention-3 kernels used by the streaming attention"
        )


try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except Exception:
    # The V3 wrapper below raises ImportError when called, so the absence is reported at the call site
    # instead of on every process at import time.
    flash_attn_varlen_func_v3 = None


def fa3_output(result):
    """The attention output of a FlashAttention-3 call.

    FlashAttention-3 builds differ in what `flash_attn_func` / `flash_attn_varlen_func` return: `(out, softmax_lse)`
    in the 3.0.0b1 builds this code was validated with, `out` alone in builds that return the LSE only on request.
    """
    return result[0] if isinstance(result, tuple) else result


def flash_attn(
    q, k, v, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False
):
    """FlashAttention-2 over dense, unpadded q / k / v [B, S, H, D]; returns [B, S, H, D]."""
    _require_fa2()
    return flash_attn_func(
        q=q,
        k=k,
        v=v,
        causal=causal,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        deterministic=deterministic,
    )

def flash_attn_no_pad(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False
):
    """FlashAttention-2 self-attention over packed qkv [B, S, 3, H, D] with a [B, S] keep-mask (1 = valid).

    With B == 1 the valid tokens must be a prefix: attention runs on that prefix only and the padded positions
    return their query unchanged. Larger batches are unpadded and repadded, the padded positions being zero.
    Returns [B, S, H, D].
    """
    # adapted from https://github.com/Dao-AILab/flash-attention/blob/13403e81157ba37ca525890f2f0f2137edf75311/flash_attn/flash_attention.py#L27
    _require_fa2()
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    if batch_size == 1:
        valid_len = int(key_padding_mask[0].sum().item())
        cu_seqlens = torch.tensor([0, valid_len], dtype=torch.int32, device=qkv.device)
        max_seqlen = valid_len

        q_remained = qkv[:, valid_len:, 0]
        output = flash_attn_varlen_qkvpacked_func(
            qkv.squeeze(0)[:valid_len],
            cu_seqlens,
            max_seqlen,
            dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )

        return torch.cat([output.unsqueeze(0), q_remained], dim=1)

    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(
        x, key_padding_mask
    )

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output



def flash_attn_no_pad_v3(
    qkv, key_padding_mask, causal=False, softmax_scale=None, deterministic=False
):
    if flash_attn_varlen_func_v3 is None:
        raise ImportError("FlashAttention V3 backend not available")

    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query, key, value = qkv.unbind(dim=2)

    # batch_size == 1: skip unpad / repad to save memory and compute
    if batch_size == 1:
        valid_len = int(key_padding_mask[0].sum().item())
        cu_seqlens_q = torch.tensor([0, valid_len], dtype=torch.int32, device=query.device)
        cu_seqlens_k = torch.tensor([0, valid_len], dtype=torch.int32, device=key.device)
        max_seqlen_q = valid_len
        max_seqlen_k = valid_len

        query_remained = query[:, valid_len:]

        output = fa3_output(flash_attn_varlen_func_v3(
            query.squeeze(0)[:valid_len],
            key.squeeze(0)[:valid_len],
            value.squeeze(0)[:valid_len],
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        ))

        return torch.cat([output.unsqueeze(0), query_remained], dim=1)

    query_unpad, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
        rearrange(query, "b s h d -> b s (h d)"), key_padding_mask
    )
    key_unpad, _, cu_seqlens_k, _, _ = unpad_input(  # max_seqlen_q is reused for k
        rearrange(key, "b s h d -> b s (h d)"), key_padding_mask
    )
    value_unpad, _, _, _, _ = unpad_input(
        rearrange(value, "b s h d -> b s (h d)"), key_padding_mask
    )

    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)

    output_unpad = fa3_output(flash_attn_varlen_func_v3(
        query_unpad, key_unpad, value_unpad,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic
    ))

    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
        "b s (h d) -> b s h d", h=nheads
    )
    return output
