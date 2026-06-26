import os
import torch
import torch.nn.functional as F

sparge_attn_func = None
flash_attn_func = None
flash_attn_varlen_func = None
sageattn = None
sageattn_varlen = None
xformers_attn_func = None

ATTN_BACKEND = os.environ.get("GIGAWORLD_ATTN_BACKEND", "auto").lower()


# ============================================================
# Local FlashAttention2
# ============================================================

def _try_load_flash_attn():
    global flash_attn_func, flash_attn_varlen_func

    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        print("✅ Local Flash Attn 2 is installed!")
    except Exception as e:
        print(f"⚠️ Local Flash Attn 2 unavailable: {repr(e)}")
        flash_attn_func = None
        flash_attn_varlen_func = None


_try_load_flash_attn()


# ============================================================
# SageAttention
# ============================================================

try:
    from sageattention import sageattn, sageattn_varlen
    print("✅ Sage Attn is installed!")
except Exception as e:
    print(f"⚠️ Sage Attn is not installed: {repr(e)}")
    sageattn = None
    sageattn_varlen = None

# ============================================================
# SpargeAttention
# ============================================================

try:
    from spas_sage_attn import spas_sage2_attn_meansim_cuda
    sparge_attn_func = spas_sage2_attn_meansim_cuda
    print("✅ SpargeAttn is installed!")
except Exception as e:
    print(f"⚠️ SpargeAttn is not installed: {repr(e)}")
    sparge_attn_func = None

# ============================================================
# xFormers
# ============================================================

try:
    from xformers.ops import memory_efficient_attention as xformers_attn_func
    print("✅ Xformers is installed!")
except Exception as e:
    print(f"⚠️ Xformers is not installed: {repr(e)}")
    xformers_attn_func = None


print(f"🚀 GIGAWORLD_ATTN_BACKEND = {ATTN_BACKEND}")


# ============================================================
# Mask creation
# ============================================================

def create_navit_attention_masks(
    batch_size: int,
    original_context_length_list: list,
    history_context_length: int,
    encoder_hidden_states_seq_len: int,
    device: torch.device,
    restrict_self_attn: bool = False,
    guidance_cross_attn: bool = False,
):
    if restrict_self_attn:
        cu_seqlens_q = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cu_seqlens_q.append(cu_seqlens_q[-1] + length)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, device=device, dtype=torch.int32)
        max_seqlen_q = max(original_context_length_list)

        cu_seqlens_kv = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cu_seqlens_kv.append(
                    cu_seqlens_kv[-1] + length + history_context_length
                )
        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
        max_seqlen_kv = max(original_context_length_list) + history_context_length

    else:
        cu_seqlens_kv = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cu_seqlens_kv.append(
                    cu_seqlens_kv[-1] + length + history_context_length
                )
        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
        max_seqlen_kv = max(original_context_length_list) + history_context_length

        cu_seqlens_q = cu_seqlens_kv
        max_seqlen_q = max_seqlen_kv

    navit_hidden_attention_mask = (
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
    )

    navit_history_hidden_attention_mask = None

    if restrict_self_attn:
        cu_seqlens_kv = [0]
        for _ in range(batch_size):
            for _length in original_context_length_list:
                cu_seqlens_kv.append(cu_seqlens_kv[-1] + history_context_length)

        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
        max_seqlen_kv = history_context_length

        cu_seqlens_q = cu_seqlens_kv
        max_seqlen_q = max_seqlen_kv

        navit_history_hidden_attention_mask = (
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )

    if guidance_cross_attn:
        cross_cu_seqlens_q = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cross_cu_seqlens_q.append(cross_cu_seqlens_q[-1] + length)
        cross_cu_seqlens_q = torch.tensor(
            cross_cu_seqlens_q,
            device=device,
            dtype=torch.int32,
        )
        cross_max_seqlen_q = max(original_context_length_list)

    else:
        cross_cu_seqlens_q = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cross_cu_seqlens_q.append(
                    cross_cu_seqlens_q[-1] + length + history_context_length
                )
        cross_cu_seqlens_q = torch.tensor(
            cross_cu_seqlens_q,
            device=device,
            dtype=torch.int32,
        )
        cross_max_seqlen_q = max(original_context_length_list) + history_context_length

    cu_seqlens_kv = [0]
    for _ in range(batch_size):
        for _length in original_context_length_list:
            cu_seqlens_kv.append(cu_seqlens_kv[-1] + encoder_hidden_states_seq_len)

    cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
    max_seqlen_kv = encoder_hidden_states_seq_len

    navit_encoder_attention_mask = (
        cross_cu_seqlens_q,
        cu_seqlens_kv,
        cross_max_seqlen_q,
        max_seqlen_kv,
    )

    return (
        navit_hidden_attention_mask,
        navit_encoder_attention_mask,
        navit_history_hidden_attention_mask,
    )


# ============================================================
# Wrappers
# ============================================================

@torch.compiler.disable
def _flash_attn_wrapper(q, k, v):
    return flash_attn_func(q, k, v, causal=False)


@torch.compiler.disable
def _flash_attn_varlen_wrapper(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
):
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        causal=False,
    )


@torch.compiler.disable
def _sage_attn_wrapper(q, k, v):
    return sageattn(
        q,
        k,
        v,
        tensor_layout="NHD",
        is_causal=False,
    )


@torch.compiler.disable
def _sage_attn_varlen_wrapper(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
):
    return sageattn_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
    )

@torch.compiler.disable
def _sparge_attn_wrapper(q, k, v):
    return sparge_attn_func(
        q,
        k,
        v,
        simthreshd1=float(os.environ.get("SPARGE_SIMTHRESHD1", "0.6")),
        cdfthreshd=float(os.environ.get("SPARGE_CDFTHRESHD", "0.98")),
        is_causal=False,
    )

def _torch_sdpa_nhd(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)


def _torch_sdpa_varlen_fallback(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
):
    outputs = []
    num_segments = cu_seqlens_q.numel() - 1

    for i in range(num_segments):
        qs = int(cu_seqlens_q[i].item())
        qe = int(cu_seqlens_q[i + 1].item())
        ks = int(cu_seqlens_kv[i].item())
        ke = int(cu_seqlens_kv[i + 1].item())

        qi = q[qs:qe].unsqueeze(0)
        ki = k[ks:ke].unsqueeze(0)
        vi = v[ks:ke].unsqueeze(0)

        oi = _torch_sdpa_nhd(qi, ki, vi).squeeze(0)
        outputs.append(oi)

    return torch.cat(outputs, dim=0)


# ============================================================
# Backend selection
# ============================================================

def _dense_attention(q, k, v):
    if ATTN_BACKEND == "fa2":
        if flash_attn_func is None:
            raise RuntimeError("GIGAWORLD_ATTN_BACKEND=fa2 but flash-attn is not available")
        return _flash_attn_wrapper(q, k, v)
    
    if ATTN_BACKEND in ["sparge", "sparge_ckpt"]:
        if sparge_attn_func is None:
            raise RuntimeError("GIGAWORLD_ATTN_BACKEND=sparge but SpargeAttn is not available")
        return _sparge_attn_wrapper(q, k, v)

    if ATTN_BACKEND == "sage":
        if sageattn is None:
            raise RuntimeError("GIGAWORLD_ATTN_BACKEND=sage but sageattention is not available")
        return _sage_attn_wrapper(q, k, v)

    if ATTN_BACKEND == "xformers":
        if xformers_attn_func is None:
            raise RuntimeError("GIGAWORLD_ATTN_BACKEND=xformers but xformers is not available")
        return xformers_attn_func(q, k, v)

    if ATTN_BACKEND == "sdpa":
        return _torch_sdpa_nhd(q, k, v)

    # auto: FA2 > Sage > xFormers > SDPA
    # if flash_attn_func is not None:
    #     return _flash_attn_wrapper(q, k, v)

    # if sageattn is not None:
    #     return _sage_attn_wrapper(q, k, v)

    if xformers_attn_func is not None:
        return xformers_attn_func(q, k, v)

    return _torch_sdpa_nhd(q, k, v)


def _varlen_attention(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
):
    if ATTN_BACKEND == "fa2":
        if flash_attn_varlen_func is None:
            raise RuntimeError("GIGAWORLD_ATTN_BACKEND=fa2 but flash-attn varlen is not available")
        return _flash_attn_varlen_wrapper(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )

    if ATTN_BACKEND == "sage":
        if sageattn_varlen is None:
            raise RuntimeError("GIGAWORLD_ATTN_BACKEND=sage but sageattn_varlen is not available")
        return _sage_attn_varlen_wrapper(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )

    if ATTN_BACKEND == "xformers":
        return _torch_sdpa_varlen_fallback(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
        )

    if ATTN_BACKEND == "sdpa":
        return _torch_sdpa_varlen_fallback(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
        )

    # auto: FA2 > Sage > SDPA fallback
    if flash_attn_varlen_func is not None:
        return _flash_attn_varlen_wrapper(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )

    if sageattn_varlen is not None:
        return _sage_attn_varlen_wrapper(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )

    return _torch_sdpa_varlen_fallback(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
    )


# ============================================================
# Main attention dispatch
# ============================================================

def attn_varlen_func(q, k, v, attention_mask=None):
    """
    q/k/v layout: [B, L, H, D]

    GIGAWORLD_ATTN_BACKEND:
        auto      : FA2 > Sage > xFormers > SDPA
        fa2       : force FlashAttention2
        sage      : force SageAttention
        xformers  : force xFormers
        sdpa      : force PyTorch SDPA
    """

    if attention_mask is None:
        return _dense_attention(q, k, v)

    B, L, H, D = q.shape

    q_flat = q.flatten(0, 1)
    k_flat = k.flatten(0, 1)
    v_flat = v.flatten(0, 1)

    cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = attention_mask

    x = _varlen_attention(
        q_flat,
        k_flat,
        v_flat,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
    )

    return x.unflatten(0, (B, L))