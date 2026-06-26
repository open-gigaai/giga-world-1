import torch
import triton
import triton.language as tl

from .utils import calculate_settings, torch_gpu_device


# ------------------------------- replace funtion -------------------------------


def apply_rotary_emb_transposed_flash(x, freqs_cis):
    return Flash_RoPE_Transposed.apply(x, freqs_cis)


def replace_rope_with_flash_rope():
    from ...diffusers_version import transformer_gigaworld_diffusers
    from .. import transformer_gigaworld

    transformer_gigaworld_diffusers.apply_rotary_emb_transposed = apply_rotary_emb_transposed_flash
    transformer_gigaworld.apply_rotary_emb_transposed = apply_rotary_emb_transposed_flash
    print("Patched Flash_RoPE globally\n")


# ------------------------------- layer norm -------------------------------


@triton.jit
def _apply_rope_transposed_kernel(
    X,
    Out,
    cos,
    sin,
    n_heads: tl.constexpr,
    stride_x: tl.constexpr,
    stride_out: tl.constexpr,
    stride_freq: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    freq_row_idx = row_idx // n_heads

    half_head_dim = head_dim // 2
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < half_head_dim

    x_ptr = X + row_idx * stride_x
    out_ptr = Out + row_idx * stride_out
    cos_ptr = cos + freq_row_idx * stride_freq
    sin_ptr = sin + freq_row_idx * stride_freq

    x_real = tl.load(x_ptr + col_offsets * 2, mask=mask, other=0.0)
    x_imag = tl.load(x_ptr + col_offsets * 2 + 1, mask=mask, other=0.0)
    cos_even = tl.load(cos_ptr + col_offsets * 2, mask=mask, other=0.0)
    sin_odd = tl.load(sin_ptr + col_offsets * 2 + 1, mask=mask, other=0.0)

    out_even = x_real * cos_even - x_imag * sin_odd
    out_odd = x_real * sin_odd + x_imag * cos_even

    tl.store(out_ptr + col_offsets * 2, out_even, mask=mask)
    tl.store(out_ptr + col_offsets * 2 + 1, out_odd, mask=mask)


class Flash_RoPE_Transposed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freqs_cis):
        # x: [B, seq_len, n_heads, head_dim]
        # freqs_cis: [B, seq_len, head_dim*2]

        B, seq_len, n_heads, head_dim = x.shape

        x_flat = x.reshape(-1, head_dim).contiguous()
        device = x_flat.device
        out = torch.empty_like(x_flat)

        freqs_flat = freqs_cis.reshape(B * seq_len, -1).contiguous()
        half_dim = freqs_flat.shape[-1] // 2
        cos = freqs_flat[:, :half_dim].contiguous()  # [B*seq_len, head_dim]
        sin = freqs_flat[:, half_dim:].contiguous()  # [B*seq_len, head_dim]

        n_rows = x_flat.shape[0]  # B*seq_len*n_heads
        BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)

        with torch_gpu_device(device):
            _apply_rope_transposed_kernel[(n_rows,)](
                x_flat,
                out,
                cos,
                sin,
                n_heads,
                x_flat.stride(0),
                out.stride(0),
                cos.stride(0),
                head_dim,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        out = out.reshape(B, seq_len, n_heads, head_dim)

        ctx.save_for_backward(cos, sin)
        ctx.n_heads = n_heads
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.head_dim = head_dim

        return out

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin = ctx.saved_tensors

        B, seq_len, n_heads, head_dim = grad_output.shape
        grad_flat = grad_output.reshape(-1, head_dim).contiguous()
        device = grad_flat.device
        grad_x = torch.empty_like(grad_flat)

        sin_neg = -sin

        n_rows = grad_flat.shape[0]

        with torch_gpu_device(device):
            _apply_rope_transposed_kernel[(n_rows,)](
                grad_flat,
                grad_x,
                cos,
                sin_neg,
                ctx.n_heads,
                grad_flat.stride(0),
                grad_x.stride(0),
                cos.stride(0),
                ctx.head_dim,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps,
            )

        grad_x = grad_x.reshape(B, seq_len, n_heads, head_dim)
        return grad_x, None


# ------------------------------- For test -------------------------------
def test_zero_error():
    def apply_rotary_emb_transposed_orig(x, freqs_cis):
        cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
        x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
        out = torch.empty_like(x)
        out[..., 0::2] = x_real * cos[..., 0::2] - x_imag * sin[..., 1::2]
        out[..., 1::2] = x_real * sin[..., 1::2] + x_imag * cos[..., 0::2]
        return out

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.randn(1, 128, 12, 128, device="cuda", dtype=dtype)
        freqs_cis = torch.randn(1, 128, 256, device="cuda", dtype=dtype)

        out_orig = apply_rotary_emb_transposed_orig(x, freqs_cis)
        out_fast = apply_rotary_emb_transposed_flash(x, freqs_cis)

        diff = (out_orig - out_fast).abs().max()

        eps = torch.finfo(dtype).eps
        print(f"{dtype}: max_diff={diff.item():.2e}, machine_eps={eps:.2e}")

        if diff < eps * 100:
            print(f"  ✅ Essentially zero error for {dtype}")
        else:
            print(f"  ⚠️ Significant error: {diff / eps:.1f}x machine epsilon")


def test_comparison():
    def apply_rotary_emb_transposed_orig(x, freqs_cis):
        cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
        x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
        out = torch.empty_like(x)
        out[..., 0::2] = x_real * cos[..., 0::2] - x_imag * sin[..., 1::2]
        out[..., 1::2] = x_real * sin[..., 1::2] + x_imag * cos[..., 0::2]
        return out

    x = torch.randn(1, 14040, 12, 128, device="cuda", dtype=torch.float32)
    freqs_cis = torch.randn(1, 14040, 256, device="cuda", dtype=torch.float32)

    out_orig = apply_rotary_emb_transposed_orig(x, freqs_cis)
    out_fast = apply_rotary_emb_transposed_flash(x, freqs_cis)

    diff = (out_orig - out_fast).abs().max()
    print(f"Max difference: {diff.item():.6e}")

    if diff < 1e-5:
        print("✅ Test passed!")
        print(f"Input shapes: x={x.shape}, freqs_cis={freqs_cis.shape}")
        print(f"Output shape: {out_fast.shape}")
    else:
        print(f"❌ Test failed! Max diff: {diff.item()}")


def test_backward_comparison():
    def apply_rotary_emb_transposed_orig(x, freqs_cis):
        cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
        x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
        out = torch.empty_like(x)
        out[..., 0::2] = x_real * cos[..., 0::2] - x_imag * sin[..., 1::2]
        out[..., 1::2] = x_real * sin[..., 1::2] + x_imag * cos[..., 0::2]
        return out

    x1 = torch.randn(1, 128, 12, 128, device="cuda", requires_grad=True)
    x2 = x1.clone().detach().requires_grad_(True)
    freqs_cis = torch.randn(1, 128, 256, device="cuda")

    out_orig = apply_rotary_emb_transposed_orig(x1, freqs_cis)
    out_fast = apply_rotary_emb_transposed_flash(x2, freqs_cis)

    grad_output = torch.randn_like(out_orig)

    out_orig.backward(grad_output)
    out_fast.backward(grad_output)

    grad_diff = (x1.grad - x2.grad).abs()
    max_diff = grad_diff.max().item()
    mean_diff = grad_diff.mean().item()

    print("Gradient comparison:")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-5:
        print("✅ Backward gradients match!")
    else:
        print(f"⚠️ Gradients differ by {max_diff:.6e}")
        max_idx = grad_diff.argmax()
        print(f"  Max diff location: {torch.unravel_index(max_idx, grad_diff.shape)}")
        print(f"  Original grad: {x1.grad.flatten()[max_idx]:.6f}")
        print(f"  Fast grad: {x2.grad.flatten()[max_idx]:.6f}")


def test_backward():
    from torch.autograd import gradcheck

    B, seq_len, n_heads, head_dim = 2, 16, 4, 32
    x = torch.randn(B, seq_len, n_heads, head_dim, device="cuda", dtype=torch.float64, requires_grad=True)
    freqs_cis = torch.randn(B, seq_len, head_dim * 2, device="cuda", dtype=torch.float64)

    test = gradcheck(
        Flash_RoPE_Transposed.apply,
        (x, freqs_cis),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )

    if test:
        print("✅ Backward pass is correct (gradcheck passed)")
    else:
        print("❌ Backward pass has errors")


def test_in_training_loop_comparison():
    def apply_rotary_emb_transposed_orig(x, freqs_cis):
        cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
        x_real, x_imag = x.unflatten(-1, (-1, 2)).unbind(-1)
        out = torch.empty_like(x)
        out[..., 0::2] = x_real * cos[..., 0::2] - x_imag * sin[..., 1::2]
        out[..., 1::2] = x_real * sin[..., 1::2] + x_imag * cos[..., 0::2]
        return out

    class SimpleModel(torch.nn.Module):
        def __init__(self, use_fast=False):
            super().__init__()
            self.linear = torch.nn.Linear(128, 128, device="cuda")
            self.use_fast = use_fast

        def forward(self, x, freqs_cis):
            x = self.linear(x)
            if self.use_fast:
                x = apply_rotary_emb_transposed_flash(x, freqs_cis)
            else:
                x = apply_rotary_emb_transposed_orig(x, freqs_cis)
            return x.mean()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model_orig = SimpleModel(use_fast=False)
    model_fast = SimpleModel(use_fast=True)

    model_fast.load_state_dict(model_orig.state_dict())

    optimizer_orig = torch.optim.Adam(model_orig.parameters(), lr=1e-3)
    optimizer_fast = torch.optim.Adam(model_fast.parameters(), lr=1e-3)

    losses_orig = []
    losses_fast = []

    print("=" * 80)
    print("Training comparison: Original vs Optimized RoPE")
    print("=" * 80)
    print(f"{'Step':<6} {'Original Loss':<15} {'Fast Loss':<15} {'Diff':<12} {'Status':<10}")
    print("-" * 80)

    torch.manual_seed(42)
    inputs = [
        (torch.randn(1, 128, 12, 128, device="cuda"), torch.randn(1, 128, 256, device="cuda")) for _ in range(10)
    ]

    for step, (x, freqs_cis) in enumerate(inputs):
        optimizer_orig.zero_grad()
        loss_orig = model_orig(x.clone(), freqs_cis)
        loss_orig.backward()
        optimizer_orig.step()

        optimizer_fast.zero_grad()
        loss_fast = model_fast(x.clone(), freqs_cis)
        loss_fast.backward()
        optimizer_fast.step()

        has_nan_orig = any(p.grad is not None and torch.isnan(p.grad).any() for p in model_orig.parameters())
        has_nan_fast = any(p.grad is not None and torch.isnan(p.grad).any() for p in model_fast.parameters())

        if has_nan_orig or has_nan_fast:
            print(f"❌ Step {step}: Found NaN in gradients")
            return False

        loss_orig_val = loss_orig.item()
        loss_fast_val = loss_fast.item()
        losses_orig.append(loss_orig_val)
        losses_fast.append(loss_fast_val)

        diff = abs(loss_orig_val - loss_fast_val)
        rel_diff = diff / abs(loss_orig_val) if abs(loss_orig_val) > 1e-10 else 0

        if diff < 1e-6:
            status = "✅ Match"
        elif diff < 1e-4:
            status = "✓ Close"
        else:
            status = "⚠️ Differ"

        print(
            f"{step:<6} {loss_orig_val:<15.6f} {loss_fast_val:<15.6f} "
            f"{diff:<12.2e} {status:<10}"
            f"{rel_diff:<12.2e} {status:<10}"
        )

    print("-" * 80)

    avg_diff = sum(abs(o - f) for o, f in zip(losses_orig, losses_fast)) / len(losses_orig)
    max_diff = max(abs(o - f) for o, f in zip(losses_orig, losses_fast))

    print(f"\n{'Summary':<20} {'Original':<15} {'Optimized':<15} {'Difference':<15}")
    print("-" * 65)
    print(
        f"{'Initial loss:':<20} {losses_orig[0]:<15.6f} {losses_fast[0]:<15.6f} "
        f"{abs(losses_orig[0] - losses_fast[0]):<15.2e}"
    )
    print(
        f"{'Final loss:':<20} {losses_orig[-1]:<15.6f} {losses_fast[-1]:<15.6f} "
        f"{abs(losses_orig[-1] - losses_fast[-1]):<15.2e}"
    )
    print(
        f"{'Average loss:':<20} {sum(losses_orig) / len(losses_orig):<15.6f} "
        f"{sum(losses_fast) / len(losses_fast):<15.6f} {avg_diff:<15.2e}"
    )
    print(f"{'Max difference:':<20} {'':<15} {'':<15} {max_diff:<15.2e}")

    weight_diffs = []
    for (name_o, param_o), (name_f, param_f) in zip(model_orig.named_parameters(), model_fast.named_parameters()):
        diff = (param_o - param_f).abs().max().item()
        weight_diffs.append(diff)

    max_weight_diff = max(weight_diffs)
    print(f"{'Max weight diff:':<20} {'':<15} {'':<15} {max_weight_diff:<15.2e}")

    print("=" * 80)

    if max_diff < 1e-4 and max_weight_diff < 1e-4:
        print("✅ Training consistency test PASSED")
        print("   Original and optimized versions produce nearly identical results")
        return True
    elif max_diff < 1e-2:
        print("✓ Training consistency test ACCEPTABLE")
        print("   Small numerical differences detected (within tolerance)")
        return True
    else:
        print("⚠️ Training consistency test WARNING")
        print(f"   Differences detected: loss_diff={max_diff:.2e}, weight_diff={max_weight_diff:.2e}")
        return False


if __name__ == "__main__":
    test_zero_error()
    test_comparison()
    test_backward_comparison()
    test_in_training_loop_comparison()
    # test_backward()
