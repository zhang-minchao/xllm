#!/usr/bin/env python3

"""
Benchmark the Triton chunk_gated_delta_rule forward-h kernel on Ascend NPU.

Triton implementation inlined from vllm-ascend (chunk_delta_h.py).
Currently triton-only; TileLang comparison can be added when a TileLang
kernel is available in this repo.

  - normal mode: print usage info (performance must be measured with msprof)
  - worker mode: run warmup + measure loops without profiling (for msprof)

msprof orchestration: kernel_msprof_task_duration.py
"""

import argparse
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[6]
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Utility functions and triton kernel inlined from vllm-ascend:
#   repo: https://github.com/vllm-project/vllm-ascend
#   files: vllm_ascend/ops/triton/fla/chunk_delta_h.py,
#          vllm_ascend/ops/triton/fla/utils.py
#   commit: fe7097f3 ([kernel] Recompilation optimization triggered by triton function para… (#7481))
# ---------------------------------------------------------------------------

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


# ---------------------------------------------------------------------------
# Triton kernel: chunk_gated_delta_rule_fwd_kernel_h (from vllm-ascend)
# ---------------------------------------------------------------------------

@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "H", "Hg", "K", "V"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    h_update,
    T,
    H,
    Hg,
    K,
    V,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    T_max = 1 * T
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    stride_v = H * V
    stride_k = Hg * K
    stride_w = H * K

    b_h1_bv1 = tl.zeros([128, 64], dtype=tl.float32)
    b_h1_bv2 = tl.zeros([128, 64], dtype=tl.float32)

    v_start1 = 0
    v_start2 = 64

    offs_k = tl.arange(0, 128)[:, None]
    offs_v1 = v_start1 + tl.arange(0, 64)[None, :]
    offs_v2 = v_start2 + tl.arange(0, 64)[None, :]
    mask_kv1 = (offs_k < K) & (offs_v1 < V)
    mask_kv2 = (offs_k < K) & (offs_v2 < V)

    if USE_INITIAL_STATE:
        h0_ptr = h0 + i_nh * K * V
        ptr_h0_bv1 = h0_ptr + offs_k * V + offs_v1 * 1
        b_h1_bv1 += tl.load(ptr_h0_bv1, mask=mask_kv1, other=0.0).to(tl.float32)
        ptr_h0_bv2 = h0_ptr + offs_k * V + offs_v2 * 1
        b_h1_bv2 += tl.load(ptr_h0_bv2, mask=mask_kv2, other=0.0).to(tl.float32)

    for i_t in range(NT):
        h_base = h + (boh + i_t) * H * K * V + i_h * K * V

        p_h1_bv1 = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start1), (128, 64), (1, 0))
        tl.store(p_h1_bv1, b_h1_bv1.to(p_h1_bv1.dtype.element_ty), boundary_check=(0, 1))
        p_h1_bv2 = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start2), (128, 64), (1, 0))
        tl.store(p_h1_bv2, b_h1_bv2.to(p_h1_bv2.dtype.element_ty), boundary_check=(0, 1))

        offs_t_wv = (i_t * BT + tl.arange(0, BT))[:, None]
        offs_k_wv = tl.arange(0, 128)[None, :]
        mask_w = (offs_t_wv < T) & (offs_k_wv < K)

        w_base = w + bos * H * K + i_h * K
        ptr_w = w_base + offs_t_wv * stride_w + offs_k_wv * 1
        b_w = tl.load(ptr_w, mask=mask_w, other=0.0)

        k_base = k + bos * Hg * K + (i_h // (H // Hg)) * K
        p_k = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (0, i_t * BT), (128, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        v_new_base = v_new + bos * H * V + i_h * V

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + bos + i_h * T_max + last_idx)

        offs_t = i_t * BT + tl.arange(0, BT)
        mask_t = offs_t < T
        g_ptr = g + bos + i_h * T_max
        b_g = tl.load(g_ptr + offs_t, mask=mask_t, other=0.0)

        b_g = safe_exp(b_g_last - b_g)
        b_g_last = tl.exp(b_g_last)

        offs_t_v = (i_t * BT + tl.arange(0, BT))[:, None]
        mask_v1 = (offs_t_v < T) & (offs_v1 < V)

        v_base = v + bos * H * V + i_h * V
        ptr_v1 = v_base + offs_t_v * stride_v + offs_v1 * 1
        b_v1 = tl.load(ptr_v1, mask=mask_v1, other=0.0)
        b_v_new1 = b_v1.to(tl.float32)
        b_v_new1 -= tl.dot(b_w, b_h1_bv1.to(b_w.dtype))

        if SAVE_NEW_VALUE:
            p_v_new1 = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (i_t * BT, v_start1), (BT, 64), (1, 0))
            tl.store(p_v_new1, b_v_new1.to(p_v_new1.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            b_v_new1 = b_v_new1 * b_g[:, None]
            b_h1_bv1 = b_h1_bv1 * b_g_last

        b_v_new1 = b_v_new1.to(k.dtype.element_ty)
        b_h1_bv1 += tl.dot(b_k, b_v_new1)

        mask_v2 = (offs_t_v < T) & (offs_v2 < V)
        ptr_v2 = v_base + offs_t_v * stride_v + offs_v2 * 1
        b_v2 = tl.load(ptr_v2, mask=mask_v2, other=0.0)
        b_v_new2 = b_v2.to(tl.float32)
        b_v_new2 -= tl.dot(b_w, b_h1_bv2.to(b_w.dtype))

        if SAVE_NEW_VALUE:
            p_v_new2 = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (i_t * BT, v_start2), (BT, 64), (1, 0))
            tl.store(p_v_new2, b_v_new2.to(p_v_new2.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            b_v_new2 = b_v_new2 * b_g[:, None]
            b_h1_bv2 = b_h1_bv2 * b_g_last

        b_v_new2 = b_v_new2.to(k.dtype.element_ty)
        b_h1_bv2 += tl.dot(b_k, b_v_new2)

    if STORE_FINAL_STATE:
        ht_ptr = ht + i_nh * K * V
        p_ht1_bv1 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start1), (128, 64), (1, 0))
        tl.store(p_ht1_bv1, b_h1_bv1.to(p_ht1_bv1.dtype.element_ty), boundary_check=(0, 1))
        p_ht1_bv2 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start2), (128, 64), (1, 0))
        tl.store(p_ht1_bv2, b_h1_bv2.to(p_ht1_bv2.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# Python launcher
# ---------------------------------------------------------------------------

def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            chunk_offsets,
        )

    h = k.new_empty(B, NT, H, K, V)
    h_update = k.new_empty(B, NT, H, K, K)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    g_transposed = g.transpose(1, 2).contiguous() if g is not None else None

    def grid(meta):
        return (1, N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g_transposed,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        h_update=h_update,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        num_warps=4,
        num_stages=2,
    )
    return h, v_new, final_state


# ---------------------------------------------------------------------------
# Test case definitions from design doc
# ---------------------------------------------------------------------------

DEFAULT_K = 128
DEFAULT_V = 128
DEFAULT_CHUNK_SIZE = 64
DEFAULT_WARMUP_ITERS = 20
DEFAULT_MEASURE_ITERS = 200
DEFAULT_SEED = 42

BENCHMARK_CASES = [
    {"name": "short_real", "T": 31, "Hqk": 4, "Hv": 8, "cu_seqlens": [0, 31]},
    {"name": "grouped_4k_h4_to_h8", "T": 4096, "Hqk": 4, "Hv": 8, "cu_seqlens": [0, 4096]},
    {"name": "grouped_4k_h8_to_h16", "T": 4096, "Hqk": 8, "Hv": 16, "cu_seqlens": [0, 4096]},
    {"name": "grouped_4k_h16_to_h32", "T": 4096, "Hqk": 16, "Hv": 32, "cu_seqlens": [0, 4096]},
    {"name": "long_real", "T": 10240, "Hqk": 16, "Hv": 16, "cu_seqlens": [0, 10240]},
    {"name": "tail_stress", "T": 4097, "Hqk": 8, "Hv": 16, "cu_seqlens": [0, 4097]},
    {"name": "multi_seq_packed", "T": 4096, "Hqk": 8, "Hv": 16, "cu_seqlens": [0, 1024, 3072, 4096]},
]


def make_case_tensors(
    *,
    T: int,
    Hqk: int,
    Hv: int,
    K: int,
    V: int,
    cu_seqlens_list: list[int],
    initial_state_dtype: torch.dtype,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device("npu")
    B = 1
    N = len(cu_seqlens_list) - 1

    q = torch.randn((B, T, Hqk, K), device=device, dtype=torch.bfloat16)
    k = torch.randn((B, T, Hqk, K), device=device, dtype=torch.bfloat16)
    v = torch.randn((B, T, Hv, V), device=device, dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn((B, T, Hv), device=device, dtype=torch.float32))
    # w is the preprocessed write-weight with shape [B, T, H, K], not the raw beta [B, T, Hv].
    # In FLA, w = beta.unsqueeze(-1) * k (after GQA head replication).
    # For benchmarking we just need the correct shape and value range.
    w = torch.randn((B, T, Hv, K), device=device, dtype=torch.bfloat16)
    initial_state = torch.randn((N, Hv, K, V), device=device, dtype=initial_state_dtype)
    cu_seqlens = torch.tensor(cu_seqlens_list, device=device, dtype=torch.int32).to(torch.long)

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "w": w,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
    }


def make_case_runner(
    *,
    case: dict[str, Any],
    initial_state_dtype: torch.dtype,
    seed: int,
) -> dict[str, Any]:
    tensors = make_case_tensors(
        T=case["T"],
        Hqk=case["Hqk"],
        Hv=case["Hv"],
        K=DEFAULT_K,
        V=DEFAULT_V,
        cu_seqlens_list=case["cu_seqlens"],
        initial_state_dtype=initial_state_dtype,
        seed=seed,
    )

    def run_triton() -> None:
        chunk_gated_delta_rule_fwd_h(
            k=tensors["k"],
            w=tensors["w"],
            u=tensors["v"],
            g=tensors["g"],
            initial_state=tensors["initial_state"],
            output_final_state=True,
            chunk_size=DEFAULT_CHUNK_SIZE,
            save_new_value=True,
            cu_seqlens=tensors["cu_seqlens"],
        )

    return {"tensors": tensors, "run_triton": run_triton}


def run_worker(args: argparse.Namespace) -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")

    cases = BENCHMARK_CASES
    if args.case_name:
        cases = [c for c in BENCHMARK_CASES if c["name"] == args.case_name]
        if not cases:
            raise ValueError(f"Unknown case: {args.case_name}")

    for case in cases:
        for dtype_tag, dtype in [("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            runner = make_case_runner(
                case=case,
                initial_state_dtype=dtype,
                seed=args.seed,
            )
            run_fn = runner["run_triton"]

            for _ in range(args.warmup_iters):
                run_fn()
            torch.npu.synchronize()

            for _ in range(args.measure_iters):
                run_fn()
            torch.npu.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton chunk_gated_delta_rule forward-h kernel."
    )
    parser.add_argument(
        "--case-name", type=str, default=None,
        help="Run a specific case. If omitted, run all cases.",
    )
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--measure-iters", type=int, default=DEFAULT_MEASURE_ITERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.worker:
        run_worker(args)
        return

    print(
        "Performance must be measured with msprof via kernel_msprof_task_duration.py.\n"
        "Example:\n"
        "  python kernel_msprof_task_duration.py \\\n"
        "    --runner-script chunk_gated_delta_rule_python_perf_compare.py \\\n"
        "    --worker-cmd-template '{python} {runner_script} --worker "
        "--case-name <CASE> --warmup-iters {warmup_iters} "
        "--measure-iters {measure_iters} --seed {seed}' \\\n"
        "    --kernel 'triton=chunk_gated_delta_rule_fwd_kernel_h_blockdim64' \\\n"
        "    --kernel-match-mode exact"
    )


if __name__ == "__main__":
    main()
