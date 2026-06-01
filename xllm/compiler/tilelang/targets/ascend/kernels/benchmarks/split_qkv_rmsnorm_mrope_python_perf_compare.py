#!/usr/bin/env python3

"""
Run the TileLang and/or Triton split_qkv_rmsnorm_mrope kernels on the Python path.

  - normal mode: compare TileLang vs Triton outputs
  - --triton-only: benchmark only the Triton baseline (no TileLang dependency)
  - worker mode: run warmup + measure loops without profiling (for msprof)

msprof orchestration and task-duration extraction: kernel_msprof_task_duration.py
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

DEFAULT_MROPE_SECTION = (11, 11, 10)
DEFAULT_NUM_Q_HEADS = 16
DEFAULT_NUM_KV_HEADS = 4
REF_CHECK_EPS = 1e-6
DEFAULT_HEAD_SIZE = 256
DEFAULT_ROPE_DIM = 64


def _import_tilelang():
    from compiler.tilelang.targets.ascend.kernels.split_qkv_rmsnorm_mrope import (
        SUPPORTED_HEAD_SPECS,
        build_mrope_gather_pattern_merged,
        detect_vec_core_num,
        split_qkv_rmsnorm_mrope_kernel_jit,
    )
    return SUPPORTED_HEAD_SPECS, build_mrope_gather_pattern_merged, detect_vec_core_num, split_qkv_rmsnorm_mrope_kernel_jit
DEFAULT_NUM_TOKENS_LIST = (1, 16, 128, 512)
DEFAULT_WARMUP_ITERS = 20
DEFAULT_MEASURE_ITERS = 200
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Triton kernel inlined from vllm-ascend:
#   repo: https://github.com/vllm-project/vllm-ascend
#   file: vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py
#   commit: fc5cbd6e ([Misc][Upgrade] Upgrade CANN to 9.0.0 and triton-ascend to 3.2.1 (#9085))
# ---------------------------------------------------------------------------

def get_vectorcore_num() -> int:
    props = triton.runtime.driver.active.utils.get_device_properties(
        torch.npu.current_device()
    )
    return int(props["num_vectorcore"])


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "front_core_num",
        "num_tokens_each_front_core",
        "num_tokens_each_tail_core",
    ]
)
def triton_split_qkv_rmsnorm_mrope_kernel(
    in_qkv_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    cos_sin_ptr,
    out_q_ptr,
    out_k_ptr,
    out_v_ptr,
    out_gate_ptr,
    num_tokens,
    front_core_num,
    num_tokens_each_front_core,
    num_tokens_each_tail_core,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    q_size: tl.constexpr,
    kv_size: tl.constexpr,
    eps: tl.constexpr,
    mrope_section_t,
    mrope_section_h,
    mrope_section_w,
    has_bias: tl.constexpr,
    is_interleaved: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    gate_size: tl.constexpr,
):
    block_idx = tl.program_id(0)
    loop_num = num_tokens_each_front_core
    if block_idx >= front_core_num:
        loop_num = num_tokens_each_tail_core
    block_offset = num_tokens_each_front_core * block_idx
    if block_idx >= front_core_num:
        block_offset = (
            num_tokens_each_front_core * front_core_num
            + (block_idx - front_core_num) * num_tokens_each_tail_core
        )

    q_rmsnorm_weight = tl.load(q_weight_ptr + tl.arange(0, head_size))
    k_rmsnorm_weight = tl.load(k_weight_ptr + tl.arange(0, head_size))
    if has_bias:
        q_bias = tl.load(q_bias_ptr + tl.arange(0, head_size))
        k_bias = tl.load(k_bias_ptr + tl.arange(0, head_size))

    for index in range(loop_num):
        in_q_offset = (
            in_qkv_ptr + (block_offset + index) * (q_size + gate_size + 2 * kv_size)
        )
        if gate_size > 0:
            in_q_gate_tensor = (
                tl.load(in_q_offset + tl.arange(0, q_size + gate_size))
                .to(tl.float32)
                .reshape(num_q_heads, head_size * 2)
            )
            in_q_tensor = tl.extract_slice(
                in_q_gate_tensor, offsets=(0, 0),
                sizes=(num_q_heads, head_size), strides=(1, 1),
            )
            in_gate_tensor = tl.extract_slice(
                in_q_gate_tensor, offsets=(0, head_size),
                sizes=(num_q_heads, head_size), strides=(1, 1),
            ).reshape(q_size)
        else:
            in_q_tensor = (
                tl.load(in_q_offset + tl.arange(0, q_size))
                .to(tl.float32)
                .reshape(num_q_heads, head_size)
            )
        in_k_offset = in_q_offset + q_size + gate_size
        in_k_tensor = (
            tl.load(in_k_offset + tl.arange(0, kv_size))
            .to(tl.float32)
            .reshape(num_kv_heads, head_size)
        )
        in_v_offset = in_k_offset + kv_size
        in_v_tensor = tl.load(in_v_offset + tl.arange(0, kv_size))

        cos_offsets = tl.arange(0, half_rope_dim)
        if is_interleaved:
            h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
            w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
            t_mask = ~(h_mask | w_mask)
        else:
            t_mask = cos_offsets < mrope_section_t
            h_mask = (mrope_section_t - 1 < cos_offsets) & (
                cos_offsets < mrope_section_t + mrope_section_h
            )
            w_mask = (mrope_section_t + mrope_section_h - 1 < cos_offsets) & (
                cos_offsets < mrope_section_t + mrope_section_h + mrope_section_w
            )

        t_cos_offset = cos_sin_ptr + (block_offset + index) * rope_dim
        h_cos_offset = t_cos_offset + num_tokens * rope_dim
        w_cos_offset = h_cos_offset + num_tokens * rope_dim
        t_sin_offset = t_cos_offset + half_rope_dim
        h_sin_offset = t_sin_offset + num_tokens * rope_dim
        w_sin_offset = h_sin_offset + num_tokens * rope_dim

        t_cos = tl.load(t_cos_offset + cos_offsets, mask=t_mask, other=0)
        h_cos = tl.load(h_cos_offset + cos_offsets, mask=h_mask, other=0)
        w_cos = tl.load(w_cos_offset + cos_offsets, mask=w_mask, other=0)
        t_sin = tl.load(t_sin_offset + cos_offsets, mask=t_mask, other=0)
        h_sin = tl.load(h_sin_offset + cos_offsets, mask=h_mask, other=0)
        w_sin = tl.load(w_sin_offset + cos_offsets, mask=w_mask, other=0)

        cos_tensor = (
            (t_cos + h_cos + w_cos)
            .to(tl.float32)
            .reshape(1, half_rope_dim)
        )
        cos_tensor = tl.broadcast_to(cos_tensor, (2, half_rope_dim)).reshape(
            1, rope_dim
        )
        sin_tensor = (
            (t_sin + h_sin + w_sin)
            .to(tl.float32)
            .reshape(1, half_rope_dim)
        )
        sin_tensor = tl.broadcast_to(sin_tensor, (2, half_rope_dim)).reshape(
            1, rope_dim
        )

        squares = in_q_tensor * in_q_tensor
        variances = tl.sum(squares, axis=1) / head_size
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_q_heads, 1)
        q_normalized = in_q_tensor * reciprocal_std * q_rmsnorm_weight
        if has_bias:
            q_normalized = q_normalized + q_bias

        squares = in_k_tensor * in_k_tensor
        variances = tl.sum(squares, axis=1) / head_size
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_kv_heads, 1)
        k_normalized = in_k_tensor * reciprocal_std * k_rmsnorm_weight
        if has_bias:
            k_normalized = k_normalized + k_bias

        x1 = tl.extract_slice(q_normalized, offsets=(0, 0),
                               sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        x2 = tl.extract_slice(q_normalized, offsets=(0, half_rope_dim),
                               sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        cat_x = tl.zeros((num_q_heads, rope_dim), dtype=tl.float32)
        cat_x = tl.insert_slice(cat_x, -x2, offsets=(0, 0),
                                 sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        cat_x = tl.insert_slice(cat_x, x1, offsets=(0, half_rope_dim),
                                 sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        if IS_PARTIAL_ROPE:
            orig_qk = tl.extract_slice(q_normalized, offsets=(0, 0),
                                        sizes=(num_q_heads, rope_dim), strides=(1, 1))
        else:
            orig_qk = q_normalized
        roped_q = cat_x * sin_tensor + orig_qk * cos_tensor

        y1 = tl.extract_slice(k_normalized, offsets=(0, 0),
                               sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        y2 = tl.extract_slice(k_normalized, offsets=(0, half_rope_dim),
                               sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        cat_y = tl.zeros((num_kv_heads, rope_dim), dtype=tl.float32)
        cat_y = tl.insert_slice(cat_y, -y2, offsets=(0, 0),
                                 sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        cat_y = tl.insert_slice(cat_y, y1, offsets=(0, half_rope_dim),
                                 sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        if IS_PARTIAL_ROPE:
            orig_qk = tl.extract_slice(k_normalized, offsets=(0, 0),
                                        sizes=(num_kv_heads, rope_dim), strides=(1, 1))
        else:
            orig_qk = k_normalized
        roped_k = cat_y * sin_tensor + orig_qk * cos_tensor

        if IS_PARTIAL_ROPE:
            q_normalized = tl.insert_slice(q_normalized, roped_q, offsets=(0, 0),
                                            sizes=(num_q_heads, rope_dim), strides=(1, 1))
            k_normalized = tl.insert_slice(k_normalized, roped_k, offsets=(0, 0),
                                            sizes=(num_kv_heads, rope_dim), strides=(1, 1))
        else:
            q_normalized = roped_q
            k_normalized = roped_k

        out_q_offset = out_q_ptr + (block_offset + index) * q_size
        tl.store(out_q_offset + tl.arange(0, q_size), q_normalized.reshape(q_size))
        out_k_offset = out_k_ptr + (block_offset + index) * kv_size
        tl.store(out_k_offset + tl.arange(0, kv_size), k_normalized.reshape(kv_size))
        out_v_offset = out_v_ptr + (block_offset + index) * kv_size
        tl.store(out_v_offset + tl.arange(0, kv_size), in_v_tensor)
        if gate_size > 0:
            out_gate_offset = out_gate_ptr + (block_offset + index) * gate_size
            tl.store(out_gate_offset + tl.arange(0, gate_size), in_gate_tensor)


def run_triton(
    *,
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    q_output: torch.Tensor,
    k_output: torch.Tensor,
    v_output: torch.Tensor,
    gate_output: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: tuple[int, int, int],
    is_interleaved: bool,
    rope_dim: int,
) -> None:
    core_num = get_vectorcore_num()
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    num_tokens = qkv.shape[0]
    gate_size = q_size
    IS_PARTIAL_ROPE = rope_dim != head_size

    front_core_num = core_num
    if num_tokens % core_num != 0:
        front_core_num = num_tokens % core_num
    num_tokens_each_front_core = (num_tokens + core_num - 1) // core_num
    tail_core_num = 0
    if num_tokens > core_num:
        tail_core_num = core_num - front_core_num
    num_tokens_each_tail_core = num_tokens // core_num
    total_core = front_core_num + tail_core_num
    block_dim = min(total_core, core_num)

    triton_split_qkv_rmsnorm_mrope_kernel[(block_dim,)](
        qkv, q_weight, None, k_weight, None, cos_sin,
        q_output, k_output, v_output, gate_output,
        num_tokens, front_core_num,
        num_tokens_each_front_core, num_tokens_each_tail_core,
        num_q_heads, num_kv_heads, head_size, q_size, kv_size, eps,
        mrope_section[0], mrope_section[1], mrope_section[2],
        False, is_interleaved, rope_dim, rope_dim // 2,
        IS_PARTIAL_ROPE, gate_size,
    )


# ---------------------------------------------------------------------------
# Benchmark infrastructure
# ---------------------------------------------------------------------------

def _make_base_tensors(
    *,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    rope_dim: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device("npu")
    half_rope_dim = rope_dim // 2
    q_width = num_q_heads * head_size
    kv_width = num_kv_heads * head_size
    qkv_stride = q_width * 2 + kv_width * 2

    qkv = torch.randn((num_tokens, qkv_stride), device=device, dtype=torch.bfloat16)
    q_weight = torch.randn((1, head_size), device=device, dtype=torch.bfloat16)
    k_weight = torch.randn((1, head_size), device=device, dtype=torch.bfloat16)
    phase = torch.randn((3, num_tokens, half_rope_dim), device=device)
    cos_sin = torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1).to(
        torch.bfloat16
    )

    return {
        "qkv": qkv,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "cos_sin": cos_sin,
        "tr_q": torch.empty((num_tokens, q_width), device=device, dtype=torch.bfloat16),
        "tr_k": torch.empty((num_tokens, kv_width), device=device, dtype=torch.bfloat16),
        "tr_v": torch.empty((num_tokens, kv_width), device=device, dtype=torch.bfloat16),
        "tr_gate": torch.empty((num_tokens, q_width), device=device, dtype=torch.bfloat16),
    }


def make_case_tensors(
    *,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    rope_dim: int,
    mrope_section: tuple[int, int, int],
    is_interleaved: bool,
    seed: int,
    build_mrope_gather_fn=None,
) -> dict[str, torch.Tensor]:
    tensors = _make_base_tensors(
        num_tokens=num_tokens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rope_dim=rope_dim,
        seed=seed,
    )
    device = torch.device("npu")
    q_width = num_q_heads * head_size
    kv_width = num_kv_heads * head_size

    cos_sin_reshaped = tensors["cos_sin"].permute(1, 0, 2).contiguous().view(num_tokens, 3 * rope_dim)
    tensors["cos_sin_reshaped"] = cos_sin_reshaped

    if build_mrope_gather_fn is not None:
        half_rope_dim = rope_dim // 2
        gather_pad_dim = ((3 * rope_dim + 127) // 128) * 128
        tensors["gather_pat"] = build_mrope_gather_fn(
            half_rope_dim=half_rope_dim,
            rope_dim=rope_dim,
            mrope_section=mrope_section,
            is_interleaved=is_interleaved,
            gather_pad_dim=gather_pad_dim,
            device=str(device),
        )
        tensors["tl_q"] = torch.empty((num_tokens, q_width), device=device, dtype=torch.bfloat16)
        tensors["tl_k"] = torch.empty((num_tokens, kv_width), device=device, dtype=torch.bfloat16)
        tensors["tl_v"] = torch.empty((num_tokens, kv_width), device=device, dtype=torch.bfloat16)
        tensors["tl_gate"] = torch.empty((num_tokens, q_width), device=device, dtype=torch.bfloat16)

    return tensors


def make_triton_runner(
    *,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    rope_dim: int,
    eps: float,
    mrope_section: tuple[int, int, int],
    is_interleaved: bool,
    seed: int,
) -> dict[str, Any]:
    tensors = _make_base_tensors(
        num_tokens=num_tokens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rope_dim=rope_dim,
        seed=seed,
    )

    def run_tr() -> None:
        run_triton(
            qkv=tensors["qkv"],
            q_weight=tensors["q_weight"].view(-1),
            k_weight=tensors["k_weight"].view(-1),
            cos_sin=tensors["cos_sin"],
            q_output=tensors["tr_q"],
            k_output=tensors["tr_k"],
            v_output=tensors["tr_v"],
            gate_output=tensors["tr_gate"],
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            eps=eps,
            mrope_section=mrope_section,
            is_interleaved=is_interleaved,
            rope_dim=rope_dim,
        )

    return {"tensors": tensors, "run_triton": run_tr}


def make_case_runners(
    *,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    rope_dim: int,
    eps: float,
    mrope_section: tuple[int, int, int],
    is_interleaved: bool,
    tilelang_kernel,
    seed: int,
    build_mrope_gather_fn=None,
) -> dict[str, Any]:
    tensors = make_case_tensors(
        num_tokens=num_tokens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rope_dim=rope_dim,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
        seed=seed,
        build_mrope_gather_fn=build_mrope_gather_fn,
    )

    def run_tl() -> None:
        tilelang_kernel(
            tensors["qkv"],
            tensors["q_weight"],
            tensors["k_weight"],
            tensors["cos_sin_reshaped"],
            tensors["gather_pat"],
            tensors["tl_q"],
            tensors["tl_k"],
            tensors["tl_v"],
            tensors["tl_gate"],
            num_tokens,
            eps,
        )

    def run_tr() -> None:
        run_triton(
            qkv=tensors["qkv"],
            q_weight=tensors["q_weight"].view(-1),
            k_weight=tensors["k_weight"].view(-1),
            cos_sin=tensors["cos_sin"],
            q_output=tensors["tr_q"],
            k_output=tensors["tr_k"],
            v_output=tensors["tr_v"],
            gate_output=tensors["tr_gate"],
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            eps=eps,
            mrope_section=mrope_section,
            is_interleaved=is_interleaved,
            rope_dim=rope_dim,
        )

    return {"tensors": tensors, "run_tilelang": run_tl, "run_triton": run_tr}


def benchmark_case(
    *,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    rope_dim: int,
    eps: float,
    mrope_section: tuple[int, int, int],
    is_interleaved: bool,
    tilelang_kernel,
    seed: int,
) -> dict[str, float | int]:
    runners = make_case_runners(
        num_tokens=num_tokens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rope_dim=rope_dim,
        eps=eps,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
        tilelang_kernel=tilelang_kernel,
        seed=seed,
    )
    t = runners["tensors"]
    runners["run_tilelang"]()
    runners["run_triton"]()
    torch.npu.synchronize()

    q_diff = float((t["tl_q"].float() - t["tr_q"].float()).abs().max().item())
    k_diff = float((t["tl_k"].float() - t["tr_k"].float()).abs().max().item())

    return {
        "num_tokens": num_tokens,
        "q_max_diff": q_diff,
        "k_max_diff": k_diff,
    }



def run_worker(args: argparse.Namespace) -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")
    if len(args.num_tokens_list) != 1:
        raise ValueError("--worker requires exactly one num_tokens value")

    num_tokens = args.num_tokens_list[0]
    mrope_section = tuple(args.mrope_section)

    if args.triton_only:
        runner = make_triton_runner(
            num_tokens=num_tokens,
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            rope_dim=args.rope_dim,
            eps=args.eps,
            mrope_section=mrope_section,
            is_interleaved=args.is_interleaved,
            seed=args.seed,
        )
        run_fn = runner["run_triton"]
        for _ in range(args.warmup_iters):
            run_fn()
        torch.npu.synchronize()
        for _ in range(args.measure_iters):
            run_fn()
        torch.npu.synchronize()
        return

    _, build_mrope_gather_fn, detect_vec_core_num, split_qkv_rmsnorm_mrope_kernel_jit = _import_tilelang()
    tilelang_kernel = split_qkv_rmsnorm_mrope_kernel_jit(
        head_size=args.head_size,
        rope_dim=args.rope_dim,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        vec_core_num=detect_vec_core_num(),
        max_num_tokens=num_tokens,
    )
    runners = make_case_runners(
        num_tokens=num_tokens,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        rope_dim=args.rope_dim,
        eps=args.eps,
        mrope_section=mrope_section,
        is_interleaved=args.is_interleaved,
        tilelang_kernel=tilelang_kernel,
        seed=args.seed,
        build_mrope_gather_fn=build_mrope_gather_fn,
    )
    run_tl = runners["run_tilelang"]
    run_tr = runners["run_triton"]

    for _ in range(args.warmup_iters):
        run_tl()
        run_tr()
    torch.npu.synchronize()
    for _ in range(args.measure_iters):
        run_tl()
        run_tr()
    torch.npu.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run TileLang/Triton split_qkv_rmsnorm_mrope kernels on the Python path."
        )
    )
    parser.add_argument(
        "--num-tokens-list", type=int, nargs="+",
        default=list(DEFAULT_NUM_TOKENS_LIST),
    )
    parser.add_argument("--num-q-heads", type=int, default=DEFAULT_NUM_Q_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=DEFAULT_NUM_KV_HEADS)
    parser.add_argument("--head-size", type=int, default=DEFAULT_HEAD_SIZE)
    parser.add_argument("--rope-dim", type=int, default=DEFAULT_ROPE_DIM)
    parser.add_argument("--eps", type=float, default=REF_CHECK_EPS)
    parser.add_argument(
        "--mrope-section", type=int, nargs=3,
        default=list(DEFAULT_MROPE_SECTION), metavar=("T", "H", "W"),
    )
    parser.add_argument("--is-interleaved", action="store_true", default=True)
    parser.add_argument("--no-interleaved", dest="is_interleaved", action="store_false")
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--measure-iters", type=int, default=DEFAULT_MEASURE_ITERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--triton-only", action="store_true",
        help="Benchmark only the Triton baseline (skip TileLang).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.worker:
        run_worker(args)
        return

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")

    mrope_section = tuple(args.mrope_section)
    il_tag = "interleaved" if args.is_interleaved else "non-interleaved"

    if args.triton_only:
        print(
            "Performance must be measured with msprof via kernel_msprof_task_duration.py.\n"
            "Use --worker --triton-only mode with msprof orchestration."
        )
        return

    _, build_mrope_gather_fn, detect_vec_core_num, split_qkv_rmsnorm_mrope_kernel_jit = _import_tilelang()

    print(
        f"# qh={args.num_q_heads} kvh={args.num_kv_heads} "
        f"hs={args.head_size} rd={args.rope_dim} {il_tag}"
    )
    print("# Performance must be measured with msprof via kernel_msprof_task_duration.py")
    print("case,num_tokens,q_max_diff,k_max_diff")
    for idx, num_tokens in enumerate(args.num_tokens_list):
        tilelang_kernel = split_qkv_rmsnorm_mrope_kernel_jit(
            head_size=args.head_size,
            rope_dim=args.rope_dim,
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            vec_core_num=detect_vec_core_num(),
            max_num_tokens=num_tokens,
        )
        result = benchmark_case(
            num_tokens=num_tokens,
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            rope_dim=args.rope_dim,
            eps=args.eps,
            mrope_section=mrope_section,
            is_interleaved=args.is_interleaved,
            tilelang_kernel=tilelang_kernel,
            seed=args.seed + idx,
        )
        print(
            f"t{num_tokens},"
            f"{result['num_tokens']},"
            f"{result['q_max_diff']:.6e},"
            f"{result['k_max_diff']:.6e}"
        )


if __name__ == "__main__":
    main()
