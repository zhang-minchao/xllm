/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "compressor.h"

#include <glog/logging.h>

#include <tuple>

#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#endif
#endif

#include "kernels/ops_api.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {

CompressorImpl::CompressorImpl(int64_t compress_ratio, int64_t head_dim)
    : CompressorImpl(
          compress_ratio,
          head_dim,
          64,
          2,
          1e-6,
          torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)) {}

CompressorImpl::CompressorImpl(int64_t compress_ratio,
                               int64_t head_dim,
                               const torch::TensorOptions& options)
    : CompressorImpl(compress_ratio, head_dim, 64, 2, 1e-6, options) {}

CompressorImpl::CompressorImpl(int64_t compress_ratio,
                               int64_t head_dim,
                               int64_t rope_head_dim,
                               int64_t rot_mode,
                               double norm_eps,
                               const torch::TensorOptions& options)
    : compress_ratio_(compress_ratio),
      head_dim_(head_dim),
      rope_head_dim_(rope_head_dim),
      rot_mode_(rot_mode),
      eps_(norm_eps),
      options_(options) {
  enable_compressor_overlap_ = (compress_ratio == 4);
}

torch::Tensor CompressorImpl::forward(
    const DSAMetadata& attn_metadata,
    torch::Tensor& hidden_states,
    std::tuple<torch::Tensor, torch::Tensor>& kv_states,
    std::tuple<torch::Tensor, torch::Tensor>& block_tables,
    torch::Tensor& compressed_sin,
    torch::Tensor& compressed_cos,
    torch::Tensor actual_seq_lengths_query) {
  auto [kv_state, score_state] = kv_states;
  auto [kv_block_table, score_block_table] = block_tables;

  const int64_t sin_last_dim = compressed_sin.size(compressed_sin.dim() - 1);
  const int64_t cos_last_dim = compressed_cos.size(compressed_cos.dim() - 1);

  torch::Tensor compressed_kv;
  // TODO - replace opfunc; cu_seqlens/start_pos need Tensor from DSA metadata
  xllm::kernel::CompressorParams params;
  params.x = hidden_states;
  params.wkv = cmp_wkv_;
  params.wgate = cmp_wgate_;
  params.kv_state = kv_state;
  params.score_state = score_state;
  params.ape = cmp_ape_;
  params.norm_weight = cmp_norm_;
  params.rope_sin = compressed_sin.view({-1, sin_last_dim});
  params.rope_cos = compressed_cos.view({-1, cos_last_dim});
  params.kv_block_table = c10::optional<torch::Tensor>(kv_block_table);
  params.score_block_table = c10::optional<torch::Tensor>(score_block_table);
  params.cu_seqlens = c10::optional<torch::Tensor>(actual_seq_lengths_query);
  params.seqused = c10::nullopt;
  params.start_pos = c10::optional<torch::Tensor>(attn_metadata.start_pos);
  params.rope_head_dim = rope_head_dim_;
  params.cmp_ratio = compress_ratio_;
  params.coff = enable_compressor_overlap_ ? 2 : 1;
  params.norm_eps = eps_;
  params.rotary_mode = rot_mode_;
  params.enable_grad = false;

  std::tie(compressed_kv, std::ignore, std::ignore, std::ignore, std::ignore) =
      xllm::kernel::compressor(params);
  return compressed_kv;
}

void CompressorImpl::load_state_dict(const StateDict& state_dict) {
  const auto wkv = state_dict.get_tensor("wkv.weight");
  const auto wgate = state_dict.get_tensor("wgate.weight");
  const auto norm = state_dict.get_tensor("norm.weight");
  auto ape = state_dict.get_tensor("ape");
  if (!ape.defined()) {
    ape = state_dict.get_tensor("ape.weight");
  }

  CHECK(wkv.defined()) << "missing weight: " << state_dict.prefix()
                       << "wkv.weight";
  CHECK(wgate.defined()) << "missing weight: " << state_dict.prefix()
                         << "wgate.weight";
  CHECK(norm.defined()) << "missing weight: " << state_dict.prefix()
                        << "norm.weight";
  CHECK(ape.defined()) << "missing weight: " << state_dict.prefix()
                       << "ape (or ape.weight)";

  auto coff = enable_compressor_overlap_ ? 2 : 1;
  CHECK_EQ(ape.dim(), 2) << "ape weight dim should be 2, but got " << ape.dim();
  CHECK_EQ(ape.size(0), compress_ratio_)
      << "ape weight size mismatch on dim0, expected " << compress_ratio_
      << " but got " << ape.size(0);
  CHECK_EQ(ape.size(1), coff * head_dim_)
      << "ape weight size mismatch on dim1, expected " << (coff * head_dim_)
      << " but got " << ape.size(1);

  cmp_wkv_ = wkv.to(options_);
  cmp_wgate_ = wgate.to(options_);
  cmp_norm_ = norm.to(options_);
  cmp_ape_ = ape.to(options_.dtype(torch::kFloat32));

  CHECK(cmp_wkv_.defined()) << "cmp_wkv is undefined before format cast";
  CHECK(cmp_wgate_.defined()) << "cmp_wgate is undefined before format cast";
  const bool is_npu_device =
      options_.device().type() == c10::DeviceType::PrivateUse1;

#if defined(USE_NPU)
  if (is_npu_device) {
    cmp_wkv_ = at_npu::native::npu_format_cast(cmp_wkv_, 29);
    cmp_wgate_ = at_npu::native::npu_format_cast(cmp_wgate_, 29);
  }
#else
  CHECK(!is_npu_device)
      << "Compressor weights are on NPU device, but xllm is built without "
         "USE_NPU.";
#endif

  CHECK_EQ(cmp_wkv_.device(), options_.device())
      << "cmp_wkv device mismatch, expected: " << options_.device()
      << ", actual: " << cmp_wkv_.device();
  CHECK_EQ(cmp_wgate_.device(), options_.device())
      << "cmp_wgate device mismatch, expected: " << options_.device()
      << ", actual: " << cmp_wgate_.device();
  CHECK_EQ(cmp_norm_.device(), options_.device())
      << "cmp_norm device mismatch, expected: " << options_.device()
      << ", actual: " << cmp_norm_.device();
  CHECK_EQ(cmp_ape_.device(), options_.device())
      << "cmp_ape device mismatch, expected: " << options_.device()
      << ", actual: " << cmp_ape_.device();
}

}  // namespace layer
}  // namespace xllm
