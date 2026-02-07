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
#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include <tuple>

#include "kernels/ops_api.h"
#include "xllm/core/kernels/npu/xllm_ops/xllm_ops_api.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {

CompressorImpl::CompressorImpl(int64_t compress_ratio, int64_t head_dim)
    : CompressorImpl(compress_ratio, head_dim, 64, 2, 1e-6) {}

CompressorImpl::CompressorImpl(int64_t compress_ratio,
                               int64_t head_dim,
                               int64_t rope_head_dim,
                               int64_t rot_mode,
                               double norm_eps)
    : compress_ratio_(compress_ratio),
      head_dim_(head_dim),
      rope_head_dim_(rope_head_dim),
      rot_mode_(rot_mode),
      eps_(norm_eps) {
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
  std::tie(compressed_kv, std::ignore, std::ignore, std::ignore, std::ignore) =
      xllm::kernel::npu::compressor(
          hidden_states,
          cmp_wkv_,
          cmp_wgate_,
          kv_state,
          score_state,
          cmp_ape_,
          cmp_norm_,
          compressed_sin.view({-1, sin_last_dim}),
          compressed_cos.view({-1, cos_last_dim}),
          c10::optional<torch::Tensor>(kv_block_table),
          c10::optional<torch::Tensor>(score_block_table),
          c10::optional<torch::Tensor>(actual_seq_lengths_query),  // cu_seqlens
          c10::nullopt,                                            // seqused
          c10::optional<torch::Tensor>(attn_metadata.start_pos),   // start_pos
          rope_head_dim_,
          compress_ratio_,
          enable_compressor_overlap_ ? 2 : 1,
          eps_,
          rot_mode_,
          false);
  return compressed_kv;
}

// TODO:trans weight to_device
void CompressorImpl::load_state_dict(const StateDict& state_dict) {
  // state_dict.get_tensor(tensor_name).to(device_)
  xllm::weight::load_weight(
      state_dict, "wkv.weight", cmp_wkv_, weight_is_loaded_);
  xllm::weight::load_weight(
      state_dict, "wgate.weight", cmp_wgate_, weight_is_loaded_);
  cmp_wkv_ = at_npu::native::npu_format_cast(cmp_wkv_, 29);
  cmp_wgate_ = at_npu::native::npu_format_cast(cmp_wgate_, 29);
  xllm::weight::load_weight(
      state_dict, "norm.weight", cmp_norm_, weight_is_loaded_);

  auto coff = enable_compressor_overlap_ ? 2 : 1;
  cmp_ape_ = torch::empty(
      {compress_ratio_, coff * head_dim_},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  // TODO - check if ape in cpu or npu
}

}  // namespace layer
}  // namespace xllm
