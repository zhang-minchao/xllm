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

#pragma once

#include <torch/torch.h>

#include <tuple>

namespace xllm::kernel::npu {

void beam_search(const torch::Tensor& logprobs,
                 const torch::Tensor& top_tokens,
                 const torch::Tensor& top_logprobs,
                 torch::Tensor& src_seq_idxes,
                 torch::Tensor& out_logprobs,
                 torch::Tensor& out_token_ids);

void top_k_top_p(torch::Tensor& logits,
                 const torch::Tensor& topK,
                 const torch::Tensor& topP);

void replace_token(torch::Tensor& dst, torch::Tensor& src);

at::Tensor hc_post(const at::Tensor& x,
                   const at::Tensor& residual,
                   const at::Tensor& post,
                   const at::Tensor& comb);

std::tuple<at::Tensor, at::Tensor> quant_lightning_indexer(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& weights,
    const at::Tensor& query_dequant_scale,
    const at::Tensor& key_dequant_scale,
    int64_t query_quant_mode,
    int64_t key_quant_mode,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_key,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& metadata,
    c10::string_view layout_query,
    c10::string_view layout_key,
    int64_t sparse_count,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t cmp_ratio,
    bool return_value);
at::Tensor hc_pre_inv_rms(const at::Tensor& x, double epsilon = 1e-6);

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k_hash(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& bias = c10::nullopt,
    const c10::optional<at::Tensor>& input_ids = c10::nullopt,
    const c10::optional<at::Tensor>& tid2eid = c10::nullopt,
    int64_t k = 1,
    int64_t k_group = 1,
    int64_t group_count = 1,
    int64_t group_select_mode = 0,
    int64_t renorm = 0,
    int64_t norm_type = 0,
    bool out_flag = false,
    double routed_scaling_factor = 1.0,
    double eps = 1e-20);

std::tuple<at::Tensor, at::Tensor> sparse_attn_sharedkv(
    const at::Tensor& q,
    const c10::optional<at::Tensor>& ori_kv = c10::nullopt,
    const c10::optional<at::Tensor>& cmp_kv = c10::nullopt,
    const c10::optional<at::Tensor>& ori_sparse_indices = c10::nullopt,
    const c10::optional<at::Tensor>& cmp_sparse_indices = c10::nullopt,
    const c10::optional<at::Tensor>& ori_block_table = c10::nullopt,
    const c10::optional<at::Tensor>& cmp_block_table = c10::nullopt,
    const c10::optional<at::Tensor>& cu_seqlens_q = c10::nullopt,
    const c10::optional<at::Tensor>& cu_seqlens_ori_kv = c10::nullopt,
    const c10::optional<at::Tensor>& cu_seqlens_cmp_kv = c10::nullopt,
    const c10::optional<at::Tensor>& seqused_q = c10::nullopt,
    const c10::optional<at::Tensor>& seqused_kv = c10::nullopt,
    const c10::optional<at::Tensor>& sinks = c10::nullopt,
    const c10::optional<at::Tensor>& metadata = c10::nullopt,
    double softmax_scale = 1.0,
    int64_t cmp_ratio = 1,
    int64_t ori_mask_mode = 3,
    int64_t cmp_mask_mode = 3,
    int64_t ori_win_left = 128,
    int64_t ori_win_right = 0,
    const char* layout_q = "BSND",
    const char* layout_kv = "PA_ND",
    bool return_softmax_lse = false);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
compressor(const at::Tensor& x,
           const at::Tensor& wkv,
           const at::Tensor& wgate,
           at::Tensor& kv_state,
           at::Tensor& score_state,
           const at::Tensor& ape,
           const at::Tensor& norm_weight,
           const at::Tensor& rope_sin,
           const at::Tensor& rope_cos,
           const c10::optional<at::Tensor>& kv_block_table,
           const c10::optional<at::Tensor>& score_block_table,
           const c10::optional<at::Tensor>& cu_seqlens,
           const c10::optional<at::Tensor>& seqused,
           const c10::optional<at::Tensor>& start_pos,
           int64_t rope_head_dim,
           int64_t cmp_ratio,
           int64_t coff,
           double norm_eps,
           int64_t rotary_mode,
           bool enable_grad);

}  // namespace xllm::kernel::npu
