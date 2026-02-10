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

#include "deepseek_v4_decoder_layer.h"

#include <algorithm>

#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

DeepseekV4DecoderLayerImpl::DeepseekV4DecoderLayerImpl(
    const ModelContext& context)
    : DeepseekV4DecoderLayerImpl(context, /*layer_id=*/-1) {}

DeepseekV4DecoderLayerImpl::DeepseekV4DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();

  int64_t hidden_size = args.hidden_size();
  int64_t intermediate_size = args.intermediate_size();
  if (intermediate_size <= 0) {
    if (args.moe_intermediate_size() > 0) {
      intermediate_size = args.moe_intermediate_size();
    } else if (hidden_size > 0) {
      intermediate_size = hidden_size * 4;
    }
  }
  std::string hidden_act =
      args.hidden_act().empty() ? "silu" : args.hidden_act();

  hc_mult_ = std::max<int64_t>(args.hc_mult(), 1);
  hc_sinkhorn_iters_ = args.hc_sinkhorn_iters();
  hc_eps_ = static_cast<double>(args.hc_eps());
  norm_eps_ = static_cast<double>(args.rms_norm_eps());

  attention_ = register_module("attn", DSAttention(context, layer_id));
  attn_norm_ = register_module(
      "attn_norm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  ffn_norm_ = register_module(
      "ffn_norm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  mlp_ = register_module("ffn",
                         DenseMLP(hidden_size,
                                  intermediate_size,
                                  /*is_gated=*/true,
                                  /*has_bias=*/false,
                                  hidden_act,
                                  /*enable_result_reduction=*/true,
                                  quant_args,
                                  parallel_args.tp_group_,
                                  options));

  const int64_t mix_hc = (2 + hc_mult_) * hc_mult_;
  const int64_t hc_dim = hc_mult_ * hidden_size;
  auto hc_options = options.dtype(torch::kFloat32);
  hc_attn_fn_ = register_parameter("hc_attn_fn",
                                   torch::empty({mix_hc, hc_dim}, hc_options),
                                   /*requires_grad=*/false);
  hc_ffn_fn_ = register_parameter("hc_ffn_fn",
                                  torch::empty({mix_hc, hc_dim}, hc_options),
                                  /*requires_grad=*/false);
  hc_attn_base_ = register_parameter("hc_attn_base",
                                     torch::empty({mix_hc}, hc_options),
                                     /*requires_grad=*/false);
  hc_ffn_base_ = register_parameter("hc_ffn_base",
                                    torch::empty({mix_hc}, hc_options),
                                    /*requires_grad=*/false);
  hc_attn_scale_ = register_parameter("hc_attn_scale",
                                      torch::empty({3}, hc_options),
                                      /*requires_grad=*/false);
  hc_ffn_scale_ = register_parameter("hc_ffn_scale",
                                     torch::empty({3}, hc_options),
                                     /*requires_grad=*/false);
}

void DeepseekV4DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  auto attn_state = state_dict.get_dict_with_prefix("attn.");
  if (attn_state.size() == 0) {
    attn_state = state_dict.get_dict_with_prefix("self_attn.");
  }
  if (attn_state.size() > 0) {
    attention_->load_state_dict(attn_state);
  }

  auto attn_norm_state = state_dict.get_dict_with_prefix("attn_norm.");
  if (attn_norm_state.size() == 0) {
    attn_norm_state = state_dict.get_dict_with_prefix("input_layernorm.");
  }
  if (attn_norm_state.size() > 0) {
    attn_norm_->load_state_dict(attn_norm_state);
  }

  auto ffn_norm_state = state_dict.get_dict_with_prefix("ffn_norm.");
  if (ffn_norm_state.size() == 0) {
    ffn_norm_state =
        state_dict.get_dict_with_prefix("post_attention_layernorm.");
  }
  if (ffn_norm_state.size() > 0) {
    ffn_norm_->load_state_dict(ffn_norm_state);
  }

  auto ffn_state = state_dict.get_dict_with_prefix("ffn.");
  if (ffn_state.size() > 0) {
    mlp_->load_state_dict(ffn_state, {"w1.", "w3."}, "w2.");
  } else {
    auto mlp_state = state_dict.get_dict_with_prefix("mlp.");
    if (mlp_state.size() > 0) {
      mlp_->load_state_dict(mlp_state);
    }
  }

  LOAD_WEIGHT(hc_attn_fn);
  LOAD_WEIGHT(hc_ffn_fn);
  LOAD_WEIGHT(hc_attn_base);
  LOAD_WEIGHT(hc_ffn_base);
  LOAD_WEIGHT(hc_attn_scale);
  LOAD_WEIGHT(hc_ffn_scale);
}

torch::Tensor DeepseekV4DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  (void)positions;
  (void)input_params;

  residual = std::nullopt;

  CHECK(attn_metadata.dsa_metadata)
      << "DeepseekV4DecoderLayer requires DSA metadata for DSAttention path.";

  auto residual_attn = x;
  auto [attn_input, post_attn, comb_attn] =
      hc_pre(x, hc_attn_fn_, hc_attn_scale_, hc_attn_base_);
  attn_input = std::get<0>(attn_norm_->forward(attn_input));

  auto& dsa = *(attn_metadata.dsa_metadata);
  const auto compress_metadata = std::make_tuple(
      dsa.c1_metadata, dsa.c4_metadata, dsa.c128_metadata, dsa.qli_metadata);
  KVState kv_state{kv_cache.get_swa_cache(),
                   kv_cache.get_compress_kv_state(),
                   kv_cache.get_compress_score_state(),
                   kv_cache.get_compress_index_kv_state(),
                   kv_cache.get_compress_index_score_state()};
  auto [attn_output, attn_lse] = attention_->forward(
      dsa,
      attn_input,
      kv_cache,
      kv_state,
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill,
      std::to_string(dsa.layer_id),
      compress_metadata);
  (void)attn_lse;
  attn_input = attn_output;
  x = hc_post(attn_input, residual_attn, post_attn, comb_attn);

  auto residual_ffn = x;
  auto [ffn_input, post_ffn, comb_ffn] =
      hc_pre(x, hc_ffn_fn_, hc_ffn_scale_, hc_ffn_base_);
  ffn_input = std::get<0>(ffn_norm_->forward(ffn_input));
  ffn_input = mlp_->forward(ffn_input);
  x = hc_post(ffn_input, residual_ffn, post_ffn, comb_ffn);

  return x;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
DeepseekV4DecoderLayerImpl::hc_pre(const torch::Tensor& x,
                                   const torch::Tensor& hc_fn,
                                   const torch::Tensor& hc_scale,
                                   const torch::Tensor& hc_base) {
  kernel::HcPreParams params;
  params.x = x;
  params.hc_fn = hc_fn;
  params.hc_scale = hc_scale;
  params.hc_base = hc_base;
  params.hc_mult = hc_mult_;
  params.hc_sinkhorn_iters = hc_sinkhorn_iters_;
  params.norm_eps = norm_eps_;
  params.hc_eps = hc_eps_;
  return kernel::hc_pre(params);
}

torch::Tensor DeepseekV4DecoderLayerImpl::hc_post(const torch::Tensor& x,
                                                  const torch::Tensor& residual,
                                                  const torch::Tensor& post,
                                                  const torch::Tensor& comb) {
  kernel::HcPostParams params;
  if (x.dim() == 2 && residual.dim() == 3 && post.dim() == 2 &&
      comb.dim() == 3) {
    params.x = x.unsqueeze(0);
    params.residual = residual.unsqueeze(0);
    params.post = post.unsqueeze(0);
    params.comb = comb.unsqueeze(0);
    return kernel::hc_post(params).squeeze(0);
  }

  params.x = x;
  params.residual = residual;
  params.post = post;
  params.comb = comb;
  return kernel::hc_post(params);
}

}  // namespace layer
}  // namespace xllm
