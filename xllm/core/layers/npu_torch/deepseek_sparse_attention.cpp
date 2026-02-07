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
#include "deepseek_sparse_attention.h"

#include <glog/logging.h>

#include <tuple>

#include "kernels/ops_api.h"
#include "layers/common/rotary_embedding.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {

DSAttentionImpl::DSAttentionImpl(const ModelContext& context) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  DSAttentionImpl(args, quant_args, parallel_args, options);
}

DSAttentionImpl::DSAttentionImpl(const ModelArgs& args,
                                 const QuantArgs& quant_args,
                                 const ParallelArgs& parallel_args,
                                 const torch::TensorOptions& options)
    : num_heads_(args.n_heads()),
      head_size_(args.head_dim()),
      head_dim_(args.head_dim()),
      n_kv_heads_(args.n_kv_heads().value()),
      sliding_window_(-1),
      q_lora_rank_(args.q_lora_rank()),
      o_lora_rank_(args.o_lora_rank()),
      o_groups_(args.o_groups()),
      rope_head_dim_(args.rope_head_dim()),
      window_size_(args.window_size()),
      compress_ratio_(args.compress_ratios().empty()
                          ? 1.0
                          : static_cast<double>(args.compress_ratios()[0])),
      eps_(args.rms_norm_eps()) {
  softmax_scale_ = std::pow(head_dim_, static_cast<double>(-0.5));
  scale_ = static_cast<float>(softmax_scale_);
  nope_head_dim_ = head_dim_ - rope_head_dim_;
  qk_head_dim_ = nope_head_dim_ + rope_head_dim_;

  const int64_t tp_size = parallel_args.tp_group_->world_size();
  int64_t hidden_size = args.hidden_size();
  int64_t num_heads = args.n_heads();
  int64_t max_position_embeddings = args.max_position_embeddings();

  CHECK_EQ(o_groups_ % tp_size, 0)
      << "o_groups must be divisible by tensor parallel size";
  CHECK_EQ(num_heads % tp_size, 0)
      << "num_heads must be divisible by tensor parallel size";
  n_local_heads_ = num_heads / tp_size;
  n_local_groups_ = o_groups_ / tp_size;

  // is_per_token_smoothquant_ = quant_args.quant_method() == "smoothquant";

  q_a_proj_ = register_module(
      "q_a_proj",
      ReplicatedLinear(hidden_size, q_lora_rank_, false, QuantArgs(), options));

  q_layernorm_ =
      register_module("q_a_layernorm", RMSNorm(q_lora_rank_, eps_, options));

  q_b_proj_ = register_module("q_b_proj",
                              ColumnParallelLinear(q_lora_rank_,
                                                   num_heads * head_dim_,
                                                   false,
                                                   false,
                                                   quant_args,
                                                   parallel_args.tp_group_,
                                                   options));

  kv_proj_ = register_module(
      "kv_proj",
      ReplicatedLinear(hidden_size, head_dim_, false, QuantArgs(), options));
  kv_layernorm_ =
      register_module("kv_layernorm", RMSNorm(head_dim_, eps_, options));

  if (compress_ratio_ > 1) {
    compressor_ =
        register_module("compressor",
                        Compressor(static_cast<int64_t>(compress_ratio_),
                                   head_dim_));  // TODO - ADD Indexer
    if (compress_ratio_ == 4) {
      //     self.indexer = Indexer(config, f"{prefix}.indexer",
      //     self.compress_ratio)
    }
    // else:
    //     self.indexer = None
  }
  // TODO - ADD Indexer when compress_ratio_ == 4

  o_a_proj_ =
      register_module("o_a_proj",
                      ColumnParallelLinear(num_heads * head_dim_ / o_groups_,
                                           o_groups_ * o_lora_rank_,
                                           false,
                                           true,
                                           quant_args,
                                           parallel_args.tp_group_,
                                           options));

  o_b_proj_ = register_module("o_b_proj",
                              RowParallelLinear(o_groups_ * o_lora_rank_,
                                                hidden_size,
                                                false,
                                                true,
                                                /*reduce=*/false,
                                                quant_args,
                                                parallel_args.tp_group_,
                                                options));
  // rotary_emb_ =
  //     register_module("rotary_emb",
  //                     DeepseekScalingRotaryEmbedding(
  //                         qk_rope_head_dim_,
  //                         qk_rope_head_dim_,
  //                         max_position_embeddings,
  //                         args.rope_scaling_original_max_position_embeddings(),
  //                         args.rope_theta(),
  //                         interleaved_,
  //                         args.rope_scaling_factor(),
  //                         args.rope_extrapolation_factor(),
  //                         args.rope_scaling_attn_factor(),
  //                         args.rope_scaling_beta_fast(),
  //                         args.rope_scaling_beta_slow(),
  //                         args.rope_scaling_mscale(),
  //                         args.rope_scaling_mscale_all_dim(),
  //                         options));

  // if (args.rope_scaling_rope_type() == "deepseek_yarn") {
  //   float mscale = layer::rotary::yarn_get_mscale(
  //       args.rope_scaling_factor(), args.rope_scaling_mscale_all_dim());
  //   scaling *= mscale * mscale;
  // }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
DSAttentionImpl::forward(const DSAMetadata& attn_metadata,
                         torch::Tensor& hidden_states,
                         KVCache& kv_cache,
                         KVState& kv_state,
                         bool isprefill,
                         std::string layer_name,
                         const std::tuple<torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor>& compress_metadata) {
  auto [c1_metadata, c4_metadata, c128_metadata, qli_metadata] =
      compress_metadata;

  // TODO - implement full DSA forward (indexer, compressor, attention, output
  // proj)
  // (cmp_kv) = \
    //   kv_cache
  // (ori_kv, compressor_kv_state, compressor_score_state) = \
    //   kv_state
  // (cmp_block_table, ori_block_table, kv_block_table, score_block_table) = \
    //   attn_metadata.block_tables
  // (compressed_kv_slot, ori_kv_slot, _, _) = \
    //   attn_metadata.slot_mapping

  torch::Tensor cos, sin;
  // cos = attn_metadata.cos;
  // sin = attn_metadata.sin;
  // c4_cos = attn_metadata.c4_cos;
  // c4_sin = attn_metadata.c4_sin;
  // c128_cos = attn_metadata.c128_cos;
  // c128_sin = attn_metadata.c128_sin;

  // cos = cos[layer_name]
  // sin = sin[layer_name]

  torch::Tensor output, output_lse;
  // output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

void DSAttentionImpl::load_state_dict(const StateDict& state_dict) {
  q_a_proj_->load_state_dict(state_dict.get_dict_with_prefix("wq_a."));
  q_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("wq_b."));
  q_layernorm_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));

  kv_proj_->load_state_dict(state_dict.get_dict_with_prefix("wkv."));
  kv_layernorm_->load_state_dict(state_dict.get_dict_with_prefix("kv_norm."));
  o_a_proj_->load_state_dict(state_dict.get_dict_with_prefix("wo_a."));
  o_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("wo_b."));

  if (compressor_ && compress_ratio_ >= 4) {
    compressor_->load_state_dict(
        state_dict.get_dict_with_prefix("compressor."));
  }
}

}  // namespace layer
}  // namespace xllm
