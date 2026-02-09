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

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <unordered_map>

#include "core/framework/state_dict/utils.h"
#include "core/layers/common/dsa_metadata.h"
#include "core/layers/common/dsa_metadata_builder.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/deepseek_v4_decoder_layer.h"
#include "layers/npu/deepseek_v4_rotary_embedding.h"
#include "llm_model_base.h"
#include "xllm/core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm {

inline int64_t deepseek_v4_next_power_of_two(int64_t n) {
  int64_t value = 1;
  while (value < n) {
    value <<= 1;
  }
  return value;
}

inline torch::Tensor deepseek_v4_create_hadamard_matrix(
    int64_t n,
    torch::ScalarType dtype,
    const torch::Device& device) {
  auto options = torch::TensorOptions().dtype(dtype).device(device);
  torch::Tensor matrix = torch::ones({1, 1}, options);
  for (int64_t m = 1; m < n; m <<= 1) {
    auto top = torch::cat({matrix, matrix}, 1);
    auto bottom = torch::cat({matrix, -matrix}, 1);
    matrix = torch::cat({top, bottom}, 0);
  }
  return matrix;
}

// Group key: (ratio, type, block_size) -> group_id
struct DSAGroupKey {
  int32_t ratio;
  DSACacheType type;
  int32_t block_size;
  bool operator==(const DSAGroupKey& o) const {
    return ratio == o.ratio && type == o.type && block_size == o.block_size;
  }
};

struct DSAGroupKeyHash {
  size_t operator()(const DSAGroupKey& k) const {
    size_t h = std::hash<int32_t>()(k.ratio);
    h ^= std::hash<int32_t>()(static_cast<int32_t>(k.type)) << 16;
    h ^= std::hash<int32_t>()(k.block_size) << 8;
    return h;
  }
};

class DeepseekV4ModelImpl
    : public LlmModelImplBase<layer::DeepseekV4DecoderLayer> {
 public:
  explicit DeepseekV4ModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::DeepseekV4DecoderLayer>(
            "deepseek_v4",
            context.get_model_args()) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));

    hc_mult_ = std::max<int64_t>(model_args.hc_mult(), 1);
    hc_eps_ = static_cast<double>(model_args.hc_eps());
    norm_eps_ = static_cast<double>(model_args.rms_norm_eps());

    num_heads_ = model_args.n_heads();
    head_dim_ = model_args.head_dim();
    window_size_ =
        model_args.window_size() > 0 ? model_args.window_size() : 128;
    index_n_heads_ = model_args.index_n_heads();
    index_head_dim_ = model_args.index_head_dim();
    index_topk_ = model_args.index_topk() > 0 ? model_args.index_topk() : 512;

    const int64_t hc_dim = hc_mult_ * model_args.hidden_size();
    auto hc_options = options.dtype(torch::kFloat32);
    hc_head_fn_ =
        register_parameter("hc_head_fn",
                           torch::empty({hc_mult_, hc_dim}, hc_options),
                           /*requires_grad=*/false);
    hc_head_base_ = register_parameter("hc_head_base",
                                       torch::empty({hc_mult_}, hc_options),
                                       /*requires_grad=*/false);
    hc_head_scale_ = register_parameter("hc_head_scale",
                                        torch::empty({1}, hc_options),
                                        /*requires_grad=*/false);

    const int64_t rope_head_dim = model_args.rope_head_dim();
    const int64_t max_pos = model_args.max_position_embeddings();
    if (rope_head_dim > 0 && max_pos > 0) {
      const int64_t original_max_pos =
          model_args.rope_scaling_original_max_position_embeddings() > 0
              ? model_args.rope_scaling_original_max_position_embeddings()
              : max_pos;
      const float scaling_factor =
          model_args.factor() > 0.0f ? model_args.factor() : 1.0f;
      const float attn_factor = model_args.rope_scaling_attn_factor() > 0.0f
                                    ? model_args.rope_scaling_attn_factor()
                                    : 1.0f;
      // MindIE alignment: hardcode RoPE theta for DS4.0 DSA path.
      const float rope_theta = 10000.0f;
      const float compress_rope_theta = 40000.0f;
      dsa_rotary_embedding_ =
          std::make_shared<layer::DeepseekV4RotaryEmbedding>(
              /*rotary_dim=*/rope_head_dim,
              /*max_position_embeddings=*/max_pos,
              /*interleaved=*/false,
              /*rope_theta=*/rope_theta,
              /*compress_rope_theta=*/compress_rope_theta,
              /*scaling_factor=*/scaling_factor,
              /*extrapolation_factor=*/model_args.rope_extrapolation_factor(),
              /*beta_fast=*/model_args.beta_fast(),
              /*beta_slow=*/model_args.beta_slow(),
              /*attn_factor=*/attn_factor,
              /*mscale=*/model_args.rope_scaling_mscale(),
              /*mscale_all_dim=*/model_args.rope_scaling_mscale_all_dim(),
              /*original_max_position_embeddings=*/original_max_pos,
              options);
      dsa_cos_sin_ = dsa_rotary_embedding_->get_cos_sin_cache("default");
    }

    if (model_args.index_head_dim() > 0) {
      auto hadamard_dim_padded =
          deepseek_v4_next_power_of_two(model_args.index_head_dim());
      dsa_hadamard_ =
          deepseek_v4_create_hadamard_matrix(hadamard_dim_padded,
                                             options.dtype().toScalarType(),
                                             options.device());
    }

    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto layer = layer::DeepseekV4DecoderLayer(context);
      layers_.push_back(layer);
    }

    // Build DSA caches_info from compress_ratios
    const auto& compress_ratios = model_args.compress_ratios();
    const int32_t window_size =
        model_args.window_size() > 0 ? model_args.window_size() : 128;
    const int32_t base_block_size = 128;  // default block size

    std::unordered_map<DSAGroupKey, int32_t, DSAGroupKeyHash> group_key_map;
    caches_info_.resize(model_args.n_layers());

    for (int32_t layer_id = 0; layer_id < model_args.n_layers(); ++layer_id) {
      int32_t cr = (layer_id < static_cast<int32_t>(compress_ratios.size()))
                       ? compress_ratios[layer_id]
                       : 1;
      // Build per-layer cache specs based on compress_ratio
      struct CacheEntry {
        DSACacheType type;
        int32_t ratio;
        int32_t block_size;
      };
      std::vector<CacheEntry> layer_caches;

      if (cr == 1) {
        // C1: 1 cache (swa)
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
      } else if (cr == 4) {
        // C4: 8 caches
        // compress_kv(TOKEN,4,128), compress_index(TOKEN,4,128),
        // swa(SW,1,window), kv_state(SW,1,window), score_state(SW,1,window),
        // idx_kv_state(SW,1,window), idx_score_state(SW,1,window),
        // indexer_scale(TOKEN,4,128)
        layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
        layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::TOKEN, 4, base_block_size});
      } else if (cr == 128) {
        // C128: 4 caches
        // compress_kv(TOKEN,128,128), swa(SW,1,window),
        // kv_state(SW,1,window), score_state(SW,1,window)
        layer_caches.push_back({DSACacheType::TOKEN, 128, base_block_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
        layer_caches.push_back({DSACacheType::SLIDING_WINDOW, 1, window_size});
      }

      for (const auto& ce : layer_caches) {
        DSAGroupKey gk{ce.ratio, ce.type, ce.block_size};
        int32_t gid;
        auto it = group_key_map.find(gk);
        if (it == group_key_map.end()) {
          gid = static_cast<int32_t>(group_infos_.size());
          group_key_map[gk] = gid;
          group_infos_.push_back({ce.type, ce.ratio, ce.block_size});
        } else {
          gid = it->second;
        }
        caches_info_[layer_id].push_back(
            {gid, ce.type, ce.ratio, ce.block_size});
      }
    }
  }

  void load_state_dict(const StateDict& state_dict) override {
    LlmModelImplBase<layer::DeepseekV4DecoderLayer>::load_state_dict(
        state_dict);
    LOAD_WEIGHT(hc_head_fn);
    LOAD_WEIGHT(hc_head_base);
    LOAD_WEIGHT(hc_head_scale);
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
    }

    auto inputs_embeds = input_params.input_embedding;
    torch::Tensor h =
        inputs_embeds.defined() ? inputs_embeds : embed_tokens_(tokens);

    if (h.dim() == 2) {
      h = h.unsqueeze(1).repeat({1, hc_mult_, 1});
    }

    auto modified_input_params = input_params;
    auto& dp_token_nums = modified_input_params.dp_global_token_nums;
    // DP helper: keep zero entries at least 1 to avoid empty slices/padding
    // in xllm DP utilities. DeepSeek V4 not use DP today.
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);

    if (!modified_input_params.attn_metadata) {
      modified_input_params.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::DSAMetadataBuilder::build(modified_input_params,
                                               positions,
                                               dsa_cos_sin_,
                                               caches_info_,
                                               group_infos_));
    }
    auto& attn_metadata = *(modified_input_params.attn_metadata);

    if (attn_metadata.dsa_metadata) {
      auto& dsa = *(attn_metadata.dsa_metadata);

      if (dsa_hadamard_.defined()) {
        dsa.hadamard = dsa_hadamard_;
      }

      if (dsa_rotary_embedding_) {
        std::unordered_map<std::string, torch::Tensor> positions_map;

        auto append_group_positions = [&positions_map](
                                          const std::string& group,
                                          const torch::Tensor& positions) {
          if (!positions.defined() || positions.numel() == 0) {
            return;
          }
          auto group_positions = positions;
          if (group_positions.scalar_type() != torch::kInt64) {
            group_positions = group_positions.to(torch::kInt64);
          }
          positions_map[group] = group_positions;
        };

        append_group_positions("default", dsa.input_positions);
        append_group_positions("c4", dsa.c4_pad_positions);
        append_group_positions("c128", dsa.c128_pad_positions);

        if (!positions_map.empty()) {
          auto group_cos_sin = dsa_rotary_embedding_->build(positions_map);

          auto default_it = group_cos_sin.find("default");
          if (default_it != group_cos_sin.end()) {
            dsa.cos = default_it->second.first;
            dsa.sin = default_it->second.second;
          }

          auto c4_it = group_cos_sin.find("c4");
          if (c4_it != group_cos_sin.end()) {
            dsa.c4_cos = c4_it->second.first;
            dsa.c4_sin = c4_it->second.second;
          }

          auto c128_it = group_cos_sin.find("c128");
          if (c128_it != group_cos_sin.end()) {
            dsa.c128_cos = c128_it->second.first;
            dsa.c128_sin = c128_it->second.second;
          }
        }
      }

      if (dsa.actual_seq_lengths_kv.defined() && dsa.seq_lens_q.defined()) {
        dsa.start_pos =
            (dsa.actual_seq_lengths_kv - dsa.seq_lens_q).to(torch::kInt32);
      }

      build_precomputed_metadata(dsa);
    }

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      if (attn_metadata.dsa_metadata) {
        auto& dsa = *(attn_metadata.dsa_metadata);
        const int32_t layer_id = static_cast<int32_t>(i);
        dsa.layer_id = layer_id;

        if (layer_id < static_cast<int32_t>(dsa.block_tables.size()) &&
            layer_id < static_cast<int32_t>(dsa.slot_mappings.size()) &&
            !dsa.block_tables[layer_id].empty() &&
            !dsa.slot_mappings[layer_id].empty()) {
          size_t attn_cache_idx = 0;
          if (layer_id < static_cast<int32_t>(caches_info_.size())) {
            const auto& layer_caches = caches_info_[layer_id];
            for (size_t cache_idx = 0; cache_idx < layer_caches.size();
                 ++cache_idx) {
              if (layer_caches[cache_idx].type ==
                  DSACacheType::SLIDING_WINDOW) {
                attn_cache_idx = cache_idx;
                break;
              }
            }
          }

          if (attn_cache_idx < dsa.block_tables[layer_id].size() &&
              dsa.block_tables[layer_id][attn_cache_idx].defined()) {
            attn_metadata.block_table =
                dsa.block_tables[layer_id][attn_cache_idx];
          }
          if (attn_cache_idx < dsa.slot_mappings[layer_id].size() &&
              dsa.slot_mappings[layer_id][attn_cache_idx].defined()) {
            attn_metadata.slot_mapping =
                dsa.slot_mappings[layer_id][attn_cache_idx];
          }
        }
      }

      h = layers_[i](h,
                     residual,
                     positions,
                     attn_metadata,
                     kv_caches[i],
                     modified_input_params);
    }
    h = hc_head(h);
    auto [hidden_states, residual_out] = norm_(h, std::nullopt);
    return ModelOutput(hidden_states, residual_out);
  }

 private:
  static c10::optional<torch::Tensor> as_optional_tensor(
      const torch::Tensor& tensor) {
    if (tensor.defined() && tensor.numel() > 0) {
      return c10::optional<torch::Tensor>(tensor);
    }
    return c10::nullopt;
  }

  static int64_t tensor_max_or_zero(const torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.numel() == 0) {
      return 0;
    }
    return tensor.max().item<int64_t>();
  }

  static int64_t pick_max_seqlen(const torch::Tensor& max_seqlen_tensor,
                                 const torch::Tensor& fallback_tensor) {
    if (max_seqlen_tensor.defined() && max_seqlen_tensor.numel() > 0) {
      return max_seqlen_tensor.max().item<int64_t>();
    }
    return tensor_max_or_zero(fallback_tensor);
  }

  void build_precomputed_metadata(layer::DSAMetadata& dsa) const {
    dsa.c1_metadata = torch::Tensor();
    dsa.c4_metadata = torch::Tensor();
    dsa.c128_metadata = torch::Tensor();
    dsa.qli_metadata = torch::Tensor();

    if (!dsa.actual_seq_lengths_query.defined() ||
        !dsa.actual_seq_lengths_kv.defined()) {
      return;
    }

    const int64_t batch_size =
        std::max<int64_t>(dsa.actual_seq_lengths_kv.size(0), 1);
    const int64_t max_seqlen_q =
        pick_max_seqlen(dsa.max_seqlen_q, dsa.seq_lens_q);
    const int64_t max_seqlen_kv =
        pick_max_seqlen(dsa.max_seqlen_kv, dsa.actual_seq_lengths_kv);
    const int64_t ori_win_left = std::max<int64_t>(window_size_ - 1, 0);
    const int64_t sparse_topk = std::max<int64_t>(index_topk_, 1);

    dsa.c1_metadata = xllm::kernel::npu::sparse_attn_sharedkv_metadata(
        /*num_heads_q=*/num_heads_,
        /*num_heads_kv=*/1,
        /*head_dim=*/head_dim_,
        as_optional_tensor(dsa.actual_seq_lengths_query),
        as_optional_tensor(dsa.actual_seq_lengths_kv),
        /*batch_size=*/batch_size,
        /*max_seqlen_q=*/max_seqlen_q,
        /*max_seqlen_kv=*/max_seqlen_kv,
        /*topk=*/0,
        /*cmp_ratio=*/1,
        /*ori_mask_mode=*/4,
        /*cmp_mask_mode=*/3,
        /*ori_win_left=*/ori_win_left,
        /*ori_win_right=*/0,
        /*layout_q=*/"TND",
        /*layout_kv=*/"PA_ND",
        /*has_ori_kv=*/true,
        /*has_cmp_kv=*/false);

    dsa.c4_metadata = xllm::kernel::npu::sparse_attn_sharedkv_metadata(
        /*num_heads_q=*/num_heads_,
        /*num_heads_kv=*/1,
        /*head_dim=*/head_dim_,
        as_optional_tensor(dsa.actual_seq_lengths_query),
        as_optional_tensor(dsa.actual_seq_lengths_kv),
        /*batch_size=*/batch_size,
        /*max_seqlen_q=*/max_seqlen_q,
        /*max_seqlen_kv=*/max_seqlen_kv,
        /*topk=*/sparse_topk,
        /*cmp_ratio=*/4,
        /*ori_mask_mode=*/4,
        /*cmp_mask_mode=*/3,
        /*ori_win_left=*/ori_win_left,
        /*ori_win_right=*/0,
        /*layout_q=*/"TND",
        /*layout_kv=*/"PA_ND",
        /*has_ori_kv=*/true,
        /*has_cmp_kv=*/true);

    dsa.c128_metadata = xllm::kernel::npu::sparse_attn_sharedkv_metadata(
        /*num_heads_q=*/num_heads_,
        /*num_heads_kv=*/1,
        /*head_dim=*/head_dim_,
        as_optional_tensor(dsa.actual_seq_lengths_query),
        as_optional_tensor(dsa.actual_seq_lengths_kv),
        /*batch_size=*/batch_size,
        /*max_seqlen_q=*/max_seqlen_q,
        /*max_seqlen_kv=*/max_seqlen_kv,
        /*topk=*/0,
        /*cmp_ratio=*/128,
        /*ori_mask_mode=*/4,
        /*cmp_mask_mode=*/3,
        /*ori_win_left=*/ori_win_left,
        /*ori_win_right=*/0,
        /*layout_q=*/"TND",
        /*layout_kv=*/"PA_ND",
        /*has_ori_kv=*/true,
        /*has_cmp_kv=*/true);

    torch::Tensor query_lens;
    if (dsa.actual_seq_lengths_query.defined() &&
        dsa.actual_seq_lengths_query.dim() > 0 &&
        dsa.actual_seq_lengths_query.size(0) > 1) {
      query_lens = dsa.actual_seq_lengths_query.slice(
          /*dim=*/0,
          /*start=*/1,
          /*end=*/dsa.actual_seq_lengths_query.size(0));
    } else if (dsa.seq_lens_q.defined()) {
      query_lens = dsa.seq_lens_q;
    }

    torch::Tensor key_lens;
    if (dsa.seq_lens.defined()) {
      key_lens = dsa.seq_lens;
    } else if (dsa.actual_seq_lengths_kv.defined()) {
      key_lens = dsa.actual_seq_lengths_kv;
    }

    if (!query_lens.defined() || !key_lens.defined() ||
        query_lens.numel() == 0 || key_lens.numel() == 0) {
      return;
    }

    const int64_t index_num_heads =
        std::max<int64_t>(index_n_heads_ > 0 ? index_n_heads_ : num_heads_, 1);
    const int64_t index_head_dim =
        std::max<int64_t>(index_head_dim_ > 0 ? index_head_dim_ : head_dim_, 1);
    const int64_t qli_batch_size = std::max<int64_t>(key_lens.size(0), 1);
    const int64_t qli_max_seqlen_q =
        pick_max_seqlen(dsa.max_seqlen_q, query_lens);
    const int64_t qli_max_seqlen_k =
        pick_max_seqlen(dsa.max_seqlen_kv, key_lens);

    dsa.qli_metadata = xllm::kernel::npu::quant_lightning_indexer_metadata(
        /*num_heads_q=*/index_num_heads,
        /*num_heads_k=*/1,
        /*head_dim=*/index_head_dim,
        /*query_quant_mode=*/0,
        /*key_quant_mode=*/0,
        as_optional_tensor(query_lens),
        as_optional_tensor(key_lens),
        /*batch_size=*/qli_batch_size,
        /*max_seqlen_q=*/qli_max_seqlen_q,
        /*max_seqlen_k=*/qli_max_seqlen_k,
        /*layout_query=*/"TND",
        /*layout_key=*/"PA_BSND",
        /*sparse_count=*/sparse_topk,
        /*sparse_mode=*/3,
        /*pre_tokens=*/std::numeric_limits<int64_t>::max(),
        /*next_tokens=*/std::numeric_limits<int64_t>::max(),
        /*cmp_ratio=*/4,
        /*device=*/query_lens.device().str());
  }

  torch::Tensor hc_head(const torch::Tensor& x) {
    auto x_float = x.to(torch::kFloat32);
    auto x_flatten = x_float.flatten(-2, -1);
    auto rsqrt = torch::rsqrt(x_flatten.pow(2).mean(-1, true) + norm_eps_);
    auto mixes = torch::matmul(x_flatten, hc_head_fn_.transpose(0, 1));
    mixes = mixes * rsqrt;
    auto pre = torch::sigmoid(mixes * hc_head_scale_ + hc_head_base_) + hc_eps_;
    auto y = (pre.unsqueeze(-1) * x_float).sum(-2);
    return y.to(x.dtype());
  }

  torch::Tensor dsa_cos_sin_;
  torch::Tensor dsa_hadamard_;
  std::shared_ptr<layer::DeepseekV4RotaryEmbedding> dsa_rotary_embedding_;

  int64_t hc_mult_ = 1;
  double hc_eps_ = 0.0;
  double norm_eps_ = 1e-6;

  int64_t num_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t window_size_ = 128;
  int64_t index_n_heads_ = 0;
  int64_t index_head_dim_ = 0;
  int64_t index_topk_ = 512;

  // DSA cache group info: built once at model init from compress_ratios
  // caches_info_[layer_id] = vector of DSACacheInfo for each cache in that
  // layer
  std::vector<std::vector<DSACacheInfo>> caches_info_;
  // group_infos_[group_id] = DSAGroupInfo
  std::vector<DSAGroupInfo> group_infos_;

  DEFINE_WEIGHT(hc_head_fn);
  DEFINE_WEIGHT(hc_head_base);
  DEFINE_WEIGHT(hc_head_scale);
};
TORCH_MODULE(DeepseekV4Model);

class DeepseekV4ForCausalLMImpl
    : public LlmForCausalLMImplBase<DeepseekV4Model> {
 public:
  explicit DeepseekV4ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV4Model>(context) {}
};
TORCH_MODULE(DeepseekV4ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(deepseek_v4, DeepseekV4ForCausalLM);

// register the model args
REGISTER_MODEL_ARGS(deepseek_v4, [&] {
  LOAD_ARG_OR(model_type, "model_type", "deepseek_v4");
  LOAD_ARG_OR(dtype, "torch_dtype", "");

  // Basic model structure
  LOAD_ARG_OR_FUNC(hidden_size, "dim", [&] { return args->hidden_size(); });
  LOAD_ARG_OR_FUNC(
      hidden_size, "hidden_size", [&] { return args->hidden_size(); });
  LOAD_ARG_OR_FUNC(
      n_layers, "num_hidden_layers", [&] { return args->n_layers(); });
  LOAD_ARG_OR_FUNC(n_heads, "n_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR_FUNC(
      n_heads, "num_attention_heads", [&] { return args->n_heads(); });
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 1);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    if (args->head_dim() > 0) {
      return args->head_dim();
    }
    if (args->hidden_size() > 0 && args->n_heads() > 0) {
      return args->hidden_size() / args->n_heads();
    }
    return int64_t{0};
  });
  LOAD_ARG_OR_FUNC(
      vocab_size, "vocab_size", [&] { return args->vocab_size(); });
  LOAD_ARG_OR_FUNC(max_position_embeddings, "max_position_embeddings", [&] {
    return args->max_position_embeddings();
  });
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR_FUNC(intermediate_size, "intermediate_size", [&] {
    if (args->intermediate_size() > 0) {
      return args->intermediate_size();
    }
    if (args->moe_intermediate_size() > 0) {
      return static_cast<int64_t>(args->moe_intermediate_size());
    }
    if (args->hidden_size() > 0) {
      return args->hidden_size() * 4;
    }
    return int64_t{0};
  });

  // Norm / RoPE
  LOAD_ARG_OR_FUNC(
      rms_norm_eps, "norm_eps", [&] { return args->rms_norm_eps(); });
  LOAD_ARG_OR_FUNC(
      rms_norm_eps, "rms_norm_eps", [&] { return args->rms_norm_eps(); });
  LOAD_ARG_OR_FUNC(
      rope_theta, "rope_theta", [&] { return args->rope_theta(); });
  LOAD_ARG_OR_FUNC(
      rope_head_dim, "rope_head_dim", [&] { return args->rope_head_dim(); });

  // LoRA / groups
  LOAD_ARG_OR_FUNC(
      q_lora_rank, "q_lora_rank", [&] { return args->q_lora_rank(); });
  LOAD_ARG_OR_FUNC(
      o_lora_rank, "o_lora_rank", [&] { return args->o_lora_rank(); });
  LOAD_ARG_OR_FUNC(o_groups, "o_groups", [&] { return args->o_groups(); });

  // KV compression / windowing
  LOAD_ARG(compress_ratios, "compress_ratios");
  LOAD_ARG_OR_FUNC(compress_rope_theta, "compress_rope_theta", [&] {
    return args->compress_rope_theta();
  });
  LOAD_ARG_OR_FUNC(
      window_size, "window_size", [&] { return args->window_size(); });

  // MoE routing (DeepSeek V4)
  LOAD_ARG_OR_FUNC(n_routed_experts, "n_routed_experts", [&] {
    return args->n_routed_experts();
  });
  LOAD_ARG_OR_FUNC(n_activated_experts, "n_activated_experts", [&] {
    return args->n_activated_experts();
  });
  LOAD_ARG_OR_FUNC(
      n_hash_layers, "n_hash_layers", [&] { return args->n_hash_layers(); });
  LOAD_ARG_OR_FUNC(
      route_scale, "route_scale", [&] { return args->route_scale(); });
  LOAD_ARG_OR_FUNC(
      score_func, "score_func", [&] { return args->score_func(); });

  // Indexer
  LOAD_ARG_OR_FUNC(
      index_head_dim, "index_head_dim", [&] { return args->index_head_dim(); });
  LOAD_ARG_OR_FUNC(
      index_n_heads, "index_n_heads", [&] { return args->index_n_heads(); });
  LOAD_ARG_OR_FUNC(
      index_topk, "index_topk", [&] { return args->index_topk(); });

  // HC / DSA helpers
  LOAD_ARG_OR_FUNC(hc_mult, "hc_mult", [&] { return args->hc_mult(); });
  LOAD_ARG_OR_FUNC(hc_sinkhorn_iters, "hc_sinkhorn_iters", [&] {
    return args->hc_sinkhorn_iters();
  });
  LOAD_ARG_OR_FUNC(hc_eps, "hc_eps", [&] { return args->hc_eps(); });
  LOAD_ARG_OR_FUNC(factor, "factor", [&] { return args->factor(); });
  LOAD_ARG_OR_FUNC(beta_fast, "beta_fast", [&] { return args->beta_fast(); });
  LOAD_ARG_OR_FUNC(beta_slow, "beta_slow", [&] { return args->beta_slow(); });
  LOAD_ARG_OR_FUNC(scale_fmt, "scale_fmt", [&] { return args->scale_fmt(); });

  // Runtime sizing hints
  LOAD_ARG_OR_FUNC(
      max_batch_size, "max_batch_size", [&] { return args->max_batch_size(); });
  LOAD_ARG_OR_FUNC(
      max_seq_len, "max_seq_len", [&] { return args->max_seq_len(); });

  // Token ids
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm
