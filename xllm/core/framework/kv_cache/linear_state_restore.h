/* Copyright 2025-2026 The xLLM Authors.

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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"

namespace xllm {

// Convert logical sequence KV cursors to the warm/cold state mask consumed by
// active linear-attention rows. Data-parallel expansion keeps each logical
// row contiguous in the execution batch.
LinearStateValidityMask build_linear_state_mask(
    const std::vector<int32_t>& cached_tokens,
    int64_t active_rows);

// Materialize the worker-produced linear-state validity result for a backend
// kernel that consumes a device bool tensor. Empty data-parallel shards have
// one synthetic, always-cold dummy metadata row.
inline torch::Tensor materialize_linear_state_mask(
    const LinearStateValidityMask& validity_mask,
    int64_t metadata_rows,
    bool is_dummy,
    const torch::Device& device) {
  const int64_t mask_rows = static_cast<int64_t>(validity_mask.size());
  if (is_dummy && mask_rows == 0 && metadata_rows == 1) {
    return torch::zeros({1}, torch::dtype(torch::kBool).device(device));
  }
  CHECK_EQ(mask_rows, metadata_rows)
      << "linear state mask row count mismatch: mask_rows=" << mask_rows
      << ", metadata_rows=" << metadata_rows;
  return torch::tensor(validity_mask,
                       torch::dtype(torch::kBool).device(device));
}

// Apply each sequence's recurrent-state initialization plan in place. Fresh
// sequences clear their live slot, restored sequences copy a resolved
// checkpoint into it, and continued sequences preserve it. The caller derives
// `validity_mask` from context lengths before this call; this function
// corrects entries after clear/copy operations have been enqueued.
//
// Slot 0 is reserved for null/padding rows and is never a valid live or source
// slot. A malformed restore descriptor or cache layout is fatal: continuing
// with a reused full-attention KV prefix but missing recurrent state would
// produce semantically inconsistent output.
//
// Operations run on the current stream. The caller must preserve stream order
// before model forward consumes the restored slots.
void restore_linear_state_slots(
    std::vector<KVCache>& kv_caches,
    const std::vector<LinearStateCacheOp>& cache_ops,
    LinearStateValidityMask& validity_mask);

}  // namespace xllm
