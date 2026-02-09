include(cc_library)

cc_library(
  NAME
    npu_torch_layers
  HDRS
    attention.h
    deepseek_sparse_attention.h
    compressor.h
    deepseek_v4_indexer.h
  SRCS
    attention.cpp
    deepseek_sparse_attention.cpp
    compressor.cpp
    deepseek_v4_indexer.cpp
  DEPS
    :common_layers
    :xllm_ops
)

add_subdirectory(tests)
