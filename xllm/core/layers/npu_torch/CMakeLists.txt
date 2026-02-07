include(cc_library)

cc_library(
  NAME
    npu_torch_layers
  HDRS
    attention.h
    deepseek_sparse_attention.h
    compressor.h
  SRCS
    attention.cpp
    deepseek_sparse_attention.cpp
    compressor.cpp
  DEPS
    :common_layers
)
