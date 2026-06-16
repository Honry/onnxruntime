// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

// This contains the utility functions which will be used to build a webnn model

#pragma once

#include "core/common/status.h"
#include "core/graph/basic_types.h"
#include <emscripten/val.h>

namespace onnxruntime {
namespace webnn {

class ModelBuilder;  // Forward declaration.

// Try to see if we can map explicit padding to auto padding for Conv/Pool.
// Since usually use auto padding is more efficient.
common::Status HandleAutoPad(const std::vector<int64_t> input_shape,
                             const int64_t weight_size_y,
                             const int64_t weight_size_x,
                             const std::vector<int64_t>& onnx_pads,
                             const std::vector<int64_t>& onnx_strides,
                             const std::vector<int64_t>& onnx_dilations,
                             AutoPadType auto_pad_type,
                             std::vector<int64_t>& pads_out) ORT_MUST_USE_RESULT;

// Compute pads and output shape for ConvTranspose.
common::Status ComputeConvTransposePadsAndOutputShape(const std::vector<int64_t> input_shape,
                                                      const int64_t weight_size_y,
                                                      const int64_t weight_size_x,
                                                      const std::vector<int64_t>& onnx_pads,
                                                      const std::vector<int64_t>& onnx_strides,
                                                      const std::vector<int64_t>& onnx_dilations,
                                                      const std::vector<int64_t>& onnx_output_padding,
                                                      AutoPadType auto_pad_type,
                                                      std::vector<int64_t>& pads_out,
                                                      std::vector<int64_t>& output_shape_out) ORT_MUST_USE_RESULT;

// Compute SAME_UPPER/SAME_LOWER padding dynamically using WebNN graph ops and apply dynamicPad.
// Used when spatial dimensions are dynamic and explicit padding cannot be pre-computed.
// Returns the padded input operand.
emscripten::val ComputeDynamicSamePadding(ModelBuilder& model_builder,
                                          const emscripten::val& input,
                                          const std::vector<int64_t>& input_shape,
                                          int64_t kernel_h, int64_t kernel_w,
                                          const std::vector<int64_t>& strides,
                                          AutoPadType auto_pad_type,
                                          const std::string& node_name);

}  // namespace webnn
}  // namespace onnxruntime
