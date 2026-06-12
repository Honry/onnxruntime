// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

// Common utilities for building WebNN dynamic shape subgraphs.
// These helpers construct shape operand chains (shape → slice → concat → dynamicReshape)
// that are shared across multiple op builders (Flatten, Squeeze, Unsqueeze, etc.).

#pragma once

#include <emscripten.h>
#include <emscripten/val.h>

#include "core/providers/webnn/builders/model_builder.h"

namespace onnxruntime {
namespace webnn {
namespace shape_utils {

// Get a constant scalar [1] operand matching the shape op's output data type.
// WebNN shape() outputs int64 when supported, int32 otherwise.
// This is commonly needed for Unsqueeze (inserting size-1 dims) and Flatten (empty product).
inline emscripten::val GetShapeConstantOne(ModelBuilder& model_builder) {
  if (model_builder.IsInt64Supported()) {
    return model_builder.CreateOrGetConstant<int64_t>(
        ONNX_NAMESPACE::TensorProto_DataType_INT64, 1, {1});
  } else {
    return model_builder.CreateOrGetConstant<int32_t>(
        ONNX_NAMESPACE::TensorProto_DataType_INT32, 1, {1});
  }
}

// Slice a contiguous range [start, start+size) from a 1-D shape operand.
// The shape_operand is typically the output of builder.shape(input).
inline emscripten::val SliceShapeRange(const emscripten::val& wnn_builder,
                                       const emscripten::val& shape_operand,
                                       int32_t start, int32_t size) {
  emscripten::val starts = emscripten::val::array();
  starts.call<void>("push", start);
  emscripten::val sizes = emscripten::val::array();
  sizes.call<void>("push", size);
  return wnn_builder.call<emscripten::val>("slice", shape_operand, starts, sizes);
}

// Compute the product of a contiguous range of dims from a shape operand.
// Returns a 1-D operand of length 1 suitable for concat.
// - If size == 0: returns constant [1] (empty product).
// - If size == 1: returns a slice of that single dim.
// - If size > 1: slices the range and reduces with product (keepDimensions=true).
inline emscripten::val ReduceShapeRange(ModelBuilder& model_builder,
                                        const emscripten::val& shape_operand,
                                        int32_t start, int32_t size,
                                        const std::string& label) {
  emscripten::val wnn_builder = model_builder.GetBuilder();

  if (size == 0) {
    return GetShapeConstantOne(model_builder);
  }
  if (size == 1) {
    return SliceShapeRange(wnn_builder, shape_operand, start, 1);
  }
  // Multiple dims — slice then reduce with product.
  emscripten::val segment = SliceShapeRange(wnn_builder, shape_operand, start, size);

  emscripten::val reduce_options = emscripten::val::object();
  reduce_options.set("label", label);
  reduce_options.set("keepDimensions", true);
  emscripten::val axes_array = emscripten::val::array();
  axes_array.call<void>("push", 0);
  reduce_options.set("axes", axes_array);
  return wnn_builder.call<emscripten::val>("reduceProduct", segment, reduce_options);
}

// Build a dynamic reshape: concat segments into a 1-D target shape, then call dynamicReshape.
// segments: a JS array of 1-D operands to concat into the target shape.
// Returns the dynamicReshape output operand.
inline emscripten::val DynamicReshapeWithSegments(ModelBuilder& model_builder,
                                                  const emscripten::val& input,
                                                  const emscripten::val& segments,
                                                  const std::string& node_name) {
  emscripten::val wnn_builder = model_builder.GetBuilder();

  emscripten::val concat_options = emscripten::val::object();
  concat_options.set("label", node_name + "_shape_concat");
  emscripten::val target_shape = wnn_builder.call<emscripten::val>(
      "concat", segments, 0, concat_options);

  emscripten::val options = emscripten::val::object();
  options.set("label", node_name);
  return wnn_builder.call<emscripten::val>("dynamicReshape", input, target_shape, options);
}

}  // namespace shape_utils
}  // namespace webnn
}  // namespace onnxruntime
