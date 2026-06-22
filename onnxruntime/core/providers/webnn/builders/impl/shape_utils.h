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


// Builds a 1-D shape operand for dynamicReshape from a target shape specification
// that may include 0 (copy from input) and -1 (infer).
//
// Emitted sub-op patterns per dimension (matching reshapeFusion's recognized patterns):
//   target_dims[i] == 0:   Shape → Gather(i) → Unsqueeze([0])          → fuses to 0
//   target_dims[i] == -1:  Div(total_elements, known_product) → Unsqueeze([0])  → fuses to -1
//   target_dims[i] > 0:    Constant([value])                            → fuses to value
//
// reshapeFusion recognizes:
//   Pattern 1: Shape → Gather(i) → Unsqueeze(axes=0) → Concat  →  fused as 0
//   Pattern 2: one-element subgraph with Div/Mul → Unsqueeze(axes=0) → Concat  →  fused as -1
//   Constant: literal value in concat input  →  fused as the value
//
// Data type:
//   WebNN shape() always returns uint32.
//   - IsInt64Supported: cast shape to int64, all constants int64.
//   - !IsInt64Supported: keep everything uint32, no cast.
//
// Parameters:
//   input:       the WebNN operand whose shape is queried (the reshape input).
//   target_dims: 0 = copy from input at same position,
//                -1 = infer (one per shape),
//                positive = static dim value.
//   label:       node name for labeling intermediate ops.
inline emscripten::val ComputeShape(ModelBuilder& model_builder,
                                    const emscripten::val& input,
                                    const std::vector<int64_t>& target_dims,
                                    const std::string& label) {
  emscripten::val wnn_builder = model_builder.GetBuilder();
  const bool is_int64 = model_builder.IsInt64Supported();
  const int32_t data_type = is_int64 ? ONNX_NAMESPACE::TensorProto_DataType_INT64
                                     : ONNX_NAMESPACE::TensorProto_DataType_UINT32;

  // Step 1: Get runtime shape of input (uint32), cast to int64 if needed.
  emscripten::val common_options = emscripten::val::object();
  common_options.set("label", label + "_shape");
  emscripten::val shape_op = wnn_builder.call<emscripten::val>("shape", input, common_options);
  if (is_int64) {
    common_options.set("label", label + "_cast");
    shape_op = wnn_builder.call<emscripten::val>(
        "cast", shape_op, emscripten::val("int64"), common_options);
  }

  // Step 2: If -1 is present, precompute total_elements and known_product for inference.
  //   inferred_dim = total_elements / known_product
  //   where known_product = product of all non-(-1) target dims at runtime.
  emscripten::val inferred_dim = emscripten::val::undefined();
  bool has_infer = std::find(target_dims.begin(), target_dims.end(), -1) != target_dims.end();
  if (has_infer) {
    // total_elements = reduceProduct(shape(input))
    emscripten::val reduce_options = emscripten::val::object();
    reduce_options.set("label", label + "_total_reduce");
    reduce_options.set("keepDimensions", true);
    reduce_options.set("axes", emscripten::val::array(std::vector<uint32_t>{0}));
    emscripten::val total = wnn_builder.call<emscripten::val>(
        "reduceProduct", shape_op, reduce_options);

    // known_product = product of all other target dims (built from their segments).
    // If there are no other dims (target is just [-1]), inferred_dim = total directly.
    size_t non_infer_count = std::count_if(target_dims.begin(), target_dims.end(),
                                           [](int64_t d) { return d != -1; });
    if (non_infer_count == 0) {
      // Only -1 in target: inferred_dim = total_elements (known_product = 1).
      inferred_dim = total;
    } else {
      emscripten::val known_segments = emscripten::val::array();
      for (size_t i = 0; i < target_dims.size(); ++i) {
        if (target_dims[i] == -1) continue;
        if (target_dims[i] == 0) {
          const emscripten::val& idx = is_int64
              ? model_builder.CreateOrGetConstant<int64_t>(data_type, static_cast<int64_t>(i), {})
              : model_builder.CreateOrGetConstant<uint32_t>(data_type, static_cast<uint32_t>(i), {});
          common_options.set("label", label + "_known_gather_" + std::to_string(i));
          emscripten::val dim_scalar = wnn_builder.call<emscripten::val>(
              "gather", shape_op, idx, common_options);
          emscripten::val unsqueeze_options = emscripten::val::object();
          unsqueeze_options.set("label", label + "_known_unsqueeze_" + std::to_string(i));
          emscripten::val axes_arr = emscripten::val::array();
          axes_arr.call<void>("push", 0u);
          emscripten::val dim_1d = wnn_builder.call<emscripten::val>(
              "unsqueeze", dim_scalar, axes_arr, unsqueeze_options);
          known_segments.call<void>("push", dim_1d);
        } else {
          const emscripten::val& c = is_int64
              ? model_builder.CreateOrGetConstant<int64_t>(data_type, target_dims[i], {1})
              : model_builder.CreateOrGetConstant<uint32_t>(data_type, static_cast<uint32_t>(target_dims[i]), {1});
          known_segments.call<void>("push", c);
        }
      }
      common_options.set("label", label + "_known_concat");
      emscripten::val known_shape = wnn_builder.call<emscripten::val>(
          "concat", known_segments, 0, common_options);

      reduce_options.set("label", label + "_known_reduce");
      emscripten::val known_product = wnn_builder.call<emscripten::val>(
          "reduceProduct", known_shape, reduce_options);

      // inferred_dim = total / known_product (scalar [1]-shaped)
      common_options.set("label", label + "_infer_div");
      inferred_dim = wnn_builder.call<emscripten::val>("div", total, known_product, common_options);
    }
  }

  // Step 3: Build per-dim segments for the final shape operand.
  emscripten::val segments = emscripten::val::array();
  for (size_t i = 0; i < target_dims.size(); ++i) {
    if (target_dims[i] == 0) {
      // Shape → Gather(i) → Unsqueeze([0])
      const std::string i_str = std::to_string(i);
      const emscripten::val& idx = is_int64
          ? model_builder.CreateOrGetConstant<int64_t>(data_type, static_cast<int64_t>(i), {})
          : model_builder.CreateOrGetConstant<uint32_t>(data_type, static_cast<uint32_t>(i), {});

      common_options.set("label", label + "_gather_" + i_str);
      emscripten::val dim_scalar = wnn_builder.call<emscripten::val>(
          "gather", shape_op, idx, common_options);

      emscripten::val unsqueeze_options = emscripten::val::object();
      unsqueeze_options.set("label", label + "_unsqueeze_" + i_str);
      emscripten::val axes_arr = emscripten::val::array();
      axes_arr.call<void>("push", 0u);
      emscripten::val dim_1d = wnn_builder.call<emscripten::val>(
          "unsqueeze", dim_scalar, axes_arr, unsqueeze_options);
      segments.call<void>("push", dim_1d);
    } else if (target_dims[i] == -1) {
      // Div(total, known_product) already computed → [1]-shaped, push directly.
      segments.call<void>("push", inferred_dim);
    } else {
      // Static constant [value].
      const emscripten::val& c = is_int64
          ? model_builder.CreateOrGetConstant<int64_t>(data_type, target_dims[i], {1})
          : model_builder.CreateOrGetConstant<uint32_t>(data_type, static_cast<uint32_t>(target_dims[i]), {1});
      segments.call<void>("push", c);
    }
  }

  // Step 4: Concat all segments into final 1-D shape operand.
  common_options.set("label", label + "_concat");
  return wnn_builder.call<emscripten::val>("concat", segments, 0, common_options);
}

}  // namespace shape_utils
}  // namespace webnn
}  // namespace onnxruntime
