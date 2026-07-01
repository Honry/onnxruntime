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
                                       int32_t start, int32_t size,
                                       const std::string& label) {
  emscripten::val starts = emscripten::val::array();
  starts.call<void>("push", start);
  emscripten::val sizes = emscripten::val::array();
  sizes.call<void>("push", size);
  emscripten::val options = emscripten::val::object();
  if (!label.empty()) {
    options.set("label", label);
  }
  return wnn_builder.call<emscripten::val>("slice", shape_operand, starts, sizes, options);
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
    return SliceShapeRange(wnn_builder, shape_operand, start, 1, label + "_slice");
  }
  // Multiple dims — slice then reduce with product.
  emscripten::val segment = SliceShapeRange(wnn_builder, shape_operand, start, size, label + "_slice");

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
  // The int64 chain enables reshapeFusion in the ORT backend.
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
  emscripten::val result = wnn_builder.call<emscripten::val>("concat", segments, 0, common_options);

  // Step 5: Cast back to uint32 for dynamicReshape (requires uint32 shape operand).
  if (is_int64) {
    common_options.set("label", label + "_cast_uint32");
    result = wnn_builder.call<emscripten::val>(
        "cast", result, emscripten::val("uint32"), common_options);
  }
  return result;
}

// Reshape with automatic static/dynamic dispatch.
// If input_shape is all-static, resolves target_dims to concrete values and calls reshape.
// If input_shape has dynamic dims, calls ComputeShape + dynamicReshape.
//
// target_dims convention: 0 = copy from input at same position, -1 = infer, >0 = static value.
inline emscripten::val Reshape(ModelBuilder& model_builder,
                               const emscripten::val& input,
                               const std::vector<int64_t>& input_shape,
                               const std::vector<int64_t>& target_dims,
                               const std::string& label) {
  emscripten::val wnn_builder = model_builder.GetBuilder();
  emscripten::val options = emscripten::val::object();
  options.set("label", label);

  // Check if all referenced input dims are positive (static and non-zero).
  // target_dims 0 means "copy from input[i]" — need that dim to be known and positive.
  bool can_use_static = !HasDynamicShape(input_shape);
  if (can_use_static) {
    for (size_t i = 0; i < target_dims.size(); ++i) {
      if (target_dims[i] == 0 && (i >= input_shape.size() || input_shape[i] <= 0)) {
        can_use_static = false;
        break;
      }
    }
  }

  if (can_use_static) {
    // Static path: resolve 0/-1 to concrete values.
    std::vector<uint32_t> concrete_shape;
    concrete_shape.reserve(target_dims.size());
    int64_t product_known = 1;
    int64_t total_elements = 1;
    for (auto d : input_shape) total_elements *= d;
    for (auto d : target_dims) {
      if (d > 0) product_known *= d;
    }
    for (size_t i = 0; i < target_dims.size(); ++i) {
      if (target_dims[i] == 0) {
        concrete_shape.push_back(SafeInt<uint32_t>(input_shape[i]));
      } else if (target_dims[i] == -1) {
        int64_t product_with_zeros = product_known;
        for (size_t j = 0; j < target_dims.size(); ++j) {
          if (target_dims[j] == 0 && j < input_shape.size()) {
            product_with_zeros *= input_shape[j];
          }
        }
        concrete_shape.push_back(SafeInt<uint32_t>(total_elements / product_with_zeros));
      } else {
        concrete_shape.push_back(SafeInt<uint32_t>(target_dims[i]));
      }
    }
    return wnn_builder.call<emscripten::val>(
        "reshape", input, emscripten::val::array(concrete_shape), options);
  } else {
    // Dynamic path: ComputeShape + dynamicReshape.
    emscripten::val shape_operand = ComputeShape(model_builder, input, target_dims, label);
    return wnn_builder.call<emscripten::val>("dynamicReshape", input, shape_operand, options);
  }
}

// Get the input's runtime shape as a 1-D operand cast to the specified signed type.
// WebNN shape() returns uint32; this casts to the target type for arithmetic compatibility.
// use_int64: true → cast to int64, false → cast to int32.
inline emscripten::val GetShapeInWorkingType(ModelBuilder& model_builder,
                                             const emscripten::val& input,
                                             bool use_int64,
                                             const std::string& label) {
  emscripten::val wnn_builder = model_builder.GetBuilder();
  const std::string type_str = use_int64 ? "int64" : "int32";

  emscripten::val options = emscripten::val::object();
  options.set("label", label + "_shape");
  emscripten::val shape_op = wnn_builder.call<emscripten::val>("shape", input, options);
  options.set("label", label + "_shape_cast");
  return wnn_builder.call<emscripten::val>("cast", shape_op, emscripten::val(type_str), options);
}

// Normalize negative indices by wrapping relative to dim_sizes, then clamp to [0, dim_size].
// Handles both negative indices (ONNX relative-to-end semantics) and out-of-bounds values
// like INT_MAX (ONNX "to end of axis" convention in Slice).
//
// All operands (indices, dim_sizes, zero_const) must be the same signed data type.
// use_int64: true if operands are int64, false if int32.
// Returns the normalized and clamped indices in the same type.
inline emscripten::val NormalizeAndClampIndices(ModelBuilder& model_builder,
                                               const emscripten::val& indices,
                                               const emscripten::val& dim_sizes,
                                               bool use_int64,
                                               uint32_t length,
                                               const std::string& label) {
  emscripten::val wnn_builder = model_builder.GetBuilder();
  const int32_t data_type = use_int64 ? ONNX_NAMESPACE::TensorProto_DataType_INT64
                                      : ONNX_NAMESPACE::TensorProto_DataType_INT32;

  const emscripten::val zero_const = use_int64
      ? model_builder.CreateOrGetConstant<int64_t>(data_type, int64_t{0},
            std::vector<uint32_t>{length})
      : model_builder.CreateOrGetConstant<int32_t>(data_type, int32_t{0},
            std::vector<uint32_t>{length});

  emscripten::val options = emscripten::val::object();

  // Wrap negative: val = where(val < 0, val + dim_size, val)
  options.set("label", label + "_is_neg");
  emscripten::val is_neg = wnn_builder.call<emscripten::val>(
      "lesser", indices, zero_const, options);
  options.set("label", label + "_add_dim");
  emscripten::val wrapped = wnn_builder.call<emscripten::val>(
      "add", indices, dim_sizes, options);
  options.set("label", label + "_wrap");
  emscripten::val normalized = wnn_builder.call<emscripten::val>(
      "where", is_neg, wrapped, indices, options);

  // Clamp to [0, dim_size]: min(dim_sizes), then max(0)
  options.set("label", label + "_clamp_max");
  normalized = wnn_builder.call<emscripten::val>("min", normalized, dim_sizes, options);
  options.set("label", label + "_clamp_min");
  return wnn_builder.call<emscripten::val>("max", normalized, zero_const, options);
}

// Resolve ONNX -1/0 semantics in a runtime shape operand for dynamicReshape.
//
// When Reshape's shape input is a non-constant operand (e.g., from an unfused Concat),
// it may contain -1 (infer dimension) or 0 (copy from input). WebNN's dynamicReshape
// requires all positive uint32 values, so we insert sub-ops to resolve these at runtime:
//
//   0 → replaced with the corresponding dimension from shape(input)
//  -1 → replaced with total_elements / product(other_dims)
//
// The shape_operand is expected to be int64 (ONNX's Reshape shape type).
// input_rank and output_rank are known at build time from ONNX shape info.
// Returns a uint32 operand suitable for dynamicReshape.
inline emscripten::val ResolveReshapeShape(ModelBuilder& model_builder,
                                           const emscripten::val& input,
                                           const emscripten::val& shape_operand,
                                           uint32_t input_rank,
                                           uint32_t output_rank,
                                           const std::string& label) {
  emscripten::val wnn_builder = model_builder.GetBuilder();
  const bool is_int64 = model_builder.IsInt64Supported();
  const int32_t working_type = is_int64 ? ONNX_NAMESPACE::TensorProto_DataType_INT64
                                        : ONNX_NAMESPACE::TensorProto_DataType_INT32;

  emscripten::val shape_op = shape_operand;
  emscripten::val options = emscripten::val::object();

  // Step 1: Get input's runtime shape in working type.
  emscripten::val input_shape_typed = GetShapeInWorkingType(model_builder, input, is_int64, label);

  // Step 2: Resolve 0 → copy from input shape at same position.
  // ONNX 0 means "copy input_shape[i]". When output_rank != input_rank, we pad/slice
  // input_shape to match output_rank so `where` operands have compatible shapes.
  emscripten::val input_shape_aligned = input_shape_typed;
  if (input_rank != output_rank) {
    if (input_rank > output_rank) {
      // Slice input_shape to output_rank (only first output_rank dims are relevant for 0-copy).
      options.set("label", label + "_input_shape_slice");
      input_shape_aligned = wnn_builder.call<emscripten::val>(
          "slice", input_shape_typed,
          emscripten::val::array(std::vector<uint32_t>{0}),
          emscripten::val::array(std::vector<uint32_t>{output_rank}),
          options);
    } else {
      // Pad input_shape with 1s to output_rank (positions beyond input_rank won't have 0
      // in valid ONNX models, but padding ensures no shape mismatch in `where`).
      uint32_t pad_size = output_rank - input_rank;
      const emscripten::val pad_ones = is_int64
          ? model_builder.CreateOrGetConstant<int64_t>(working_type, label + "_pad_ones",
                std::vector<int64_t>(pad_size, 1), {pad_size})
          : model_builder.CreateOrGetConstant<int32_t>(working_type, label + "_pad_ones",
                std::vector<int32_t>(pad_size, 1), {pad_size});
      options.set("label", label + "_input_shape_pad");
      emscripten::val pad_segments = emscripten::val::array();
      pad_segments.call<void>("push", input_shape_typed);
      pad_segments.call<void>("push", pad_ones);
      input_shape_aligned = wnn_builder.call<emscripten::val>(
          "concat", pad_segments, 0, options);
    }
  }

  const emscripten::val zero_const = is_int64
      ? model_builder.CreateOrGetConstant<int64_t>(working_type, int64_t{0}, {})
      : model_builder.CreateOrGetConstant<int32_t>(working_type, int32_t{0}, {});
  options.set("label", label + "_is_zero");
  emscripten::val is_zero = wnn_builder.call<emscripten::val>(
      "equal", shape_op, zero_const, options);
  options.set("label", label + "_resolve_zero");
  emscripten::val shape_no_zero = wnn_builder.call<emscripten::val>(
      "where", is_zero, input_shape_aligned, shape_op, options);

  // Step 3: Resolve -1 → infer = total_elements / product(other_dims).
  const emscripten::val neg1_const = is_int64
      ? model_builder.CreateOrGetConstant<int64_t>(working_type, int64_t{-1}, {})
      : model_builder.CreateOrGetConstant<int32_t>(working_type, int32_t{-1}, {});
  options.set("label", label + "_is_neg1");
  emscripten::val is_neg1 = wnn_builder.call<emscripten::val>(
      "equal", shape_no_zero, neg1_const, options);

  // Replace -1 with 1 for product computation.
  const emscripten::val one_const = is_int64
      ? model_builder.CreateOrGetConstant<int64_t>(working_type, int64_t{1}, {})
      : model_builder.CreateOrGetConstant<int32_t>(working_type, int32_t{1}, {});
  options.set("label", label + "_shape_for_product");
  emscripten::val shape_for_product = wnn_builder.call<emscripten::val>(
      "where", is_neg1, one_const, shape_no_zero, options);

  // product_other = reduceProduct(shape_for_product)
  emscripten::val reduce_options = emscripten::val::object();
  reduce_options.set("label", label + "_product_other");
  reduce_options.set("axes", emscripten::val::array(std::vector<uint32_t>{0}));
  emscripten::val product_other = wnn_builder.call<emscripten::val>(
      "reduceProduct", shape_for_product, reduce_options);

  // total_elements = reduceProduct(input_shape_typed)
  reduce_options.set("label", label + "_total_elements");
  emscripten::val total_elements = wnn_builder.call<emscripten::val>(
      "reduceProduct", input_shape_typed, reduce_options);

  // inferred_dim = total_elements / product_other
  options.set("label", label + "_inferred_div");
  emscripten::val inferred_dim = wnn_builder.call<emscripten::val>(
      "div", total_elements, product_other, options);

  // Replace -1 positions with the inferred value.
  options.set("label", label + "_resolve_neg1");
  emscripten::val shape_resolved = wnn_builder.call<emscripten::val>(
      "where", is_neg1, inferred_dim, shape_no_zero, options);

  // Step 4: Cast to uint32 for dynamicReshape.
  options.set("label", label + "_cast_uint32");
  return wnn_builder.call<emscripten::val>(
      "cast", shape_resolved, emscripten::val("uint32"), options);
}

}  // namespace shape_utils
}  // namespace webnn
}  // namespace onnxruntime
