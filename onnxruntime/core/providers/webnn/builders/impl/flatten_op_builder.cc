// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class FlattenOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
};

// Add operator related.

Status FlattenOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  ORT_RETURN_IF(input_defs.size() < 1, "Flatten has no input tensor");

  // Get rank from shape proto (always available, validated by base class).
  const auto* shape_proto = input_defs[0]->Shape();
  ORT_RETURN_IF(!shape_proto, "Flatten input has no shape proto");
  const int64_t rank = shape_proto->dim_size();

  NodeAttrHelper helper(node);
  int64_t axis = helper.Get("axis", 1);
  ORT_ENFORCE(axis >= -rank && axis <= rank, "axis ", axis,
              " is not in valid range [-", rank, ",", rank, "]");
  if (axis < 0) {
    axis += rank;
  }

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  // Check if input has any dynamic dimensions.
  bool has_dynamic_shape = false;
  for (const auto& dim : shape_proto->dim()) {
    if (!dim.has_dim_value()) {
      has_dynamic_shape = true;
      break;
    }
  }

  emscripten::val output = emscripten::val::undefined();
  if (!has_dynamic_shape) {
    // === Static path: compute new_shape at build time and use WebNN reshape. ===
    std::vector<int64_t> input_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");

    int64_t num_pre_axis_elements = std::accumulate(
        input_shape.begin(), input_shape.begin() + static_cast<int32_t>(axis), 1, std::multiplies<int64_t>());
    int64_t num_post_axis_elements = std::accumulate(
        input_shape.begin() + static_cast<int32_t>(axis), input_shape.end(), 1, std::multiplies<int64_t>());

    std::vector<uint32_t> new_shape = {SafeInt<uint32_t>(num_pre_axis_elements),
                                       SafeInt<uint32_t>(num_post_axis_elements)};
    output = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", input, emscripten::val::array(new_shape), options);
  } else {
    // === Dynamic path: use shape() + slice/reduceMul + concat + dynamicReshape. ===
    // Flatten(input, axis) produces shape [product(dims[:axis]), product(dims[axis:])].
    // We use WebNN shape() to get the runtime shape, slice into pre/post segments,
    // reduce each with mul, then concat into [pre, post] for dynamicReshape.
    emscripten::val wnn_builder = model_builder.GetBuilder();
    emscripten::val shape_operand = wnn_builder.call<emscripten::val>(
        "shape", input, emscripten::val::object());

    // Helper lambda to compute the product of a contiguous range of shape dims.
    // Returns a scalar operand (reshaped to [1] for concat).
    auto reduce_shape_range = [&](int32_t start, int32_t size,
                                  const std::string& label_suffix) -> emscripten::val {
      if (size == 0) {
        // Empty range means product is 1 (e.g., axis=0 → pre-axis is empty).
        if (model_builder.IsInt64Supported()) {
          return model_builder.CreateOrGetConstant<int64_t>(
              ONNX_NAMESPACE::TensorProto_DataType_INT64, 1, {1});
        } else {
          return model_builder.CreateOrGetConstant<int32_t>(
              ONNX_NAMESPACE::TensorProto_DataType_INT32, 1, {1});
        }
      }
      if (size == 1) {
        // Single dim — just slice it out.
        emscripten::val starts = emscripten::val::array();
        starts.call<void>("push", start);
        emscripten::val sizes = emscripten::val::array();
        sizes.call<void>("push", 1);
        return wnn_builder.call<emscripten::val>("slice", shape_operand, starts, sizes);
      }
      // Multiple dims — slice then reduce with mul.
      emscripten::val starts = emscripten::val::array();
      starts.call<void>("push", start);
      emscripten::val sizes = emscripten::val::array();
      sizes.call<void>("push", size);
      emscripten::val segment = wnn_builder.call<emscripten::val>(
          "slice", shape_operand, starts, sizes);

      emscripten::val reduce_options = emscripten::val::object();
      reduce_options.set("label", node.Name() + label_suffix);
      reduce_options.set("keepDimensions", true);
      emscripten::val axes_array = emscripten::val::array();
      axes_array.call<void>("push", 0);
      reduce_options.set("axes", axes_array);
      return wnn_builder.call<emscripten::val>("reduceProduct", segment, reduce_options);
    };

    int32_t axis32 = static_cast<int32_t>(axis);
    int32_t rank32 = static_cast<int32_t>(rank);

    emscripten::val pre_product = reduce_shape_range(0, axis32, "_pre_reduce");
    emscripten::val post_product = reduce_shape_range(axis32, rank32 - axis32, "_post_reduce");

    // Concat [pre_product, post_product] into a 1-D shape tensor of length 2.
    emscripten::val concat_inputs = emscripten::val::array();
    concat_inputs.call<void>("push", pre_product);
    concat_inputs.call<void>("push", post_product);
    emscripten::val concat_options = emscripten::val::object();
    concat_options.set("label", node.Name() + std::string("_concat_shape"));
    emscripten::val new_shape_operand = wnn_builder.call<emscripten::val>(
        "concat", concat_inputs, 0, concat_options);

    output = wnn_builder.call<emscripten::val>("dynamicReshape", input, new_shape_operand, options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

void CreateFlattenOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<FlattenOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
