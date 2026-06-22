// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"
#include "shape_utils.h"

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

  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  const int64_t rank = input_shape.size();

  NodeAttrHelper helper(node);
  int64_t raw_axis = helper.Get("axis", 1);
  ORT_RETURN_IF(raw_axis < -rank || raw_axis > rank,
                "Flatten: axis ", raw_axis, " is out of range [-", rank, ", ", rank, "]");
  const uint32_t axis = static_cast<uint32_t>(raw_axis < 0 ? raw_axis + rank : raw_axis);

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val output = emscripten::val::undefined();

  const emscripten::val& wnn_limits = model_builder.GetOpSupportLimits();
  if (!wnn_limits["flatten"].isUndefined()) {
    emscripten::val flatten_options = emscripten::val::object();
    flatten_options.set("axis", axis);
    flatten_options.set("label", node.Name());
    output = model_builder.GetBuilder().call<emscripten::val>("flatten", input, flatten_options);
  } else if (!HasDynamicShape(input_shape)) {
    // Fallback: static reshape to [pre_axis_product, post_axis_product].
    int64_t pre = std::accumulate(
        input_shape.begin(), input_shape.begin() + axis, int64_t{1}, std::multiplies<int64_t>());
    int64_t post = std::accumulate(
        input_shape.begin() + axis, input_shape.end(), int64_t{1}, std::multiplies<int64_t>());
    std::vector<uint32_t> new_shape{SafeInt<uint32_t>(pre), SafeInt<uint32_t>(post)};
    emscripten::val options = emscripten::val::object();
    options.set("label", node.Name());
    output = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", input, emscripten::val::array(new_shape), options);
  } else {
    // Fallback: dynamic reshape via ReduceShapeRange.
    emscripten::val wnn_builder = model_builder.GetBuilder();
    emscripten::val shape_options = emscripten::val::object();
    shape_options.set("label", node.Name() + "_shape");
    emscripten::val shape_operand = wnn_builder.call<emscripten::val>("shape", input, shape_options);

    emscripten::val pre_product = shape_utils::ReduceShapeRange(
        model_builder, shape_operand, 0, static_cast<int32_t>(axis), node.Name() + "_pre_reduce");
    emscripten::val post_product = shape_utils::ReduceShapeRange(
        model_builder, shape_operand, static_cast<int32_t>(axis),
        static_cast<int32_t>(rank) - static_cast<int32_t>(axis), node.Name() + "_post_reduce");

    emscripten::val segments = emscripten::val::array();
    segments.call<void>("push", pre_product);
    segments.call<void>("push", post_product);
    output = shape_utils::DynamicReshapeWithSegments(model_builder, input, segments, node.Name());
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
