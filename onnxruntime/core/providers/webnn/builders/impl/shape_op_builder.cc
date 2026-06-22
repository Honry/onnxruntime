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

class ShapeOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
};

Status ShapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                             const Node& node,
                                             const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());

  // Use WebNN shape() to get the shape of the input tensor as a 1-D uint32 tensor.
  // Cast to int64 if supported (ONNX Shape output is int64; this also enables reshapeFusion
  // in the ORT backend which expects int64 shape tensors).
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("shape", input, options);

  if (model_builder.IsInt64Supported()) {
    emscripten::val cast_options = emscripten::val::object();
    cast_options.set("label", node.Name() + "_cast");
    output = model_builder.GetBuilder().call<emscripten::val>(
        "cast", output, emscripten::val("int64"), cast_options);
  }

  // Handle start/end attributes (opset 15+).
  // Only need to slice when start or end is explicitly specified.
  NodeAttrHelper helper(node);
  const auto start = helper.GetInt64("start");
  const auto end = helper.GetInt64("end");

  if (start.has_value() || end.has_value()) {
    const auto rank = static_cast<int64_t>(input_defs[0]->Shape()->dim_size());
    auto true_start = start.value_or(0);
    auto true_end = end.value_or(rank);

    // Normalize negative values and clamp.
    true_start = std::clamp(true_start + (true_start < 0 ? rank : 0), int64_t{0}, rank);
    true_end = std::clamp(true_end + (true_end < 0 ? rank : 0), true_start, rank);

    if (!(true_start == 0 && true_end == rank)) {
      // Use slice to extract the subset [start, end).
      auto slice_length = true_end - true_start;
      emscripten::val starts = emscripten::val::array();
      starts.call<void>("push", static_cast<int32_t>(true_start));
      emscripten::val sizes = emscripten::val::array();
      sizes.call<void>("push", static_cast<int32_t>(slice_length));

      emscripten::val slice_options = emscripten::val::object();
      slice_options.set("label", node.Name() + "_slice");
      output = model_builder.GetBuilder().call<emscripten::val>("slice", output, starts, sizes, slice_options);
    }
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

void CreateShapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ShapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
