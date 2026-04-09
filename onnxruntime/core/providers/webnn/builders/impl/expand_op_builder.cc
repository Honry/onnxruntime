// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/reshape_helper.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class ExpandOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer& graph_viewer, const Node& node,
                              const emscripten::val& wnn_limits,
                              const logging::Logger& logger) const override;
};

// Add operator related.

void ExpandOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& shape_name = node.InputDefs()[1]->Name();
  // Only skip the shape input when it is a constant initializer (consumed at build time).
  // When it is an operand, we need it as the newShape input for dynamicExpand.
  if (model_builder.GetGraphViewer().GetConstantInitializer(shape_name)) {
    model_builder.AddInitializerToSkip(shape_name);
  }
}

Status ExpandOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const Node& node,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  const auto& initializers(model_builder.GetInitializerTensors());
  const bool is_constant_shape = initializers.count(input_defs[1]->Name()) > 0;

  emscripten::val output = emscripten::val::undefined();
  if (is_constant_shape) {
    // Constant shape path: compute broadcast shape at build time and use WebNN expand.
    const auto& shape_tensor = *initializers.at(input_defs[1]->Name());
    std::vector<int64_t> new_shape;
    ORT_RETURN_IF_NOT(ReadIntArrayFrom1DTensor(shape_tensor, new_shape, model_builder.GetGraphViewer(), logger),
                      "Cannot get shape.");
    std::vector<int64_t> input_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input's shape.");

    std::vector<int64_t> output_shape;
    ORT_RETURN_IF_NOT(GetBidirectionalBroadcastShape(input_shape, new_shape, output_shape),
                      "Cannot get output shape.");

    emscripten::val output_shape_arr = emscripten::val::array(GetNarrowedIntFromInt64<uint32_t>(output_shape));
    output = model_builder.GetBuilder().call<emscripten::val>("expand", input, output_shape_arr, options);
  } else {
    // Operand shape path: use dynamicExpand with the shape operand.
    emscripten::val shape_operand = model_builder.GetOperand(input_defs[1]->Name());
    output = model_builder.GetBuilder().call<emscripten::val>("dynamicExpand", input, shape_operand, options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool ExpandOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer,
                                        const Node& node,
                                        const WebnnDeviceType /* device_type */,
                                        const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& shape_name = input_defs[1]->Name();

  // When the shape input is a constant initializer, validate its contents.
  const auto* shape_init = graph_viewer.GetConstantInitializer(shape_name);
  if (shape_init) {
    const auto& shape_tensor = *shape_init;
    if (shape_tensor.data_type() != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      LOGS(logger, VERBOSE) << "The type of tensor's element data must be INT64.";
      return false;
    }

    std::vector<int64_t> new_shape;
    if (!ReadIntArrayFrom1DTensor(shape_tensor, new_shape, graph_viewer, logger)) {
      return false;
    }
    if (std::any_of(new_shape.begin(), new_shape.end(), [](int64_t dimension) { return dimension == 0; })) {
      LOGS(logger, VERBOSE) << "WebNN expand does not support new shape with 0 dimension.";
      return false;
    }

    std::vector<int64_t> input_shape;
    if (!GetShape(*input_defs[0], input_shape, logger)) {
      LOGS(logger, VERBOSE) << "Cannot get input's shape.";
      return false;
    }

    std::vector<int64_t> output_shape;
    if (!GetBidirectionalBroadcastShape(input_shape, new_shape, output_shape)) {
      LOGS(logger, VERBOSE) << "The input cannot expand to shape " << GetShapeString(new_shape);
      return false;
    }
  }
  // When shape is an operand (not a constant initializer), dynamicExpand handles
  // the shape at runtime so no static shape validation is needed.

  return true;
}

bool ExpandOpBuilder::HasSupportedInputsImpl(const GraphViewer& graph_viewer,
                                             const Node& node,
                                             const emscripten::val& wnn_limits,
                                             const logging::Logger& logger) const {
  // When shape is a constant initializer, it is consumed at build time.
  // Delegate to the base class which checks input 0 against WebNN expand's limits.
  if (graph_viewer.GetConstantInitializer(node.InputDefs()[1]->Name())) {
    return BaseOpBuilder::HasSupportedInputsImpl(graph_viewer, node, wnn_limits, logger);
  }

  // When shape is an operand, check inputs against dynamicExpand's limits.
  const auto& input_defs = node.InputDefs();
  const std::string_view webnn_op_type = "dynamicExpand";

  // Check input 0 (data tensor) against dynamicExpand's "input" parameter.
  int32_t input_type;
  if (!GetType(*input_defs[0], input_type, logger)) {
    return false;
  }
  if (!IsDataTypeSupportedByWebNNOp("Expand", webnn_op_type, input_type, wnn_limits,
                                    "input", "input", logger)) {
    return false;
  }
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger) ||
      !IsInputRankSupported(wnn_limits, webnn_op_type, "input",
                            input_shape.size(), node.Name(), logger)) {
    return false;
  }

  // Check input 1 (shape operand) against dynamicExpand's "newShape" parameter.
  int32_t shape_type;
  if (!GetType(*input_defs[1], shape_type, logger)) {
    return false;
  }
  if (!IsDataTypeSupportedByWebNNOp("Expand", webnn_op_type, shape_type, wnn_limits,
                                    "newShape", "shape", logger)) {
    return false;
  }

  return true;
}

void CreateExpandOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ExpandOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
