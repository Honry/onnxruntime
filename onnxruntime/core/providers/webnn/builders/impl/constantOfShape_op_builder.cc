// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class ConstantOfShapeOpBuilder : public BaseOpBuilder {
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
  // The output is validated in HasSupportedInputsImpl: it needs to know whether the shape is a constant (built with
  // expand) or an operand (built with expandDynamic).
  bool HasSupportedInputsImpl(const GraphViewer& graph_viewer, const Node& node, const emscripten::val& wnn_limits,
                              const logging::Logger& logger) const override;
  bool HasSupportedOutputsImpl(const Node& /*node*/, const emscripten::val& /*wnn_limits*/,
                               const logging::Logger& /*logger*/) const override { return true; }
};

// Add operator related.

void ConstantOfShapeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& shape_name = node.InputDefs()[0]->Name();
  // Only skip the shape input when it is a constant initializer (consumed at build time).
  // When it is an operand, we need it as the newShape input for expandDynamic.
  if (model_builder.GetGraphViewer().GetConstantInitializer(shape_name)) {
    model_builder.AddInitializerToSkip(shape_name);
    model_builder.AddInputToSkip(shape_name);
  }
}

Status ConstantOfShapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                       const Node& node,
                                                       const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& initializers(model_builder.GetInitializerTensors());

  // Check if the shape input is a constant initializer.
  const bool is_constant_shape = initializers.count(input_defs[0]->Name()) > 0;

  // The 'value' attribute is a 1-element TensorProto, default is float32 0. Build it as a scalar constant.
  emscripten::val output = emscripten::val::undefined();
  const auto& attrs = node.GetAttributes();
  auto value_it = attrs.find("value");
  if (value_it != attrs.end()) {
    const auto& value_tensor = value_it->second.t();
    emscripten::val scalar_shape = emscripten::val::array();
    emscripten::val desc = emscripten::val::object();
    desc.set("shape", scalar_shape);
    desc.set("dimensions", scalar_shape);
    ORT_RETURN_IF_NOT(SetWebnnDataType(desc, value_tensor.data_type()),
                      "WebNN backend does not support data type: ", value_tensor.data_type());
    ORT_RETURN_IF_ERROR(model_builder.RegisterConstant(value_tensor, output, desc, logger));
  } else {
    output = model_builder.CreateOrGetConstant<float>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 0.0f);
  }

  if (is_constant_shape) {
    std::vector<int64_t> output_shape;
    ORT_RETURN_IF_NOT(ReadIntArrayFrom1DTensor(*initializers.at(input_defs[0]->Name()), output_shape,
                                               model_builder.GetGraphViewer(), logger),
                      "Cannot get output shape from initializer.");
    // An empty shape is a scalar output, which is the constant itself.
    if (!output_shape.empty()) {
      emscripten::val options = emscripten::val::object();
      options.set("label", node.Name());
      output = model_builder.GetBuilder().call<emscripten::val>(
          "expand", output, emscripten::val::array(GetNarrowedIntFromInt64<uint32_t>(output_shape)), options);
    }
  } else {
    // For operand shape, use expandDynamic to expand the scalar constant to the dynamic shape.
    // Cast to uint32 (ONNX shape is int64, expandDynamic requires uint32).
    emscripten::val shape_operand = model_builder.GetOperand(input_defs[0]->Name());
    emscripten::val cast_options = emscripten::val::object();
    cast_options.set("label", node.Name() + "_cast_shape_uint32");
    shape_operand = model_builder.GetBuilder().call<emscripten::val>(
        "cast", shape_operand, emscripten::val("uint32"), cast_options);
    emscripten::val options = emscripten::val::object();
    options.set("label", node.Name());
    output = model_builder.GetBuilder().call<emscripten::val>("expandDynamic", output, shape_operand, options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool ConstantOfShapeOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer,
                                                 const Node& node,
                                                 const WebnnDeviceType /* device_type */,
                                                 const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& shape_name = input_defs[0]->Name();

  // When the shape input is a constant initializer, validate its contents.
  const auto* shape_init = graph_viewer.GetConstantInitializer(shape_name);
  if (shape_init) {
    const auto& shape_tensor = *shape_init;
    if (shape_tensor.data_type() != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      LOGS(logger, VERBOSE) << "The shape input data type must be INT64.";
      return false;
    }

    std::vector<int64_t> output_shape;
    if (!ReadIntArrayFrom1DTensor(shape_tensor, output_shape, graph_viewer, logger)) {
      return false;
    }

    // Reject negative or zero dimensions since WebNN expand requires a valid static shape.
    if (std::any_of(output_shape.begin(), output_shape.end(), [](int64_t dim) { return dim <= 0; })) {
      LOGS(logger, VERBOSE) << "ConstantOfShape requires all output dimensions to be known and positive.";
      return false;
    }
  }
  // When shape is an operand (not a constant initializer), expandDynamic handles
  // the shape at runtime so no static shape validation is needed.

  return true;
}

bool ConstantOfShapeOpBuilder::HasSupportedInputsImpl(const GraphViewer& graph_viewer, const Node& node,
                                                      const emscripten::val& wnn_limits,
                                                      const logging::Logger& logger) const {
  int32_t output_type = 0;
  if (!GetType(*node.OutputDefs()[0], output_type, logger)) {
    return false;
  }

  // The value is a scalar constant, expanded to the output shape by expand (constant shape) or expandDynamic.
  if (!IsSupportedDataType(output_type, wnn_limits, "constant", "")) {
    LOGS(logger, VERBOSE) << "[ConstantOfShape] Output's data type: [" << output_type
                          << "] is not supported by WebNN op [constant] for now";
    return false;
  }

  const std::string_view webnn_op_type =
      graph_viewer.GetConstantInitializer(node.InputDefs()[0]->Name()) ? GetWebNNOpType(node.OpType())
                                                                       : "expandDynamic";
  return IsDataTypeSupportedByWebNNOp(node.OpType(), webnn_op_type, output_type, wnn_limits, "output", "Output",
                                      logger) &&
         IsOutputRankSupportedByOp(node, webnn_op_type, wnn_limits, logger);
}

void CreateConstantOfShapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ConstantOfShapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
