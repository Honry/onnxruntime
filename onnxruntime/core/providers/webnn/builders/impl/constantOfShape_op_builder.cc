// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

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
  bool HasSupportedInputsImpl(const GraphViewer& graph_viewer, const Node& node,
                              const emscripten::val& wnn_limits,
                              const logging::Logger& logger) const override;

  // ConstantOfShape is supported since opset 9.
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 9; }
};

// Add operator related.

void ConstantOfShapeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& shape_name = node.InputDefs()[0]->Name();
  // Only skip the shape input when it is a constant initializer (consumed at build time).
  // When it is an operand, we need it as the newShape input for dynamicExpand.
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

  // For constant shape: create a full-sized WebNN constant directly.
  // For operand shape: create a scalar constant and use dynamicExpand to expand it.
  std::vector<uint32_t> dims;
  if (is_constant_shape) {
    std::vector<int64_t> output_shape;
    ORT_RETURN_IF_NOT(ReadIntArrayFrom1DTensor(*initializers.at(input_defs[0]->Name()), output_shape,
                                               model_builder.GetGraphViewer(), logger),
                      "Cannot get output shape from initializer.");
    dims = GetNarrowedIntFromInt64<uint32_t>(output_shape);
  }
  // For operand path, dims stays empty (scalar constant).

  // Get the 'value' attribute (1-element TensorProto). Default is float32 0.
  int32_t data_type = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
  emscripten::val scalar_value = emscripten::val(0.0f);

  const auto& attrs = node.GetAttributes();
  auto value_it = attrs.find("value");
  if (value_it != attrs.end()) {
    const auto& value_tensor = value_it->second.t();
    data_type = value_tensor.data_type();
    ORT_RETURN_IF_NOT(ReadScalarTensorData(value_tensor, scalar_value,
                                           model_builder.GetGraphViewer(), logger),
                      "Failed to read value attribute.");
  }

  // Handle int64 -> int32 fallback if int64 is not supported.
  int32_t effective_data_type = data_type;
  if (!model_builder.IsInt64Supported() && data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    effective_data_type = ONNX_NAMESPACE::TensorProto_DataType_INT32;
    // ReadScalarTensorData already returned the int64 value as emscripten::val,
    // which JavaScript will narrow when stored in an Int32Array.
  }

  // Build the WebNN constant descriptor.
  emscripten::val desc = emscripten::val::object();
  desc.set("shape", emscripten::val::array(dims));
  desc.set("dimensions", emscripten::val::array(dims));
  ORT_RETURN_IF_NOT(SetWebnnDataType(desc, effective_data_type),
                    "Unsupported data type for ConstantOfShape: ", effective_data_type);

  // Compute the total number of elements (1 for scalar when dims is empty).
  uint32_t num_elements = 1;
  for (auto d : dims) {
    num_elements *= d;
  }

  // Create a typed JS buffer filled with the scalar value.
  emscripten::val buffer = emscripten::val::undefined();
  switch (effective_data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      buffer = emscripten::val::global("Uint8Array").new_(num_elements);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      buffer = emscripten::val::global("Int8Array").new_(num_elements);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      buffer = emscripten::val::global("Float16Array").isUndefined()
                   ? emscripten::val::global("Uint16Array").new_(num_elements)
                   : emscripten::val::global("Float16Array").new_(num_elements);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      buffer = emscripten::val::global("Float32Array").new_(num_elements);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      buffer = emscripten::val::global("Int32Array").new_(num_elements);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      buffer = emscripten::val::global("BigInt64Array").new_(num_elements);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      buffer = emscripten::val::global("Uint32Array").new_(num_elements);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      buffer = emscripten::val::global("BigUint64Array").new_(num_elements);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "ConstantOfShape unsupported data type: ", effective_data_type);
  }
  buffer.call<void>("fill", scalar_value);

  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("constant", desc, buffer);

  // For operand shape, use dynamicExpand to expand the scalar constant to the dynamic shape.
  if (!is_constant_shape) {
    emscripten::val shape_operand = model_builder.GetOperand(input_defs[0]->Name());
    emscripten::val options = emscripten::val::object();
    options.set("label", node.Name());
    output = model_builder.GetBuilder().call<emscripten::val>("dynamicExpand", output, shape_operand, options);
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

    // Reject negative or zero dimensions since WebNN constant requires a valid static shape.
    if (std::any_of(output_shape.begin(), output_shape.end(), [](int64_t dim) { return dim <= 0; })) {
      LOGS(logger, VERBOSE) << "ConstantOfShape requires all output dimensions to be known and positive.";
      return false;
    }
  }
  // When shape is an operand (not a constant initializer), dynamicExpand handles
  // the shape at runtime so no static shape validation is needed.

  return true;
}

bool ConstantOfShapeOpBuilder::HasSupportedInputsImpl(const GraphViewer& graph_viewer,
                                                      const Node& node,
                                                      const emscripten::val& wnn_limits,
                                                      const logging::Logger& logger) const {
  // When shape is a constant initializer, it is consumed at build time.
  // No WebNN input to validate.
  if (graph_viewer.GetConstantInitializer(node.InputDefs()[0]->Name())) {
    return true;
  }

  // When shape is an operand, delegate to the base class which checks
  // input data type and rank against dynamicExpand's limits via op_inputs_map.
  return BaseOpBuilder::HasSupportedInputsImpl(graph_viewer, node, wnn_limits, logger);
}

void CreateConstantOfShapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ConstantOfShapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
