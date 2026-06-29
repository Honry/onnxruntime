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

class SizeOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
  // WebNN shape() outputs uint32 regardless of ONNX's int64 output type declaration.
  // Skip output type validation to avoid rejecting the node.
  bool HasSupportedOutputsImpl(const Node& /*node*/, const emscripten::val& /*wnn_limits*/,
                               const logging::Logger& /*logger*/) const override {
    return true;
  }
};

Status SizeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                            const Node& node,
                                            const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");

  emscripten::val output = emscripten::val::undefined();

  if (!HasDynamicShape(input_shape)) {
    // Static path: compute size at build time.
    int64_t size = 1;
    for (const auto dim : input_shape) {
      size *= dim;
    }
    if (model_builder.IsInt64Supported()) {
      output = model_builder.CreateOrGetConstant<int64_t>(
          ONNX_NAMESPACE::TensorProto_DataType_INT64, size, {});
    } else {
      output = model_builder.CreateOrGetConstant<int32_t>(
          ONNX_NAMESPACE::TensorProto_DataType_INT32, static_cast<int32_t>(size), {});
    }
  } else {
    // Dynamic path: shape() → reduceProduct() to compute element count at runtime.
    emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
    emscripten::val wnn_builder = model_builder.GetBuilder();

    emscripten::val common_options = emscripten::val::object();
    common_options.set("label", node.Name() + "_shape");
    emscripten::val shape_operand = wnn_builder.call<emscripten::val>("shape", input, common_options);

    std::string shape_type_str = model_builder.IsInt64Supported() ? "int64" : "int32";
    common_options.set("label", node.Name() + "_cast_shape_" + shape_type_str);
    emscripten::val cast_shape_operand = wnn_builder.call<emscripten::val>(
        "cast", shape_operand, emscripten::val(shape_type_str), common_options);

    common_options.set("label", node.Name() + "_reduceProduct");
    output = wnn_builder.call<emscripten::val>("reduceProduct", cast_shape_operand, common_options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

bool SizeOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                           const emscripten::val& wnn_limits,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const std::string_view op_type = node.OpType();
  int32_t input_type = 0;
  if (!GetType(*input_defs[0], input_type, logger)) {
    return false;
  }

  // Check if the input data type is supported by each decomposed WebNN op.
  for (const std::string_view decomposed_op_type : decomposed_op_map.at(op_type)) {
    const std::string_view webnn_op_type = GetWebNNOpType(decomposed_op_type);
    const std::string_view webnn_input_name = GetWebNNOpFirstInputName(decomposed_op_type);
    if (!IsDataTypeSupportedByWebNNOp(op_type, webnn_op_type, input_type, wnn_limits, webnn_input_name, "input", logger)) {
      return false;
    }
  }

  return true;
}

void CreateSizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SizeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
