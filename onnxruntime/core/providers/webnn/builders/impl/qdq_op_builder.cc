// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "core/providers/webnn/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class QDQOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
  bool IsOpSupportedImpl(const GraphViewer&, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
};

Status QDQOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                           const Node& node,
                                           const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();

  std::vector<int64_t> input_shape;
  std::vector<int64_t> scale_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[1], scale_shape, logger), "Cannot get scale shape");

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val scale = model_builder.GetOperand(input_defs[1]->Name());
  emscripten::val zero_point = emscripten::val::undefined();
  emscripten::val common_options = emscripten::val::object();
  const bool has_zero_point = TensorExists(input_defs, 2);

  if (has_zero_point) {
    zero_point = model_builder.GetOperand(node.InputDefs()[2]->Name());
  }

  const auto input_rank = input_shape.size();
  NodeAttrHelper helper(node);
  int32_t block_size = helper.Get("block_size", 0);
  int32_t axis = helper.Get("axis", 1);
  if (axis < 0) {
    axis = SafeInt<int32_t>(HandleNegativeAxis(axis, input_rank));
  }

  // For per-tensor quantization/dequantization, the scale and zero_point tensors are scalars.
  // WebNN requires the scale and zero_point tensors to have the same rank as the input tensor.
  // We need to reshape them to match the input rank.
  if (scale_shape.size() == 0 && input_rank > 0) {
    std::vector<uint32_t> target_shape(input_rank, 1);
    common_options.set("label", node.Name() + "_reshape_scale");
    scale = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", scale, emscripten::val::array(target_shape), common_options);

    if (has_zero_point) {
      // Reshape the zero_point tensor too.
      common_options.set("label", node.Name() + "_reshape_zero_point");
      zero_point = model_builder.GetBuilder().call<emscripten::val>(
          "reshape", zero_point, emscripten::val::array(target_shape), common_options);
    }
  }

  // For per-axis quantization/dequantization, the scale is 1-D.
  // WebNN requires the scale and zero_point tensors to have the same rank as the input tensor.
  // We need to reshape them to make them broadcastable with the input tensor.
  if (scale_shape.size() == 1 && input_rank > 1 &&
      block_size == 0) {
    // Scale/zero_point are always static (constants). Use reshape with concrete values.
    std::vector<uint32_t> target_shape(input_rank, 1);
    if (input_shape[axis] > 0) {
      target_shape[axis] = SafeInt<uint32_t>(input_shape[axis]);
    } else {
      target_shape[axis] = SafeInt<uint32_t>(scale_shape[0]);
    }
    common_options.set("label", node.Name() + "_reshape_scale");
    scale = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", scale, emscripten::val::array(target_shape), common_options);
    if (has_zero_point) {
      common_options.set("label", node.Name() + "_reshape_zero_point");
      zero_point = model_builder.GetBuilder().call<emscripten::val>(
          "reshape", zero_point, emscripten::val::array(target_shape), common_options);
    }
  }

  // For old API, create a default zero_point constant if not provided.
  if (!has_zero_point && !IsZeroPointOptional()) {
    int32_t input_type = 0;
    int32_t output_type = 0;
    ORT_RETURN_IF_NOT(GetType(*input_defs[0], input_type, logger), "Cannot get input data type");
    ORT_RETURN_IF_NOT(GetType(*output_defs[0], output_type, logger), "Cannot get output data type");
    int32_t zero_point_type = op_type == "DequantizeLinear" ? input_type : output_type;
    // Compute the broadcast shape matching what scale was reshaped to.
    std::vector<uint32_t> zero_point_shape;
    if (scale_shape.size() == 1 && input_rank > 1 && block_size == 0) {
      zero_point_shape.resize(input_rank, 1);
      if (input_shape[axis] > 0) {
        zero_point_shape[axis] = SafeInt<uint32_t>(input_shape[axis]);
      } else {
        zero_point_shape[axis] = SafeInt<uint32_t>(scale_shape[0]);
      }
    } else if (scale_shape.size() == 0 && input_rank > 0) {
      zero_point_shape.resize(input_rank, 1);
    } else {
      zero_point_shape = GetNarrowedIntFromInt64<uint32_t>(scale_shape);
    }
    zero_point = model_builder.CreateOrGetConstant<uint8_t>(zero_point_type, 0, zero_point_shape);
  }

  common_options.set("label", node.Name());
  const std::string_view webnn_op_type = GetWebNNOpType(op_type);
  ORT_RETURN_IF(webnn_op_type.empty(), "Cannot get WebNN op type");

  emscripten::val output = emscripten::val::undefined();
  if (IsZeroPointOptional()) {
    if (has_zero_point) {
      common_options.set("zeroPoint", zero_point);
    }
    output = model_builder.GetBuilder().call<emscripten::val>(
        std::string(webnn_op_type).c_str(), input, scale, common_options);
  } else {
    output = model_builder.GetBuilder().call<emscripten::val>(
        std::string(webnn_op_type).c_str(), input, scale, zero_point, common_options);
  }

  model_builder.AddOperand(output_defs[0]->Name(), std::move(output));

  return Status::OK();
}

// Operator support related.
bool QDQOpBuilder::IsOpSupportedImpl(const GraphViewer&,
                                     const Node& node,
                                     const WebnnDeviceType /* device_type */,
                                     const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  std::vector<int64_t> scale_shape;

  if (!GetShape(*input_defs[0], input_shape, logger) || !GetShape(*input_defs[1], scale_shape, logger)) {
    return false;
  }

  if (scale_shape.size() > input_shape.size()) {
    LOGS(logger, VERBOSE) << "The rank of scale is larger than the rank of input";
    return false;
  }

  return true;
}

bool QDQOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                          const emscripten::val& wnn_limits, const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const std::string_view op_type = node.OpType();
  int32_t input0_type = 0;
  int32_t input1_type = 0;

  if (!GetType(*input_defs[0], input0_type, logger) ||
      !GetType(*input_defs[1], input1_type, logger)) {
    return false;
  }

  return IsInputRankSupportedByOp(node, wnn_limits, logger) &&
         IsDataTypeSupportedByOp(op_type, input0_type, wnn_limits, "input", "x", logger) &&
         IsDataTypeSupportedByOp(op_type, input1_type, wnn_limits, "scale", "x_scale", logger);
}

void CreateQDQOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "DequantizeLinear",
          "QuantizeLinear",
      };

  op_registrations.builders.push_back(std::make_unique<QDQOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
