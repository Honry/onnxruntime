// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"
#include "builder_utils.h"

namespace onnxruntime {
namespace webnn {

class PoolOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const logging::Logger& logger) const override;
};

// Add operator related

Status PoolOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                            const Node& node,
                                            const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();

  ::wnn::Operand input = model_builder.GetOperand(input_defs[0]->Name());

  bool is_global_pooling = false;
  bool is_average_pool = false;
  if (op_type == "GlobalAveragePool") {
    is_global_pooling = true;
    is_average_pool = true;
  } else if (op_type == "GlobalMaxPool") {
    is_global_pooling = true;
  } else if (op_type == "AveragePool") {
    is_average_pool = true;
  } else if (op_type == "MaxPool") {
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "PoolOpBuilder, unknown op: ", op_type);
  }

  ::wnn::Pool2dOptions options;
  NodeAttrHelper helper(node);

  const auto kernel_shape = helper.Get("kernel_shape", std::vector<int32_t>{0, 0});
  if (!is_global_pooling) {
    options.windowDimensions = kernel_shape.data();
    options.windowDimensionsCount = SafeInt<uint32_t>(kernel_shape.size());
  }
  const auto strides = helper.Get("strides", std::vector<int32_t>{1, 1});
  options.strides = strides.data();
  options.stridesCount = SafeInt<uint32_t>(strides.size());
  const auto dilations = helper.Get("dilations", std::vector<int32_t>{1, 1});
  options.dilations = dilations.data();
  options.dilationsCount = SafeInt<uint32_t>(dilations.size());

  // Add Padding
  // Usually using autopadding is more efficient than using explicit padding
  // Try to see if we can map explicit padding to auto padding
  const auto onnx_kernel_shape = helper.Get("kernel_shape", std::vector<int64_t>{0, 0});
  const auto onnx_strides = helper.Get("strides", std::vector<int64_t>{1, 1});
  const auto onnx_pads = helper.Get("pads", std::vector<int64_t>{0, 0, 0, 0});
  const auto pads = helper.Get("pads", std::vector<int32_t>{0, 0, 0, 0});
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  AutoPadType auto_pad_type;
  ORT_RETURN_IF_ERROR(HandleAutoPad(input_shape, onnx_kernel_shape[0], onnx_kernel_shape[1],
                                    onnx_pads, onnx_strides, {1, 1} /* dilations */,
                                    StringToAutoPadType(helper.Get("auto_pad", "NOTSET")),
                                    auto_pad_type));

  if (AutoPadType::SAME_UPPER == auto_pad_type || AutoPadType::SAME_LOWER == auto_pad_type) {
    if (AutoPadType::SAME_LOWER == auto_pad_type) {  // default is SAME_UPPER
      options.autoPad = ::wnn::AutoPad::SameLower;
    } else {
      options.autoPad = ::wnn::AutoPad::SameUpper;
    }
  } else {
    options.padding = pads.data();
    options.paddingCount = SafeInt<uint32_t>(pads.size());
  }

  const auto ceil_mode = helper.Get("ceil_mode", 0);
  options.roundingType = ceil_mode == 0 ? ::wnn::RoundingType::Floor
                                        : ::wnn::RoundingType::Ceil;

  ::wnn::Operand output;
  if (is_average_pool) {
    output = model_builder.GetBuilder().AveragePool2d(input, &options);
  } else {
    output = model_builder.GetBuilder().MaxPool2d(input, &options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related
bool PoolOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                      const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS(logger, VERBOSE)
        << op_type << " only supports rank-4 tensor, input ["
        << input_defs[0]->Name() << "] has actual dim count " << input_size;
    return false;
  }

  if (op_type == "AveragePool" || op_type == "MaxPool") {
    NodeAttrHelper helper(node);
    const auto storage_order = helper.Get("storage_order", 0);
    if (storage_order == 1) {
      LOGS(logger, VERBOSE) << "storage_order == 1 is not supported";
      return false;
    }

    if (helper.Get("kernel_shape", std::vector<int32_t>{1, 1}).size() != 2) {
      LOGS(logger, VERBOSE) << "Only pooling 2d is supported";
      return false;
    }

    if (node.OutputDefs().size() != 1) {
      LOGS(logger, VERBOSE) << "Argmax in maxpooling is not supported";
      return false;
    }
  }

  return true;
}

void CreatePoolOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.find(op_type) != op_registrations.op_builder_map.cend())
    return;

  static std::vector<std::string> op_types =
      {
          "GlobalAveragePool",
          "GlobalMaxPool",
          "AveragePool",
          "MaxPool",
      };

  op_registrations.builders.push_back(std::make_unique<PoolOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
