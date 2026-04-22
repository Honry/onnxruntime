// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class SplitOpBuilder : public BaseOpBuilder {
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
  bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                               const logging::Logger& logger) const override;
};

// Add operator related.

void SplitOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Skip split initializer if present and is a constant initializer.
  // When it is an operand, we need it as the splits input for dynamicSplit.
  if (node.InputDefs().size() > 1) {
    const auto& split_name = node.InputDefs()[1]->Name();
    if (model_builder.GetGraphViewer().GetConstantInitializer(split_name)) {
      model_builder.AddInitializerToSkip(split_name);
    }
  }
}

Status SplitOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                             const Node& node,
                                             const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  const size_t rank = input_shape.size();
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  NodeAttrHelper helper(node);
  int32_t axis = helper.Get("axis", 0);
  axis = SafeInt<int32_t>(HandleNegativeAxis(axis, rank));
  options.set("axis", axis);

  // Check if the split input is an operand (not a constant initializer).
  const auto& initializers(model_builder.GetInitializerTensors());
  const std::string split_name = GetTensorName(input_defs, 1);
  const bool is_operand_split = !split_name.empty() && initializers.count(split_name) == 0;

  emscripten::val output_array = emscripten::val::undefined();
  if (is_operand_split) {
    // Operand split path: use dynamicSplit with the splits operand.
    emscripten::val splits_operand = model_builder.GetOperand(split_name);
    output_array = model_builder.GetBuilder().call<emscripten::val>("dynamicSplit", input, splits_operand, options);
  } else {
    // Constant split path: read split count or explicit split lengths.
    uint32_t split_count = 0;
    std::vector<uint32_t> splits = helper.Get("split", std::vector<uint32_t>{});

    if (helper.HasAttr("num_outputs")) {
      split_count = helper.Get("num_outputs", 0);
    } else if (!split_name.empty()) {
      const auto& split_tensor = *initializers.at(split_name);
      ORT_RETURN_IF_NOT(ReadIntArrayFrom1DTensor(split_tensor, splits, model_builder.GetGraphViewer(), logger),
                        "Cannot get input for split.");
    } else if (!helper.HasAttr("split")) {
      split_count = node.OutputDefs().size();
    }

    // Check that the splits evenly divide.
    // When axis dim is dynamic (0 from GetShape), skip uneven-split computation;
    // WebNN split with split_count will handle even division at runtime.
    if (split_count > 0 && splits.empty() && input_shape[axis] > 0 && input_shape[axis] % split_count != 0) {
      // Divide inputs into variable size outputs:
      splits.insert(splits.end(), split_count - 1, SafeInt<uint32_t>(input_shape[axis]) / split_count);
      splits.insert(splits.end(), SafeInt<uint32_t>(input_shape[axis]) % split_count);
    }

    if (splits.empty()) {
      output_array = model_builder.GetBuilder().call<emscripten::val>(
          "split", input, split_count, options);
    } else {
      output_array = model_builder.GetBuilder().call<emscripten::val>(
          "split", input, emscripten::val::array(splits), options);
    }
  }

  for (size_t i = 0, count = output_array["length"].as<size_t>(); i < count; i++) {
    model_builder.AddOperand(node.OutputDefs()[i]->Name(), std::move(output_array[i]));
  }
  return Status::OK();
}

// Operator support related.

bool SplitOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer,
                                       const Node& node,
                                       const WebnnDeviceType /* device_type */,
                                       const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    LOGS(logger, VERBOSE) << "Cannot get input's shape.";
    return false;
  }
  const size_t rank = input_shape.size();

  NodeAttrHelper helper(node);
  int32_t axis = helper.Get("axis", 0);
  axis = SafeInt<int32_t>(HandleNegativeAxis(axis, rank));
  std::vector<uint32_t> split = helper.Get("split", std::vector<uint32_t>{});

  const std::string split_name = GetTensorName(input_defs, 1);
  // Inputs contain optional 'split' input.
  if (!split_name.empty()) {
    const auto* split_init = graph_viewer.GetConstantInitializer(split_name);
    if (split_init) {
      // When split is a constant initializer, validate its contents.
      const auto& split_tensor = *split_init;
      if (split_tensor.data_type() != ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        LOGS(logger, VERBOSE) << "The type of tensor's element data must be INT64.";
        return false;
      }
      if (!ReadIntArrayFrom1DTensor(split_tensor, split, graph_viewer, logger)) {
        return false;
      }
    }
    // When split is an operand (not a constant initializer), dynamicSplit handles
    // the splits at runtime so no static validation is needed.
  } else {
    if (helper.HasAttr("num_outputs")) {
      // Split has 'num_outputs' attribute when opset is 18.
      const int32_t num_outputs = helper.Get("num_outputs", 1);
      if (num_outputs < 1) {
        LOGS(logger, VERBOSE) << "The 'num_outputs' must be a positive integer.";
        return false;
      }
    } else {
      const auto opset = node.SinceVersion();
      if (opset >= 18) {
        LOGS(logger, VERBOSE) << "The 'num_outputs' should be specified when 'split' isn't specified.";
        return false;
      }
    }
  }

  if (!split.empty()) {
    int64_t sum = 0;
    // TODO: Allow 0 size dimensions.
    // https://github.com/webmachinelearning/webnn/issues/391
    for (uint32_t split_value : split) {
      if (split_value <= 0) {
        LOGS(logger, VERBOSE) << "Value of split should be greater than 0.";
        return false;
      }
      sum += split_value;
    }
    if (sum != input_shape[axis] && input_shape[axis] > 0) {
      LOGS(logger, VERBOSE) << "Sum of the split's values must be equal to the dim value at 'axis' specified.";
      return false;
    }
  }
  return true;
}

bool SplitOpBuilder::HasSupportedInputsImpl(const GraphViewer& graph_viewer,
                                            const Node& node,
                                            const emscripten::val& wnn_limits,
                                            const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const std::string split_name = GetTensorName(input_defs, 1);

  // When split input is absent or is a constant initializer, use the constant path.
  // Delegate to the base class which checks input 0 against WebNN split's limits.
  if (split_name.empty() || graph_viewer.GetConstantInitializer(split_name)) {
    return BaseOpBuilder::HasSupportedInputsImpl(graph_viewer, node, wnn_limits, logger);
  }

  // When split is an operand, check inputs against dynamicSplit's limits.
  const std::string_view webnn_op_type = "dynamicSplit";

  // Check input 0 (data tensor) against dynamicSplit's "input" parameter.
  int32_t input_type;
  if (!GetType(*input_defs[0], input_type, logger)) {
    return false;
  }
  if (!IsDataTypeSupportedByWebNNOp("Split", webnn_op_type, input_type, wnn_limits,
                                    "input", "input", logger)) {
    return false;
  }
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger) ||
      !IsInputRankSupported(wnn_limits, webnn_op_type, "input",
                            input_shape.size(), node.Name(), logger)) {
    return false;
  }

  // Check input 1 (splits operand) against dynamicSplit's "splits" parameter.
  int32_t splits_type;
  if (!GetType(*input_defs[1], splits_type, logger)) {
    return false;
  }
  if (!IsDataTypeSupportedByWebNNOp("Split", webnn_op_type, splits_type, wnn_limits,
                                    "splits", "split", logger)) {
    return false;
  }

  return true;
}

bool SplitOpBuilder::HasSupportedOutputsImpl(const Node& node,
                                             const emscripten::val& wnn_limits,
                                             const logging::Logger& logger) const {
  const auto& output_defs = node.OutputDefs();
  const std::string_view op_type = node.OpType();
  int32_t output_type = 0;

  if (!GetType(*output_defs[0], output_type, logger)) {
    return false;
  }

  // Chromium has changed the output name of split from 'output' to 'outputs',
  // to avoid breaking the existing API, we need to check both names.
  const std::string_view wnn_output_name = wnn_limits["split"]["output"].isUndefined() ? "outputs" : "output";
  return IsDataTypeSupportedByOp(op_type, output_type, wnn_limits, wnn_output_name, "outputs", logger);
}

void CreateSplitOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SplitOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
