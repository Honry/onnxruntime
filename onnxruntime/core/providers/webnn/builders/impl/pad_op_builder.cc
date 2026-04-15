// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"
#include "builder_utils.h"

namespace onnxruntime {
namespace webnn {

class PadOpBuilder : public BaseOpBuilder {
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
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
};

// Add operator related.

// ONNX mode to WebNN mode mapping.
const InlinedHashMap<std::string, std::string> supported_mode = {
    {"constant", "constant"},
    {"reflect", "reflection"},
    {"edge", "edge"},
};

void PadOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() < 2) {
    return;  // Opset < 11: pads is an attribute, nothing to skip.
  }

  const auto& initializers(model_builder.GetInitializerTensors());
  const bool is_constant_pads = initializers.count(input_defs[1]->Name()) > 0;

  if (is_constant_pads) {
    // Constant path: skip all non-data inputs (consumed at build time).
    for (size_t i = 1; i < input_defs.size(); i++) {
      model_builder.AddInitializerToSkip(input_defs[i]->Name());
      model_builder.AddInputToSkip(input_defs[i]->Name());
    }
  } else {
    // Dynamic path: pads is a runtime operand, do NOT skip it.
    // Skip constant_value and axes (consumed at build time as constants).
    for (size_t i = 2; i < input_defs.size(); i++) {
      const std::string name = GetTensorName(input_defs, i);
      if (!name.empty() && initializers.count(name) > 0) {
        model_builder.AddInitializerToSkip(name);
        model_builder.AddInputToSkip(name);
      }
    }
  }
}

bool clampNegativeValues(const std::vector<int64_t>& padding,
                         /*out*/ std::vector<uint32_t>& clamped_padding) {
  if (std::any_of(padding.begin(), padding.end(), [](auto pad) { return pad < 0; })) {
    std::transform(padding.begin(), padding.end(), std::back_inserter(clamped_padding),
                   [](int64_t x) -> uint32_t { return SafeInt<uint32_t>(std::max(x, 0LL)); });
    return true;  // Values were coerced.
  } else {
    std::transform(padding.begin(), padding.end(), std::back_inserter(clamped_padding),
                   [](int64_t x) -> uint32_t { return SafeInt<uint32_t>(x); });
  }
  return false;
}

Status PadOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& initializers = model_builder.GetInitializerTensors();
  const auto& graph_viewer = model_builder.GetGraphViewer();
  emscripten::val builder = model_builder.GetBuilder();
  const std::string& label = node.Name();

  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  const size_t rank = input_shape.size();

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());

  // Common setup: mode.
  emscripten::val options = emscripten::val::object();
  options.set("label", label);
  NodeAttrHelper helper(node);
  const auto pad_mode = helper.Get("mode", std::string("constant"));
  ORT_RETURN_IF(supported_mode.find(pad_mode) == supported_mode.end(),
                "WebNN does not support mode", pad_mode);
  options.set("mode", emscripten::val(supported_mode.find(pad_mode)->second));

  const auto opset = node.SinceVersion();
  const bool is_constant_pads = input_defs.size() < 2 || initializers.count(input_defs[1]->Name()) > 0;
  std::vector<int64_t> start_padding;
  std::vector<int64_t> end_padding;

  if (opset < 11) {
    // Before opset 11, pads and constant_value are attributes (always constant).
    ORT_RETURN_IF_NOT(helper.HasAttr("pads"), "Pads is required as attribute in opset ", opset);
    const auto pads = helper.Get("pads", std::vector<int>());
    options.set("value", helper.Get("value", 0.0f));
    start_padding.assign(pads.begin(), pads.begin() + pads.size() / 2);
    end_padding.assign(pads.begin() + pads.size() / 2, pads.end());
  } else {
    // opset >= 11: constant_value is an optional initializer input (shared by both paths).
    if (TensorExists(input_defs, 2)) {
      const auto value_tensor = *initializers.at(input_defs[2]->Name());
      emscripten::val value = emscripten::val::object();
      ORT_RETURN_IF_NOT(ReadScalarTensorData(value_tensor, value, graph_viewer, logger),
                        "Cannot read constant value");
      options.set("value", value);
    }

    if (is_constant_pads) {
      // Read pads from initializer and expand axes to full rank.
      std::vector<int64_t> pads;
      const auto& pads_tensor = *initializers.at(input_defs[1]->Name());
      ORT_RETURN_IF_NOT(ReadIntArrayFrom1DTensor(pads_tensor, pads, graph_viewer, logger),
                        "Error while reading pads tensor");
      if (TensorExists(input_defs, 3)) {
        std::vector<int64_t> axes;
        const auto& axes_tensor = *initializers.at(input_defs[3]->Name());
        ORT_RETURN_IF_NOT(ReadIntArrayFrom1DTensor(axes_tensor, axes, graph_viewer, logger),
                          "Error while reading axes tensor");
        std::vector<size_t> axes_index;
        std::transform(
            axes.begin(), axes.end(), std::back_inserter(axes_index),
            [rank](int64_t axis) -> int32_t { return SafeInt<int32_t>(HandleNegativeAxis(axis, rank)); });
        start_padding.resize(rank, 0);
        end_padding.resize(rank, 0);
        for (size_t i = 0; i < axes_index.size(); i++) {
          size_t index = axes_index[i];
          start_padding[index] = pads[i];
          end_padding[index] = pads[i + pads.size() / 2];
        }
      } else {
        start_padding.assign(pads.begin(), pads.begin() + pads.size() / 2);
        end_padding.assign(pads.begin() + pads.size() / 2, pads.end());
      }
    }
  }

  emscripten::val output = emscripten::val::undefined();

  if (is_constant_pads) {
    // Constant path: emit WebNN pad with uint32 arrays. Handle negative padding via clamp + slice.
    std::vector<uint32_t> webnn_start, webnn_end;
    bool negative_padding = clampNegativeValues(start_padding, webnn_start);
    negative_padding |= clampNegativeValues(end_padding, webnn_end);
    output = builder.call<emscripten::val>("pad", input,
                                           emscripten::val::array(webnn_start),
                                           emscripten::val::array(webnn_end), options);
    if (negative_padding) {
      std::vector<uint32_t> starts, sizes;
      for (size_t i = 0; i < start_padding.size(); i++) {
        starts.push_back(start_padding[i] >= 0 ? SafeInt<uint32_t>(0) : SafeInt<uint32_t>(-start_padding[i]));
        sizes.push_back(SafeInt<uint32_t>(input_shape[i] + start_padding[i] + end_padding[i]));
      }
      emscripten::val slice_opts = emscripten::val::object();
      slice_opts.set("label", label + "_slice_output");
      output = builder.call<emscripten::val>("slice", output,
                                             emscripten::val::array(starts),
                                             emscripten::val::array(sizes), slice_opts);
    }
  } else {
    // Dynamic path: pads is a runtime operand. Call dynamicPad.
    emscripten::val pads_op = model_builder.GetOperand(input_defs[1]->Name());

    // If axes is specified (constant), expand pads from partial axes to full rank via gather + where.
    if (TensorExists(input_defs, 3)) {
      std::vector<int64_t> axes;
      const auto* axes_init = graph_viewer.GetConstantInitializer(input_defs[3]->Name());
      ORT_RETURN_IF_NOT(axes_init && ReadIntArrayFrom1DTensor(*axes_init, axes, graph_viewer, logger),
                        "Cannot read axes initializer.");
      const size_t num_axes = axes.size();
      for (auto& axis : axes) {
        axis = HandleNegativeAxis(axis, rank);
      }

      // ONNX pads layout: [begin_ax0, ..., begin_axN, end_ax0, ..., end_axN]
      // Full pads layout: [begin_dim0, ..., begin_dimR, end_dim0, ..., end_dimR]
      std::vector<int32_t> gather_indices(2 * rank, 0);
      std::vector<uint8_t> mask(2 * rank, 0);
      for (size_t i = 0; i < num_axes; ++i) {
        const size_t d = static_cast<size_t>(axes[i]);
        gather_indices[d] = static_cast<int32_t>(i);
        gather_indices[rank + d] = static_cast<int32_t>(num_axes + i);
        mask[d] = 1;
        mask[rank + d] = 1;
      }

      int32_t pads_type;
      ORT_RETURN_IF_NOT(GetType(*input_defs[1], pads_type, logger), "Cannot get pads type");
      const bool use_int64 = pads_type == ONNX_NAMESPACE::TensorProto_DataType_INT64 &&
                             model_builder.IsInt64Supported();
      const int32_t pads_data_type = use_int64 ? ONNX_NAMESPACE::TensorProto_DataType_INT64
                                               : ONNX_NAMESPACE::TensorProto_DataType_INT32;
      constexpr int32_t INT32 = ONNX_NAMESPACE::TensorProto_DataType_INT32;
      const auto shape_1d = [](size_t n) { return std::vector<uint32_t>{static_cast<uint32_t>(n)}; };

      const emscripten::val& zeros_const = use_int64
          ? model_builder.CreateOrGetConstant<int64_t>(pads_data_type, int64_t{0}, shape_1d(2 * rank))
          : model_builder.CreateOrGetConstant<int32_t>(pads_data_type, int32_t{0}, shape_1d(2 * rank));
      const emscripten::val& gather_idx_const = model_builder.CreateOrGetConstant<int32_t>(
          INT32, label + "_gather_idx", gather_indices, shape_1d(2 * rank));
      const emscripten::val& mask_const = model_builder.CreateOrGetConstant<uint8_t>(
          ONNX_NAMESPACE::TensorProto_DataType_UINT8, label + "_mask", mask, shape_1d(2 * rank));

      emscripten::val opts = emscripten::val::object();
      opts.set("label", label + "_pads_gather");
      emscripten::val pads_expanded = builder.call<emscripten::val>(
          "gather", pads_op, gather_idx_const, opts);
      opts.set("label", label + "_pads_select");
      pads_op = builder.call<emscripten::val>(
          "where", mask_const, pads_expanded, zeros_const, opts);
    }

    output = builder.call<emscripten::val>("dynamicPad", input, pads_op, options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.
bool PadOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer,
                                     const Node& node,
                                     const WebnnDeviceType /* device_type */,
                                     const logging::Logger& logger) const {
  const auto& name = node.Name();
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  const auto opset = node.SinceVersion();

  NodeAttrHelper helper(node);
  const auto pad_mode = helper.Get("mode", "constant");
  if (supported_mode.find(pad_mode) == supported_mode.end()) {
    LOGS(logger, VERBOSE) << op_type << " [" << name << "] WebNN does not support mode " << pad_mode;
    return false;
  }

  if (input_defs.size() < 1) {
    LOGS(logger, VERBOSE) << op_type << " [" << name << "] requires at least one input (data)";
    return false;
  }

  if (opset >= 11) {
    if (input_defs.size() < 2) {
      LOGS(logger, VERBOSE) << op_type << " [" << name << "] at opset " << opset
                            << " requires at least two inputs (data and pads)";
      return false;
    }

    // constant_value and axes must always be constant initializers if present.
    for (size_t i = 2; i < input_defs.size(); i++) {
      const std::string input_name = GetTensorName(input_defs, i);
      if (!input_name.empty() && !graph_viewer.GetConstantInitializer(input_name)) {
        LOGS(logger, VERBOSE) << "Input '" << input_name << "' of " << op_type << " [" << name
                              << "] must be a constant initializer";
        return false;
      }
    }
  }

  return true;
}

bool PadOpBuilder::HasSupportedInputsImpl(const GraphViewer& graph_viewer, const Node& node,
                                          const emscripten::val& wnn_limits,
                                          const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  // Constant path: pads is absent or a constant initializer.
  // Delegate to base class which checks input 0 against WebNN pad's limits.
  const std::string pads_name = GetTensorName(input_defs, 1);
  if (pads_name.empty() || graph_viewer.GetConstantInitializer(pads_name)) {
    return BaseOpBuilder::HasSupportedInputsImpl(graph_viewer, node, wnn_limits, logger);
  }

  // Dynamic path: check inputs against dynamicPad's limits.
  const std::string_view webnn_op_type = "dynamicPad";

  // Check input 0 (data tensor) against dynamicPad's "input" parameter.
  int32_t input_type;
  if (!GetType(*input_defs[0], input_type, logger)) {
    return false;
  }
  if (!IsDataTypeSupportedByWebNNOp("Pad", webnn_op_type, input_type, wnn_limits,
                                    "input", "data", logger)) {
    return false;
  }
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger) ||
      !IsInputRankSupported(wnn_limits, webnn_op_type, "input",
                            input_shape.size(), node.Name(), logger)) {
    return false;
  }

  // Check pads operand type against dynamicPad's "pads" parameter.
  int32_t pads_type;
  if (!GetType(*input_defs[1], pads_type, logger)) {
    return false;
  }
  if (!IsDataTypeSupportedByWebNNOp("Pad", webnn_op_type, pads_type, wnn_limits,
                                    "pads", "pads", logger)) {
    return false;
  }

  return true;
}

void CreatePadOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<PadOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
