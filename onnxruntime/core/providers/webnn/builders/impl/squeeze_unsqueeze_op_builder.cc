// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "core/optimizer/initializer.h"

#include "base_op_builder.h"
#include "builder_utils.h"

namespace onnxruntime {
namespace webnn {

class SqueezeUnsqueezeOpBuilder : public BaseOpBuilder {
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
};

// Add operator related.
void SqueezeUnsqueezeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Squeeze/Unsqueeze opset 13 uses input 1 as axes, add it to initializer skip list.
  const auto& input_defs = node.InputDefs();
  if (node.SinceVersion() >= 13 && input_defs.size() > 1) {
    model_builder.AddInitializerToSkip(input_defs[1]->Name());  // "axes"
    model_builder.AddInputToSkip(input_defs[1]->Name());
  }
}

// Add operator related.

Status SqueezeUnsqueezeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                        const Node& node,
                                                        const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());

  // Get rank from shape proto (always available, validated by base class).
  const auto* shape_proto = input_defs[0]->Shape();
  const auto input_rank = static_cast<size_t>(shape_proto->dim_size());

  // Check if input has any dynamic dimensions.
  bool has_dynamic_shape = false;
  for (const auto& dim : shape_proto->dim()) {
    if (!dim.has_dim_value()) {
      has_dynamic_shape = true;
      break;
    }
  }

  // Resolve axes (always static — from attribute or constant initializer).
  std::vector<int32_t> axes_data;
  auto rank = input_rank;

  if (node.SinceVersion() >= 13 && !GetTensorName(input_defs, 1).empty()) {
    // Input axes is provided, use axes initializer data.
    const auto& initializers = model_builder.GetInitializerTensors();
    const auto& axes_tensor = *initializers.at(input_defs[1]->Name());
    Initializer axes_initializer(axes_tensor);
    const auto axes_data_span = axes_initializer.DataAsSpan<int64_t>();
    if (op_type == "Unsqueeze") {
      rank = input_rank + axes_data_span.size();
    }
    std::transform(
        axes_data_span.begin(), axes_data_span.end(), std::back_inserter(axes_data),
        [rank](int64_t axis) -> int32_t { return SafeInt<int32_t>(HandleNegativeAxis(axis, rank)); });
  } else {
    NodeAttrHelper helper(node);
    if (helper.HasAttr("axes")) {
      auto axes = helper.Get("axes", std::vector<int64_t>{});
      if (op_type == "Unsqueeze") {
        rank = input_rank + axes.size();
      }
      std::transform(
          axes.begin(), axes.end(), std::back_inserter(axes_data),
          [rank](int64_t axis) -> int32_t { return SafeInt<int32_t>(HandleNegativeAxis(axis, rank)); });
    }
  }

  // Sort axes in ascending order.
  std::sort(axes_data.begin(), axes_data.end());

  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  emscripten::val output = emscripten::val::undefined();

  if (!has_dynamic_shape) {
    // === Static path: compute new_shape at build time and use WebNN reshape. ===
    std::vector<int64_t> input_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
    std::vector<uint32_t> new_shape = GetNarrowedIntFromInt64<uint32_t>(input_shape);

    if (op_type == "Squeeze") {
      if (!axes_data.empty()) {
        for (auto it = axes_data.rbegin(); it != axes_data.rend(); ++it) {
          size_t index = *it;
          new_shape.erase(new_shape.begin() + index);
        }
      } else {
        // Remove all dimensions that are 1.
        new_shape.erase(
            std::remove_if(new_shape.begin(), new_shape.end(), [](uint32_t axis) { return axis == 1; }),
            new_shape.end());
      }
    } else if (op_type == "Unsqueeze") {
      for (const int32_t& axis : axes_data) {
        new_shape.insert(new_shape.begin() + axis, 1);
      }
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "SqueezeUnsqueezeOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
    }

    output = model_builder.GetBuilder().call<emscripten::val>("reshape",
                                                              input,
                                                              emscripten::val::array(new_shape),
                                                              options);
  } else {
    // === Dynamic path: axes are always known at build time, so we can determine
    // which shape indices to keep (Squeeze) or where to insert 1s (Unsqueeze). ===
    emscripten::val wnn_builder = model_builder.GetBuilder();

    if (op_type == "Squeeze") {
      // Use shape() + slice/concat + dynamicReshape to remove the squeezed axes.
      emscripten::val shape_operand = wnn_builder.call<emscripten::val>(
          "shape", input, emscripten::val::object());

      // Build segments of the shape tensor that skip the squeezed axis indices.
      // E.g., input_rank=5, axes=[1,3] → segments: shape[0:1], shape[2:1], shape[4:1]
      std::vector<emscripten::val> segments;
      int32_t prev = 0;
      for (int32_t axis : axes_data) {
        if (axis > prev) {
          emscripten::val starts = emscripten::val::array();
          starts.call<void>("push", prev);
          emscripten::val sizes = emscripten::val::array();
          sizes.call<void>("push", axis - prev);
          segments.push_back(wnn_builder.call<emscripten::val>("slice", shape_operand, starts, sizes));
        }
        prev = axis + 1;
      }
      if (prev < static_cast<int32_t>(input_rank)) {
        emscripten::val starts = emscripten::val::array();
        starts.call<void>("push", prev);
        emscripten::val sizes = emscripten::val::array();
        sizes.call<void>("push", static_cast<int32_t>(input_rank) - prev);
        segments.push_back(wnn_builder.call<emscripten::val>("slice", shape_operand, starts, sizes));
      }

      // Concatenate segments into the new shape tensor and use dynamicReshape.
      emscripten::val new_shape_operand;
      if (segments.size() == 1) {
        new_shape_operand = std::move(segments[0]);
      } else {
        emscripten::val inputs_array = emscripten::val::array();
        for (auto& seg : segments) {
          inputs_array.call<void>("push", seg);
        }
        emscripten::val concat_options = emscripten::val::object();
        concat_options.set("label", node.Name() + std::string("_concat_shape"));
        new_shape_operand = wnn_builder.call<emscripten::val>("concat", inputs_array, 0, concat_options);
      }
      output = wnn_builder.call<emscripten::val>("dynamicReshape", input, new_shape_operand, options);
    } else {  // Unsqueeze
      // For Unsqueeze, build the new shape as a JS array by reading the input operand's
      // dimension values and inserting literal 1s at the specified axes.
      // Using static reshape() (instead of dynamicReshape) preserves the static-ness of
      // each dimension — inserted 1s are known static, and input dims carry their original
      // static/dynamic status. This is important because downstream ops like ScatterND
      // may require certain dimensions to be statically known.
      emscripten::val new_shape = emscripten::val::array();
      size_t axes_idx = 0;
      uint32_t input_dim = 0;
      const auto output_rank = static_cast<int32_t>(rank);

      for (int32_t i = 0; i < output_rank; i++) {
        if (axes_idx < axes_data.size() && axes_data[axes_idx] == i) {
          new_shape.call<void>("push", 1u);  // Inserted dimension, always 1.
          axes_idx++;
        } else {
          new_shape.call<void>("push", input["shape"][input_dim]);  // From operand.
          input_dim++;
        }
      }
      output = wnn_builder.call<emscripten::val>("reshape", input, new_shape, options);
    }
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool SqueezeUnsqueezeOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer,
                                                  const Node& node,
                                                  const WebnnDeviceType /* device_type */,
                                                  const logging::Logger& logger) const {
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();

  if (input_defs.size() < 1) {
    LOGS(logger, ERROR) << op_type << " has no input tensor";
    return false;
  }

  bool has_explicit_axes = false;

  // Squeeze/Unsqueeze opset 13 uses input 1 as axes, it needs to be an initializer.
  if (node.SinceVersion() >= 13) {
    const std::string axes_name = GetTensorName(input_defs, 1);
    if (!axes_name.empty()) {
      const auto* init = graph_viewer.GetConstantInitializer(axes_name);
      if (!init) {
        LOGS(logger, ERROR) << "Input axes of " << op_type << " is not present and constant";
        return false;
      }
      has_explicit_axes = true;
    } else if (op_type == "Unsqueeze") {
      // The axes are optional for Squeeze, but not Unsqueeze.
      LOGS(logger, ERROR) << "Input axes of Unsqueeze must be provided";
      return false;
    }
  } else {
    NodeAttrHelper helper(node);
    has_explicit_axes = helper.HasAttr("axes");
  }

  // Squeeze without explicit axes removes all dims that equal 1.
  // With dynamic shapes, we don't know which dims are 1 at build time.
  if (op_type == "Squeeze" && !has_explicit_axes) {
    if (HasDynamicShape(*input_defs[0], logger)) {
      LOGS(logger, VERBOSE) << "Squeeze without explicit axes requires static input shape";
      return false;
    }
  }

  return true;
}

void CreateSqueezeUnsqueezeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "Squeeze",
          "Unsqueeze",
      };

  op_registrations.builders.push_back(std::make_unique<SqueezeUnsqueezeOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
