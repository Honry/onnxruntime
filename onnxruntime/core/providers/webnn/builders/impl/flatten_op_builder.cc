// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"
#include "shape_utils.h"

namespace onnxruntime {
namespace webnn {

class FlattenOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

};

// Add operator related.

Status FlattenOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  ORT_RETURN_IF(input_defs.size() < 1, "Flatten has no input tensor");

  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  const int64_t rank = input_shape.size();

  NodeAttrHelper helper(node);
  const int32_t axis = SafeInt<int32_t>(HandleNegativeAxis(helper.Get("axis", 1), rank));

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  emscripten::val output = emscripten::val::undefined();
  if (!HasDynamicShape(input_shape)) {
    // === Static path: compute new_shape at build time and use WebNN reshape. ===
    int64_t num_pre_axis_elements = std::accumulate(
        input_shape.begin(), input_shape.begin() + axis, 1, std::multiplies<int64_t>());
    int64_t num_post_axis_elements = std::accumulate(
        input_shape.begin() + axis, input_shape.end(), 1, std::multiplies<int64_t>());

    std::vector<uint32_t> new_shape = {SafeInt<uint32_t>(num_pre_axis_elements),
                                       SafeInt<uint32_t>(num_post_axis_elements)};
    output = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", input, emscripten::val::array(new_shape), options);
  } else {
    // === Dynamic path: prefer static reshape with dim descriptors when possible. ===
    // Flatten(input, axis) produces shape [product(dims[:axis]), product(dims[axis:])].
    // When one side of the axis split is a single dim or all-static, we can use static reshape
    // to preserve dim descriptor info. Only fall back to dynamicReshape when a side has
    // multiple dynamic dims requiring runtime product computation.

    // Analyze pre-axis dims.
    bool pre_all_static = true;
    int64_t pre_static_product = 1;
    int pre_dynamic_count = 0;
    for (int32_t i = 0; i < axis; ++i) {
      if (input_shape[i] == kDynamicDim) {
        pre_dynamic_count++;
        pre_all_static = false;
      } else {
        pre_static_product *= input_shape[i];
      }
    }

    // Analyze post-axis dims.
    bool post_all_static = true;
    int64_t post_static_product = 1;
    int post_dynamic_count = 0;
    for (int32_t i = axis; i < static_cast<int32_t>(rank); ++i) {
      if (input_shape[i] == kDynamicDim) {
        post_dynamic_count++;
        post_all_static = false;
      } else {
        post_static_product *= input_shape[i];
      }
    }

    // Try to build a static reshape shape array with dim descriptors.
    emscripten::val new_shape = emscripten::val::array();
    bool can_use_static_reshape = true;

    // Pre-axis output dimension.
    if (pre_all_static) {
      uint32_t pre_val = SafeInt<uint32_t>(pre_static_product);
      new_shape.call<void>("push", pre_val);
    } else if (axis == 1) {
      // Pre-axis is a single dim — use its dim descriptor directly.
      new_shape.call<void>("push", input["shape"][0]);
    } else if (pre_dynamic_count == 1 && pre_static_product == 1) {
      // All static dims in pre-axis are 1, so product = the single dynamic dim.
      for (int32_t i = 0; i < axis; ++i) {
        if (input_shape[i] == kDynamicDim) {
          new_shape.call<void>("push", input["shape"][static_cast<uint32_t>(i)]);
          break;
        }
      }
    } else {
      can_use_static_reshape = false;
    }

    // Post-axis output dimension.
    if (can_use_static_reshape) {
      if (post_all_static) {
        uint32_t post_val = SafeInt<uint32_t>(post_static_product);
        new_shape.call<void>("push", post_val);
      } else if (axis == static_cast<int32_t>(rank) - 1) {
        // Post-axis is a single dim — use its dim descriptor directly.
        new_shape.call<void>("push", input["shape"][static_cast<uint32_t>(axis)]);
      } else if (post_dynamic_count == 1 && post_static_product == 1) {
        // All static dims in post-axis are 1, so product = the single dynamic dim.
        for (int32_t i = axis; i < static_cast<int32_t>(rank); ++i) {
          if (input_shape[i] == kDynamicDim) {
            new_shape.call<void>("push", input["shape"][static_cast<uint32_t>(i)]);
            break;
          }
        }
      } else {
        can_use_static_reshape = false;
      }
    }

    if (can_use_static_reshape) {
      output = model_builder.GetBuilder().call<emscripten::val>("reshape", input, new_shape, options);
    } else {
      // Fall back to dynamicReshape for complex cases (products include multiple dynamic dims).
      emscripten::val wnn_builder = model_builder.GetBuilder();
      emscripten::val shape_operand = wnn_builder.call<emscripten::val>(
          "shape", input, emscripten::val::object());

      emscripten::val pre_product = shape_utils::ReduceShapeRange(
          model_builder, shape_operand, 0, axis, node.Name() + "_pre_reduce");
      emscripten::val post_product = shape_utils::ReduceShapeRange(
          model_builder, shape_operand, axis, static_cast<int32_t>(rank) - axis, node.Name() + "_post_reduce");

      emscripten::val segments = emscripten::val::array();
      segments.call<void>("push", pre_product);
      segments.call<void>("push", post_product);

      output = shape_utils::DynamicReshapeWithSegments(model_builder, input, segments, node.Name());
    }
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

void CreateFlattenOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<FlattenOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
