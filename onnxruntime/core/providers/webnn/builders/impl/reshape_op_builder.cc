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

class ReshapeOpBuilder : public BaseOpBuilder {
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

void ReshapeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& shape_name = node.InputDefs()[1]->Name();
  // Only skip the shape input when it is a constant initializer (consumed at build time).
  // When it is an operand, we need it as the newShape input for dynamicReshape.
  if (model_builder.GetGraphViewer().GetConstantInitializer(shape_name)) {
    model_builder.AddInitializerToSkip(shape_name);
  }
}

Status ReshapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
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
    // Constant shape path: resolve the target shape at build time and use WebNN reshape.
    const auto& target_shape_tensor = *initializers.at(input_defs[1]->Name());
    const auto& target_shape_tensor_dims = target_shape_tensor.dims();

    if (!target_shape_tensor_dims.empty()) {
      const int64_t* raw_target_shape = target_shape_tensor.int64_data().empty()
                                            ? reinterpret_cast<const int64_t*>(target_shape_tensor.raw_data().data())
                                            : target_shape_tensor.int64_data().data();

      const auto size = target_shape_tensor_dims[0];
      TensorShapeVector target_shape{raw_target_shape, raw_target_shape + size};
      std::vector<int64_t> input_shape;
      ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");

      if (!HasDynamicShape(input_shape)) {
        // ReshapeHelper validates the reshape and resolves -1 to a concrete value.
        ReshapeHelper helper(TensorShape(input_shape), target_shape);
      }

      // Build the new shape array. input["shape"][i] returns the dim descriptor object
      // for dynamic dims and a plain integer for static dims, so it works in both cases.
      emscripten::val new_shape = emscripten::val::array();
      for (size_t i = 0; i < static_cast<size_t>(size); ++i) {
        if (target_shape[i] == 0) {
          // ONNX allowzero=0 semantics: copy from input at the same axis.
          new_shape.call<void>("push", input["shape"][static_cast<uint32_t>(i)]);
        } else if (target_shape[i] == -1) {
          // Only reached when has_dynamic_input (ReshapeHelper resolves -1 otherwise).
          // Use the output shape proto's dim_param/dim_value if available so that all Reshape
          // nodes inferring the same symbolic dimension share the same dim descriptor name.
          // This is critical for downstream ops (e.g., GQA) that broadcast between tensors
          // from different Reshape outputs — mismatched dim descriptor names cause GPU crashes.
          const auto* output_shape_proto = node.OutputDefs()[0]->Shape();
          if (output_shape_proto && static_cast<int>(i) < output_shape_proto->dim_size()) {
            const auto& dim = output_shape_proto->dim(static_cast<int>(i));
            if (dim.has_dim_value()) {
              // Output dim is statically known despite dynamic input.
              uint32_t dim_value = SafeInt<uint32_t>(dim.dim_value());
              new_shape.call<void>("push", dim_value);
            } else if (dim.has_dim_param()) {
              // Use the symbolic dim name from the model's shape annotations.
              emscripten::val dim_desc = emscripten::val::object();
              dim_desc.set("name", emscripten::val(dim.dim_param()));
              new_shape.call<void>("push", dim_desc);
            } else {
              // No dim info available; fall back to a unique name.
              emscripten::val dim_desc = emscripten::val::object();
              dim_desc.set("name", emscripten::val(node.Name() + "_inferred"));
              new_shape.call<void>("push", dim_desc);
            }
          } else {
            // No output shape proto; fall back to a unique name.
            emscripten::val dim_desc = emscripten::val::object();
            dim_desc.set("name", emscripten::val(node.Name() + "_inferred"));
            new_shape.call<void>("push", dim_desc);
          }
        } else {
          uint32_t dim_value = SafeInt<uint32_t>(target_shape[i]);
          new_shape.call<void>("push", dim_value);
        }
      }
      output = model_builder.GetBuilder().call<emscripten::val>("reshape", input, new_shape, options);
    } else {
      // Empty target shape → converting to a scalar.
      emscripten::val new_shape = emscripten::val::array();
      output = model_builder.GetBuilder().call<emscripten::val>("reshape", input, new_shape, options);
    }
  } else {
    // Operand shape path: use dynamicReshape with the shape operand.
    emscripten::val shape_operand = model_builder.GetOperand(input_defs[1]->Name());
    output = model_builder.GetBuilder().call<emscripten::val>("dynamicReshape", input, shape_operand, options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool ReshapeOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer,
                                         const Node& node,
                                         const WebnnDeviceType /* device_type */,
                                         const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& shape_name = input_defs[1]->Name();

  // When the shape input is a constant initializer, validate its contents.
  const auto* shape_init = graph_viewer.GetConstantInitializer(shape_name);

  // WebNN reshape/dynamicReshape does not support 0 as dimension.
  NodeAttrHelper helper(node);
  const bool allow_zero = helper.Get("allowzero", 0) == 1;
  if (allow_zero) {
    if (!shape_init) {
      // Cannot validate shape values at build time, reject outright.
      LOGS(logger, VERBOSE) << "Reshape with allowzero=1 is not supported when shape is not a constant initializer.";
      return false;
    }

    const auto& shape_tensor = *shape_init;
    std::vector<uint8_t> unpacked_tensor;
    if (!UnpackInitializerData(shape_tensor, unpacked_tensor, graph_viewer, logger)) {
      return false;
    }

    const int64_t* raw_new_shape = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
    const auto& shape_dims = shape_tensor.dims();
    if (!shape_dims.empty()) {
      for (int64_t i = 0; i < shape_dims[0]; i++) {
        if (raw_new_shape[i] == 0) {
          LOGS(logger, VERBOSE) << "Reshape doesn't support 0 reshape dimension when allowzero is enabled.";
          return false;
        }
      }
    }
  }

  return true;
}

bool ReshapeOpBuilder::HasSupportedInputsImpl(const GraphViewer& graph_viewer,
                                              const Node& node,
                                              const emscripten::val& wnn_limits,
                                              const logging::Logger& logger) const {
  // When shape is a constant initializer, it is consumed at build time.
  // Delegate to the base class which checks input 0 against WebNN reshape's limits.
  if (graph_viewer.GetConstantInitializer(node.InputDefs()[1]->Name())) {
    return BaseOpBuilder::HasSupportedInputsImpl(graph_viewer, node, wnn_limits, logger);
  }

  // When shape is an operand, check inputs against dynamicReshape's limits.
  const auto& input_defs = node.InputDefs();
  const std::string_view webnn_op_type = "dynamicReshape";

  // Check input 0 (data tensor) against dynamicReshape's "input" parameter.
  int32_t input_type;
  if (!GetType(*input_defs[0], input_type, logger)) {
    return false;
  }
  if (!IsDataTypeSupportedByWebNNOp("Reshape", webnn_op_type, input_type, wnn_limits,
                                    "input", "input", logger)) {
    return false;
  }
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger) ||
      !IsInputRankSupported(wnn_limits, webnn_op_type, "input",
                            input_shape.size(), node.Name(), logger)) {
    return false;
  }

  // Check input 1 (shape operand) against dynamicReshape's "newShape" parameter.
  int32_t shape_type;
  if (!GetType(*input_defs[1], shape_type, logger)) {
    return false;
  }
  if (!IsDataTypeSupportedByWebNNOp("Reshape", webnn_op_type, shape_type, wnn_limits,
                                    "newShape", "shape", logger)) {
    return false;
  }

  return true;
}

void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReshapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
