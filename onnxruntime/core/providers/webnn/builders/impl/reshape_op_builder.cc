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
          //
          // Try to reuse an input dim descriptor instead of creating a new one.
          // When reshape merely splits/merges static dimensions (e.g., [B, S, 3840] → [B, S, 30, 128]),
          // the -1 dim is mathematically equal to an existing input dimension. Reusing that input's
          // dim descriptor preserves static dim info.
          bool reused_dim = false;

          // Find the single unclaimed dynamic input dim and compute static products.
          std::vector<bool> claimed(input_shape.size(), false);
          for (size_t t = 0; t < static_cast<size_t>(size); ++t) {
            if (target_shape[t] == 0 && t < input_shape.size()) {
              claimed[t] = true;
            }
          }

          int64_t static_target_product = 1;
          for (size_t t = 0; t < static_cast<size_t>(size); ++t) {
            if (target_shape[t] > 0) {
              static_target_product *= target_shape[t];
            }
          }

          int64_t static_input_product = 1;
          int unclaimed_dynamic_count = 0;
          int unclaimed_dynamic_idx = -1;
          for (size_t j = 0; j < input_shape.size(); ++j) {
            if (claimed[j]) continue;
            if (input_shape[j] <= 0) {
              unclaimed_dynamic_count++;
              unclaimed_dynamic_idx = static_cast<int>(j);
            } else {
              static_input_product *= input_shape[j];
            }
          }

          if (unclaimed_dynamic_count == 1) {
            if (static_input_product == static_target_product) {
              // Direct case: -1 equals the single unclaimed dynamic input dim. Reuse its descriptor.
              new_shape.call<void>("push", input["shape"][static_cast<uint32_t>(unclaimed_dynamic_idx)]);
              reused_dim = true;
            } else if (static_input_product > 0 && static_target_product > 0) {
              // Check provenance: the input's dynamic dim may itself be a scaled version of an
              // original dim. If this reshape inverts the scaling (merge-then-split pattern),
              // we can reuse the original source descriptor.
              emscripten::val input_dim_val = input["shape"][static_cast<uint32_t>(unclaimed_dynamic_idx)];
              if (input_dim_val.typeOf().as<std::string>() == "object" &&
                  !input_dim_val["name"].isUndefined()) {
                std::string dim_name = input_dim_val["name"].as<std::string>();
                const auto* prov = model_builder.GetDimProvenance(dim_name);
                if (prov != nullptr) {
                  int64_t new_factor_num = prov->factor_num * static_input_product;
                  int64_t new_factor_den = prov->factor_den * static_target_product;
                  if (new_factor_num == new_factor_den) {
                    // Factor is 1: -1 equals the original source dim. Reuse its descriptor.
                    const emscripten::val& source_operand =
                        model_builder.GetOperand(prov->source_operand_name);
                    new_shape.call<void>("push",
                                         source_operand["shape"][prov->source_dim_index]);
                    reused_dim = true;
                  }
                }
              }
            }
          }

          if (!reused_dim) {
            // Fall back: use the output shape proto's dim_param/dim_value if available so that all
            // Reshape nodes inferring the same symbolic dimension share the same dim descriptor name.
            const auto* output_shape_proto = node.OutputDefs()[0]->Shape();
            std::string new_dim_name;
            if (output_shape_proto && static_cast<int>(i) < output_shape_proto->dim_size()) {
              const auto& dim = output_shape_proto->dim(static_cast<int>(i));
              if (dim.has_dim_value()) {
                uint32_t dim_value = SafeInt<uint32_t>(dim.dim_value());
                new_shape.call<void>("push", dim_value);
              } else {
                new_dim_name = dim.has_dim_param() ? dim.dim_param()
                                                   : node.Name() + "_inferred";
                emscripten::val dim_desc = emscripten::val::object();
                dim_desc.set("name", emscripten::val(new_dim_name));
                new_shape.call<void>("push", dim_desc);
              }
            } else {
              new_dim_name = node.Name() + "_inferred";
              emscripten::val dim_desc = emscripten::val::object();
              dim_desc.set("name", emscripten::val(new_dim_name));
              new_shape.call<void>("push", dim_desc);
            }

            // Record provenance for this new dim descriptor so future reshapes can trace back.
            if (!new_dim_name.empty() && unclaimed_dynamic_count == 1 &&
                static_input_product > 0 && static_target_product > 0) {
              emscripten::val input_dim_val = input["shape"][static_cast<uint32_t>(unclaimed_dynamic_idx)];
              if (input_dim_val.typeOf().as<std::string>() == "object" &&
                  !input_dim_val["name"].isUndefined()) {
                std::string input_dim_name = input_dim_val["name"].as<std::string>();
                const auto* prov = model_builder.GetDimProvenance(input_dim_name);
                if (prov != nullptr) {
                  // Chain: new = source * (old_factor * P_in / P_out)
                  model_builder.RecordDimProvenance(new_dim_name, {
                      prov->source_operand_name,
                      prov->source_dim_index,
                      prov->factor_num * static_input_product,
                      prov->factor_den * static_target_product,
                  });
                } else {
                  model_builder.RecordDimProvenance(new_dim_name, {
                      input_defs[0]->Name(),
                      static_cast<uint32_t>(unclaimed_dynamic_idx),
                      static_input_product,
                      static_target_product,
                  });
                }
              } else {
                model_builder.RecordDimProvenance(new_dim_name, {
                    input_defs[0]->Name(),
                    static_cast<uint32_t>(unclaimed_dynamic_idx),
                    static_input_product,
                    static_target_product,
                });
              }
            }
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
