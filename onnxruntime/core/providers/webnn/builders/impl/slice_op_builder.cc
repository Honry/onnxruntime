// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <numeric>

#include "core/common/logging/logging.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class SliceOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer& graph_viewer, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
  // TODO: Support Slice opset < 10, which uses attributes for starts and ends.
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 10; }
};

// Add operator related.

void SliceOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& input_defs = node.InputDefs();
  const auto& initializers(model_builder.GetInitializerTensors());
  const auto& graph_viewer = model_builder.GetGraphViewer();

  const bool is_constant_starts = initializers.count(input_defs[1]->Name()) > 0;
  const bool is_constant_ends = initializers.count(input_defs[2]->Name()) > 0;

  // The constant path can be used when starts/ends are constant AND all SLICED axes
  // have static dimensions. Non-sliced axes may be dynamic.
  bool can_use_constant_path = false;
  if (is_constant_starts && is_constant_ends) {
    const auto* shape_proto = input_defs[0]->Shape();
    const bool has_dynamic_input = shape_proto &&
        std::any_of(shape_proto->dim().begin(), shape_proto->dim().end(),
                    [](const auto& dim) { return !dim.has_dim_value(); });
    if (!has_dynamic_input) {
      can_use_constant_path = true;
    } else {
      // Check if only non-sliced axes are dynamic.
      std::vector<int64_t> axes;
      const std::string axes_name = GetTensorName(input_defs, 3);
      if (!axes_name.empty()) {
        const auto* axes_init = graph_viewer.GetConstantInitializer(axes_name);
        if (axes_init) {
          const logging::Logger& logger = logging::LoggingManager::DefaultLogger();
          ReadIntArrayFrom1DTensor(*axes_init, axes, graph_viewer, logger);
        }
      }
      if (!axes.empty() && shape_proto) {
        can_use_constant_path = true;
        const size_t rank = shape_proto->dim_size();
        for (int64_t axis : axes) {
          const size_t dim_idx = SafeInt<size_t>(HandleNegativeAxis(axis, rank));
          if (!shape_proto->dim(dim_idx).has_dim_value()) {
            can_use_constant_path = false;
            break;
          }
        }
      }
    }
  }

  if (can_use_constant_path) {
    // Constant path: skip all initializer inputs (consumed at build time).
    for (size_t i = 1; i < input_defs.size(); i++) {
      model_builder.AddInitializerToSkip(input_defs[i]->Name());
    }
  } else {
    // Dynamic path (at least one of starts/ends is a runtime operand):
    // Skip axes and steps (consumed at build time as constants).
    // Do NOT skip starts/ends — they are needed as WebNN operands.
    for (size_t i = 3; i < input_defs.size(); i++) {
      const std::string name = GetTensorName(input_defs, i);
      if (!name.empty() && initializers.count(name) > 0) {
        model_builder.AddInitializerToSkip(name);
      }
    }
  }
}

// Constant path: all of starts/ends/axes/steps are constant initializers.
// Reads them at build time, handles negative steps via reverse, then emits WebNN slice.
Status BuildConstantSlice(ModelBuilder& model_builder, const Node& node,
                          emscripten::val input, const std::vector<int64_t>& input_shape,
                          emscripten::val& output, const logging::Logger& logger) {
  const auto& input_defs = node.InputDefs();
  const size_t rank = input_shape.size();

  std::vector<int64_t> input_starts;
  std::vector<int64_t> input_ends;
  std::vector<int64_t> input_axes;
  std::vector<int64_t> input_steps;
  SliceOp::PrepareForComputeMetadata compute_metadata(input_shape);
  const auto CopyInputData = [&input_defs, &model_builder, &logger](size_t input_idx,
                                                                    std::vector<int64_t>& data,
                                                                    bool is_required = false) {
    data.clear();
    std::string input_name;
    if (!is_required) {
      if (input_defs.size() <= input_idx)
        return Status::OK();
      input_name = input_defs[input_idx]->Name();
      if (input_name.empty())
        return Status::OK();
    }
    input_name = input_defs[input_idx]->Name();
    const auto& inits(model_builder.GetInitializerTensors());
    const auto& tensor = *inits.at(input_name);
    ORT_RETURN_IF_NOT(ReadIntArrayFrom1DTensor(tensor, data, model_builder.GetGraphViewer(), logger),
                      "Data type for starts or ends inputs is not supported in this build.");
    return Status::OK();
  };
  ORT_RETURN_IF_ERROR(CopyInputData(1, input_starts, true));
  ORT_RETURN_IF_ERROR(CopyInputData(2, input_ends, true));
  ORT_RETURN_IF_ERROR(CopyInputData(3, input_axes));
  ORT_RETURN_IF_ERROR(CopyInputData(4, input_steps));
  ORT_RETURN_IF_ERROR(
      SliceOp::PrepareForComputeHelper(input_starts, input_ends, input_axes, input_steps, compute_metadata));

  // Handle negative steps via reverse.
  std::vector<uint32_t> reverse_axes;
  emscripten::val reverse_output = input;
  for (size_t i = 0; i < rank; ++i) {
    if (compute_metadata.steps_[i] < 0) {
      reverse_axes.push_back(SafeInt<uint32_t>(i));
      compute_metadata.steps_[i] = -compute_metadata.steps_[i];
      compute_metadata.starts_[i] = input_shape[i] - 1 - compute_metadata.starts_[i];
      compute_metadata.ends_[i] = input_shape[i] - 1 - compute_metadata.ends_[i];
    }
  }
  if (!reverse_axes.empty()) {
    emscripten::val reverse_options = emscripten::val::object();
    reverse_options.set("axes", emscripten::val::array(reverse_axes));
    reverse_options.set("label", node.Name() + "_reverse");
    reverse_output = model_builder.GetBuilder().call<emscripten::val>("reverse", input, reverse_options);
  }

  // Identify which axes are actually sliced (not identity pass-through).
  // For non-sliced axes: starts=0, ends=input_shape[i], steps=1 (set by PrepareForComputeHelper).
  // For dynamic non-sliced axes: ends=-1 == input_shape[i]=-1, so the check still works.
  std::vector<size_t> sliced_axes;
  for (size_t i = 0; i < rank; ++i) {
    if (compute_metadata.steps_[i] != 1 || compute_metadata.starts_[i] != 0 ||
        compute_metadata.ends_[i] != input_shape[i]) {
      sliced_axes.push_back(i);
    }
  }

  output = reverse_output;
  if (!sliced_axes.empty()) {
    const bool has_dynamic_non_sliced = HasDynamicShape(input_shape);
    if (has_dynamic_non_sliced) {
      // Dynamic non-sliced axes: WebNN slice requires concrete sizes for ALL axes,
      // but dynamic dims have no concrete size at build time.
      // Use gather on each sliced axis instead — gather preserves dim descriptors
      // on non-gathered axes, avoiding the cascading dim descriptor breakage.
      for (size_t axis : sliced_axes) {
        const int32_t start = static_cast<int32_t>(compute_metadata.starts_[axis]);
        const int32_t step = static_cast<int32_t>(compute_metadata.steps_[axis]);
        const int64_t extent = compute_metadata.ends_[axis] - compute_metadata.starts_[axis];
        const uint32_t count = static_cast<uint32_t>((extent + step - 1) / step);

        // Build indices: [start, start+step, start+2*step, ...].
        std::vector<int32_t> indices_data(count);
        for (uint32_t j = 0; j < count; ++j) {
          indices_data[j] = start + static_cast<int32_t>(j) * step;
        }

        const std::string indices_name = node.Name() + "_gather_indices_axis" + std::to_string(axis);
        emscripten::val indices_operand = model_builder.CreateOrGetConstant<int32_t>(
            ONNX_NAMESPACE::TensorProto_DataType_INT32, indices_name, indices_data, {count});

        emscripten::val gather_options = emscripten::val::object();
        gather_options.set("axis", static_cast<uint32_t>(axis));
        gather_options.set("label", node.Name() + "_gather_axis" + std::to_string(axis));
        output = model_builder.GetBuilder().call<emscripten::val>(
            "gather", output, indices_operand, gather_options);
      }
    } else {
      // All dims are static — use slice directly.
      std::vector<uint32_t> starts(rank);
      std::vector<uint32_t> sizes(rank);
      std::vector<uint32_t> steps(rank);
      for (size_t i = 0; i < rank; ++i) {
        starts[i] = static_cast<uint32_t>(compute_metadata.starts_[i]);
        steps[i] = static_cast<uint32_t>(compute_metadata.steps_[i]);
        int64_t extent = compute_metadata.ends_[i] - compute_metadata.starts_[i];
        int64_t step = compute_metadata.steps_[i];
        sizes[i] = static_cast<uint32_t>((extent + step - 1) / step);
      }
      emscripten::val options = emscripten::val::object();
      options.set("strides", emscripten::val::array(steps));
      options.set("label", node.Name());
      output = model_builder.GetBuilder().call<emscripten::val>(
          "slice", reverse_output, emscripten::val::array(starts),
          emscripten::val::array(sizes), options);
    }
  }

  return Status::OK();
}

// Dynamic path: at least one of starts/ends is a runtime operand.
// WebNN dynamicSlice accepts starts/ends as operands with native support for negative
// indices and clamping. The preprocessing subgraph only needs to:
//   1. Reverse negative-step axes on input (constant decision)
//   2. Transform starts/ends for reversed axes: new_val = -1 - old_val
//   3. Expand partial axes to full rank (if axes ⊂ all dims)
//   4. Call dynamicSlice(input, starts, ends, {strides})
Status BuildDynamicSlice(ModelBuilder& model_builder, const Node& node,
                         emscripten::val input, const std::vector<int64_t>& input_shape,
                         emscripten::val& output, const logging::Logger& logger) {
  const auto& input_defs = node.InputDefs();
  const auto& graph_viewer = model_builder.GetGraphViewer();
  emscripten::val builder = model_builder.GetBuilder();
  const std::string& label = node.Name();
  const size_t rank = input_shape.size();

  // Read constant axes (default: all dimensions).
  std::vector<int64_t> axes;
  const std::string axes_name = GetTensorName(input_defs, 3);
  if (!axes_name.empty()) {
    const auto* axes_init = graph_viewer.GetConstantInitializer(axes_name);
    ORT_RETURN_IF_NOT(axes_init && ReadIntArrayFrom1DTensor(*axes_init, axes, graph_viewer, logger),
                      "Cannot read axes initializer.");
  }
  if (axes.empty()) {
    axes.resize(rank);
    std::iota(axes.begin(), axes.end(), 0);
  }
  for (auto& axis : axes) {
    axis = HandleNegativeAxis(axis, rank);
  }
  const size_t num_axes = axes.size();

  // Read constant steps (default: all 1s). Negative steps are handled via reverse.
  std::vector<int64_t> steps;
  const std::string steps_name = GetTensorName(input_defs, 4);
  if (!steps_name.empty()) {
    const auto* steps_init = graph_viewer.GetConstantInitializer(steps_name);
    ORT_RETURN_IF_NOT(steps_init && ReadIntArrayFrom1DTensor(*steps_init, steps, graph_viewer, logger),
                      "Cannot read steps initializer.");
  }
  if (steps.empty()) {
    steps.assign(num_axes, 1);
  }

  // Step 1: Reverse negative-step axes on input and negate the steps.
  bool has_neg_steps = false;
  std::vector<bool> is_neg_step(num_axes, false);
  {
    std::vector<uint32_t> reverse_axes;
    for (size_t i = 0; i < num_axes; ++i) {
      if (steps[i] < 0) {
        reverse_axes.push_back(SafeInt<uint32_t>(axes[i]));
        is_neg_step[i] = true;
        has_neg_steps = true;
        steps[i] = -steps[i];
      }
    }
    if (!reverse_axes.empty()) {
      emscripten::val reverse_options = emscripten::val::object();
      reverse_options.set("axes", emscripten::val::array(reverse_axes));
      reverse_options.set("label", label + "_reverse");
      input = builder.call<emscripten::val>("reverse", input, reverse_options);
    }
  }

  // Get starts/ends as operands.
  emscripten::val starts_op = model_builder.GetOperand(input_defs[1]->Name());
  emscripten::val ends_op = model_builder.GetOperand(input_defs[2]->Name());

  // Step 2: For negative-step axes, transform to reversed coordinate space.
  // After reversing an axis of dimension D, original index i maps to (D-1)-i.
  // Using the identity (D-1)-i ≡ -1-i (with WebNN's native negative index handling),
  // we compute: new_val = -1 - old_val = sub(-1, old_val).
  // For positive-step axes, starts/ends pass through unchanged.
  if (has_neg_steps) {
    // Determine effective operand type for type-matched constants.
    int32_t tind_type;
    ORT_RETURN_IF_NOT(GetType(*input_defs[1], tind_type, logger), "Cannot get starts type");
    const bool use_int64 = tind_type == ONNX_NAMESPACE::TensorProto_DataType_INT64 &&
                           model_builder.IsInt64Supported();
    const int32_t index_data_type = use_int64 ? ONNX_NAMESPACE::TensorProto_DataType_INT64
                                              : ONNX_NAMESPACE::TensorProto_DataType_INT32;
    const auto shape_1d = [](size_t n) { return std::vector<uint32_t>{static_cast<uint32_t>(n)}; };
    emscripten::val neg_one_const = use_int64
                                        ? model_builder.CreateOrGetConstant<int64_t>(index_data_type, int64_t{-1}, shape_1d(num_axes))
                                        : model_builder.CreateOrGetConstant<int32_t>(index_data_type, int32_t{-1}, shape_1d(num_axes));

    emscripten::val options = emscripten::val::object();

    // Check if all specified axes have negative steps (no mixing).
    const bool all_neg = std::all_of(is_neg_step.begin(), is_neg_step.end(), [](bool b) { return b; });

    if (all_neg) {
      // All axes have negative steps — transform all directly.
      options.set("label", label + "_starts_transform");
      starts_op = builder.call<emscripten::val>("sub", neg_one_const, starts_op, options);
      options.set("label", label + "_ends_transform");
      ends_op = builder.call<emscripten::val>("sub", neg_one_const, ends_op, options);
    } else {
      // Mixed positive/negative steps — use where to select per axis.
      std::vector<uint8_t> neg_mask(num_axes, 0);
      for (size_t i = 0; i < num_axes; ++i) {
        if (is_neg_step[i]) neg_mask[i] = 1;
      }
      emscripten::val neg_mask_const = model_builder.CreateOrGetConstant<uint8_t>(
          ONNX_NAMESPACE::TensorProto_DataType_UINT8, label + "_neg_mask", neg_mask,
          {static_cast<uint32_t>(num_axes)});

      options.set("label", label + "_starts_transform");
      emscripten::val starts_transformed = builder.call<emscripten::val>(
          "sub", neg_one_const, starts_op, options);
      options.set("label", label + "_starts_select_neg");
      starts_op = builder.call<emscripten::val>(
          "where", neg_mask_const, starts_transformed, starts_op, options);

      options.set("label", label + "_ends_transform");
      emscripten::val ends_transformed = builder.call<emscripten::val>(
          "sub", neg_one_const, ends_op, options);
      options.set("label", label + "_ends_select_neg");
      ends_op = builder.call<emscripten::val>(
          "where", neg_mask_const, ends_transformed, ends_op, options);
    }
  }

  // Step 3: Expand to full rank if axes is a subset of all dimensions.
  emscripten::val starts_full = starts_op;
  emscripten::val ends_full = ends_op;
  if (num_axes < rank) {
    // Determine effective operand type for type-matched constants.
    int32_t tind_type;
    ORT_RETURN_IF_NOT(GetType(*input_defs[1], tind_type, logger), "Cannot get starts type");
    const bool use_int64 = tind_type == ONNX_NAMESPACE::TensorProto_DataType_INT64 &&
                           model_builder.IsInt64Supported();
    const int32_t index_data_type = use_int64 ? ONNX_NAMESPACE::TensorProto_DataType_INT64
                                              : ONNX_NAMESPACE::TensorProto_DataType_INT32;
    const auto shape_1d = [](size_t n) { return std::vector<uint32_t>{static_cast<uint32_t>(n)}; };
    const auto CreateIndexConstant = [&](const std::string& name,
                                         const std::vector<int64_t>& values,
                                         size_t n) -> const emscripten::val& {
      if (use_int64) {
        return model_builder.CreateOrGetConstant<int64_t>(index_data_type, name, values, shape_1d(n));
      } else {
        std::vector<int32_t> i32_values(values.begin(), values.end());
        return model_builder.CreateOrGetConstant<int32_t>(index_data_type, name, i32_values, shape_1d(n));
      }
    };

    // Use gather + where to expand partial-axis tensors to full-rank.
    std::vector<int32_t> gather_indices(rank, 0);
    std::vector<uint8_t> mask(rank, 0);
    for (size_t i = 0; i < num_axes; ++i) {
      const size_t d = SafeInt<size_t>(axes[i]);
      gather_indices[d] = static_cast<int32_t>(i);
      mask[d] = 1;
    }
    // Non-sliced axes: start=0, end=dim (full range).
    // For dynamic dims (-1 placeholder), use INT32_MAX so dynamicSlice clamps to actual size.
    std::vector<int64_t> starts_default(rank, 0);
    std::vector<int64_t> ends_default(rank);
    for (size_t d = 0; d < rank; ++d) {
      ends_default[d] = input_shape[d] > 0 ? input_shape[d]
                                           : static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    }

    emscripten::val gather_idx_const = model_builder.CreateOrGetConstant<int32_t>(
        ONNX_NAMESPACE::TensorProto_DataType_INT32, label + "_gather_idx",
        std::vector<int32_t>(gather_indices.begin(), gather_indices.end()),
        {static_cast<uint32_t>(rank)});
    emscripten::val mask_const = model_builder.CreateOrGetConstant<uint8_t>(
        ONNX_NAMESPACE::TensorProto_DataType_UINT8, label + "_mask", mask,
        {static_cast<uint32_t>(rank)});
    emscripten::val starts_default_const = CreateIndexConstant(
        label + "_starts_default", starts_default, rank);
    emscripten::val ends_default_const = CreateIndexConstant(
        label + "_ends_default", ends_default, rank);

    emscripten::val options = emscripten::val::object();
    options.set("label", label + "_starts_gather");
    emscripten::val starts_expanded = builder.call<emscripten::val>(
        "gather", starts_op, gather_idx_const, options);
    options.set("label", label + "_ends_gather");
    emscripten::val ends_expanded = builder.call<emscripten::val>(
        "gather", ends_op, gather_idx_const, options);
    options.set("label", label + "_starts_select");
    starts_full = builder.call<emscripten::val>(
        "where", mask_const, starts_expanded, starts_default_const, options);
    options.set("label", label + "_ends_select");
    ends_full = builder.call<emscripten::val>(
        "where", mask_const, ends_expanded, ends_default_const, options);
  }

  // Step 4: Build full-rank strides and call dynamicSlice.
  // dynamicSlice requires uint32 starts/ends operands — cast if needed.
  std::vector<uint32_t> strides_full(rank, 1);
  for (size_t i = 0; i < num_axes; ++i) {
    strides_full[SafeInt<size_t>(axes[i])] = SafeInt<uint32_t>(steps[i]);
  }

  emscripten::val cast_options = emscripten::val::object();
  cast_options.set("label", label + "_cast_starts_uint32");
  starts_full = builder.call<emscripten::val>("cast", starts_full, emscripten::val("uint32"), cast_options);
  cast_options.set("label", label + "_cast_ends_uint32");
  ends_full = builder.call<emscripten::val>("cast", ends_full, emscripten::val("uint32"), cast_options);

  emscripten::val slice_options = emscripten::val::object();
  slice_options.set("strides", emscripten::val::array(strides_full));
  slice_options.set("label", label);
  output = builder.call<emscripten::val>(
      "dynamicSlice", input, starts_full, ends_full, slice_options);

  return Status::OK();
}

Status SliceOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                             const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());

  const auto& initializers(model_builder.GetInitializerTensors());
  const auto& graph_viewer = model_builder.GetGraphViewer();
  const bool is_constant_starts = initializers.count(input_defs[1]->Name()) > 0;
  const bool is_constant_ends = initializers.count(input_defs[2]->Name()) > 0;

  // The constant path (WebNN slice) can be used when starts/ends are constant AND
  // all SLICED axes have static dimensions. Non-sliced axes may be dynamic — their
  // output dims pass through unchanged from the input operand (preserving dim descriptors).
  bool can_use_constant_path = false;
  if (is_constant_starts && is_constant_ends) {
    if (!HasDynamicShape(input_shape)) {
      // All dims are static — trivially safe for constant path.
      can_use_constant_path = true;
    } else {
      // Check if only non-sliced axes are dynamic.
      // Read axes to determine which dims are actually sliced.
      std::vector<int64_t> axes;
      const std::string axes_name = GetTensorName(input_defs, 3);
      if (!axes_name.empty()) {
        const auto* axes_init = graph_viewer.GetConstantInitializer(axes_name);
        if (axes_init) {
          ReadIntArrayFrom1DTensor(*axes_init, axes, graph_viewer, logger);
        }
      }
      if (axes.empty()) {
        // Default: all dims are sliced — need all static.
        can_use_constant_path = !HasDynamicShape(input_shape);
      } else {
        // Only the specified axes are sliced — check only those are static.
        const size_t rank = input_shape.size();
        can_use_constant_path = true;
        for (int64_t axis : axes) {
          const size_t dim_idx = SafeInt<size_t>(HandleNegativeAxis(axis, rank));
          if (input_shape[dim_idx] <= 0) {
            can_use_constant_path = false;
            break;
          }
        }
      }
    }
  }

  emscripten::val output = emscripten::val::undefined();

  if (can_use_constant_path) {
    ORT_RETURN_IF_ERROR(BuildConstantSlice(model_builder, node, input, input_shape, output, logger));
  } else {
    ORT_RETURN_IF_ERROR(BuildDynamicSlice(model_builder, node, input, input_shape, output, logger));
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

bool SliceOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const Node& node,
                                       const WebnnDeviceType /* device_type */, const logging::Logger& logger) const {
  const auto& name = node.Name();
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();

  if (input_defs.size() < 3) {
    LOGS(logger, VERBOSE) << op_type << " [" << name << "] requires at least 3 inputs (data, starts, ends) but got "
                          << input_defs.size();
    return false;
  }

  const std::string starts_name = input_defs[1]->Name();
  const std::string ends_name = input_defs[2]->Name();
  const bool is_constant_starts = graph_viewer.GetConstantInitializer(starts_name) != nullptr;
  const bool is_constant_ends = graph_viewer.GetConstantInitializer(ends_name) != nullptr;
  const bool has_dynamic_input = HasDynamicShape(*input_defs[0]);

  // Determine if the constant path can be used: starts/ends constant AND all sliced axes static.
  bool use_constant_path = false;
  if (is_constant_starts && is_constant_ends) {
    if (!has_dynamic_input) {
      use_constant_path = true;
    } else {
      // Check if only non-sliced axes are dynamic.
      std::vector<int64_t> axes;
      const std::string axes_name = GetTensorName(input_defs, 3);
      if (!axes_name.empty()) {
        const auto* axes_init = graph_viewer.GetConstantInitializer(axes_name);
        if (axes_init) {
          ReadIntArrayFrom1DTensor(*axes_init, axes, graph_viewer, logger);
        }
      }
      if (!axes.empty()) {
        const auto* shape_proto = input_defs[0]->Shape();
        if (shape_proto) {
          const size_t rank = shape_proto->dim_size();
          use_constant_path = true;
          for (int64_t axis : axes) {
            const size_t dim_idx = SafeInt<size_t>(HandleNegativeAxis(axis, rank));
            if (!shape_proto->dim(dim_idx).has_dim_value()) {
              use_constant_path = false;
              break;
            }
          }
        }
      }
    }
  }

  if (use_constant_path) {
    // Constant path: axes and steps must also be constant initializers if present.
    for (size_t i = 3; i < input_defs.size(); i++) {
      const std::string input_name = GetTensorName(input_defs, i);
      if (!input_name.empty() && !graph_viewer.GetConstantInitializer(input_name)) {
        LOGS(logger, VERBOSE) << "Input [" << input_name << "] of " << op_type << " [" << name
                              << "] must be known as initializer";
        return false;
      }
    }
  } else {
    // Dynamic path (at least one of starts/ends is not constant):
    // Axes and steps must be constant. Negative steps are handled via reverse.
    for (size_t i = 3; i < input_defs.size(); i++) {
      const std::string input_name = GetTensorName(input_defs, i);
      if (!input_name.empty() && !graph_viewer.GetConstantInitializer(input_name)) {
        LOGS(logger, VERBOSE) << "Input '" << input_name << "' (axes or steps) of " << op_type << " [" << name
                              << "] must be a constant when starts or ends is not a constant";
        return false;
      }
    }

    // Zero steps are not allowed per ONNX spec. Negative steps are handled via reverse.
    const std::string steps_name = GetTensorName(input_defs, 4);
    if (!steps_name.empty()) {
      const auto* steps_init = graph_viewer.GetConstantInitializer(steps_name);
      if (steps_init) {
        std::vector<int64_t> steps;
        if (!ReadIntArrayFrom1DTensor(*steps_init, steps, graph_viewer, logger)) {
          return false;
        }
        if (std::any_of(steps.begin(), steps.end(), [](int64_t s) { return s == 0; })) {
          LOGS(logger, VERBOSE) << "Zero steps are not supported for " << op_type << " [" << name << "]";
          return false;
        }
      }
    }
  }

  return true;
}

bool SliceOpBuilder::HasSupportedInputsImpl(const GraphViewer& graph_viewer, const Node& node,
                                            const emscripten::val& wnn_limits, const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  // Slice opset >= 10 requires at least 3 inputs (data, starts, ends).
  // Opset < 10 uses attributes instead and has only 1 input — reject early.
  if (input_defs.size() < 3) {
    return false;
  }
  const std::string starts_name = input_defs[1]->Name();
  const std::string ends_name = input_defs[2]->Name();

  // Determine if the constant path will be used (same logic as AddToModelBuilderImpl).
  // Constant path requires starts/ends constant AND all sliced axes static.
  const bool has_dynamic_input = HasDynamicShape(*input_defs[0]);
  bool use_constant_path = false;
  if (graph_viewer.GetConstantInitializer(starts_name) && graph_viewer.GetConstantInitializer(ends_name)) {
    if (!has_dynamic_input) {
      use_constant_path = true;
    } else {
      // Check if only non-sliced axes are dynamic.
      std::vector<int64_t> axes;
      const std::string axes_name = GetTensorName(input_defs, 3);
      if (!axes_name.empty()) {
        const auto* axes_init = graph_viewer.GetConstantInitializer(axes_name);
        if (axes_init) {
          ReadIntArrayFrom1DTensor(*axes_init, axes, graph_viewer, logger);
        }
      }
      if (!axes.empty()) {
        const auto* shape_proto = input_defs[0]->Shape();
        if (shape_proto) {
          const size_t rank = shape_proto->dim_size();
          use_constant_path = true;
          for (int64_t axis : axes) {
            const size_t dim_idx = SafeInt<size_t>(HandleNegativeAxis(axis, rank));
            if (!shape_proto->dim(dim_idx).has_dim_value()) {
              use_constant_path = false;
              break;
            }
          }
        }
      }
    }
  }

  if (use_constant_path) {
    const auto& input = *input_defs[0];
    std::vector<int64_t> input_shape;
    if (!GetShape(*input_defs[0], input_shape, logger)) {
      return false;
    }

    int32_t input_type;
    if (!GetType(input, input_type, logger)) {
      return false;
    }

    const std::string_view op_type = node.OpType();

    // If there is step < 0, check data type support of reverse.
    if (TensorExists(input_defs, 4)) {
      std::vector<int64_t> steps;
      const auto* init = graph_viewer.GetConstantInitializer(input_defs[4]->Name());
      if (!init || !ReadIntArrayFrom1DTensor(*init, steps, graph_viewer, logger))
        return false;
      if (std::any_of(steps.begin(), steps.end(), [](int64_t step) { return step < 0; })) {
        if (!IsDataTypeSupportedByWebNNOp(op_type, "reverse", input_type, wnn_limits, "input", "data", logger) ||
            !IsInputRankSupported(wnn_limits, "reverse", "input", input_shape.size(), node.Name(), logger)) {
          return false;
        }
      }
    }

    // When input has dynamic non-sliced axes, we use gather instead of slice.
    // Check gather support in that case.
    if (has_dynamic_input) {
      if (!IsDataTypeSupportedByWebNNOp(op_type, "gather", input_type, wnn_limits, "input", "data", logger) ||
          !IsInputRankSupported(wnn_limits, "gather", "input", input_shape.size(), node.Name(), logger)) {
        return false;
      }
    }

    return IsDataTypeSupportedByOp(op_type, input_type, wnn_limits, "input", "data", logger) &&
           IsInputRankSupportedByOp(node, wnn_limits, logger);
  }

  // Dynamic path: check inputs against dynamicSlice's limits.
  const std::string_view webnn_op_type = "dynamicSlice";

  // Check input 0 (data tensor) against dynamicSlice's "input" parameter.
  int32_t input_type;
  if (!GetType(*input_defs[0], input_type, logger)) {
    return false;
  }
  if (!IsDataTypeSupportedByWebNNOp("Slice", webnn_op_type, input_type, wnn_limits,
                                    "input", "data", logger)) {
    return false;
  }
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger) ||
      !IsInputRankSupported(wnn_limits, webnn_op_type, "input",
                            input_shape.size(), node.Name(), logger)) {
    return false;
  }

  // Check that dynamicSlice supports the type for starts/ends.
  int32_t starts_type;
  if (!GetType(*input_defs[1], starts_type, logger)) {
    return false;
  }
  if (!IsDataTypeSupportedByWebNNOp("Slice", webnn_op_type, starts_type, wnn_limits,
                                    "starts", "starts", logger) ||
      !IsDataTypeSupportedByWebNNOp("Slice", webnn_op_type, starts_type, wnn_limits,
                                    "ends", "ends", logger)) {
    return false;
  }

  // If any steps are negative, reverse is also needed — check its limits.
  if (TensorExists(input_defs, 4)) {
    const auto* steps_init = graph_viewer.GetConstantInitializer(input_defs[4]->Name());
    if (steps_init) {
      std::vector<int64_t> steps;
      if (ReadIntArrayFrom1DTensor(*steps_init, steps, graph_viewer, logger) &&
          std::any_of(steps.begin(), steps.end(), [](int64_t s) { return s < 0; })) {
        if (!IsDataTypeSupportedByWebNNOp("Slice", "reverse", input_type,
                                          wnn_limits, "input", "data", logger) ||
            !IsInputRankSupported(wnn_limits, "reverse", "input",
                                  input_shape.size(), node.Name(), logger)) {
          return false;
        }
      }
    }
  }

  return true;
}

void CreateSliceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SliceOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
