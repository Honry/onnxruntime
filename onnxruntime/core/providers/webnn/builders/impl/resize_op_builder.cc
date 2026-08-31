// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <math.h>

#include <algorithm>
#include <string>

#include "core/providers/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/cpu/tensor/reshape_helper.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"
#include "shape_utils.h"

namespace onnxruntime {
namespace webnn {

class ResizeOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  // Allow roi and scales potentially being empty inputs that are ignored during processing.
  ResizeOpBuilder() : BaseOpBuilder(/*allow empty inputs*/ true) {}
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const GraphViewer&, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer& graph_viewer, const Node& node,
                              const emscripten::val& wnn_limits,
                              const logging::Logger& logger) const override;

  // Resize opset 10- is very different than Resize opset 11+, with many key attributes missing.
  // We only support Resize opset 11+ here.
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 11; }
};

// Helper functions

// Given the indices of the (up to 4) input dims that Resize actually changes,
// determine the two axes WebNN's resample2d should operate on.
// resample2d always resamples exactly 2 axes (its scales/sizes/axes are length-2 lists),
// so at most two dims may change. WebNN spec allows those two axes to be any two
// distinct dims of the input.
// Returns false only if more than two dims change, which resample2d cannot express.
bool GetResample2dAxes(const std::vector<int64_t>& changed_axes,
                       std::vector<int64_t>& axes,
                       const logging::Logger& logger) {
  // changed_axes is expected to be sorted ascending (built by scanning dims 0..3).
  if (changed_axes.size() > 2) {
    LOGS(logger, VERBOSE) << "Resize: WebNN resample2d can resample at most 2 axes, but "
                          << changed_axes.size() << " dims are being resized";
    return false;
  }

  if (changed_axes.size() == 2) {
    axes = changed_axes;
  } else if (changed_axes.size() == 1) {
    // Only one dim is scaled; pad with an adjacent axis so axes has the required length 2.
    // Resampling an axis whose scale is 1 / size is unchanged is a no-op, so the partner is
    // arbitrary; an adjacent one keeps the common single-spatial-axis case on the trailing axes.
    const int64_t a = changed_axes[0];
    axes = (a < 3) ? std::vector<int64_t>{a, a + 1} : std::vector<int64_t>{a - 1, a};
  } else {
    // No dim changes (identity resize); default to the trailing spatial axes.
    axes = {2, 3};
  }

  return true;
}

bool GetResizeScalesAndAxes(const GraphViewer& graph_viewer,
                            const Node& node,
                            std::vector<float>& scales,
                            std::vector<int64_t>& axes,
                            const logging::Logger& logger) {
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() < 3)
    return false;

  const bool has_axes = !axes.empty();
  const auto* scales_init = graph_viewer.GetConstantInitializer(input_defs[2]->Name());
  if (!scales_init || scales_init->dims_size() != 1) {
    LOGS(logger, ERROR) << "Expecting 'scales' as a 1D constant initialized tensor.";
    return false;
  }

  // Number of elements of 'scales' tensor.
  const auto& scales_tensor = *scales_init;
  const auto num_of_scales = scales_tensor.dims()[0];

  if (has_axes && num_of_scales != 2) {
    LOGS(logger, ERROR) << "When 'axes' is provided, 'scales' should have 2 elements.";
    return false;
  }

  if (!has_axes && num_of_scales != 4) {
    LOGS(logger, ERROR) << "When 'axes' is not provided, 'scales' should have 4 elements.";
    return false;
  }

  std::vector<uint8_t> unpacked_tensor;
  if (!UnpackInitializerData(scales_tensor, unpacked_tensor, graph_viewer, logger)) {
    return false;
  };
  const float* scales_data = reinterpret_cast<const float*>(unpacked_tensor.data());

  if (has_axes) {
    // 'axes' is specified since opset 18+, 'scales' should have 2 elements.
    scales = std::vector<float>{scales_data, scales_data + 2};
  } else {
    // Before opset 18, 'scales' should have 4 elements.
    // Infer the two axes to resample from whichever dims are actually scaled (scale != 1),
    // so that both NCHW ([1,1,sh,sw]) and NHWC ([1,sh,sw,1]) layouts are supported.
    std::vector<float> onnx_scales{scales_data, scales_data + 4};
    std::vector<int64_t> changed_axes;
    for (size_t i = 0; i < 4; ++i) {
      if (onnx_scales[i] != 1.0f) {
        changed_axes.push_back(static_cast<int64_t>(i));
      }
    }

    if (!GetResample2dAxes(changed_axes, axes, logger)) {
      return false;
    }

    scales = {onnx_scales[static_cast<size_t>(axes[0])], onnx_scales[static_cast<size_t>(axes[1])]};
  }

  return true;
}

bool GetResizeSizesAndAxes(const GraphViewer& graph_viewer,
                           const Node& node,
                           std::vector<int64_t>& sizes,
                           std::vector<int64_t>& axes,
                           const gsl::span<int64_t>& input_shape,
                           const logging::Logger& logger) {
  const auto& input_defs = node.InputDefs();
  if (input_defs.size() < 4)
    return false;

  const bool has_axes = !axes.empty();
  const auto* sizes_init = graph_viewer.GetConstantInitializer(input_defs[3]->Name());
  if (!sizes_init || sizes_init->dims_size() != 1) {
    LOGS(logger, ERROR) << "'sizes' should be a 1D constant initializer tensor.";
    return false;
  }

  const auto& sizes_tensor = *sizes_init;
  // Number of elements of sizes tensor.
  const auto num_of_sizes = sizes_tensor.dims()[0];
  if (has_axes && num_of_sizes != 2) {
    LOGS(logger, ERROR) << "When 'axes' is provided, 'sizes' should have 2 elements.";
    return false;
  }

  if (!has_axes && num_of_sizes != 4) {
    LOGS(logger, ERROR) << "When 'axes' is not provided, 'sizes' should have 4 elements.";
    return false;
  }

  std::vector<uint8_t> unpacked_tensor;
  if (!UnpackInitializerData(sizes_tensor, unpacked_tensor, graph_viewer, logger)) {
    return false;
  }
  const int64_t* sizes_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());

  if (has_axes) {
    // 'axes' is specified since opset 18+, 'sizes' should have 2 elements.
    sizes = std::vector<int64_t>{sizes_data, sizes_data + 2};
  } else {
    // Before opset 18, 'sizes' should have 4 elements.
    // Infer the two axes to resample from whichever dims actually change size,
    // so that both NCHW and NHWC layouts are supported.
    std::vector<int64_t> onnx_sizes{sizes_data, sizes_data + 4};
    // A dynamic input dim (kDynamicDim) can't be compared at build time. Dims that provably
    // change are resampled first; dynamic dims fill any remaining slots, trailing ones first
    // since the (typically dynamic) batch dim is dim 0.
    std::vector<int64_t> changed_axes;
    std::vector<int64_t> dynamic_axes;
    for (size_t i = 0; i < 4; ++i) {
      if (input_shape[i] == kDynamicDim) {
        dynamic_axes.push_back(static_cast<int64_t>(i));
      } else if (onnx_sizes[i] != input_shape[i]) {
        changed_axes.push_back(static_cast<int64_t>(i));
      }
    }
    while (changed_axes.size() < 2 && !dynamic_axes.empty()) {
      changed_axes.push_back(dynamic_axes.back());
      dynamic_axes.pop_back();
    }
    std::sort(changed_axes.begin(), changed_axes.end());

    if (!GetResample2dAxes(changed_axes, axes, logger)) {
      return false;
    }

    sizes = {onnx_sizes[static_cast<size_t>(axes[0])], onnx_sizes[static_cast<size_t>(axes[1])]};
  }

  return true;
}

// Add operator related.

void ResizeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // We don't really use ROI here, so add it to skipped list if it's an initializer tensor.
  model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());  // ROI
  model_builder.AddInputToSkip(node.InputDefs()[1]->Name());        // ROI

  // We will still add scales to the skipped list even sizes are present,
  // since there is no use of it, we will not process it later.
  model_builder.AddInitializerToSkip(node.InputDefs()[2]->Name());  // scales
  model_builder.AddInputToSkip(node.InputDefs()[2]->Name());        // scales

  if (node.InputDefs().size() > 3) {
    const auto& sizes_name = node.InputDefs()[3]->Name();
    // Only skip sizes when it is a constant initializer (consumed at build time).
    // When it is an operand, we need it as the sizes input for resample2dDynamic.
    if (model_builder.GetGraphViewer().GetConstantInitializer(sizes_name)) {
      model_builder.AddInitializerToSkip(sizes_name);  // sizes
      model_builder.AddInputToSkip(sizes_name);        // sizes
    }
  }
}

Status ResizeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                              const Node& node,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");

  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node);

  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  const auto mode = helper.Get("mode", "nearest");
  if (mode == "linear") {
    options.set("mode", emscripten::val("linear"));
  } else {  // we already checked the mode must be NN or Bilinear in IsOpSupportedImpl.
    options.set("mode", emscripten::val("nearest-neighbor"));
  }

  std::vector<int64_t> axes = GetResolvedAxes(helper, 4);  // We already checked input shape is 4D in IsOpSupportedImpl.

  std::string sizes_name = GetTensorName(input_defs, 3);
  const bool is_constant_sizes = !sizes_name.empty() && Contains(initializers, sizes_name);
  const bool is_dynamic_sizes = !sizes_name.empty() && !is_constant_sizes;

  // Resolve the two resample axes (and the matching sizes / scales) before setting options.
  std::vector<int64_t> sizes;
  std::vector<float> scales;
  if (is_constant_sizes) {
    ORT_RETURN_IF_NOT(GetResizeSizesAndAxes(model_builder.GetGraphViewer(), node, sizes, axes, input_shape, logger),
                      "Error getting Resize sizes");
  } else if (is_dynamic_sizes) {
    // Without an 'axes' attribute the resampled dims can't be inferred from a runtime 'sizes'
    // operand, so assume NCHW.
    if (axes.empty()) {
      axes = {2, 3};
    }
  } else {
    ORT_RETURN_IF_NOT(GetResizeScalesAndAxes(model_builder.GetGraphViewer(), node, scales, axes, logger),
                      "Error getting Resize scales");
  }

  std::vector<uint32_t> webnn_axes = GetNarrowedIntFromInt64<uint32_t>(axes);
  options.set("axes", emscripten::val::array(webnn_axes));

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val output = emscripten::val::undefined();
  emscripten::val common_options = emscripten::val::object();

  if (is_dynamic_sizes) {
    // Dynamic sizes operand path: slice spatial dims + cast to uint32.
    emscripten::val sizes_operand = model_builder.GetOperand(input_defs[3]->Name());

    // When sizes has 4 elements [N,C,H,W], extract only spatial dims for WebNN.
    std::vector<int64_t> sizes_shape;
    if (GetShape(*input_defs[3], sizes_shape, logger) && !sizes_shape.empty() && sizes_shape[0] == 4) {
      common_options.set("label", node.Name() + "_sizes_slice");
      sizes_operand = model_builder.GetBuilder().call<emscripten::val>(
          "slice", sizes_operand,
          emscripten::val::array(std::vector<uint32_t>{static_cast<uint32_t>(axes[0])}),
          emscripten::val::array(std::vector<uint32_t>{static_cast<uint32_t>(webnn_axes.size())}),
          common_options);
    }

    // Cast to uint32 (ONNX sizes is int64, resample2dDynamic requires uint32).
    common_options.set("label", node.Name() + "_cast_sizes_uint32");
    sizes_operand = model_builder.GetBuilder().call<emscripten::val>(
        "cast", sizes_operand, emscripten::val("uint32"), common_options);

    options.set("sizes", sizes_operand);
    output = model_builder.GetBuilder().call<emscripten::val>("resample2dDynamic", input, options);
  } else if (!HasDynamicShape(input_shape)) {
    // Static path: use WebNN resample2d with sizes or scales.
    if (is_constant_sizes) {
      options.set("sizes", emscripten::val::array(GetNarrowedIntFromInt64<uint32_t>(sizes)));
    } else {
      options.set("scales", emscripten::val::array(scales));
    }
    output = model_builder.GetBuilder().call<emscripten::val>("resample2d", input, options);
  } else if (is_constant_sizes) {
    // Dynamic input + constant sizes: create uint32 constant for resample2dDynamic.
    std::vector<uint32_t> webnn_sizes = GetNarrowedIntFromInt64<uint32_t>(sizes);
    const emscripten::val& sizes_operand = model_builder.CreateOrGetConstant<uint32_t>(
        ONNX_NAMESPACE::TensorProto_DataType_UINT32, node.Name() + "_sizes",
        webnn_sizes, {static_cast<uint32_t>(webnn_sizes.size())});
    options.set("sizes", sizes_operand);
    output = model_builder.GetBuilder().call<emscripten::val>("resample2dDynamic", input, options);
  } else {
    // Dynamic input + constant scales: compute sizes at runtime via shape sub-ops.
    emscripten::val wnn_builder = model_builder.GetBuilder();
    common_options.set("label", node.Name() + "_input_shape");
    emscripten::val input_shape_op = wnn_builder.call<emscripten::val>("shape", input, common_options);

    // Extract the resampled dims in 'axes' order; they may be non-adjacent (e.g. {1, 3}) or
    // descending when given by the 'axes' attribute.
    emscripten::val spatial_shape = emscripten::val::undefined();
    if (axes[1] == axes[0] + 1) {
      spatial_shape = shape_utils::SliceShapeRange(
          wnn_builder, input_shape_op, static_cast<int32_t>(axes[0]), 2, node.Name() + "_spatial_slice");
    } else {
      emscripten::val dims = emscripten::val::array();
      for (size_t i = 0; i < 2; ++i) {
        dims.call<void>("push", shape_utils::SliceShapeRange(
                                    wnn_builder, input_shape_op, static_cast<int32_t>(axes[i]), 1,
                                    node.Name() + "_spatial_slice_" + std::to_string(i)));
      }
      common_options.set("label", node.Name() + "_spatial_concat");
      spatial_shape = wnn_builder.call<emscripten::val>("concat", dims, 0, common_options);
    }

    // shape(uint32) → float32 → mul(scales) → floor → uint32
    common_options.set("label", node.Name() + "_shape_to_float");
    emscripten::val spatial_float = wnn_builder.call<emscripten::val>(
        "cast", spatial_shape, emscripten::val("float32"), common_options);

    const emscripten::val& scales_const = model_builder.CreateOrGetConstant<float>(
        ONNX_NAMESPACE::TensorProto_DataType_FLOAT, node.Name() + "_scales",
        scales, {static_cast<uint32_t>(scales.size())});
    common_options.set("label", node.Name() + "_sizes_mul");
    emscripten::val sizes_float = wnn_builder.call<emscripten::val>(
        "mul", spatial_float, scales_const, common_options);

    common_options.set("label", node.Name() + "_sizes_floor");
    sizes_float = wnn_builder.call<emscripten::val>("floor", sizes_float, common_options);
    common_options.set("label", node.Name() + "_sizes_to_uint32");
    emscripten::val sizes_operand = wnn_builder.call<emscripten::val>(
        "cast", sizes_float, emscripten::val("uint32"), common_options);

    options.set("sizes", sizes_operand);
    output = model_builder.GetBuilder().call<emscripten::val>("resample2dDynamic", input, options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool ResizeOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer,
                                        const Node& node,
                                        const WebnnDeviceType /* device_type */,
                                        const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  NodeAttrHelper helper(node);

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS(logger, VERBOSE) << "Resize only support 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  {  // Check attributes.
    // antialias
    if (helper.Get("antialias", 0) != 0) {
      LOGS(logger, VERBOSE) << "Resize does not support antialias";
      return false;
    }

    // Ignore coordinate_transformation_mode because WebNN only supports half_pixel mode.
    // TODO: Validate coordinate_transformation_mode. Related spec issue for supporting attribute coordinate
    // transformation modes: https://github.com/webmachinelearning/webnn/issues/270

    // exclude_outside
    const auto exclude_outside = helper.Get("exclude_outside", 0);
    if (exclude_outside != 0) {
      LOGS(logger, VERBOSE) << "Resize does not support exclude_outside for now";
      return false;
    }

    // keep_aspect_ratio_policy
    const auto keep_aspect_ratio_policy = helper.Get("keep_aspect_ratio_policy", "stretch");
    if (keep_aspect_ratio_policy != "stretch") {
      LOGS(logger, VERBOSE) << "Resize does not support keep_aspect_ratio_policy: " << keep_aspect_ratio_policy;
      return false;
    }

    // mode
    const auto mode = helper.Get("mode", "nearest");
    bool is_linear_resize = mode == "linear";
    bool is_nearest_resize = mode == "nearest";
    // WebNN only supports "linear" and "nearest" modes.
    if (!is_linear_resize && !is_nearest_resize) {
      LOGS(logger, VERBOSE) << "Resize does not support input mode: " << mode;
      return false;
    }
  }

  {  // 'scales' and 'sizes' (if present) must be non-empty initializers, or sizes can be a dynamic operand.
    const std::string scales_name = GetTensorName(input_defs, 2);
    const std::string sizes_name = GetTensorName(input_defs, 3);

    // Check for 'sizes' first.
    // This handles Resize-11 where 'scales' was a required input but 'sizes' were used if provided.
    // 'scales' or 'sizes' may be empty tensor.
    bool using_sizes = !IsEmptyTensor(graph_viewer, sizes_name);
    bool using_scales = !using_sizes && !IsEmptyTensor(graph_viewer, scales_name);

    if (!using_scales && !using_sizes) {
      LOGS(logger, VERBOSE) << "Resize: only one of 'scales' and 'sizes' can be specified";
      return false;
    }

    // 'axes' is from opset 18 on and allows 'scales' or 'sizes' to have entries for the subset of 'axes'.
    // We fill with default values if necessary so that the processing is consistent across all supported opsets.
    std::vector<int64_t> axes = GetResolvedAxes(helper, input_size);
    if (!axes.empty()) {  // We have 'axes' attribute.
      if (axes.size() != 2 || axes[0] >= input_size || axes[1] >= input_size) {
        LOGS(logger, VERBOSE) << "Resize: invalid axes attribute";
        return false;
      }
    }

    if (using_sizes) {
      // sizes can be either a constant initializer or a dynamic operand.
      const auto* sizes_init = graph_viewer.GetConstantInitializer(sizes_name);
      if (sizes_init) {
        // Constant sizes path: validate the initializer contents.
        std::vector<int64_t> sizes;
        if (!GetResizeSizesAndAxes(graph_viewer, node, sizes, axes, input_shape, logger)) {
          return false;
        }
      }
      // Dynamic sizes: accepted, will use resample2dDynamic at build time.
    } else {  // We are using 'scales'.
      // 'scales' must be a constant initializer.
      std::vector<float> scales;
      if (!GetResizeScalesAndAxes(graph_viewer, node, scales, axes, logger)) {
        return false;
      }
    }
  }

  return true;
}

bool ResizeOpBuilder::HasSupportedInputsImpl(const GraphViewer& graph_viewer,
                                             const Node& node,
                                             const emscripten::val& wnn_limits,
                                             const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const std::string sizes_name = GetTensorName(input_defs, 3);

  // When sizes is a constant initializer (or absent/empty), the op maps to resample2d.
  // Delegate to the base class which checks input 0 against WebNN resample2d's limits.
  // When sizes is a non-constant operand, it's the dynamic path using resample2dDynamic.
  if (sizes_name.empty() || graph_viewer.GetConstantInitializer(sizes_name)) {
    return BaseOpBuilder::HasSupportedInputsImpl(graph_viewer, node, wnn_limits, logger);
  }

  // When sizes is a dynamic operand, check inputs against resample2dDynamic's limits.
  const std::string_view webnn_op_type = "resample2dDynamic";

  // Check input 0 (data tensor) against resample2dDynamic's "input" parameter.
  int32_t input_type;
  if (!GetType(*input_defs[0], input_type, logger)) {
    return false;
  }
  if (!IsDataTypeSupportedByWebNNOp("Resize", webnn_op_type, input_type, wnn_limits,
                                    "input", "input", logger)) {
    return false;
  }
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger) ||
      !IsRankSupportedByWebNNOp(wnn_limits, webnn_op_type, "input",
                            input_shape.size(), node.Name(), logger)) {
    return false;
  }

  // resample2dDynamic's sizes is always uint32 (we cast at build time).
  // Skip type check — ONNX sizes input is int64 but we handle the conversion.

  return true;
}

void CreateResizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ResizeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
