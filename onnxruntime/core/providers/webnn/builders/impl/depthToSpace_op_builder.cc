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

class DepthToSpaceOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const GraphViewer&, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
  bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                               const logging::Logger& logger) const override;
};

// Add operator related.

Status DepthToSpaceOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                    const Node& node,
                                                    const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();

  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  const bool has_dynamic_shape = HasDynamicShape(input_shape);

  NodeAttrHelper helper(node);
  const int64_t blocksize = *helper.GetInt64("blocksize");
  const std::string mode = helper.Get("mode", "DCR");

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val options = emscripten::val::object();

  if (has_dynamic_shape) {
    // Dynamic shape path: B, H, W may be dynamic; channels and blocksize are static.
    const int64_t channels = input_shape[1];
    const int64_t new_channels = channels / (blocksize * blocksize);

    // Step 1: Reshape to 6D [B, bs, bs, C', H, W] (DCR) or [B, C', bs, bs, H, W] (CRD)
    std::vector<int64_t> shape1_dims;
    shape1_dims.push_back(0);  // B (dynamic)
    if (mode == "DCR") {
      shape1_dims.push_back(blocksize);
      shape1_dims.push_back(blocksize);
      shape1_dims.push_back(new_channels);
    } else {
      shape1_dims.push_back(new_channels);
      shape1_dims.push_back(blocksize);
      shape1_dims.push_back(blocksize);
    }
    shape1_dims.push_back(0);  // H (dynamic, dim 2 of input → dim 4 of target)
    shape1_dims.push_back(0);  // W (dynamic, dim 3 of input → dim 5 of target)
    // Note: ComputeShape gathers dim at same index — but here target dim 4 should get input dim 2.
    // ComputeShape can't handle cross-index gathering. Build manually using SliceShapeRange.
    emscripten::val wnn_builder = model_builder.GetBuilder();
    emscripten::val shape_opts = emscripten::val::object();
    shape_opts.set("label", node.Name() + "_shape1_shape");
    emscripten::val input_shape_op = wnn_builder.call<emscripten::val>("shape", input, shape_opts);

    emscripten::val shape1_segments = emscripten::val::array();
    // B from input dim 0
    shape1_segments.call<void>("push", shape_utils::SliceShapeRange(wnn_builder, input_shape_op, 0, 1));
    // Static middle dims
    if (mode == "DCR") {
      std::vector<uint32_t> mid{static_cast<uint32_t>(blocksize), static_cast<uint32_t>(blocksize),
                                static_cast<uint32_t>(new_channels)};
      shape1_segments.call<void>("push", model_builder.CreateOrGetConstant<uint32_t>(
          ONNX_NAMESPACE::TensorProto_DataType_UINT32, node.Name() + "_shape1_mid",
          mid, {3}));
    } else {
      std::vector<uint32_t> mid{static_cast<uint32_t>(new_channels), static_cast<uint32_t>(blocksize),
                                static_cast<uint32_t>(blocksize)};
      shape1_segments.call<void>("push", model_builder.CreateOrGetConstant<uint32_t>(
          ONNX_NAMESPACE::TensorProto_DataType_UINT32, node.Name() + "_shape1_mid",
          mid, {3}));
    }
    // H, W from input dims 2, 3
    shape1_segments.call<void>("push", shape_utils::SliceShapeRange(wnn_builder, input_shape_op, 2, 2));

    shape_opts.set("label", node.Name() + "_shape1_concat");
    emscripten::val shape1_operand = wnn_builder.call<emscripten::val>(
        "concat", shape1_segments, 0, shape_opts);

    options.set("label", node.Name() + "_reshape1");
    emscripten::val tmp = wnn_builder.call<emscripten::val>(
        "dynamicReshape", input, shape1_operand, options);

    // Step 2: Transpose
    const std::vector<uint32_t> perm = (mode == "DCR")
                                           ? std::vector<uint32_t>{0, 3, 4, 1, 5, 2}
                                           : std::vector<uint32_t>{0, 1, 4, 2, 5, 3};
    options = emscripten::val::object();
    options.set("label", node.Name() + "_transpose");
    options.set("permutation", emscripten::val::array(perm));
    tmp = wnn_builder.call<emscripten::val>("transpose", tmp, options);

    // Step 3: Reshape to [B, C', H*bs, W*bs]. After transpose, tensor is
    // [B, C', H, bs, W, bs]. Target is {0, C', -1, -1} but only one -1 allowed.
    // Use ComputeShape with {0, C', -1} on a flattened view... or build manually.
    // Simplest: shape(tmp) → gather B (dim 0), constant C', then for H*bs and W*bs
    // we can multiply: gather dim 2 * constant(bs), gather dim 4 * constant(bs).
    // But WebNN doesn't have elementwise multiply on shape operands easily.
    // Alternative: use {0, new_channels, -1} with flatten semantics — but input is 6D
    // and -1 would give H*bs*W*bs (wrong, we need two separate dims).
    //
    // Best approach: compute H*bs = reduceProduct(dims[2:4]), W*bs = reduceProduct(dims[4:6])
    shape_opts.set("label", node.Name() + "_shape2_shape");
    emscripten::val tmp_shape = wnn_builder.call<emscripten::val>("shape", tmp, shape_opts);

    emscripten::val shape2_segments = emscripten::val::array();
    // B from dim 0
    shape2_segments.call<void>("push", shape_utils::SliceShapeRange(wnn_builder, tmp_shape, 0, 1));
    // C' (static)
    shape2_segments.call<void>("push", model_builder.CreateOrGetConstant<uint32_t>(
        ONNX_NAMESPACE::TensorProto_DataType_UINT32, static_cast<uint32_t>(new_channels), {1}));
    // H*bs = reduceProduct(dims[2:4]) — dims 2,3 of transposed tensor are H and bs
    shape2_segments.call<void>("push", shape_utils::ReduceShapeRange(
        model_builder, tmp_shape, 2, 2, node.Name() + "_shape2_h_reduce"));
    // W*bs = reduceProduct(dims[4:6]) — dims 4,5 of transposed tensor are W and bs
    shape2_segments.call<void>("push", shape_utils::ReduceShapeRange(
        model_builder, tmp_shape, 4, 2, node.Name() + "_shape2_w_reduce"));

    shape_opts.set("label", node.Name() + "_shape2_concat");
    emscripten::val shape2_operand = wnn_builder.call<emscripten::val>(
        "concat", shape2_segments, 0, shape_opts);

    options = emscripten::val::object();
    options.set("label", node.Name());
    emscripten::val output = wnn_builder.call<emscripten::val>(
        "dynamicReshape", tmp, shape2_operand, options);

    model_builder.AddOperand(output_defs[0]->Name(), std::move(output));
  } else {
    // Static shape path: all dims are concrete.
    const int64_t batch = input_shape[0];
    const int64_t channels = input_shape[1];
    const int64_t height = input_shape[2];
    const int64_t width = input_shape[3];

    const int64_t new_channels = channels / (blocksize * blocksize);
    const int64_t new_height = height * blocksize;
    const int64_t new_width = width * blocksize;

    std::vector<uint32_t> shape1;
    std::vector<uint32_t> perm;
    if (mode == "DCR") {
      shape1 = {
          SafeInt<uint32_t>(batch),
          SafeInt<uint32_t>(blocksize),
          SafeInt<uint32_t>(blocksize),
          SafeInt<uint32_t>(new_channels),
          SafeInt<uint32_t>(height),
          SafeInt<uint32_t>(width)};
      perm = {0, 3, 4, 1, 5, 2};
    } else {
      shape1 = {
          SafeInt<uint32_t>(batch),
          SafeInt<uint32_t>(new_channels),
          SafeInt<uint32_t>(blocksize),
          SafeInt<uint32_t>(blocksize),
          SafeInt<uint32_t>(height),
          SafeInt<uint32_t>(width)};
      perm = {0, 1, 4, 2, 5, 3};
    }

    options.set("label", node.Name() + "_reshape1");
    emscripten::val tmp = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", input, emscripten::val::array(shape1), options);

    options.set("label", node.Name() + "_transpose");
    options.set("permutation", emscripten::val::array(perm));
    tmp = model_builder.GetBuilder().call<emscripten::val>("transpose", tmp, options);

    std::vector<uint32_t> shape2{
        SafeInt<uint32_t>(batch),
        SafeInt<uint32_t>(new_channels),
        SafeInt<uint32_t>(new_height),
        SafeInt<uint32_t>(new_width)};
    options = emscripten::val::object();
    options.set("label", node.Name());
    emscripten::val output = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", tmp, emscripten::val::array(shape2), options);

    model_builder.AddOperand(output_defs[0]->Name(), std::move(output));
  }
  return Status::OK();
}

// Operator support related.

bool DepthToSpaceOpBuilder::IsOpSupportedImpl(const GraphViewer&,
                                              const Node& node,
                                              const WebnnDeviceType /* device_type */,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    LOGS(logger, VERBOSE) << "Cannot get input shape";
    return false;
  }

  if (input_shape.size() != 4) {
    LOGS(logger, VERBOSE) << "DepthToSpace input must be 4D ([N,C,H,W]), got " << input_shape.size() << "D";
    return false;
  }

  NodeAttrHelper helper(node);
  const int64_t blocksize = *helper.GetInt64("blocksize");
  if (blocksize <= 0) {
    LOGS(logger, VERBOSE) << "blocksize must be positive";
    return false;
  }

  const int64_t channels = input_shape[1];
  if (channels <= 0) {
    LOGS(logger, VERBOSE) << "DepthToSpace requires static channel dimension, got dynamic";
    return false;
  }
  if (channels % (blocksize * blocksize) != 0) {
    LOGS(logger, VERBOSE) << "channels must be divisible by blocksize^2";
    return false;
  }

  return true;
}

bool DepthToSpaceOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                                   const emscripten::val& wnn_limits,
                                                   const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const std::string_view op_type = node.OpType();
  int32_t input_type = 0;
  if (!GetType(*input_defs[0], input_type, logger)) {
    return false;
  }

  // Check if the input data type is supported by each decomposed WebNN op.
  // Decomposed ops include: "Reshape" and "Transpose".
  for (const std::string_view decomposed_op_type : decomposed_op_map.at(op_type)) {
    const std::string_view webnn_op_type = GetWebNNOpType(decomposed_op_type);
    const std::string_view webnn_input_name = GetWebNNOpFirstInputName(decomposed_op_type);
    if (!IsDataTypeSupportedByWebNNOp(op_type, webnn_op_type, input_type, wnn_limits, webnn_input_name, "input", logger)) {
      return false;
    }
  }

  return true;
}

bool DepthToSpaceOpBuilder::HasSupportedOutputsImpl(const Node& node,
                                                    const emscripten::val& wnn_limits,
                                                    const logging::Logger& logger) const {
  const auto& output_defs = node.OutputDefs();
  const std::string_view op_type = node.OpType();
  int32_t output_type = 0;
  if (!GetType(*output_defs[0], output_type, logger)) {
    return false;
  }

  // Check if the output data type is supported by every decomposed WebNN op.
  for (const std::string_view decomposed_op_type : decomposed_op_map.at(op_type)) {
    const std::string_view webnn_op_type = GetWebNNOpType(decomposed_op_type);
    if (!IsDataTypeSupportedByWebNNOp(op_type, webnn_op_type, output_type, wnn_limits, "output", "output", logger)) {
      return false;
    }
  }

  return true;
}

void CreateDepthToSpaceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<DepthToSpaceOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
