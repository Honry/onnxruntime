// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

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
    // Dynamic shape path: use input["shape"][i] for dynamic dims, literals for static dims.
    // blocksize is always a constant attribute.
    const int64_t channels = input_shape[1];  // channels must be static (validated in IsOpSupportedImpl)
    const int64_t new_channels = channels / (blocksize * blocksize);

    // Step 1: Reshape to 6D
    emscripten::val shape1 = emscripten::val::array();
    shape1.call<void>("push", input["shape"][0]);  // batch (may be dynamic)
    if (mode == "DCR") {
      shape1.call<void>("push", static_cast<uint32_t>(blocksize));
      shape1.call<void>("push", static_cast<uint32_t>(blocksize));
      shape1.call<void>("push", static_cast<uint32_t>(new_channels));
    } else {
      shape1.call<void>("push", static_cast<uint32_t>(new_channels));
      shape1.call<void>("push", static_cast<uint32_t>(blocksize));
      shape1.call<void>("push", static_cast<uint32_t>(blocksize));
    }
    shape1.call<void>("push", input["shape"][2]);  // height (may be dynamic)
    shape1.call<void>("push", input["shape"][3]);  // width (may be dynamic)

    options.set("label", node.Name() + "_reshape1");
    emscripten::val tmp = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", input, shape1, options);

    // Step 2: Transpose
    const std::vector<uint32_t> perm = (mode == "DCR")
                                           ? std::vector<uint32_t>{0, 3, 4, 1, 5, 2}
                                           : std::vector<uint32_t>{0, 1, 4, 2, 5, 3};
    options = emscripten::val::object();
    options.set("label", node.Name() + "_transpose");
    options.set("permutation", emscripten::val::array(perm));
    tmp = model_builder.GetBuilder().call<emscripten::val>("transpose", tmp, options);

    // Step 3: Reshape to output shape [batch, new_channels, height*blocksize, width*blocksize]
    emscripten::val shape2 = emscripten::val::array();
    shape2.call<void>("push", input["shape"][0]);  // batch
    shape2.call<void>("push", static_cast<uint32_t>(new_channels));

    // For new_height and new_width, use output shape proto to preserve dim_param names.
    const auto* output_shape_proto = output_defs[0]->Shape();
    if (output_shape_proto && output_shape_proto->dim_size() == 4) {
      // Use output shape proto dims for height and width (preserves dim_param names).
      for (int d = 2; d <= 3; ++d) {
        const auto& dim = output_shape_proto->dim(d);
        if (dim.has_dim_value()) {
          shape2.call<void>("push", static_cast<uint32_t>(dim.dim_value()));
        } else if (dim.has_dim_param()) {
          emscripten::val dim_desc = emscripten::val::object();
          dim_desc.set("name", emscripten::val(dim.dim_param()));
          shape2.call<void>("push", dim_desc);
        } else {
          // Fallback: use concrete computation if height/width are static.
          int64_t val = input_shape[d] * blocksize;
          shape2.call<void>("push", static_cast<uint32_t>(val));
        }
      }
    } else {
      // No output shape proto — height/width must be static.
      shape2.call<void>("push", static_cast<uint32_t>(input_shape[2] * blocksize));
      shape2.call<void>("push", static_cast<uint32_t>(input_shape[3] * blocksize));
    }

    options = emscripten::val::object();
    options.set("label", node.Name());
    emscripten::val output = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", tmp, shape2, options);

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
