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

// com.microsoft.CausalConvWithState is a stateful causal depthwise 1-D convolution used by the
// gated-DeltaNet / Mamba style "linear attention" blocks (e.g. Qwen3.5). WebNN has no dedicated
// operator, so it is decomposed. For ndim=1 with input (B, C, L), weight (C, 1, K), bias (C),
// past_state (B, C, K-1):
//
//   padded  = concat([past_state, input], axis=2)          // (B, C, K-1+L); zero left-pad if no state
//   conv    = conv2d(unsqueeze(padded,3), reshape(weight,[C,1,K,1]),
//                    {groups=C, padding=0, strides=1, dilations=1, bias})   // valid conv → width L
//   output  = squeeze(conv, 3)                             // (B, C, L)
//   output  = mul(output, sigmoid(output))                 // if activation is silu/swish
//   present = last (K-1) columns of padded along axis 2    // (B, C, K-1)
//
// The convolution is parallel across positions, so this is correct for any sequence length L
// (both decode L=1 and prefill L>1), and maps onto OpenVINO's GroupConvolution + Concat + Slice
// pattern (fused to PagedCausalConv1D).
class CausalConvWithStateOpBuilder : public BaseOpBuilder {
 public:
  // Optional inputs (bias, past_state) may be absent.
  CausalConvWithStateOpBuilder() : BaseOpBuilder(/*allow_empty_tensor_as_input=*/true) {}

  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const GraphViewer&, const Node& node,
                         const WebnnDeviceType device_type, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
  bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                               const logging::Logger& logger) const override;
};

Status CausalConvWithStateOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();
  NodeAttrHelper helper(node);
  const std::string activation = helper.Get("activation", "none");
  const bool apply_silu = (activation == "silu" || activation == "swish");

  emscripten::val builder = model_builder.GetBuilder();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val filter = model_builder.GetOperand(input_defs[1]->Name());

  std::vector<int64_t> weight_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[1], weight_shape, logger), "Cannot get weight shape");
  const int64_t channels = weight_shape[0];
  const int64_t kernel = weight_shape[2];
  const int64_t pad = kernel - 1;
  const uint32_t channels_u32 = SafeInt<uint32_t>(channels);

  emscripten::val common_options = emscripten::val::object();

  // Step 1: build the padded window [past_state | input] along the causal (last) axis.
  emscripten::val padded;
  if (TensorExists(input_defs, 3)) {
    emscripten::val past_state = model_builder.GetOperand(input_defs[3]->Name());
    emscripten::val concat_inputs = emscripten::val::array();
    concat_inputs.call<void>("push", past_state);
    concat_inputs.call<void>("push", input);
    common_options.set("label", node.Name() + "_concat_state");
    padded = builder.call<emscripten::val>("concat", concat_inputs, static_cast<uint32_t>(2), common_options);
  } else if (pad > 0) {
    // No past_state: left-pad with (K-1) zeros along axis 2. Input may be dynamic, so use padDynamic.
    emscripten::val begin_pad = model_builder.CreateOrGetConstant<uint32_t>(
        ONNX_NAMESPACE::TensorProto_DataType_UINT32, node.Name() + "_begin_pad",
        std::vector<uint32_t>{0, 0, SafeInt<uint32_t>(pad)}, {3});
    emscripten::val end_pad = model_builder.CreateOrGetConstant<uint32_t>(
        ONNX_NAMESPACE::TensorProto_DataType_UINT32, node.Name() + "_end_pad",
        std::vector<uint32_t>{0, 0, 0}, {3});
    common_options.set("label", node.Name() + "_zero_pad");
    padded = builder.call<emscripten::val>("padDynamic", input, begin_pad, end_pad, common_options);
  } else {
    padded = input;  // K == 1: no causal history.
  }

  // Step 2: reshape to 4D NCHW (append a size-1 width) and run a depthwise valid conv2d.
  common_options.set("label", node.Name() + "_unsqueeze_input");
  emscripten::val conv_input = builder.call<emscripten::val>(
      "unsqueeze", padded, emscripten::val::array(std::vector<uint32_t>{3}), common_options);

  // weight (C, 1, K) → (C, 1, K, 1); weight is a constant initializer so use a static reshape.
  std::vector<uint32_t> filter_shape_4d{channels_u32, 1, SafeInt<uint32_t>(kernel), 1};
  common_options.set("label", node.Name() + "_reshape_filter");
  filter = builder.call<emscripten::val>(
      "reshape", filter, emscripten::val::array(filter_shape_4d), common_options);

  emscripten::val conv_options = emscripten::val::object();
  conv_options.set("label", node.Name() + "_conv");
  conv_options.set("groups", channels_u32);
  conv_options.set("strides", emscripten::val::array(std::vector<uint32_t>{1, 1}));
  conv_options.set("dilations", emscripten::val::array(std::vector<uint32_t>{1, 1}));
  conv_options.set("padding", emscripten::val::array(std::vector<uint32_t>{0, 0, 0, 0}));
  if (TensorExists(input_defs, 2)) {
    conv_options.set("bias", model_builder.GetOperand(input_defs[2]->Name()));
  }
  emscripten::val conv = builder.call<emscripten::val>("conv2d", conv_input, filter, conv_options);

  // Step 3: drop the size-1 width dim → (B, C, L).
  emscripten::val squeeze_options = emscripten::val::object();
  squeeze_options.set("axes", emscripten::val::array(std::vector<uint32_t>{3}));
  squeeze_options.set("label", node.Name() + "_squeeze_output");
  emscripten::val output = builder.call<emscripten::val>("squeeze", conv, squeeze_options);

  // Step 4: optional fused SiLU/Swish activation: x * sigmoid(x).
  if (apply_silu) {
    common_options.set("label", node.Name() + "_sigmoid");
    emscripten::val sig = builder.call<emscripten::val>("sigmoid", output, common_options);
    common_options.set("label", node.Name() + "_silu");
    output = builder.call<emscripten::val>("mul", output, sig, common_options);
  }
  model_builder.AddOperand(output_defs[0]->Name(), std::move(output));

  // Step 5: present_state = last (K-1) columns of padded along axis 2, via sliceDynamic.
  //   starts = [0, 0, dim2 - pad],  sizes = [B, C, pad]   (all uint32).
  if (TensorExists(output_defs, 1) && pad > 0) {
    common_options.set("label", node.Name() + "_state_shape");
    emscripten::val padded_shape = builder.call<emscripten::val>("shape", padded, common_options);

    emscripten::val pad_const = model_builder.CreateOrGetConstant<uint32_t>(
        ONNX_NAMESPACE::TensorProto_DataType_UINT32, node.Name() + "_state_pad",
        std::vector<uint32_t>{SafeInt<uint32_t>(pad)}, {1});

    // sizes = concat([shape[0:2], pad])
    emscripten::val dims01 = shape_utils::SliceShapeRange(builder, padded_shape, 0, 2,
                                                          node.Name() + "_state_dims01");
    emscripten::val sizes_segments = emscripten::val::array();
    sizes_segments.call<void>("push", dims01);
    sizes_segments.call<void>("push", pad_const);
    common_options.set("label", node.Name() + "_state_sizes");
    emscripten::val sizes =
        builder.call<emscripten::val>("concat", sizes_segments, static_cast<uint32_t>(0), common_options);

    // starts = concat([0, 0], shape[2] - pad)
    emscripten::val dim2 = shape_utils::SliceShapeRange(builder, padded_shape, 2, 1,
                                                        node.Name() + "_state_dim2");
    common_options.set("label", node.Name() + "_state_start2");
    emscripten::val start2 = builder.call<emscripten::val>("sub", dim2, pad_const, common_options);
    emscripten::val zeros2 = model_builder.CreateOrGetConstant<uint32_t>(
        ONNX_NAMESPACE::TensorProto_DataType_UINT32, node.Name() + "_state_zeros2",
        std::vector<uint32_t>{0, 0}, {2});
    emscripten::val starts_segments = emscripten::val::array();
    starts_segments.call<void>("push", zeros2);
    starts_segments.call<void>("push", start2);
    common_options.set("label", node.Name() + "_state_starts");
    emscripten::val starts =
        builder.call<emscripten::val>("concat", starts_segments, static_cast<uint32_t>(0), common_options);

    emscripten::val slice_options = emscripten::val::object();
    slice_options.set("label", node.Name() + "_present_state");
    emscripten::val present_state =
        builder.call<emscripten::val>("sliceDynamic", padded, starts, sizes, slice_options);
    model_builder.AddOperand(output_defs[1]->Name(), std::move(present_state));
  }

  return Status::OK();
}

// Operator support related.

bool CausalConvWithStateOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const Node& node,
                                                     const WebnnDeviceType /* device_type */,
                                                     const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& name = node.Name();
  NodeAttrHelper helper(node);

  const int64_t ndim = helper.Get("ndim", static_cast<int64_t>(1));
  if (ndim != 1) {
    LOGS(logger, VERBOSE) << "CausalConvWithState [" << name << "] only supports ndim=1, got ndim=" << ndim;
    return false;
  }

  const std::string activation = helper.Get("activation", "none");
  if (activation != "none" && activation != "silu" && activation != "swish") {
    LOGS(logger, VERBOSE) << "CausalConvWithState [" << name << "] unsupported activation: " << activation;
    return false;
  }

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    LOGS(logger, VERBOSE) << "Cannot get CausalConvWithState input shape.";
    return false;
  }
  if (input_shape.size() != 3) {
    LOGS(logger, VERBOSE) << "CausalConvWithState [" << name << "] input must be rank 3 (B, C, L), got rank "
                          << input_shape.size();
    return false;
  }

  std::vector<int64_t> weight_shape;
  if (!GetShape(*input_defs[1], weight_shape, logger)) {
    LOGS(logger, VERBOSE) << "Cannot get CausalConvWithState weight shape.";
    return false;
  }
  if (weight_shape.size() != 3 || weight_shape[1] != 1) {
    LOGS(logger, VERBOSE) << "CausalConvWithState [" << name << "] weight must be (C, 1, K) depthwise.";
    return false;
  }
  // Weight must be a constant so we can read C and K at build time.
  if (graph_viewer.GetConstantInitializer(input_defs[1]->Name(), true) == nullptr) {
    LOGS(logger, VERBOSE) << "CausalConvWithState [" << name << "] requires a constant weight.";
    return false;
  }

  return true;
}

bool CausalConvWithStateOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                                          const emscripten::val& wnn_limits,
                                                          const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const std::string_view op_type = node.OpType();

  std::vector<int32_t> input_types;
  for (size_t i = 0; i < input_defs.size(); ++i) {
    if (TensorExists(input_defs, i)) {
      int32_t input_type;
      if (!GetType(*input_defs[i], input_type, logger)) {
        return false;
      }
      input_types.push_back(input_type);
    }
  }
  if (!AreDataTypesSame(op_type, input_types, logger)) {
    return false;
  }

  // Validate the input data type against each decomposed WebNN op.
  for (const std::string_view decomposed_op_type : decomposed_op_map.at(op_type)) {
    const std::string_view webnn_op_type = GetWebNNOpType(decomposed_op_type);
    const std::string_view webnn_input_name = GetWebNNOpFirstInputName(decomposed_op_type);
    if (!IsDataTypeSupportedByWebNNOp(
            op_type, webnn_op_type, input_types[0], wnn_limits, webnn_input_name, "input", logger)) {
      return false;
    }
  }

  // The convolution runs on the reshaped rank-4 input; make sure conv2d accepts that rank.
  return IsRankSupportedByWebNNOp(wnn_limits, "conv2d", "input", 4, node.Name(), logger);
}

bool CausalConvWithStateOpBuilder::HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                                           const logging::Logger& logger) const {
  const auto& output_defs = node.OutputDefs();
  const std::string_view op_type = node.OpType();
  int32_t output_type = 0;
  if (!GetType(*output_defs[0], output_type, logger)) {
    return false;
  }
  for (const std::string_view decomposed_op_type : decomposed_op_map.at(op_type)) {
    const std::string_view webnn_op_type = GetWebNNOpType(decomposed_op_type);
    if (!IsDataTypeSupportedByWebNNOp(op_type, webnn_op_type, output_type, wnn_limits, "output", "output", logger)) {
      return false;
    }
  }
  return true;
}

void CreateCausalConvWithStateOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<CausalConvWithStateOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
