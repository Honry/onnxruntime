// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class MatMulNBitsBuilder : public BaseOpBuilder {
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
  bool HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
  bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                               const logging::Logger& logger) const override;
};

void MatMulNBitsBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Inputs B and zero_points (if present) must be initializers.
  // For 4-bit: they are uint8 (packed pairs) and need re-registration as int4 WebNN constants.
  // For 8-bit: B is uint8 and needs re-registration as uint8 WebNN constant with correct shape.
  //            zero_points (if present) also need re-registration.
  const auto& input_defs = node.InputDefs();
  NodeAttrHelper helper(node);
  const auto bits = helper.Get("bits", 4);

  if (bits == 8 ||
      (bits == 4 && input_defs[1]->TypeAsProto()->tensor_type().elem_type() ==
                        ONNX_NAMESPACE::TensorProto_DataType_UINT8)) {
    model_builder.AddInitializerToSkip(input_defs[1]->Name());  // B
    if (TensorExists(input_defs, 3)) {
      model_builder.AddInitializerToSkip(input_defs[3]->Name());  // zero_points
    }
  }
}

// WebNN doesn't provide a dedicated op for MatMulNBits, it can be simply decomposed by
// DequantizeLinear + Transpose + MatMul.
//
// Supports both 4-bit and 8-bit quantization:
// - 4-bit: B stored as uint8 (packed pairs), registered as 'int4' with shape
//          [N, n_blocks_per_col, blob_size * 2].
// - 8-bit: B stored as uint8 (one element per byte), registered as 'uint8' with shape
//          [N, n_blocks_per_col, block_size].
//
// Common transformations:
// 1. scales: reshape to [N, n_blocks_per_col, 1].
// 2. zero_points: optional. If present, must be a constant initializer with same shape as
//                 reshaped scales. For int4 without explicit zero_points, dequantizeLinear is
//                 called without zeroPoint (symmetric quantization). For uint8, default is 128.
Status MatMulNBitsBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                 const Node& node,
                                                 const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();
  const auto& initializers = model_builder.GetInitializerTensors();

  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val scales = model_builder.GetOperand(input_defs[2]->Name());

  std::vector<int64_t> B_shape;  // [N, n_blocks_per_col, blob_size]
  ORT_RETURN_IF_NOT(GetShape(*input_defs[1], B_shape, logger), "Cannot get B shape");

  NodeAttrHelper helper(node);
  const uint32_t K = helper.Get("K", 0);
  const uint32_t N = helper.Get("N", 0);
  const uint32_t bits = helper.Get("bits", 4);
  const uint32_t block_size = helper.Get("block_size", 32);
  const uint32_t n_blocks_per_col = SafeInt<uint32_t>(B_shape[1]);

  emscripten::val options = emscripten::val::object();
  emscripten::val dq_x = emscripten::val::undefined();
  emscripten::val x_zero_point = emscripten::val::undefined();
  const std::vector<uint32_t> x_scale_shape{N, n_blocks_per_col, 1};
  emscripten::val x_scale_shape_array = emscripten::val::array(x_scale_shape);
  const bool has_zero_points = TensorExists(input_defs, 3);

  if (bits == 4) {
    // 4-bit path: register as int4 (symmetric quantization, zp=0).
    // ONNX MatMulNBits stores weights as uint8 (packed pairs of 4-bit values) with default zp=8.
    // Dequantization: (uint4_val - 8) * scale. We reinterpret as int4 with zp=0:
    // int4_val * scale (mathematically equivalent). The conversion XORs each packed byte with
    // 0x88, which flips the sign bit (MSB) of both 4-bit nibbles, computing int4_val = uint4_val - 8.
    // The XOR is applied in-place on the .slice() copy that WebNN constant creation already
    // requires (for Wasm memory growth safety), so this adds zero extra memory allocation.

    // Helper: load tensor raw bytes as a JS Uint8Array (handles both external and in-memory data).
    auto load_as_buffer = [&](const ONNX_NAMESPACE::TensorProto& tensor,
                              emscripten::val& buffer) -> Status {
      if (utils::HasExternalData(tensor) && !utils::HasExternalDataInMemory(tensor)) {
        std::basic_string<ORTCHAR_T> external_file_path;
        onnxruntime::FileOffsetType data_offset;
        SafeInt<size_t> tensor_byte_size;
        ORT_RETURN_IF_ERROR(utils::GetExternalDataInfo(
            tensor, model_builder.GetGraphViewer().ModelPath(),
            external_file_path, data_offset, tensor_byte_size));
        auto load_fn = emscripten::val::module_property("webnnLoadExternalData");
        buffer = load_fn(emscripten::val(external_file_path),
                         static_cast<int32_t>(data_offset),
                         static_cast<int32_t>(tensor_byte_size));
        return Status::OK();
      }
      std::byte* tensor_ptr = nullptr;
      std::vector<uint8_t> unpacked_tensor;
      if (tensor.has_raw_data()) {
        tensor_ptr = reinterpret_cast<std::byte*>(
            const_cast<char*>(tensor.raw_data().c_str()));
      } else {
        ORT_RETURN_IF_NOT(UnpackInitializerData(tensor, unpacked_tensor,
                                                model_builder.GetGraphViewer(), logger),
                          "Failed to unpack tensor data");
        tensor_ptr = reinterpret_cast<std::byte*>(unpacked_tensor.data());
      }
      auto num_bytes = SafeInt<size_t>(Product(tensor.dims()));
      emscripten::val view = emscripten::val{
          emscripten::typed_memory_view(num_bytes,
                                        reinterpret_cast<uint8_t*>(tensor_ptr))};
      buffer = view.call<emscripten::val>("slice");
      return Status::OK();
    };

    // JS function to XOR each byte with 0x88 in-place (uint4 -> int4 sign bit flip).
    emscripten::val xor_fn = emscripten::val::global("Function").new_(
        emscripten::val("buf"),
        emscripten::val("const u8 = new Uint8Array(buf.buffer, buf.byteOffset, buf.length);"
                        "for (let i = 0; i < u8.length; i++) u8[i] ^= 0x88;"
                        "return buf;"));

    const uint32_t double_blob_size = SafeInt<uint32_t>(B_shape[2] * 2);
    const std::vector<uint32_t> b_shape_int4{N, n_blocks_per_col, double_blob_size};
    emscripten::val b_desc = emscripten::val::object();
    b_desc.set("dataType", emscripten::val("int4"));
    b_desc.set("shape", emscripten::val::array(b_shape_int4));
    b_desc.set("dimensions", emscripten::val::array(b_shape_int4));

    const auto B_tensor = *initializers.at(input_defs[1]->Name());
    emscripten::val b_buffer = emscripten::val::undefined();
    ORT_RETURN_IF_ERROR(load_as_buffer(B_tensor, b_buffer));
    b_buffer = xor_fn(b_buffer);
    dq_x = model_builder.GetBuilder().call<emscripten::val>(
        "constant", b_desc, b_buffer["buffer"]);

    if (has_zero_points) {
      emscripten::val zp_desc = emscripten::val::object();
      zp_desc.set("dataType", emscripten::val("int4"));
      zp_desc.set("shape", x_scale_shape_array);
      zp_desc.set("dimensions", x_scale_shape_array);

      const auto zp_tensor = *initializers.at(input_defs[3]->Name());
      emscripten::val zp_buffer = emscripten::val::undefined();
      ORT_RETURN_IF_ERROR(load_as_buffer(zp_tensor, zp_buffer));
      zp_buffer = xor_fn(zp_buffer);
      x_zero_point = model_builder.GetBuilder().call<emscripten::val>(
          "constant", zp_desc, zp_buffer["buffer"]);
    } else if (!IsZeroPointOptional()) {
      // Old API requires zeroPoint as positional arg; create a zero constant for int4.
      emscripten::val zp_desc = emscripten::val::object();
      zp_desc.set("dataType", emscripten::val("int4"));
      zp_desc.set("shape", x_scale_shape_array);
      zp_desc.set("dimensions", x_scale_shape_array);
      auto num_elements = (Product(x_scale_shape) + 1) / 2;
      emscripten::val zp_buffer =
          emscripten::val::global("Uint8Array").new_(num_elements);
      x_zero_point = model_builder.GetBuilder().call<emscripten::val>(
          "constant", zp_desc, zp_buffer);
    }
  } else {
    assert(bits == 8);
    // 8-bit path: B is stored as uint8 (one element per byte), register as uint8.
    const std::vector<uint32_t> x_shape{N, n_blocks_per_col, block_size};
    emscripten::val x_shape_array = emscripten::val::array(x_shape);
    emscripten::val x_desc = emscripten::val::object();
    x_desc.set("dataType", emscripten::val("uint8"));
    x_desc.set("shape", x_shape_array);
    x_desc.set("dimensions", x_shape_array);
    const auto B_tensor = *initializers.at(input_defs[1]->Name());
    ORT_RETURN_IF_ERROR(model_builder.RegisterConstant(B_tensor, dq_x, x_desc, logger));

    // zero_points for 8-bit
    emscripten::val zero_points_desc = emscripten::val::object();
    zero_points_desc.set("dataType", emscripten::val("uint8"));
    zero_points_desc.set("shape", x_scale_shape_array);
    zero_points_desc.set("dimensions", x_scale_shape_array);
    if (has_zero_points) {
      const auto zero_points_tensor = *initializers.at(input_defs[3]->Name());
      ORT_RETURN_IF_ERROR(model_builder.RegisterConstant(zero_points_tensor, x_zero_point, zero_points_desc, logger));
    } else if (!IsZeroPointOptional()) {
      // Old API requires zeroPoint as positional arg; create default 128 constant.
      auto num_elements = Product(x_scale_shape);
      emscripten::val default_zero_point_buffer = emscripten::val::global("Uint8Array").new_(num_elements);
      default_zero_point_buffer.call<void>("fill", 128);
      x_zero_point =
          model_builder.GetBuilder().call<emscripten::val>("constant", zero_points_desc, default_zero_point_buffer);
    }
  }

  // Prepare DequantizeLinear's x_scale input: reshape scales to [N, n_blocks_per_col, 1]
  options.set("label", node.Name() + "_reshape_scales");
  emscripten::val x_scale =
      model_builder.GetBuilder().call<emscripten::val>("reshape", scales, x_scale_shape_array, options);

  // DequantizeLinear (zeroPoint is optional for symmetric quantization)
  emscripten::val dq_options = emscripten::val::object();
  dq_options.set("label", node.Name() + "_dequantizeLinear");
  emscripten::val dq = emscripten::val::undefined();
  if (IsZeroPointOptional()) {
    if (!x_zero_point.isUndefined()) {
      dq_options.set("zeroPoint", x_zero_point);
    }
    dq = model_builder.GetBuilder().call<emscripten::val>(
        "dequantizeLinear", dq_x, x_scale, dq_options);
  } else {
    dq = model_builder.GetBuilder().call<emscripten::val>(
        "dequantizeLinear", dq_x, x_scale, x_zero_point, dq_options);
  }

  // Reshape DequantizeLinear to [N, K]
  options.set("label", node.Name() + "_reshape_dequantizeLinear");
  const std::vector<uint32_t> new_dq_shape{N, K};
  emscripten::val new_dq_shape_array = emscripten::val::array(new_dq_shape);
  emscripten::val dq_reshaped =
      model_builder.GetBuilder().call<emscripten::val>("reshape", dq, new_dq_shape_array, options);

  // Transpose reshaped DequantizeLinear to [K, N]
  options.set("label", node.Name() + "_transpose_dequantizeLinear");
  emscripten::val dq_transposed = model_builder.GetBuilder().call<emscripten::val>("transpose", dq_reshaped, options);

  // MatMul
  options.set("label", node.Name() + "_matmul");
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("matmul", input, dq_transposed, options);

  // Add output with bias if present
  if (TensorExists(input_defs, 5)) {
    emscripten::val bias = model_builder.GetOperand(input_defs[5]->Name());
    options.set("label", node.Name() + "_add_bias");
    output = model_builder.GetBuilder().call<emscripten::val>("add", output, bias, options);
  }

  model_builder.AddOperand(output_defs[0]->Name(), std::move(output));

  return Status::OK();
}

bool MatMulNBitsBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer,
                                           const Node& node,
                                           const WebnnDeviceType /* device_type */,
                                           const logging::Logger& logger) const {
  const auto& name = node.Name();
  const auto& input_defs = node.InputDefs();

  // Inputs B and zero_points (if present) must be initializers
  if (!graph_viewer.GetConstantInitializer(input_defs[1]->Name())) {  // B
    LOGS(logger, VERBOSE) << "Input B of MatMulNBits [" << name << "] must be known as initializer";
    return false;
  }
  if (TensorExists(input_defs, 3) && !graph_viewer.GetConstantInitializer(input_defs[3]->Name())) {  // zero_points
    LOGS(logger, VERBOSE) << "Input zero_points of MatMulNBits [" << name << "] must be known as initializer";
    return false;
  }

  // WebNN doesn't support g_idx input
  if (TensorExists(input_defs, 4)) {  // g_idx
    LOGS(logger, VERBOSE) << "Input g_idx of MatMulNBits [" << name << "] is not supported";
    return false;
  }

  NodeAttrHelper helper(node);
  const auto bits = helper.Get("bits", 4);
  if (bits != 4 && bits != 8) {
    LOGS(logger, VERBOSE) << "Only 4-bit and 8-bit quantization are supported for MatMulNBits [" << name
                          << "], got bits=" << bits;
    return false;
  }

  return true;
}

bool MatMulNBitsBuilder::HasSupportedInputsImpl(const GraphViewer&,
                                                const Node& node, const emscripten::val& wnn_limits,
                                                const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const std::string_view op_type = node.OpType();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    return false;
  }

  int32_t A_type = 0;
  int32_t B_type = 0;
  int32_t scales_type = 0;
  int32_t zero_points_type = 0;
  if (!GetType(*input_defs[0], A_type, logger) ||
      !GetType(*input_defs[1], B_type, logger) ||
      !GetType(*input_defs[2], scales_type, logger)) {
    return false;
  }

  const bool has_zero_points = TensorExists(input_defs, 3);
  if (has_zero_points && !GetType(*input_defs[3], zero_points_type, logger)) {
    return false;
  }

  InlinedVector<int32_t, 2> input_types = {A_type, scales_type};
  if (!AreDataTypesSame(op_type, input_types, logger)) {
    return false;
  }

  if (A_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT && A_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    LOGS(logger, VERBOSE) << "WebNN only supports float32 or float16 data type for input A of MatMulNBits";
    return false;
  }
  if (B_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    LOGS(logger, VERBOSE) << "WebNN only supports uint8 data type for input B of MatMulNBits";
    return false;
  }
  if (has_zero_points && zero_points_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    LOGS(logger, VERBOSE) << "WebNN only supports uint8 data type for input zero_points of MatMulNBits";
    return false;
  }

  // Determine the WebNN data type based on the bits attribute.
  // For 4-bit: int4 (symmetric quantization). For 8-bit: plain uint8.
  NodeAttrHelper attr_helper(node);
  const auto bits = attr_helper.Get("bits", 4);
  const int32_t dq_input_type = (bits == 8) ? ONNX_NAMESPACE::TensorProto_DataType_UINT8
                                            : ONNX_NAMESPACE::TensorProto_DataType_INT4;

  // Ensure the quantized data type is supported by WebNN's dequantizeLinear op.
  // zeroPoint is optional in WebNN dequantizeLinear, skip its type check.
  // Input rank: Only the rank of the first input (A) is flexible. Verify that its rank is supported by
  //             WebNN's matmul op.
  return IsDataTypeSupportedByOp("DequantizeLinear", dq_input_type,
                                 wnn_limits, "input", "x", logger) &&
         IsInputRankSupported(wnn_limits, "matmul", "a", input_shape.size(), node.Name(), logger);
}

bool MatMulNBitsBuilder::HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                                 const logging::Logger& logger) const {
  const auto& output_defs = node.OutputDefs();

  int32_t output_type = 0;
  if (!GetType(*output_defs[0], output_type, logger)) {
    return false;
  }

  if (output_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      output_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    LOGS(logger, VERBOSE) << "WebNN only supports float32 or float16 data type for output of MatMulNBits";
    return false;
  }

  return true;
}

void CreateMatMulNBitsOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<MatMulNBitsBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
