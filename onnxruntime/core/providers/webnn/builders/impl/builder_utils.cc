// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/common.h>
#include "core/providers/shared/utils/utils.h"

#include "builder_utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "shape_utils.h"

namespace onnxruntime {
namespace webnn {

common::Status ComputeConvPads(const std::vector<int64_t> input_shape,
                               const int64_t weight_size_y,
                               const int64_t weight_size_x,
                               const std::vector<int64_t>& onnx_pads,
                               const std::vector<int64_t>& onnx_strides,
                               const std::vector<int64_t>& onnx_dilations,
                               AutoPadType auto_pad_type,
                               std::vector<int64_t>& pads_out) {
  const int64_t input_size_y = input_shape[2];
  const int64_t input_size_x = input_shape[3];
  const int64_t stride_y = onnx_strides[0];
  const int64_t stride_x = onnx_strides[1];
  const int64_t dilation_y = onnx_dilations[0];
  const int64_t dilation_x = onnx_dilations[1];

  int64_t padding_top = onnx_pads[0];
  int64_t padding_bottom = onnx_pads[2];
  int64_t padding_left = onnx_pads[1];
  int64_t padding_right = onnx_pads[3];

  ORT_RETURN_IF_ERROR(ComputePad(input_size_y,
                                 stride_y, weight_size_y, dilation_y,
                                 auto_pad_type,
                                 padding_top, padding_bottom));
  ORT_RETURN_IF_ERROR(ComputePad(input_size_x,
                                 stride_x, weight_size_x, dilation_x,
                                 auto_pad_type,
                                 padding_left, padding_right));

  pads_out = {padding_top, padding_left, padding_bottom, padding_right};

  return Status::OK();
}

common::Status HandleAutoPad(const std::vector<int64_t> input_shape,
                             const int64_t weight_size_y,
                             const int64_t weight_size_x,
                             const std::vector<int64_t>& onnx_pads,
                             const std::vector<int64_t>& onnx_strides,
                             const std::vector<int64_t>& onnx_dilations,
                             AutoPadType auto_pad_type,
                             std::vector<int64_t>& pads_out) {
  AutoPadType pad_type = (AutoPadType::SAME_UPPER == auto_pad_type) ? AutoPadType::SAME_UPPER : AutoPadType::SAME_LOWER;

  ORT_RETURN_IF_ERROR(ComputeConvPads(input_shape, weight_size_y, weight_size_x,
                                      onnx_pads, onnx_strides, onnx_dilations,
                                      pad_type, pads_out));
  return Status::OK();
}

common::Status ComputeConvTransposePadAndOutputShape(
    const int64_t in_size,
    const int64_t stride,
    const int64_t kernel,
    const int64_t dilation,
    const int64_t adj,
    AutoPadType pad_type,
    int64_t& pad_head,
    int64_t& pad_tail,
    int64_t& out_size) {
  // Output shape is explicitly provided - pad values will have to be computed.
  if (out_size != -1) {
    // total pad
    auto total_pad = ComputeTotalPad(in_size, stride, adj, kernel, dilation, out_size);
    DistributePadding(pad_type, total_pad, pad_head, pad_tail);
    return Status::OK();
  }

  // Output shape is not provided - it needs to be computed along with pad values (if applicable).

  // Compute padding if the auto_pad attribute is SAME_UPPER/SAME_LOWER.
  if (pad_type == AutoPadType::SAME_UPPER || pad_type == AutoPadType::SAME_LOWER) {
    // The ONNX spec says if `auto_pad` attribute is set, pad until the `out_size`
    // is `in_size * stride`.
    auto total_pad = ComputeTotalPad(in_size, stride, adj,
                                     kernel, dilation, /*out_size = */ in_size * stride);
    DistributePadding(pad_type, total_pad, pad_head, pad_tail);
  }

  out_size = (in_size - 1) * stride + adj + (kernel - 1) * dilation + 1 - pad_head - pad_tail;

  return Status::OK();
}

common::Status ComputeConvTransposePadsAndOutputShape(const std::vector<int64_t> input_shape,
                                                      const int64_t weight_size_y,
                                                      const int64_t weight_size_x,
                                                      const std::vector<int64_t>& onnx_pads,
                                                      const std::vector<int64_t>& onnx_strides,
                                                      const std::vector<int64_t>& onnx_dilations,
                                                      const std::vector<int64_t>& onnx_output_padding,
                                                      AutoPadType auto_pad_type,
                                                      std::vector<int64_t>& pads_out,
                                                      std::vector<int64_t>& output_shape_out) {
  const int64_t input_size_y = input_shape[2];
  const int64_t input_size_x = input_shape[3];
  const int64_t stride_y = onnx_strides[0];
  const int64_t stride_x = onnx_strides[1];
  const int64_t dilation_y = onnx_dilations[0];
  const int64_t dilation_x = onnx_dilations[1];
  const int64_t output_padding_y = onnx_output_padding[0];
  const int64_t output_padding_x = onnx_output_padding[1];

  int64_t padding_top = onnx_pads[0];
  int64_t padding_bottom = onnx_pads[2];
  int64_t padding_left = onnx_pads[1];
  int64_t padding_right = onnx_pads[3];
  int64_t output_shape_out_y = output_shape_out[0];
  int64_t output_shape_out_x = output_shape_out[1];
  ORT_RETURN_IF_ERROR(ComputeConvTransposePadAndOutputShape(
      input_size_y,
      stride_y,
      weight_size_y,
      dilation_y,
      output_padding_y,
      auto_pad_type,
      padding_top,
      padding_bottom,
      output_shape_out_y));
  ORT_RETURN_IF_ERROR(ComputeConvTransposePadAndOutputShape(
      input_size_x,
      stride_x,
      weight_size_x,
      dilation_x,
      output_padding_x,
      auto_pad_type,
      padding_left,
      padding_right,
      output_shape_out_x));

  // WebNN only needs the height and width of the output shape.
  output_shape_out = {output_shape_out_y, output_shape_out_x};
  pads_out = {padding_top, padding_left, padding_bottom, padding_right};

  return Status::OK();
}

emscripten::val ComputeDynamicSamePadding(ModelBuilder& model_builder,
                                          const emscripten::val& input,
                                          const std::vector<int64_t>& input_shape,
                                          int64_t kernel_h, int64_t kernel_w,
                                          const std::vector<int64_t>& strides,
                                          AutoPadType auto_pad_type,
                                          const std::string& node_name) {
  const auto& builder = model_builder.GetBuilder();
  const size_t rank = input_shape.size();
  const size_t h_idx = 2;
  const size_t w_idx = 3;

  constexpr int32_t INT32 = ONNX_NAMESPACE::TensorProto_DataType_INT32;
  const auto scalar_shape = std::vector<uint32_t>{1};

  auto compute_pad_for_dim = [&](size_t dim_idx, int64_t kernel_size, int64_t stride) {
    emscripten::val shape_op = builder.call<emscripten::val>("shape", input);
    emscripten::val dim_val = shape_utils::SliceShapeRange(builder, shape_op, static_cast<int32_t>(dim_idx), 1,
                                                           node_name + "_slice_dim_" + std::to_string(dim_idx));

    emscripten::val cast_options = emscripten::val::object();
    cast_options.set("label", node_name + "_cast_dim" + std::to_string(dim_idx));
    dim_val = builder.call<emscripten::val>("cast", dim_val, emscripten::val("int32"), cast_options);

    emscripten::val stride_const = model_builder.CreateOrGetConstant<int32_t>(
        INT32, static_cast<int32_t>(stride), scalar_shape);
    emscripten::val kernel_const = model_builder.CreateOrGetConstant<int32_t>(
        INT32, static_cast<int32_t>(kernel_size), scalar_shape);
    emscripten::val one_const = model_builder.CreateOrGetConstant<int32_t>(INT32, 1, scalar_shape);
    emscripten::val two_const = model_builder.CreateOrGetConstant<int32_t>(INT32, 2, scalar_shape);
    emscripten::val zero_const = model_builder.CreateOrGetConstant<int32_t>(INT32, 0, scalar_shape);

    emscripten::val options = emscripten::val::object();
    const std::string dim_suffix = "_dim" + std::to_string(dim_idx);

    // output_size = (input_size + stride - 1) / stride
    options.set("label", node_name + "_add_stride" + dim_suffix);
    emscripten::val numerator = builder.call<emscripten::val>("add", dim_val, stride_const, options);
    options.set("label", node_name + "_sub_one" + dim_suffix);
    numerator = builder.call<emscripten::val>("sub", numerator, one_const, options);
    options.set("label", node_name + "_div_stride" + dim_suffix);
    emscripten::val output_size = builder.call<emscripten::val>("div", numerator, stride_const, options);

    // pad_needed = (output_size - 1) * stride + kernel - input_size
    options.set("label", node_name + "_out_sub_one" + dim_suffix);
    emscripten::val pad_needed = builder.call<emscripten::val>("sub", output_size, one_const, options);
    options.set("label", node_name + "_mul_stride" + dim_suffix);
    pad_needed = builder.call<emscripten::val>("mul", pad_needed, stride_const, options);
    options.set("label", node_name + "_add_kernel" + dim_suffix);
    pad_needed = builder.call<emscripten::val>("add", pad_needed, kernel_const, options);
    options.set("label", node_name + "_sub_input" + dim_suffix);
    pad_needed = builder.call<emscripten::val>("sub", pad_needed, dim_val, options);

    // pad_needed = max(0, pad_needed)
    options.set("label", node_name + "_max_zero" + dim_suffix);
    pad_needed = builder.call<emscripten::val>("max", pad_needed, zero_const, options);

    emscripten::val pad_begin, pad_end;
    if (auto_pad_type == AutoPadType::SAME_UPPER) {
      options.set("label", node_name + "_pad_begin" + dim_suffix);
      pad_begin = builder.call<emscripten::val>("div", pad_needed, two_const, options);
    } else {
      options.set("label", node_name + "_pad_add_one" + dim_suffix);
      emscripten::val pad_plus_one = builder.call<emscripten::val>("add", pad_needed, one_const, options);
      options.set("label", node_name + "_pad_begin" + dim_suffix);
      pad_begin = builder.call<emscripten::val>("div", pad_plus_one, two_const, options);
    }
    options.set("label", node_name + "_pad_end" + dim_suffix);
    pad_end = builder.call<emscripten::val>("sub", pad_needed, pad_begin, options);

    return std::make_pair(pad_begin, pad_end);
  };

  auto [pad_begin_h, pad_end_h] = compute_pad_for_dim(h_idx, kernel_h, strides[0]);
  auto [pad_begin_w, pad_end_w] = compute_pad_for_dim(w_idx, kernel_w, strides[1]);

  // Build two separate pads operands for dynamicPad (follows WebNN pad() style):
  // dynamicPad(input, beginningPadding, endingPadding, options)
  emscripten::val zero_const = model_builder.CreateOrGetConstant<int32_t>(INT32, 0, scalar_shape);

  emscripten::val begin_segments = emscripten::val::array();
  emscripten::val end_segments = emscripten::val::array();
  for (size_t i = 0; i < rank; ++i) {
    if (i == h_idx) {
      begin_segments.call<void>("push", pad_begin_h);
      end_segments.call<void>("push", pad_end_h);
    } else if (i == w_idx) {
      begin_segments.call<void>("push", pad_begin_w);
      end_segments.call<void>("push", pad_end_w);
    } else {
      begin_segments.call<void>("push", zero_const);
      end_segments.call<void>("push", zero_const);
    }
  }

  emscripten::val common_options = emscripten::val::object();
  common_options.set("label", node_name + "_same_pad_begin_concat");
  emscripten::val beginning_pads = builder.call<emscripten::val>("concat", begin_segments, 0, common_options);
  common_options.set("label", node_name + "_same_pad_end_concat");
  emscripten::val ending_pads = builder.call<emscripten::val>("concat", end_segments, 0, common_options);

  // Cast to uint32 (dynamicPad requires uint32 padding operands).
  common_options.set("label", node_name + "_same_pad_begin_cast");
  beginning_pads = builder.call<emscripten::val>("cast", beginning_pads, emscripten::val("uint32"), common_options);
  common_options.set("label", node_name + "_same_pad_end_cast");
  ending_pads = builder.call<emscripten::val>("cast", ending_pads, emscripten::val("uint32"), common_options);

  common_options.set("label", node_name + "_same_pad");
  return builder.call<emscripten::val>("dynamicPad", input, beginning_pads, ending_pads, common_options);
}

}  // namespace webnn
}  // namespace onnxruntime
