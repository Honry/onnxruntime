// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webnn/builders/helper.h"

namespace onnxruntime {
namespace webnn {
/*
    RotaryEmbedding Helper: Apply rotary positional embedding to input tensor.
    Reused by both RotaryEmbedding and GQA ops.

    Follows the OpenVINO RoPE fusion pattern (split on last axis):

    Input [B, S, num_heads, head_size]     CosCache   PositionIds   SinCache
          |                                    |           |             |
      Split(axis=3)                           Gather      |           Gather
      /           \                             |         |             |
  first_half   second_half                  Unsqueeze(dim=2)       Unsqueeze(dim=2)
      |    \     /    |                     → [B,S,1,half]        → [B,S,1,half]
      |     \   /     |                         |                     |
      |      \ /      |                         |                     |
    first*cos  second*sin                   second*cos            first*sin
        |         |                             |                     |
        +---sub---+                             +--------add----------+
             |                                            |
           res_0                                        res_1
             \                                          /
              +----------Concat(axis=3)----------------+
                              |
                           Output [B, S, num_heads, head_size]
*/
inline Status ApplyRotaryEmbedding(
    ModelBuilder& model_builder,
    const std::string& node_name,
    emscripten::val input,         // Shape: [batch_size, sequence_length, num_heads, head_size]
    emscripten::val cos_cache,     // Shape: [max_sequence_length, head_size / 2]
    emscripten::val sin_cache,     // Shape: [max_sequence_length, head_size / 2]
    emscripten::val position_ids,  // Shape: [batch_size, sequence_length] or [1]
    int32_t input_data_type,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t rotary_embedding_dim,
    bool interleaved,
    bool has_position_ids,
    bool position_ids_is_offset,
    emscripten::val& output) {
  emscripten::val wnn_builder = model_builder.GetBuilder();
  ORT_RETURN_IF_NOT(head_size >= rotary_embedding_dim,
                    "Rotary embedding dimension must be less than or equal to head_size");
  const uint32_t half_rotary_embedding_dim = rotary_embedding_dim / 2;

  // Split the input to perform the rotary embedding only on a subregion of the tensor if needed.
  // Input shape: [batch_size, sequence_length, num_heads, head_size]
  emscripten::val rope_input = input;
  emscripten::val partial_input1 = emscripten::val::undefined();
  if (head_size > rotary_embedding_dim) {
    const std::vector<uint32_t> splits{rotary_embedding_dim, head_size - rotary_embedding_dim};
    emscripten::val split_input_options = emscripten::val::object();
    split_input_options.set("label", node_name + "_rotary_split_input");
    split_input_options.set("axis", 3);
    emscripten::val split = wnn_builder.call<emscripten::val>(
        "split", input, emscripten::val::array(splits), split_input_options);
    rope_input = split[0];
    partial_input1 = split[1];
  }

  // For interleaved mode, deinterleave first so the core RoPE formula is identical.
  // [B, S, num_heads, rotary_dim] → reshape [B, S, num_heads, half_dim, 2]
  //                                → transpose [B, S, num_heads, 2, half_dim]
  //                                → reshape [B, S, num_heads, rotary_dim] (now [first_half, second_half])
  if (interleaved) {
    emscripten::val deinterleave_shape = emscripten::val::array();
    deinterleave_shape.call<void>("push", rope_input["shape"][0]);
    deinterleave_shape.call<void>("push", rope_input["shape"][1]);
    deinterleave_shape.call<void>("push", rope_input["shape"][2]);
    deinterleave_shape.call<void>("push", half_rotary_embedding_dim);
    deinterleave_shape.call<void>("push", 2);
    emscripten::val deinterleave_reshape_options = emscripten::val::object();
    deinterleave_reshape_options.set("label", node_name + "_rotary_deinterleave_reshape");
    rope_input = wnn_builder.call<emscripten::val>(
        "reshape", rope_input, deinterleave_shape, deinterleave_reshape_options);

    const std::vector<uint32_t> deinterleave_perm{0, 1, 2, 4, 3};
    emscripten::val deinterleave_transpose_options = emscripten::val::object();
    deinterleave_transpose_options.set("label", node_name + "_rotary_deinterleave_transpose");
    deinterleave_transpose_options.set("permutation", emscripten::val::array(deinterleave_perm));
    rope_input = wnn_builder.call<emscripten::val>(
        "transpose", rope_input, deinterleave_transpose_options);

    emscripten::val flat_shape = emscripten::val::array();
    flat_shape.call<void>("push", rope_input["shape"][0]);
    flat_shape.call<void>("push", rope_input["shape"][1]);
    flat_shape.call<void>("push", rope_input["shape"][2]);
    flat_shape.call<void>("push", rotary_embedding_dim);
    emscripten::val flat_reshape_options = emscripten::val::object();
    flat_reshape_options.set("label", node_name + "_rotary_deinterleave_flat");
    rope_input = wnn_builder.call<emscripten::val>(
        "reshape", rope_input, flat_shape, flat_reshape_options);
  }

  // Split rope_input on last axis into two halves: [B, S, num_heads, half_dim] each.
  // This preserves all dim descriptors for dims 0-2 (batch, seq, num_heads).
  const std::vector<uint32_t> half_splits{half_rotary_embedding_dim, half_rotary_embedding_dim};
  emscripten::val split_halves_options = emscripten::val::object();
  split_halves_options.set("label", node_name + "_rotary_split_halves");
  split_halves_options.set("axis", 3);
  emscripten::val split_halves = wnn_builder.call<emscripten::val>(
      "split", rope_input, emscripten::val::array(half_splits), split_halves_options);
  emscripten::val first_half = split_halves[0];   // [B, S, num_heads, half_dim]
  emscripten::val second_half = split_halves[1];  // [B, S, num_heads, half_dim]

  // Helper: generate a 1D range [0, 1, ..., sequence_length-1] with dynamic sequence_length.
  auto build_sequence_range = [&](const std::string& label_suffix) -> emscripten::val {
    const bool is_int64_supported = model_builder.IsInt64Supported();
    emscripten::val value_one_constant = is_int64_supported
                         ? model_builder.CreateOrGetConstant<int64_t>(
                             ONNX_NAMESPACE::TensorProto_DataType_INT64, static_cast<int64_t>(1), {1})
                         : model_builder.CreateOrGetConstant<int32_t>(
                             ONNX_NAMESPACE::TensorProto_DataType_INT32, static_cast<int32_t>(1), {1});

    emscripten::val range_shape = emscripten::val::array();
    range_shape.call<void>("push", input["shape"][1]);
    emscripten::val range = wnn_builder.call<emscripten::val>("expand", value_one_constant, range_shape);

    emscripten::val cumsum_options = emscripten::val::object();
    cumsum_options.set("label", node_name + "_rotary_position_ids_range_cumsum" + label_suffix);
    cumsum_options.set("exclusive", false);
    cumsum_options.set("reversed", false);
    range = wnn_builder.call<emscripten::val>("cumulativeSum", range, gsl::narrow<uint32_t>(0), cumsum_options);
    range = wnn_builder.call<emscripten::val>("sub", range, value_one_constant);
    return range;
  };

  emscripten::val gather_position_ids = position_ids;
  if (position_ids_is_offset) {
    emscripten::val position_ids_range = build_sequence_range("");

    emscripten::val position_ids_range_2d_shape = emscripten::val::array();
    position_ids_range_2d_shape.call<void>("push", 1);
    position_ids_range_2d_shape.call<void>("push", input["shape"][1]);
    emscripten::val reshape_position_ids_range_options = emscripten::val::object();
    reshape_position_ids_range_options.set("label", node_name + "_rotary_position_ids_range_reshape");
    position_ids_range = wnn_builder.call<emscripten::val>(
      "reshape", position_ids_range, position_ids_range_2d_shape, reshape_position_ids_range_options);

    emscripten::val position_ids_add_range_options = emscripten::val::object();
    position_ids_add_range_options.set("label", node_name + "_rotary_position_ids_add_range");
    gather_position_ids = wnn_builder.call<emscripten::val>(
        "add", position_ids, position_ids_range, position_ids_add_range_options);
  }

  // Gather cos/sin values based on position_ids.
  emscripten::val gather_cos = cos_cache;
  emscripten::val gather_sin = sin_cache;
  if (has_position_ids) {
    emscripten::val gather_cos_options = emscripten::val::object();
    emscripten::val gather_sin_options = emscripten::val::object();
    gather_cos_options.set("label", node_name + "_rotary_gather_cos");
    gather_sin_options.set("label", node_name + "_rotary_gather_sin");
    gather_cos_options.set("axis", 0);
    gather_sin_options.set("axis", 0);
    gather_cos = wnn_builder.call<emscripten::val>("gather", gather_cos, gather_position_ids, gather_cos_options);
    gather_sin = wnn_builder.call<emscripten::val>("gather", gather_sin, gather_position_ids, gather_sin_options);
  } else {
    emscripten::val position_ids_range = build_sequence_range("_without_ids");
    emscripten::val gather_cos_options = emscripten::val::object();
    emscripten::val gather_sin_options = emscripten::val::object();
    gather_cos_options.set("label", node_name + "_rotary_gather_cos_without_ids");
    gather_sin_options.set("label", node_name + "_rotary_gather_sin_without_ids");
    gather_cos_options.set("axis", 0);
    gather_sin_options.set("axis", 0);
    gather_cos = wnn_builder.call<emscripten::val>("gather", gather_cos, position_ids_range, gather_cos_options);
    gather_sin = wnn_builder.call<emscripten::val>("gather", gather_sin, position_ids_range, gather_sin_options);
  }

  // Unsqueeze cos/sin to 4D then expand to match first_half's exact shape.
  // The key insight: static expand() with first_half["shape"] produces output that inherits
  // first_half's dim descriptors, ensuring mul compatibility in ORT backend.
  // Step 1: Reshape gather_cos from [B,S,half] or [S,half] to [B,S,1,half] using static reshape.
  emscripten::val cos_4d_shape = emscripten::val::array();
  emscripten::val sin_4d_shape = emscripten::val::array();
  if (has_position_ids) {
    // 3D [B, S, half_dim] → [B, S, 1, half_dim]
    cos_4d_shape.call<void>("push", gather_cos["shape"][0]);
    cos_4d_shape.call<void>("push", gather_cos["shape"][1]);
    cos_4d_shape.call<void>("push", 1);
    cos_4d_shape.call<void>("push", half_rotary_embedding_dim);
    sin_4d_shape.call<void>("push", gather_sin["shape"][0]);
    sin_4d_shape.call<void>("push", gather_sin["shape"][1]);
    sin_4d_shape.call<void>("push", 1);
    sin_4d_shape.call<void>("push", half_rotary_embedding_dim);
  } else {
    // 2D [S, half_dim] → [1, S, 1, half_dim]
    cos_4d_shape.call<void>("push", 1);
    cos_4d_shape.call<void>("push", gather_cos["shape"][0]);
    cos_4d_shape.call<void>("push", 1);
    cos_4d_shape.call<void>("push", half_rotary_embedding_dim);
    sin_4d_shape.call<void>("push", 1);
    sin_4d_shape.call<void>("push", gather_sin["shape"][0]);
    sin_4d_shape.call<void>("push", 1);
    sin_4d_shape.call<void>("push", half_rotary_embedding_dim);
  }

  emscripten::val reshape_cos_options = emscripten::val::object();
  emscripten::val reshape_sin_options = emscripten::val::object();
  reshape_cos_options.set("label", node_name + "_rotary_unsqueeze_cos");
  reshape_sin_options.set("label", node_name + "_rotary_unsqueeze_sin");
  emscripten::val cos_4d = wnn_builder.call<emscripten::val>(
      "reshape", gather_cos, cos_4d_shape, reshape_cos_options);
  emscripten::val sin_4d = wnn_builder.call<emscripten::val>(
      "reshape", gather_sin, sin_4d_shape, reshape_sin_options);

  // Step 2: Expand cos/sin to first_half's shape using static expand().
  // expand() with first_half["shape"] will produce output whose dim descriptors
  // are derived from first_half's shape, making mul compatible.
  emscripten::val expand_cos_options = emscripten::val::object();
  expand_cos_options.set("label", node_name + "_rotary_expand_cos");
  cos_4d = wnn_builder.call<emscripten::val>(
      "expand", cos_4d, first_half["shape"], expand_cos_options);
  emscripten::val expand_sin_options = emscripten::val::object();
  expand_sin_options.set("label", node_name + "_rotary_expand_sin");
  sin_4d = wnn_builder.call<emscripten::val>(
      "expand", sin_4d, first_half["shape"], expand_sin_options);

  // Core RoPE formula (matches OpenVINO's RoPE fusion pattern):
  // res_0 = first_half * cos - second_half * sin
  // res_1 = second_half * cos + first_half * sin

  emscripten::val mul_first_cos_options = emscripten::val::object();
  mul_first_cos_options.set("label", node_name + "_rotary_first_mul_cos");
  emscripten::val first_mul_cos = wnn_builder.call<emscripten::val>(
      "mul", first_half, cos_4d, mul_first_cos_options);

  emscripten::val mul_second_sin_options = emscripten::val::object();
  mul_second_sin_options.set("label", node_name + "_rotary_second_mul_sin");
  emscripten::val second_mul_sin = wnn_builder.call<emscripten::val>(
      "mul", second_half, sin_4d, mul_second_sin_options);

  emscripten::val sub_options = emscripten::val::object();
  sub_options.set("label", node_name + "_rotary_sub");
  emscripten::val res_0 = wnn_builder.call<emscripten::val>(
      "sub", first_mul_cos, second_mul_sin, sub_options);

  emscripten::val mul_second_cos_options = emscripten::val::object();
  mul_second_cos_options.set("label", node_name + "_rotary_second_mul_cos");
  emscripten::val second_mul_cos = wnn_builder.call<emscripten::val>(
      "mul", second_half, cos_4d, mul_second_cos_options);

  emscripten::val mul_first_sin_options = emscripten::val::object();
  mul_first_sin_options.set("label", node_name + "_rotary_first_mul_sin");
  emscripten::val first_mul_sin = wnn_builder.call<emscripten::val>(
      "mul", first_half, sin_4d, mul_first_sin_options);

  emscripten::val add_options = emscripten::val::object();
  add_options.set("label", node_name + "_rotary_add");
  emscripten::val res_1 = wnn_builder.call<emscripten::val>(
      "add", second_mul_cos, first_mul_sin, add_options);

  // Concat results back: [B, S, num_heads, rotary_dim]
  emscripten::val concat_result_options = emscripten::val::object();
  concat_result_options.set("label", node_name + "_rotary_concat_result");
  emscripten::val concat_inputs = emscripten::val::array();
  concat_inputs.call<void>("push", res_0);
  concat_inputs.call<void>("push", res_1);
  output = wnn_builder.call<emscripten::val>("concat", concat_inputs, 3, concat_result_options);

  // For interleaved mode, re-interleave the result.
  // [B, S, num_heads, rotary_dim] → reshape [B, S, num_heads, 2, half_dim]
  //                                → transpose [B, S, num_heads, half_dim, 2]
  //                                → reshape [B, S, num_heads, rotary_dim]
  if (interleaved) {
    emscripten::val reinterleave_shape = emscripten::val::array();
    reinterleave_shape.call<void>("push", output["shape"][0]);
    reinterleave_shape.call<void>("push", output["shape"][1]);
    reinterleave_shape.call<void>("push", output["shape"][2]);
    reinterleave_shape.call<void>("push", 2);
    reinterleave_shape.call<void>("push", half_rotary_embedding_dim);
    emscripten::val reinterleave_reshape_options = emscripten::val::object();
    reinterleave_reshape_options.set("label", node_name + "_rotary_reinterleave_reshape");
    output = wnn_builder.call<emscripten::val>(
        "reshape", output, reinterleave_shape, reinterleave_reshape_options);

    const std::vector<uint32_t> reinterleave_perm{0, 1, 2, 4, 3};
    emscripten::val reinterleave_transpose_options = emscripten::val::object();
    reinterleave_transpose_options.set("label", node_name + "_rotary_reinterleave_transpose");
    reinterleave_transpose_options.set("permutation", emscripten::val::array(reinterleave_perm));
    output = wnn_builder.call<emscripten::val>(
        "transpose", output, reinterleave_transpose_options);

    emscripten::val final_shape = emscripten::val::array();
    final_shape.call<void>("push", output["shape"][0]);
    final_shape.call<void>("push", output["shape"][1]);
    final_shape.call<void>("push", output["shape"][2]);
    final_shape.call<void>("push", rotary_embedding_dim);
    emscripten::val final_reshape_options = emscripten::val::object();
    final_reshape_options.set("label", node_name + "_rotary_reinterleave_flat");
    output = wnn_builder.call<emscripten::val>(
        "reshape", output, final_shape, final_reshape_options);
  }

  // Join the rotary output with the rest of the input if head_size > rotary_dim.
  if (head_size != rotary_embedding_dim) {
    emscripten::val concat_back_input_options = emscripten::val::object();
    concat_back_input_options.set("label", node_name + "_rotary_concat_back_input");
    emscripten::val concat_back = emscripten::val::array();
    concat_back.call<void>("push", output);
    concat_back.call<void>("push", partial_input1);
    output = wnn_builder.call<emscripten::val>("concat", concat_back, 3, concat_back_input_options);
  }

  return Status::OK();
}

/*
    ScaledDotProductAttention Subgraph: The basis for MultiHeadAttention and GroupQueryAttention
    inputs: query, key, value, scale, attention mask, and reshape_output_shape (for reshape)
    Abbreviations: B is batch_size, S is query sequence_length, kv_S is key/value sequence length,
                   N is number of attention heads, H is head size, W is hidden_size

  query         key
    |            |
    +---matmul---+    scale
          |             |
          +-----div-----+   attn_mask
                 |             |
                 +-----add-----+        value
                        |                 |
                        +------matmul-----+
                                 |
                   (0,2,1,3) transpose B,H,S,N -> B,S,H,N
                                 |
                              reshape B,S,H,N -> B,S,W
                                 |
                               output
*/
inline emscripten::val ScaledDotProductAttention(ModelBuilder& model_builder, const Node& node,
                                                 const logging::Logger& logger, emscripten::val query,
                                                 emscripten::val key, emscripten::val value, emscripten::val scale,
                                                 emscripten::val attn_mask,
                                                 emscripten::val reshape_output_shape) {
  emscripten::val common_options = emscripten::val::object();
  // B,H,S,N * B,H,kv_S,N = B,H,S,kv_S
  common_options.set("label", node.Name() + "_/Attention/qkv/matmul_1");
  emscripten::val matmul_output =
      model_builder.GetBuilder().call<emscripten::val>("matmul", query, key, common_options);

  common_options.set("label", node.Name() + "_/Attention/qkv/div");
  emscripten::val div_output =
      model_builder.GetBuilder().call<emscripten::val>("mul", matmul_output, scale, common_options);

  emscripten::val softmax_input = div_output;
  if (attn_mask != emscripten::val::undefined()) {
    common_options.set("label", node.Name() + "_/Attention/attn_mask/softmax_input");
    softmax_input = model_builder.GetBuilder().call<emscripten::val>("add", div_output, attn_mask, common_options);
  }

  common_options.set("label", node.Name() + "_/Attention/attn_mask/softmax_input");
  int32_t softmax_axis = 3;
  emscripten::val softmax_output =
      model_builder.GetBuilder().call<emscripten::val>("softmax", softmax_input, softmax_axis, common_options);

  // B,H,S,kv_S * B,H,kv_S,N = B,H,S,N
  common_options.set("label", node.Name() + "_/Attention/qkv/matmul_2");
  emscripten::val attn_output =
      model_builder.GetBuilder().call<emscripten::val>("matmul", softmax_output, value, common_options);

  emscripten::val options = emscripten::val::object();
  options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
  options.set("label", node.Name() + "_/Attention/qkv/transpose");
  attn_output = model_builder.GetBuilder().call<emscripten::val>("transpose", attn_output, options);

  common_options.set("label", node.Name() + "_/Attention/qkv/reshape");
  attn_output = model_builder.GetBuilder().call<emscripten::val>(
      "reshape", attn_output, reshape_output_shape, common_options);

  return attn_output;
}

}  // namespace webnn
}  // namespace onnxruntime
