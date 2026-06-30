// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/impl/shape_utils.h"

namespace onnxruntime {
namespace webnn {

// Build a 1-D integer range [0, 1, ..., N-1] where N is dynamic (from a shape dim).
// dim_shape: a 1-D uint32 operand of length 1 representing N (e.g., from SliceShapeRange).
// Returns: 1-D int32 operand of shape [N].
// Uses WebNN range op when available, otherwise decomposes to expand→cumulativeSum→sub.
inline emscripten::val BuildRange(ModelBuilder& model_builder,
                                  const emscripten::val& dim_shape,
                                  const std::string& label) {
  emscripten::val wnn_builder = model_builder.GetBuilder();
  const emscripten::val& wnn_limits = model_builder.GetOpSupportLimits();

  if (!wnn_limits["range"].isUndefined()) {
    // range(start=0, limit=N, delta=1)
    // dim_shape is [1]-shaped uint32; cast to int32 scalar for range's limit input.
    emscripten::val zero_scalar = model_builder.CreateOrGetConstant<int>(
        ONNX_NAMESPACE::TensorProto_DataType_INT32, 0, {});
    emscripten::val one_scalar = model_builder.CreateOrGetConstant<int>(
        ONNX_NAMESPACE::TensorProto_DataType_INT32, 1, {});
    // Cast dim_shape from uint32 [1] to int32 scalar for range limit.
    emscripten::val cast_options = emscripten::val::object();
    cast_options.set("label", label + "_cast_limit");
    emscripten::val limit = wnn_builder.call<emscripten::val>(
        "cast", dim_shape, emscripten::val("int32"), cast_options);
    emscripten::val reshape_options = emscripten::val::object();
    reshape_options.set("label", label + "_reshape_limit");
    limit = wnn_builder.call<emscripten::val>(
        "reshape", limit, emscripten::val::array(std::vector<uint32_t>{}), reshape_options);

    emscripten::val range_options = emscripten::val::object();
    range_options.set("label", label);
    return wnn_builder.call<emscripten::val>("range", zero_scalar, limit, one_scalar, range_options);
  }

  // Fallback: dynamicExpand([1], dim_shape) → cumulativeSum(axis=0) → sub(1)
  emscripten::val one_constant =
      model_builder.CreateOrGetConstant<int>(ONNX_NAMESPACE::TensorProto_DataType_INT32, 1, {1});

  emscripten::val options = emscripten::val::object();
  options.set("label", label + "_expand");
  emscripten::val ones = wnn_builder.call<emscripten::val>("dynamicExpand", one_constant, dim_shape, options);

  emscripten::val cumsum_options = emscripten::val::object();
  cumsum_options.set("label", label + "_cumsum");
  cumsum_options.set("exclusive", false);
  cumsum_options.set("reversed", false);
  ones = wnn_builder.call<emscripten::val>("cumulativeSum", ones, gsl::narrow<uint32_t>(0), cumsum_options);

  options.set("label", label + "_sub");
  return wnn_builder.call<emscripten::val>("sub", ones, one_constant, options);
}

/*
    RotaryEmbedding Helper: Apply rotary positional embedding to input tensor.
    Reused by both RotaryEmbedding and GQA ops.

    Follows the OpenVINO RoPE fusion pattern (RoPEFusionGPTOSS).
    Input is BSNH, transposed to BNSH internally for pattern matching.
    cos/sin are reshaped to [1, 1, max_seq, half_dim] then gathered on axis=2
    to produce [1, 1, S, half_dim] which broadcasts over [B, N, S, half_dim].

                Input [B,S,N,H]
                  |
            [Split axis=3 if H > rotary_dim]
                  |
       [Deinterleave if interleaved]
                  |
           Transpose BSNH→BNSH                    CosCache [max_seq, half_dim]    SinCache [max_seq, half_dim]
                  |                                       |                              |
                Split axis=3                     Reshape [1,1,max_seq,half_dim]  Reshape [1,1,max_seq,half_dim]
                |   |                                     |                              |
+---------------+   |                              Gather axis=2                  Gather axis=2
|                   |                          (position_ids [S])             (position_ids [S])
|       Split(first_half, second_half)                    |                              |
|         [B,N,S,half_dim] each                  cos [1,1,S,half_dim]         sin [1,1,S,half_dim]
|                   |                                     |                              |
|       +-----------+-----------+-----------+             |                              |
|       |           |           |           |             |                              |
|       |     Mul(2nd,sin)<-----------Mul(1st,sin)<-------+------------------------------+
|       |           |           |           |             |
|   Mul(1st,cos)<-----------Mul(2nd,cos)<-----------------+
|       |           |           |           |
|       |        Mul(-1)        |           |
|       |           |           |           |
|       +---+   +---+           +---+   +---+
|           |   |                   |   |
|         Add(res_0)              Add(res_1)
|           |                         |
|           +---------Concat----------+
|                       |
|         [Re-interleave if interleaved]
|                       |
|         [Transpose BNSH→BSNH if !output_bnsh]
|                       |
+-----------------------+
                       ||
                      Join (concat axis=3)

    Formula (in BNSH space):
      res_0 = first_half * cos + (second_half * sin * -1)
      res_1 = second_half * cos + first_half * sin
      output = Concat(res_0, res_1)

    output_bnsh=true:  output stays BNSH [B,N,S,H] (no back-transpose)
    output_bnsh=false: output transposed back to BSNH [B,S,N,H]

    Interleaved mode (deinterleave before, re-interleave after):
      Deinterleave: reshape [B,S,N,rotary_dim] → [B,S,N,half_dim,2]
                    transpose → [B,S,N,2,half_dim]
                    reshape → [B,S,N,rotary_dim] (first_half || second_half)
      Re-interleave: reshape [B,N,S,rotary_dim] → [B,N,S,2,half_dim]
                     transpose → [B,N,S,half_dim,2]
                     reshape → [B,N,S,rotary_dim]
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
    bool output_bnsh,              // If true, output stays in BNSH format (no back-transpose)
    bool has_dynamic_input,        // If true, use dynamic ops (dynamicReshape, unsqueeze, squeeze, dynamicExpand)
    emscripten::val& output,
    // Identity token for the input's sequence dimension (from GetDimIdentity). The [0..S-1] range
    // built below depends only on this dimension, so callers sharing a seq identity (every layer of
    // a decoder stack, plus the paired Q/K calls) reuse one range. Empty disables sharing.
    const std::string& seq_range_cache_key = "") {
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
    // [B, S, N, rotary_dim] → [B, S, N, half_dim, 2]
    emscripten::val deinterleave_reshape_options = emscripten::val::object();
    deinterleave_reshape_options.set("label", node_name + "_rotary_deinterleave_reshape");
    if (!has_dynamic_input) {
      // Static path: use reshape with concrete uint32 values.
      // Input shape is known: [B, S, num_heads, rotary_dim] where B, S are static.
      emscripten::val input_desc = wnn_builder.call<emscripten::val>("describe", rope_input);
      emscripten::val input_dims = input_desc["shape"];
      std::vector<uint32_t> deinterleave_shape{
          input_dims[0].as<uint32_t>(), input_dims[1].as<uint32_t>(),
          num_heads, half_rotary_embedding_dim, 2};
      rope_input = wnn_builder.call<emscripten::val>(
          "reshape", rope_input, emscripten::val::array(deinterleave_shape), deinterleave_reshape_options);
    } else {
      // Dynamic path: use ComputeShape + dynamicReshape.
      std::vector<int64_t> deinterleave_dims{0, 0, 0,
          static_cast<int64_t>(half_rotary_embedding_dim), 2};
      emscripten::val deinterleave_shape_op = shape_utils::ComputeShape(
          model_builder, rope_input, deinterleave_dims, node_name + "_rotary_deinterleave_reshape");
      rope_input = wnn_builder.call<emscripten::val>(
          "dynamicReshape", rope_input, deinterleave_shape_op, deinterleave_reshape_options);
    }

    const std::vector<uint32_t> deinterleave_perm{0, 1, 2, 4, 3};
    emscripten::val deinterleave_transpose_options = emscripten::val::object();
    deinterleave_transpose_options.set("label", node_name + "_rotary_deinterleave_transpose");
    deinterleave_transpose_options.set("permutation", emscripten::val::array(deinterleave_perm));
    rope_input = wnn_builder.call<emscripten::val>(
        "transpose", rope_input, deinterleave_transpose_options);

    // [B, S, N, 2, half_dim] (transposed) → [B, S, N, rotary_dim]
    emscripten::val flat_reshape_options = emscripten::val::object();
    flat_reshape_options.set("label", node_name + "_rotary_deinterleave_flat");
    if (!has_dynamic_input) {
      // Static path: use reshape with concrete uint32 values.
      emscripten::val transposed_desc = wnn_builder.call<emscripten::val>("describe", rope_input);
      emscripten::val transposed_dims = transposed_desc["shape"];
      std::vector<uint32_t> flat_shape{
          transposed_dims[0].as<uint32_t>(), transposed_dims[1].as<uint32_t>(),
          num_heads, rotary_embedding_dim};
      rope_input = wnn_builder.call<emscripten::val>(
          "reshape", rope_input, emscripten::val::array(flat_shape), flat_reshape_options);
    } else {
      // Dynamic path: use ComputeShape + dynamicReshape.
      std::vector<int64_t> flat_dims{0, 0, 0, static_cast<int64_t>(rotary_embedding_dim)};
      emscripten::val flat_shape_op = shape_utils::ComputeShape(
          model_builder, rope_input, flat_dims, node_name + "_rotary_deinterleave_flat");
      rope_input = wnn_builder.call<emscripten::val>(
          "dynamicReshape", rope_input, flat_shape_op, flat_reshape_options);
    }
  }

  // Transpose from BSNH to BNSH for OV RoPEFusionGPTOSS pattern matching.
  // OV expects cos/sin shape [?, 1, ?, half_ndims] which requires BNSH layout.
  const std::vector<uint32_t> bsnh_to_bnsh_perm{0, 2, 1, 3};
  emscripten::val to_bnsh_options = emscripten::val::object();
  to_bnsh_options.set("label", node_name + "_rotary_to_bnsh");
  to_bnsh_options.set("permutation", emscripten::val::array(bsnh_to_bnsh_perm));
  rope_input = wnn_builder.call<emscripten::val>("transpose", rope_input, to_bnsh_options);

  // Split rope_input on last axis into two halves: [B, num_heads, S, half_dim] each.
  const std::vector<uint32_t> half_splits{half_rotary_embedding_dim, half_rotary_embedding_dim};
  emscripten::val split_halves_options = emscripten::val::object();
  split_halves_options.set("label", node_name + "_rotary_split_halves");
  split_halves_options.set("axis", 3);
  emscripten::val split_halves = wnn_builder.call<emscripten::val>(
      "split", rope_input, emscripten::val::array(half_splits), split_halves_options);
  emscripten::val first_half = split_halves[0];   // [B, num_heads, S, half_dim]
  emscripten::val second_half = split_halves[1];  // [B, num_heads, S, half_dim]

  // Helper: generate a 1D range [0, 1, ..., sequence_length-1] with dynamic or static sequence_length.
  // The range depends only on the sequence length (input dim 1). Calls that share a sequence
  // identity — every layer of a decoder stack, plus the paired query/key calls — reuse one range
  // instead of rebuilding the identical shape→slice→range chain per call; those duplicates would
  // otherwise survive as distinct SSA tensors the backend's CSE cannot merge, bloating the graph and
  // blocking pattern fusion. Keying on the seq-dim identity (not a global string) keeps this correct
  // for models with more than one sequence length (e.g. encoder/decoder cross-attention).
  const std::string seq_range_full_key =
      seq_range_cache_key.empty() ? std::string() : "rotary_seq_range:" + seq_range_cache_key;
  auto build_sequence_range = [&](const std::string& label_suffix) -> emscripten::val {
    if (!seq_range_full_key.empty() && model_builder.HasCachedOperand(seq_range_full_key)) {
      return model_builder.GetCachedOperand(seq_range_full_key);
    }
    // Get [S] shape from input dim 1 via shape() → slice, then use BuildRange.
    emscripten::val shape_options = emscripten::val::object();
    shape_options.set("label", node_name + "_rotary_range_shape" + label_suffix);
    emscripten::val input_shape_op = wnn_builder.call<emscripten::val>("shape", input, shape_options);
    emscripten::val s_dim = shape_utils::SliceShapeRange(wnn_builder, input_shape_op, 1, 1,
                                                         node_name + "_rotary_range_slice_s");
    emscripten::val range = BuildRange(
        model_builder, s_dim, node_name + "_rotary_range" + label_suffix);
    // BuildRange returns INT32; cast to INT64 if needed for downstream add with position_ids.
    if (model_builder.IsInt64Supported()) {
      emscripten::val cast_options = emscripten::val::object();
      cast_options.set("label", node_name + "_rotary_range_cast" + label_suffix);
      range = wnn_builder.call<emscripten::val>("cast", range, emscripten::val("int64"), cast_options);
    }
    if (!seq_range_full_key.empty()) {
      model_builder.AddCachedOperand(seq_range_full_key, range);
    }
    return range;
  };

  emscripten::val gather_position_ids = position_ids;
  if (position_ids_is_offset) {
    emscripten::val position_ids_range = build_sequence_range("");

    // Reshape [S] → [1, S].
    emscripten::val unsqueeze_options = emscripten::val::object();
    unsqueeze_options.set("label", node_name + "_rotary_position_ids_range_reshape");
    if (!has_dynamic_input) {
      // Static path: use reshape with concrete [1, S].
      emscripten::val range_desc = wnn_builder.call<emscripten::val>("describe", position_ids_range);
      uint32_t s_len = range_desc["shape"][0].as<uint32_t>();
      std::vector<uint32_t> reshape_1s{1, s_len};
      position_ids_range = wnn_builder.call<emscripten::val>(
          "reshape", position_ids_range, emscripten::val::array(reshape_1s), unsqueeze_options);
    } else {
      // Dynamic path: use unsqueeze.
      position_ids_range = wnn_builder.call<emscripten::val>(
          "unsqueeze", position_ids_range, emscripten::val::array(std::vector<uint32_t>{0}), unsqueeze_options);
    }

    emscripten::val position_ids_add_range_options = emscripten::val::object();
    position_ids_add_range_options.set("label", node_name + "_rotary_position_ids_add_range");
    gather_position_ids = wnn_builder.call<emscripten::val>(
        "add", position_ids, position_ids_range, position_ids_add_range_options);
  }

  // Reshape cos/sin cache to 4D [1, 1, max_seq, half_dim] BEFORE gathering.
  // The reshape target is FULLY STATIC (cos_cache shape is always known constants).
  // Then gather on axis=2 with 1D indices [S] produces [1, 1, S, half_dim] directly.
  // OV can verify dim 0=1, dim 1=1 (from data input) and dim 3=half_dim (from data input),
  // satisfying shape_matches('[?, 1, ?, half_ndims]') without needing a Transpose
  // (which OV's optimization passes eliminate).
  //
  // Both paths:
  //   cos_cache [max_seq, half_dim] → reshape [1, 1, max_seq, half_dim] (4D, all static)
  //   → gather axis=2 with [S] indices → [1, 1, S, half_dim] (4D)
  //   → no transpose needed!
  //
  // Broadcasting with split output [B, N, S, half_dim] (B=1 for inference):
  //   [1, 1, S, half_dim] broadcasts as: dim 0: 1→B, dim 1: 1→N, dim 2: S=S, dim 3: half=half ✓
  emscripten::val cos_4d, sin_4d;

  // Reshape cos/sin cache [max_seq, half_dim] → [1, 1, max_seq, half_dim].
  emscripten::val reshaped_cos, reshaped_sin;
  emscripten::val unsqueeze_cache_options = emscripten::val::object();
  if (!has_dynamic_input) {
    // Static path: use reshape with concrete values (cos/sin cache are always static initializers).
    emscripten::val cos_desc = wnn_builder.call<emscripten::val>("describe", cos_cache);
    uint32_t max_seq = cos_desc["shape"][0].as<uint32_t>();
    uint32_t half_dim = cos_desc["shape"][1].as<uint32_t>();
    std::vector<uint32_t> cache_4d_shape{1, 1, max_seq, half_dim};
    unsqueeze_cache_options.set("label", node_name + "_rotary_reshape_cos_cache");
    reshaped_cos = wnn_builder.call<emscripten::val>(
        "reshape", cos_cache, emscripten::val::array(cache_4d_shape), unsqueeze_cache_options);
    unsqueeze_cache_options.set("label", node_name + "_rotary_reshape_sin_cache");
    reshaped_sin = wnn_builder.call<emscripten::val>(
        "reshape", sin_cache, emscripten::val::array(cache_4d_shape), unsqueeze_cache_options);
  } else {
    // Dynamic path: use unsqueeze.
    emscripten::val cache_axes = emscripten::val::array(std::vector<uint32_t>{0, 1});
    unsqueeze_cache_options.set("label", node_name + "_rotary_reshape_cos_cache");
    reshaped_cos = wnn_builder.call<emscripten::val>(
        "unsqueeze", cos_cache, cache_axes, unsqueeze_cache_options);
    unsqueeze_cache_options.set("label", node_name + "_rotary_reshape_sin_cache");
    reshaped_sin = wnn_builder.call<emscripten::val>(
        "unsqueeze", sin_cache, cache_axes, unsqueeze_cache_options);
  }

  // Get 1D position indices for gathering on axis=2.
  emscripten::val gather_indices_1d;
  if (has_position_ids) {
    // Squeeze/reshape gather_position_ids from [B, S] to [S] (remove batch dim, B=1 for inference).
    if (!has_dynamic_input) {
      // Static path: use reshape with concrete [S].
      emscripten::val pos_desc = wnn_builder.call<emscripten::val>("describe", gather_position_ids);
      uint32_t s_len = pos_desc["shape"][1].as<uint32_t>();
      emscripten::val reshape_pos_options = emscripten::val::object();
      reshape_pos_options.set("label", node_name + "_rotary_flatten_position_ids");
      gather_indices_1d = wnn_builder.call<emscripten::val>(
          "reshape", gather_position_ids, emscripten::val::array(std::vector<uint32_t>{s_len}), reshape_pos_options);
    } else {
      // Dynamic path: use squeeze.
      emscripten::val squeeze_options = emscripten::val::object();
      squeeze_options.set("axes", emscripten::val::array(std::vector<uint32_t>{0}));
      squeeze_options.set("label", node_name + "_rotary_flatten_position_ids");
      gather_indices_1d = wnn_builder.call<emscripten::val>(
          "squeeze", gather_position_ids, squeeze_options);
    }
  } else {
    gather_indices_1d = build_sequence_range("_for_cos_sin");
  }

  // Gather on axis=2 with 1D indices [S] → [1, 1, S, half_dim] (4D).
  // OV knows: dim 0=1 (from data), dim 1=1 (from data), dim 3=half_dim (from data).
  emscripten::val gather_cos_options = emscripten::val::object();
  gather_cos_options.set("label", node_name + "_rotary_gather_cos");
  gather_cos_options.set("axis", 2);
  cos_4d = wnn_builder.call<emscripten::val>("gather", reshaped_cos, gather_indices_1d, gather_cos_options);
  emscripten::val gather_sin_options = emscripten::val::object();
  gather_sin_options.set("label", node_name + "_rotary_gather_sin");
  gather_sin_options.set("axis", 2);
  sin_4d = wnn_builder.call<emscripten::val>("gather", reshaped_sin, gather_indices_1d, gather_sin_options);

  // Core RoPE formula (matches OpenVINO RoPEFusionGPTOSS pattern):
  //   Input in BNSH: [B, num_heads, S, head_size]
  //   cos/sin shape: [B, 1, S, half_ndims] (broadcasts over num_heads)
  //   first_half_mul_cos = Multiply(first_half, cos)
  //   second_half_mul_sin = Multiply(second_half, sin)
  //   neg = Multiply(second_half_mul_sin, -1)
  //   res_0 = Add(first_half_mul_cos, neg)
  //
  //   second_half_mul_cos = Multiply(second_half, cos)
  //   first_half_mul_sin = Multiply(first_half, sin)
  //   res_1 = Add(second_half_mul_cos, first_half_mul_sin)
  //
  //   result = Concat([res_0, res_1], axis=-1)

  emscripten::val first_mul_cos_options = emscripten::val::object();
  first_mul_cos_options.set("label", node_name + "_rotary_first_mul_cos");
  emscripten::val first_mul_cos = wnn_builder.call<emscripten::val>(
      "mul", first_half, cos_4d, first_mul_cos_options);

  emscripten::val second_mul_sin_options = emscripten::val::object();
  second_mul_sin_options.set("label", node_name + "_rotary_second_mul_sin");
  emscripten::val second_mul_sin = wnn_builder.call<emscripten::val>(
      "mul", second_half, sin_4d, second_mul_sin_options);

  // Multiply by scalar -1 to negate (matches OV's Mul(-1) pattern).
  const emscripten::val neg_one_constant =
      model_builder.CreateOrGetConstant<float>(input_data_type, -1.0f);

  emscripten::val neg_options = emscripten::val::object();
  neg_options.set("label", node_name + "_rotary_neg");
  emscripten::val neg_second_mul_sin = wnn_builder.call<emscripten::val>(
      "mul", second_mul_sin, neg_one_constant, neg_options);

  emscripten::val sub_options = emscripten::val::object();
  sub_options.set("label", node_name + "_rotary_sub");
  emscripten::val res_0 = wnn_builder.call<emscripten::val>(
      "add", first_mul_cos, neg_second_mul_sin, sub_options);

  emscripten::val second_mul_cos_options = emscripten::val::object();
  second_mul_cos_options.set("label", node_name + "_rotary_second_mul_cos");
  emscripten::val second_mul_cos = wnn_builder.call<emscripten::val>(
      "mul", second_half, cos_4d, second_mul_cos_options);

  emscripten::val first_mul_sin_options = emscripten::val::object();
  first_mul_sin_options.set("label", node_name + "_rotary_first_mul_sin");
  emscripten::val first_mul_sin = wnn_builder.call<emscripten::val>(
      "mul", first_half, sin_4d, first_mul_sin_options);

  emscripten::val add_options = emscripten::val::object();
  add_options.set("label", node_name + "_rotary_add");
  emscripten::val res_1 = wnn_builder.call<emscripten::val>(
      "add", second_mul_cos, first_mul_sin, add_options);

  // Concat results: [B, num_heads, S, rotary_dim] (BNSH)
  emscripten::val concat_result_inputs = emscripten::val::array();
  concat_result_inputs.call<void>("push", res_0);
  concat_result_inputs.call<void>("push", res_1);
  emscripten::val concat_result_options = emscripten::val::object();
  concat_result_options.set("label", node_name + "_rotary_concat_result");
  output = wnn_builder.call<emscripten::val>(
      "concat", concat_result_inputs, 3, concat_result_options);

  if (!output_bnsh) {
    // Transpose back from BNSH to BSNH.
    emscripten::val to_bsnh_options = emscripten::val::object();
    to_bsnh_options.set("label", node_name + "_rotary_to_bsnh");
    to_bsnh_options.set("permutation", emscripten::val::array(bsnh_to_bnsh_perm));
    output = wnn_builder.call<emscripten::val>("transpose", output, to_bsnh_options);
  }

  // For interleaved mode, re-interleave the result.
  // The output is [B, ?, ?, rotary_dim] (BSNH or BNSH depending on output_bnsh).
  // Re-interleave by: reshape → [.., .., .., 2, half_dim]
  //                    transpose → [.., .., .., half_dim, 2]
  //                    reshape → [.., .., .., rotary_dim]
  if (interleaved) {
    // [B, ?, ?, rotary_dim] → [B, ?, ?, 2, half_dim]
    emscripten::val reinterleave_reshape_options = emscripten::val::object();
    reinterleave_reshape_options.set("label", node_name + "_rotary_reinterleave_reshape");
    if (!has_dynamic_input) {
      // Static path: use reshape with concrete uint32 values.
      emscripten::val out_desc = wnn_builder.call<emscripten::val>("describe", output);
      emscripten::val out_dims = out_desc["shape"];
      std::vector<uint32_t> reinterleave_shape{
          out_dims[0].as<uint32_t>(), out_dims[1].as<uint32_t>(),
          out_dims[2].as<uint32_t>(), 2, half_rotary_embedding_dim};
      output = wnn_builder.call<emscripten::val>(
          "reshape", output, emscripten::val::array(reinterleave_shape), reinterleave_reshape_options);
    } else {
      // Dynamic path: use ComputeShape + dynamicReshape.
      std::vector<int64_t> reinterleave_dims{0, 0, 0, 2,
          static_cast<int64_t>(half_rotary_embedding_dim)};
      emscripten::val reinterleave_shape_op = shape_utils::ComputeShape(
          model_builder, output, reinterleave_dims, node_name + "_rotary_reinterleave_reshape");
      output = wnn_builder.call<emscripten::val>(
          "dynamicReshape", output, reinterleave_shape_op, reinterleave_reshape_options);
    }

    const std::vector<uint32_t> reinterleave_perm{0, 1, 2, 4, 3};
    emscripten::val reinterleave_transpose_options = emscripten::val::object();
    reinterleave_transpose_options.set("label", node_name + "_rotary_reinterleave_transpose");
    reinterleave_transpose_options.set("permutation", emscripten::val::array(reinterleave_perm));
    output = wnn_builder.call<emscripten::val>(
        "transpose", output, reinterleave_transpose_options);

    // [B, ?, ?, half_dim, 2] (transposed) → [B, ?, ?, rotary_dim]
    emscripten::val final_reshape_options = emscripten::val::object();
    final_reshape_options.set("label", node_name + "_rotary_reinterleave_flat");
    if (!has_dynamic_input) {
      // Static path: use reshape with concrete uint32 values.
      emscripten::val transposed_desc = wnn_builder.call<emscripten::val>("describe", output);
      emscripten::val transposed_dims = transposed_desc["shape"];
      std::vector<uint32_t> final_shape{
          transposed_dims[0].as<uint32_t>(), transposed_dims[1].as<uint32_t>(),
          transposed_dims[2].as<uint32_t>(), rotary_embedding_dim};
      output = wnn_builder.call<emscripten::val>(
          "reshape", output, emscripten::val::array(final_shape), final_reshape_options);
    } else {
      // Dynamic path: use ComputeShape + dynamicReshape.
      std::vector<int64_t> final_dims{0, 0, 0, static_cast<int64_t>(rotary_embedding_dim)};
      emscripten::val final_shape_op = shape_utils::ComputeShape(
          model_builder, output, final_dims, node_name + "_rotary_reinterleave_flat");
      output = wnn_builder.call<emscripten::val>(
          "dynamicReshape", output, final_shape_op, final_reshape_options);
    }
  }

  // Join the rotary output with the rest of the input if head_size > rotary_dim.
  if (head_size != rotary_embedding_dim) {
    // When output_bnsh=true, partial_input1 is still in BSNH (split before Transpose).
    // Transpose it to BNSH to match the output format before concatenation.
    if (output_bnsh) {
      emscripten::val partial_to_bnsh_options = emscripten::val::object();
      partial_to_bnsh_options.set("label", node_name + "_rotary_partial_to_bnsh");
      partial_to_bnsh_options.set("permutation", emscripten::val::array(bsnh_to_bnsh_perm));
      partial_input1 = wnn_builder.call<emscripten::val>("transpose", partial_input1, partial_to_bnsh_options);
    }
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
                                                 const std::vector<int64_t>& reshape_output_target,
                                                 bool has_dynamic_input) {
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

  // Reshape the transposed attn_output [B, S, N, H] to [B, S, hidden] using target_dims.
  common_options.set("label", node.Name() + "_/Attention/qkv/reshape");
  if (!has_dynamic_input) {
    // Static path: use reshape with concrete uint32 values derived from operand shape.
    emscripten::val attn_desc = model_builder.GetBuilder().call<emscripten::val>("describe", attn_output);
    emscripten::val attn_dims = attn_desc["shape"];
    std::vector<uint32_t> output_shape;
    for (size_t i = 0; i < reshape_output_target.size(); ++i) {
      if (reshape_output_target[i] == 0) {
        output_shape.push_back(attn_dims[static_cast<int>(i)].as<uint32_t>());
      } else {
        output_shape.push_back(static_cast<uint32_t>(reshape_output_target[i]));
      }
    }
    attn_output = model_builder.GetBuilder().call<emscripten::val>(
        "reshape", attn_output, emscripten::val::array(output_shape), common_options);
  } else {
    // Dynamic path: use ComputeShape + dynamicReshape.
    emscripten::val reshape_output_shape = shape_utils::ComputeShape(
        model_builder, attn_output, reshape_output_target, node.Name() + "_/Attention/qkv/reshape");
    attn_output = model_builder.GetBuilder().call<emscripten::val>(
        "dynamicReshape", attn_output, reshape_output_shape, common_options);
  }

  return attn_output;
}

}  // namespace webnn
}  // namespace onnxruntime
