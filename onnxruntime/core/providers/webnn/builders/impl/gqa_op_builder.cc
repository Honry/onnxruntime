// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"
#include <cmath>
#include <numeric>

#include "base_op_builder.h"
#include "attention_helper.h"

namespace onnxruntime {
namespace webnn {

// Broadcast a KV tensor from [B,kv_N,P,H] to [B,N,P,H] via unsqueeze → dynamicExpand → dynamicReshape.
// Used to replicate kv_num_heads to match num_heads when group_size > 1.
static emscripten::val GroupBroadcast(ModelBuilder& model_builder,
                                      const emscripten::val& present_kv,
                                      uint32_t group_size,
                                      uint32_t num_heads,
                                      const std::string& label) {
  emscripten::val wnn_builder = model_builder.GetBuilder();

  // Step 1: unsqueeze [B,kv_N,P,H] → [B,kv_N,1,P,H]
  emscripten::val unsqueeze_options = emscripten::val::object();
  unsqueeze_options.set("label", label + "_unsqueeze");
  emscripten::val unsqueezed = wnn_builder.call<emscripten::val>(
      "unsqueeze", present_kv, emscripten::val::array(std::vector<uint32_t>{2}), unsqueeze_options);

  // Step 2: dynamicExpand [B,kv_N,1,P,H] → [B,kv_N,G,P,H]
  // Build expand shape: shape(unsqueezed) with dim 2 replaced by group_size.
  emscripten::val shape_options = emscripten::val::object();
  shape_options.set("label", label + "_expand_shape");
  emscripten::val shape_op = wnn_builder.call<emscripten::val>("shape", unsqueezed, shape_options);
  emscripten::val segments = emscripten::val::array();
  segments.call<void>("push", shape_utils::SliceShapeRange(wnn_builder, shape_op, 0, 2,
                                                           label + "_slice_0_2"));
  segments.call<void>("push", model_builder.CreateOrGetConstant<uint32_t>(
      ONNX_NAMESPACE::TensorProto_DataType_UINT32, static_cast<uint32_t>(group_size), {1}));
  segments.call<void>("push", shape_utils::SliceShapeRange(wnn_builder, shape_op, 3, 2,
                                                           label + "_slice_3_2"));
  emscripten::val concat_options = emscripten::val::object();
  concat_options.set("label", label + "_expand_shape_concat");
  emscripten::val expand_target = wnn_builder.call<emscripten::val>("concat", segments, 0, concat_options);

  emscripten::val expand_options = emscripten::val::object();
  expand_options.set("label", label + "_expand");
  emscripten::val expanded = wnn_builder.call<emscripten::val>(
      "dynamicExpand", unsqueezed, expand_target, expand_options);

  // Step 3: dynamicReshape [B,kv_N,G,P,H] → [B,N,P,H]
  emscripten::val reshape_shape = shape_utils::ComputeShape(
      model_builder, present_kv,
      {0, static_cast<int64_t>(num_heads), 0, 0},
      label + "_reshape");
  emscripten::val reshape_options = emscripten::val::object();
  reshape_options.set("label", label + "_reshape");
  return wnn_builder.call<emscripten::val>("dynamicReshape", expanded, reshape_shape, reshape_options);
}

class GroupQueryAttentionOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const GraphViewer&, const Node& node, const WebnnDeviceType /* device_type */,
                         const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const GraphViewer&, const Node& node, const emscripten::val& wnn_limits,
                              const logging::Logger& logger) const override;
  bool HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                               const logging::Logger& logger) const override;
};

void GroupQueryAttentionOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // We check the value of input total_sequence_length in function IsOpSupportedImpl,
  // and it should be an initializer and does not participate in Op calculation.
  const auto input_name = node.InputDefs()[6]->Name();
  model_builder.AddInitializerToSkip(input_name);
  model_builder.AddInputToSkip(input_name);
}

/** GroupQueryAttention SubGraph.
 Abbreviations: B is batch_size, S is sequence_length, W is hidden_size, P is past_sequence_length
                N is number of attention heads, kv_N is number of attention heads for kv, H is head size
                G is group size, and G=N/kv_N, W=N*H, h=Sqrt(H).
    GQA inputs: query, key(optional), value(optional), past_key(optional), past_value(optional),
                seqlens_k, total_sequence_length, cos_cache(optional), sin_cache(optional), position_ids(optional)
    Notes:
      - key, value, past_key, past_value can be empty (optional inputs).
      - When key/value are empty, query contains packed QKV.
      - When past_key/past_value are empty, this is the first token (prefill mode).
      - When do_rotary is true, cos_cache and sin_cache must be provided.

    KV-cache update strategy is controlled by the "enableCausalLM" EP option:

    ===== enableCausalLM = true (Concat / stateful path) =====
    Cache grows each decode step: present_kv = concat(past_kv, new_kv, axis=2).
    Suitable for causal LM inference where the runtime manages growing cache tensors.

       query         key                 value
        |             |                    |
    (RotaryEmb)    (RotaryEmb)             |
        |             |                    |
      Reshape       Reshape              Reshape (B,S,kv_N,H)
        |             |                    |
     q_Transpose  Transpose(BNSH)    Transpose(BNSH)
      (0,2,1,3)       |                    |
         \   past_key |        past_value  |
          \        \  |                \   |
           \    Concat(axis=2)     Concat(axis=2)
            \          |                   |
             \     present_key       present_value -----> output[1], output[2]
              \        |                   |
               |     Expand(G)         Expand(G)    (attention_bias, causal mask)
               |       |                   |           /
               |     k_Transpose           |          /
               |     (0,1,3,2)             |         /
               |       |                   |        /
            +---------------------------------------+
            |        ScaledDotProductAttention      |
            +---------------------------------------+
                              |
                            output

    ===== enableCausalLM = false (ScatterND / stateless path, default) =====
    Fixed-size KV buffer: new tokens are scattered at position seqlens_k-(S-1).
    Suitable for models that manage the KV-cache externally (e.g., I/O binding).

       query         key                 value
        |             |                    |
    (RotaryEmb)    (RotaryEmb)             |
        |             |                    |
      Reshape       Reshape              Reshape (B,S,kv_N,H)
        |             |                    |
     q_Transpose      |                    |
      (0,2,1,3)       |   scatter_indices  |
         \            |   (B,S,kv_N,3)    |
          \   past_key|        past_value  |
           \       \  |                \   |
            \   ScatterND            ScatterND
             \        |                   |
              \   present_key       present_value -----> output[1], output[2]
               \      |                   |
               |    Expand(G)         Expand(G)    (attention_bias, causal mask)
               |      |                   |           /
               |    k_Transpose           |          /
               |    (0,1,3,2)             |         /
               |      |                   |        /
            +---------------------------------------+
            |        ScaledDotProductAttention      |
            +---------------------------------------+
                              |
                            output
*/

Status GroupQueryAttentionOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  NodeAttrHelper helper(node);
  const int32_t local_window_size = helper.Get("local_window_size", -1);
  const uint32_t kv_num_heads = helper.Get("kv_num_heads", 0);
  const uint32_t num_heads = helper.Get("num_heads", 0);
  const bool do_rotary = static_cast<bool>(helper.Get("do_rotary", 0));
  const bool rotary_interleaved = static_cast<bool>(helper.Get("rotary_interleaved", 0));

  // Check if optional inputs exist
  const bool has_key = TensorExists(input_defs, 1);
  const bool has_value = TensorExists(input_defs, 2);
  const bool has_past_key = TensorExists(input_defs, 3);
  const bool has_past_value = TensorExists(input_defs, 4);
  const bool has_cos_cache = TensorExists(input_defs, 7);
  const bool has_sin_cache = TensorExists(input_defs, 8);
  const bool has_position_ids = TensorExists(input_defs, 9);

  emscripten::val query_input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val key_input =
      has_key ? model_builder.GetOperand(input_defs[1]->Name()) : emscripten::val::undefined();
  emscripten::val value_input =
      has_value ? model_builder.GetOperand(input_defs[2]->Name()) : emscripten::val::undefined();
  emscripten::val past_key_input =
      has_past_key ? model_builder.GetOperand(input_defs[3]->Name()) : emscripten::val::undefined();
  emscripten::val past_value_input =
      has_past_value ? model_builder.GetOperand(input_defs[4]->Name()) : emscripten::val::undefined();
  emscripten::val seqlens_k_input = model_builder.GetOperand(input_defs[5]->Name());
  emscripten::val cos_cache =
      has_cos_cache ? model_builder.GetOperand(input_defs[7]->Name()) : emscripten::val::undefined();
  emscripten::val sin_cache =
      has_sin_cache ? model_builder.GetOperand(input_defs[8]->Name()) : emscripten::val::undefined();

  std::vector<int64_t> input_q_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_q_shape, logger), "Cannot get query shape");

  std::vector<int64_t> input_k_shape;
  if (has_key) {
    ORT_RETURN_IF_NOT(GetShape(*input_defs[1], input_k_shape, logger), "Cannot get key shape");
  }
  std::vector<int64_t> input_v_shape;
  if (has_value) {
    ORT_RETURN_IF_NOT(GetShape(*input_defs[2], input_v_shape, logger), "Cannot get value shape");
  }

  // Calculate hidden_size and head_size based on whether key/value are provided.
  // GetShape returns 0 for dynamic dimensions. When query dim[2] is dynamic, fall back
  // to key or past_key shape protos to derive head_size (same pattern as MHA fix).
  uint32_t qkv_hidden_size;
  uint32_t head_size = 0;
  if (input_q_shape[2] > 0) {
    if (has_key) {
      // query shape is (batch_size, sequence_length, num_heads * head_size)
      head_size = SafeInt<uint32_t>(input_q_shape[2]) / num_heads;
    } else {
      // query contains packed QKV: (batch_size, sequence_length, num_heads * head_size + 2 * kv_num_heads * head_size)
      head_size = SafeInt<uint32_t>(input_q_shape[2]) / (num_heads + 2 * kv_num_heads);
    }
  }
  // Fallback: try key shape proto dim[2] = kv_num_heads * head_size.
  if (head_size == 0 && has_key) {
    const auto* k_shape_proto = input_defs[1]->Shape();
    if (k_shape_proto && k_shape_proto->dim_size() > 2 && k_shape_proto->dim(2).has_dim_value()) {
      head_size = static_cast<uint32_t>(k_shape_proto->dim(2).dim_value() / kv_num_heads);
    }
  }
  // Fallback: try past_key shape proto dim[3] = head_size (BNSH format).
  if (head_size == 0 && has_past_key) {
    const auto* pk_shape_proto = input_defs[3]->Shape();
    if (pk_shape_proto && pk_shape_proto->dim_size() > 3 && pk_shape_proto->dim(3).has_dim_value()) {
      head_size = static_cast<uint32_t>(pk_shape_proto->dim(3).dim_value());
    }
  }
  ORT_RETURN_IF(head_size == 0,
                "GroupQueryAttention: cannot determine head_size from query, key, or past_key shape protos.");
  qkv_hidden_size = num_heads * head_size;

  emscripten::val position_ids = emscripten::val::undefined();
  bool use_position_ids_as_offset = false;
  if (has_position_ids) {
    position_ids = model_builder.GetOperand(input_defs[9]->Name());
  } else {
    // If position_ids is not provided, derive it from seqlens_k as the per-batch position offset.
    // The model computes seqlens_k = reduceSum(attention_mask) - 1 = past_seq_len + (S - 1).
    // We subtract (S - 1) to recover past_sequence_length, which serves as the position offset:
    //   - Prefill (S = L):  offset = (L-1) - (L-1) = 0  → positions [0..S-1]
    //   - Decode  (S = 1):  offset = seqlens_k - 0 = seqlens_k → position [seqlens_k]

    // Compute S-1 dynamically from query sequence_length using BuildRange + reduceMax.
    // Get [S] shape from query_input dim 1 via shape() → slice.
    emscripten::val pos_shape_options = emscripten::val::object();
    pos_shape_options.set("label", node.Name() + "_/GQA/pos/query_shape");
    emscripten::val pos_query_shape = model_builder.GetBuilder().call<emscripten::val>(
        "shape", query_input, pos_shape_options);
    emscripten::val pos_s_shape = shape_utils::SliceShapeRange(
        model_builder.GetBuilder(), pos_query_shape, 1, 1,
        node.Name() + "_/GQA/pos/query_slice_s");
    emscripten::val pos_range = BuildRange(
        model_builder, pos_s_shape, node.Name() + "_/GQA/pos/range");
    emscripten::val pos_reduce_options = emscripten::val::object();
    pos_reduce_options.set("label", node.Name() + "_/GQA/pos/s_minus_1");
    emscripten::val pos_s_minus_1 = model_builder.GetBuilder().call<emscripten::val>(
        "reduceMax", pos_range, pos_reduce_options);

    // Correct seqlens_k by subtracting (S-1) to get past_sequence_length as position offset.
    emscripten::val pos_options = emscripten::val::object();
    pos_options.set("label", node.Name() + "_/GQA/pos/corrected_seqlens_k");
    emscripten::val corrected_seqlens_k = model_builder.GetBuilder().call<emscripten::val>(
        "sub", seqlens_k_input, pos_s_minus_1, pos_options);

    // Unsqueeze [B] → [B, 1] instead of dim-descriptor reshape.
    emscripten::val reshape_options = emscripten::val::object();
    reshape_options.set("label", node.Name() + "_/GQA/seqlens_k_reshape_for_position");
    emscripten::val reshaped_seqlens_k = model_builder.GetBuilder().call<emscripten::val>(
        "unsqueeze", corrected_seqlens_k, emscripten::val::array(std::vector<uint32_t>{1}), reshape_options);

    // seqlens_k is INT32, but position_ids_range in ApplyRotaryEmbedding may be INT64
    // if int64 is supported. We need to cast to match the expected type.
    if (model_builder.IsInt64Supported()) {
      emscripten::val cast_options = emscripten::val::object();
      cast_options.set("label", node.Name() + "_/GQA/seqlens_k_cast_to_int64");
      position_ids = model_builder.GetBuilder().call<emscripten::val>(
          "cast", reshaped_seqlens_k, emscripten::val("int64"), cast_options);
    } else {
      position_ids = reshaped_seqlens_k;
    }
    use_position_ids_as_offset = true;
  }

  const uint32_t group_size = SafeInt<uint32_t>(num_heads / kv_num_heads);

  const float scale_value = helper.Get("scale", 1 / sqrt(static_cast<float>(head_size)));

  emscripten::val common_options = emscripten::val::object();

  int32_t q_type = 0;
  ORT_RETURN_IF_NOT(GetType(*input_defs[0], q_type, logger), "Could not get input data type.");

  // Split packed QKV if key and value are not provided separately
  if (!has_key) {
    // query contains packed QKV: (batch_size, sequence_length, num_heads * head_size + 2 * kv_num_heads * head_size)
    const uint32_t kv_hidden_size = kv_num_heads * head_size;
    const std::vector<uint32_t> splits{qkv_hidden_size, kv_hidden_size, kv_hidden_size};
    emscripten::val split_options = emscripten::val::object();
    split_options.set("label", node.Name() + "_/GQA/split_packed_qkv");
    split_options.set("axis", 2);
    emscripten::val split_result = model_builder.GetBuilder().call<emscripten::val>(
        "split", query_input, emscripten::val::array(splits), split_options);
    query_input = split_result[0];
    key_input = split_result[1];
    value_input = split_result[2];
  }

  // Apply rotary embedding if do_rotary is true
  bool rotary_produced_bnsh = false;
  emscripten::val rotary_key_bnsh = emscripten::val::undefined();
  if (do_rotary && has_cos_cache && has_sin_cache) {
    // Determine rotary_embedding_dim from cos_cache shape
    std::vector<int64_t> cos_cache_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[7], cos_cache_shape, logger), "Cannot get cos_cache shape");
    const uint32_t rotary_embedding_dim = static_cast<uint32_t>(cos_cache_shape[1] * 2);

    // Always output BNSH directly to avoid paired transposes that the downstream
    // TransposeOptimizer would cancel (breaking OpenVINO's RoPE pattern matching).
    const bool rotary_output_bnsh = true;

    // Reshape query to (batch_size, sequence_length, num_heads, head_size) for rotary embedding
    emscripten::val reshaped_query_for_rotary = shape_utils::Reshape(
        model_builder, query_input, input_q_shape,
        {0, 0, static_cast<int64_t>(num_heads), static_cast<int64_t>(head_size)},
        node.Name() + "_/GQA/query/reshape_for_rotary");

    // Apply rotary embedding to query
    emscripten::val rotary_query_output;
    ORT_RETURN_IF_ERROR(ApplyRotaryEmbedding(
        model_builder,
        node.Name() + "_query",
        reshaped_query_for_rotary,
        cos_cache,
        sin_cache,
        position_ids,
        q_type,
        num_heads,
        head_size,
        rotary_embedding_dim,
        rotary_interleaved,
        true,
        use_position_ids_as_offset,  // position_ids_is_offset
        rotary_output_bnsh,
        HasDynamicShape(input_q_shape),
        rotary_query_output));

    // Reshape key to (batch_size, sequence_length, kv_num_heads, head_size) for rotary embedding
    emscripten::val reshaped_key_for_rotary = shape_utils::Reshape(
        model_builder, key_input, input_k_shape,
        {0, 0, static_cast<int64_t>(kv_num_heads), static_cast<int64_t>(head_size)},
        node.Name() + "_/GQA/key/reshape_for_rotary");

    // Apply rotary embedding to key
    emscripten::val rotary_key_output;
    ORT_RETURN_IF_ERROR(ApplyRotaryEmbedding(
        model_builder,
        node.Name() + "_key",
        reshaped_key_for_rotary,
        cos_cache,
        sin_cache,
        position_ids,
        q_type,
        kv_num_heads,
        head_size,
        rotary_embedding_dim,
        rotary_interleaved,
        true,
        use_position_ids_as_offset,  // position_ids_is_offset
        rotary_output_bnsh,
        HasDynamicShape(input_q_shape),
        rotary_key_output));

    // BNSH outputs: use directly, skip reshape-to-flat + later reshape+transpose
    query_input = rotary_query_output;  // [B, N, S, H]
    rotary_key_bnsh = rotary_key_output;  // [B, kv_N, S, H]
    rotary_produced_bnsh = true;
  }

  emscripten::val new_query;
  emscripten::val key_bsnh;
  emscripten::val value_bsnh;
  emscripten::val transpose_options = emscripten::val::object();

  if (rotary_produced_bnsh) {
    // Query is already BNSH from rotary embedding output.
    new_query = query_input;

    // Key is already BNSH (rotary_key_bnsh).
    // The ScatterND path uses BNSH key directly to avoid a Transpose that would
    // form a pair with the internal RoPE transpose (breaking OV pattern matching).

    // For value (not rotary-embedded), reshape to BSNH.
    value_bsnh = shape_utils::Reshape(
        model_builder, value_input, input_v_shape,
        {0, 0, static_cast<int64_t>(kv_num_heads), static_cast<int64_t>(head_size)},
        node.Name() + "_/GQA/value/reshape_bsnh");
  } else {
    // Normal path: reshape query to 4D + transpose to BNSH.
    emscripten::val reshaped_query = shape_utils::Reshape(
        model_builder, query_input, input_q_shape,
        {0, 0, static_cast<int64_t>(num_heads), static_cast<int64_t>(head_size)},
        node.Name() + "_/GQA/query/reshape");

    transpose_options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
    transpose_options.set("label", node.Name() + "_/GQA/query/transpose");
    new_query = model_builder.GetBuilder().call<emscripten::val>("transpose", reshaped_query, transpose_options);

    // Reshape key and value from BSW to BSNH: (B, S, kv_N*H) -> (B, S, kv_N, H)
    key_bsnh = shape_utils::Reshape(
        model_builder, key_input, input_k_shape,
        {0, 0, static_cast<int64_t>(kv_num_heads), static_cast<int64_t>(head_size)},
        node.Name() + "_/GQA/key/reshape_bsnh");

    // Value uses same target shape pattern but with value_input as source.
    value_bsnh = shape_utils::Reshape(
        model_builder, value_input, input_v_shape,
        {0, 0, static_cast<int64_t>(kv_num_heads), static_cast<int64_t>(head_size)},
        node.Name() + "_/GQA/value/reshape_bsnh");
  }

  // Compute s_minus_1 = S-1 dynamically as a scalar INT32 value.
  // S-1 is used below to derive past_sequence_length from seqlens_k.
  // Build range [0, 1, ..., S-1] via BuildRange, then reduceMax to get scalar S-1.
  // S is at dim 1 in BSNH (value_bsnh) or dim 2 in BNSH (new_query).
  // Get [S] shape via shape() → slice from the appropriate operand.
  emscripten::val seq_source_operand = rotary_produced_bnsh ? new_query : value_bsnh;
  uint32_t seq_dim_index = rotary_produced_bnsh ? 2 : 1;
  emscripten::val seq_shape_options = emscripten::val::object();
  seq_shape_options.set("label", node.Name() + "_/GQA/seq_source_shape");
  emscripten::val seq_source_shape = model_builder.GetBuilder().call<emscripten::val>(
      "shape", seq_source_operand, seq_shape_options);
  emscripten::val seq_ones_shape = shape_utils::SliceShapeRange(
      model_builder.GetBuilder(), seq_source_shape, seq_dim_index, 1,
      node.Name() + "_/GQA/seq_slice_s");
  emscripten::val seq_range = BuildRange(
      model_builder, seq_ones_shape, node.Name() + "_/GQA/seq_range");
  emscripten::val seq_reduce_options = emscripten::val::object();
  seq_reduce_options.set("label", node.Name() + "_/GQA/s_minus_1");
  emscripten::val s_minus_1 = model_builder.GetBuilder().call<emscripten::val>(
      "reduceMax", seq_range, seq_reduce_options);

  // present_kv computation: branch based on enableCausalLM option.
  // - CausalLM (stateful): concat(past_kv, new_kv, axis=2) — past grows each step
  // - Stateless (ScatterND): scatter new tokens into fixed-size past buffer at seqlens_k position
  emscripten::val present_key;
  emscripten::val present_value;
  if (model_builder.IsCausalLMEnabled()) {
    // Concat path: key/value in BNSH, then concat(past_kv, new_kv, axis=2)
    emscripten::val new_key_bnsh;
    if (rotary_produced_bnsh) {
      // Key is already BNSH from rotary embedding.
      new_key_bnsh = rotary_key_bnsh;
    } else {
      // Transpose key from BSNH to BNSH.
      transpose_options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
      transpose_options.set("label", node.Name() + "_/GQA/key/transpose_to_bnsh");
      new_key_bnsh = model_builder.GetBuilder().call<emscripten::val>(
          "transpose", key_bsnh, transpose_options);
    }

    // Value always needs transpose from BSNH to BNSH.
    transpose_options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
    transpose_options.set("label", node.Name() + "_/GQA/value/transpose_to_bnsh");
    emscripten::val new_value_bnsh = model_builder.GetBuilder().call<emscripten::val>(
        "transpose", value_bsnh, transpose_options);

    if (has_past_key && has_past_value) {
      emscripten::val concat_key_inputs = emscripten::val::array();
      concat_key_inputs.call<void>("push", past_key_input);
      concat_key_inputs.call<void>("push", new_key_bnsh);
      common_options.set("label", node.Name() + "_/GQA/present_key/concat");
      present_key = model_builder.GetBuilder().call<emscripten::val>(
          "concat", concat_key_inputs, 2, common_options);

      emscripten::val concat_value_inputs = emscripten::val::array();
      concat_value_inputs.call<void>("push", past_value_input);
      concat_value_inputs.call<void>("push", new_value_bnsh);
      common_options.set("label", node.Name() + "_/GQA/present_value/concat");
      present_value = model_builder.GetBuilder().call<emscripten::val>(
          "concat", concat_value_inputs, 2, common_options);
    } else {
      // No past: new key/value ARE the present key/value (prefill).
      present_key = new_key_bnsh;
      present_value = new_value_bnsh;
    }
  } else {
    // ScatterND path: scatter new key/value into past_kv buffer at the correct position.
    // When rotary_produced_bnsh: key is BNSH (rotary_key_bnsh), value transposed to BNSH.
    // Otherwise: key_bsnh/value_bsnh are in BSNH format.
    if (has_past_key && has_past_value) {
      /* Build scatter_indices [B,kv_N,S,3] (BNSH) or [B,S,kv_N,3] (BSNH)
         where last dim = [batch_idx, head_idx, seq_idx].
         - range_b: [0..B-1] broadcast to [B,S,kv_N]
         - range_k: [0..kv_N-1] broadcast to [B,S,kv_N] or [B,kv_N,S]
         - range_s: [0..S-1] + scatter_pos, broadcast to [B,S,kv_N] or [B,kv_N,S]
         - scatter_pos = seqlens_k - (S-1) = past_sequence_length
      */

      // When rotary_produced_bnsh, key is BNSH [B,kv_N,S,H] — build indices in BNSH layout
      // to avoid a BNSH→BSNH transpose that would pair with the internal RoPE transpose.
      // Otherwise key is BSNH [B,S,kv_N,H] — build indices in BSNH layout.
      // In both cases, index last dim = [batch_idx, head_idx, seq_idx] matching past_key [B,kv_N,max_seq,H].

      // key_for_scatter and value_for_scatter: the update tensors for ScatterND
      emscripten::val key_for_scatter = rotary_produced_bnsh ? rotary_key_bnsh : key_bsnh;
      // Value: transpose to BNSH if rotary_produced_bnsh (to share indices), else keep BSNH.
      emscripten::val value_for_scatter;
      if (rotary_produced_bnsh) {
        transpose_options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
        transpose_options.set("label", node.Name() + "_/GQA/value/transpose_to_bnsh_for_scatter");
        value_for_scatter = model_builder.GetBuilder().call<emscripten::val>(
            "transpose", value_bsnh, transpose_options);
      } else {
        value_for_scatter = value_bsnh;
      }

      // Dim references for the update tensor (BNSH: [B,kv_N,S,H] or BSNH: [B,S,kv_N,H])
      // dim_b=0 always, dim_s and dim_k swap between layouts.
      const uint32_t scatter_dim_s = rotary_produced_bnsh ? 2 : 1;

      // scatter_pos_for_scatter: [B] → [B,1,1] = unsqueeze(seqlens_k, axes=[1,2])
      common_options.set("label", node.Name() + "_/GQA/scatter/scatter_pos_reshape");
      emscripten::val scatter_pos_for_scatter = model_builder.GetBuilder().call<emscripten::val>(
          "unsqueeze", seqlens_k_input, emscripten::val::array(std::vector<uint32_t>{1, 2}), common_options);

      // Correct scatter offset: seqlens_k - (S - 1) = past_sequence_length
      common_options.set("label", node.Name() + "_/GQA/scatter/scatter_pos_fix");
      scatter_pos_for_scatter = model_builder.GetBuilder().call<emscripten::val>(
          "sub", scatter_pos_for_scatter, s_minus_1, common_options);

      // expand_shape: [B, kv_N, S] (BNSH) or [B, S, kv_N] (BSNH)
      // Get first 3 dims of key_for_scatter via shape() → slice.
      emscripten::val kfs_shape_options = emscripten::val::object();
      kfs_shape_options.set("label", node.Name() + "_/GQA/scatter/key_for_scatter_shape");
      emscripten::val kfs_shape = model_builder.GetBuilder().call<emscripten::val>(
          "shape", key_for_scatter, kfs_shape_options);
      emscripten::val expand_shape = shape_utils::SliceShapeRange(
          model_builder.GetBuilder(), kfs_shape, 0, 3,
          node.Name() + "_/GQA/scatter/expand_slice_bns");

      // range_b: [0, 1, ..., B-1] → unsqueeze [B,1,1] → dynamicExpand
      emscripten::val b_shape = shape_utils::SliceShapeRange(
          model_builder.GetBuilder(), kfs_shape, 0, 1,
          node.Name() + "_/GQA/scatter/slice_b");
      emscripten::val range_b = BuildRange(
          model_builder, b_shape, node.Name() + "_/GQA/scatter/range_b");
      // [B] → [B,1,1] via unsqueeze
      common_options.set("label", node.Name() + "_/GQA/scatter/range_b_reshape");
      range_b = model_builder.GetBuilder().call<emscripten::val>(
          "unsqueeze", range_b, emscripten::val::array(std::vector<uint32_t>{1, 2}), common_options);
      common_options.set("label", node.Name() + "_/GQA/scatter/range_b_expand");
      range_b = model_builder.GetBuilder().call<emscripten::val>("dynamicExpand", range_b, expand_shape, common_options);

      // range_s: [0, 1, ..., S-1] → unsqueeze → dynamicExpand, add offset
      emscripten::val s_shape = shape_utils::SliceShapeRange(
          model_builder.GetBuilder(), kfs_shape, scatter_dim_s, 1,
          node.Name() + "_/GQA/scatter/slice_s");
      emscripten::val range_s = BuildRange(
          model_builder, s_shape, node.Name() + "_/GQA/scatter/range_s");
      // [S] → [1,1,S] (BNSH) or [1,S,1] (BSNH) via unsqueeze
      if (rotary_produced_bnsh) {
        // BNSH: S is dim 2 → unsqueeze axes=[0,1] gives [1,1,S]
        common_options.set("label", node.Name() + "_/GQA/scatter/range_s_reshape");
        range_s = model_builder.GetBuilder().call<emscripten::val>(
            "unsqueeze", range_s, emscripten::val::array(std::vector<uint32_t>{0, 1}), common_options);
      } else {
        // BSNH: S is dim 1 → unsqueeze axes=[0,2] gives [1,S,1]
        common_options.set("label", node.Name() + "_/GQA/scatter/range_s_reshape");
        range_s = model_builder.GetBuilder().call<emscripten::val>(
            "unsqueeze", range_s, emscripten::val::array(std::vector<uint32_t>{0, 2}), common_options);
      }
      common_options.set("label", node.Name() + "_/GQA/scatter/range_s_expand");
      range_s = model_builder.GetBuilder().call<emscripten::val>("dynamicExpand", range_s, expand_shape, common_options);
      common_options.set("label", node.Name() + "_/GQA/scatter/scatter_pos_expand");
      scatter_pos_for_scatter = model_builder.GetBuilder().call<emscripten::val>("dynamicExpand", scatter_pos_for_scatter,
                                                                                 expand_shape, common_options);
      common_options.set("label", node.Name() + "_/GQA/scatter/range_s_add_offset");
      range_s = model_builder.GetBuilder().call<emscripten::val>(
          "add", range_s, scatter_pos_for_scatter, common_options);

      // range_k: [kv_N] → reshape to [1,kv_N,1] (BNSH) or [1,1,kv_N] (BSNH) → dynamicExpand
      std::vector<int32_t> range_k_data(kv_num_heads);
      std::iota(range_k_data.begin(), range_k_data.end(), 0);
      std::string range_k_name = "webnn_GQA_range_k_" + std::to_string(kv_num_heads);
      emscripten::val range_k = model_builder.CreateOrGetConstant<int32_t>(
          ONNX_NAMESPACE::TensorProto_DataType_INT32, range_k_name, range_k_data,
          rotary_produced_bnsh ? std::vector<uint32_t>({1, kv_num_heads, 1})
                               : std::vector<uint32_t>({1, 1, kv_num_heads}));
      common_options.set("label", node.Name() + "_/GQA/scatter/range_k_expand");
      range_k = model_builder.GetBuilder().call<emscripten::val>("dynamicExpand", range_k, expand_shape, common_options);

      // Reshape all index components from [B,kv_N,S] or [B,S,kv_N] to [...,1] via unsqueeze(axes=[3])
      // then concat on axis 3 to get [...,3]
      common_options.set("label", node.Name() + "_/GQA/scatter/range_b_reshape_last");
      range_b = model_builder.GetBuilder().call<emscripten::val>(
          "unsqueeze", range_b, emscripten::val::array(std::vector<uint32_t>{3}), common_options);
      common_options.set("label", node.Name() + "_/GQA/scatter/range_k_reshape_last");
      range_k = model_builder.GetBuilder().call<emscripten::val>(
          "unsqueeze", range_k, emscripten::val::array(std::vector<uint32_t>{3}), common_options);
      common_options.set("label", node.Name() + "_/GQA/scatter/range_s_reshape_last");
      range_s = model_builder.GetBuilder().call<emscripten::val>(
          "unsqueeze", range_s, emscripten::val::array(std::vector<uint32_t>{3}), common_options);

      common_options.set("label", node.Name() + "_/GQA/scatter/concat_for_scatter_indices");
      emscripten::val scatter_inputs = emscripten::val::array();
      scatter_inputs.call<void>("push", range_b);
      scatter_inputs.call<void>("push", range_k);
      scatter_inputs.call<void>("push", range_s);
      emscripten::val scatter_indices = model_builder.GetBuilder().call<emscripten::val>(
          "concat", scatter_inputs, 3, common_options);

      // ScatterND: update past_kv buffer with new key/value at computed positions
      common_options.set("label", node.Name() + "_/GQA/present_key/ScatterND");
      present_key = model_builder.GetBuilder().call<emscripten::val>(
          "scatterND", past_key_input, scatter_indices, key_for_scatter, common_options);

      common_options.set("label", node.Name() + "_/GQA/present_value/ScatterND");
      present_value = model_builder.GetBuilder().call<emscripten::val>(
          "scatterND", past_value_input, scatter_indices, value_for_scatter, common_options);
    } else {
      // No past_key/past_value, use key/value directly (first token / prefill).
      // Key/value must be in BNSH format for attention.
      if (rotary_produced_bnsh) {
        present_key = rotary_key_bnsh;
      } else {
        transpose_options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
        transpose_options.set("label", node.Name() + "_/GQA/key/transpose_to_bnsh");
        present_key = model_builder.GetBuilder().call<emscripten::val>(
            "transpose", key_bsnh, transpose_options);
      }

      transpose_options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 2, 1, 3})));
      transpose_options.set("label", node.Name() + "_/GQA/value/transpose_to_bnsh");
      present_value = model_builder.GetBuilder().call<emscripten::val>(
          "transpose", value_bsnh, transpose_options);
    }
  }

  emscripten::val true_present_key;
  emscripten::val true_present_value;

  if (group_size != 1) {
    // Broadcast key and value for group query by unsqueeze, expand, and dynamicReshape.
    // present kv shape (B,kv_N,P,H)
    //   B: batch size
    //   N: total number of attention heads (query heads)
    //   kv_N: number of key/value heads
    //   P: cache sequence axis used by attention (present/past kv length dimension)
    //   H: head size
    // -> unsqueeze(axes=[2]) -> (B,kv_N,1,P,H)
    // -> dynamicExpand -> (B,kv_N,G,P,H)
    // -> dynamicReshape -> (B,N,P,H) broadcasted kv shape

    true_present_key = GroupBroadcast(model_builder, present_key, group_size, num_heads,
                                      node.Name() + "_/GQA/true_present_key");

    true_present_value = GroupBroadcast(model_builder, present_value, group_size, num_heads,
                                        node.Name() + "_/GQA/true_present_value");
  } else {  // no need for broadcast
    true_present_key = present_key;
    true_present_value = present_value;
  }

  // Transpose key for matrix multiplication
  transpose_options.set("permutation", emscripten::val::array(std::vector<uint32_t>({0, 1, 3, 2})));
  transpose_options.set("label", node.Name() + "_/GQA/present_key/transpose");
  true_present_key = model_builder.GetBuilder().call<emscripten::val>("transpose", true_present_key, transpose_options);

  emscripten::val scale_constant = model_builder.CreateOrGetConstant<float>(q_type, scale_value, {1});

  /* Calculate attention_bias for masking softmax
        ones_array (shape=B,N,S,P)                          range_of_qkv_sequence_length_constant (0,1,2,...) (shape=S)
          |                                                                 |
        CumSum (axis=3, exclusive=true, reversed=false)                    Add <--- scatter_pos
          |                                                                 |
          |                                                               Expand (shape=P,S)
          |                                                                 |
          +-------------------------------> Lesser <---------------------Transpose (1,0)
                                                |
                                      1 ---> Where (attn_mask) <--- finfo_min (minimum value of FP32)
                                                |
                                          attention_bias
  */
  // Build causal attention mask of shape [B, N, S, total_seq] where total_seq = present_key.dim[2].
  // neq_left[i,j,q,k] = k  (column index in [0..total_seq-1])
  // neq_right[q,k] = (q + past_seq + 1)  (row boundary: tokens at or after this are masked)
  // condition: k < neq_right → attend; k >= neq_right → mask
  emscripten::val value_int_one_constant =
      model_builder.CreateOrGetConstant<int>(ONNX_NAMESPACE::TensorProto_DataType_INT32, 1, {1});

  // Build mask shape [B, N, S, P] from two operands:
  // B, N, S from new_query [B,N,S,H] (dims 0,1,2); P from true_present_value [B,N,P,H] (dim 2).
  emscripten::val mask_nq_shape_options = emscripten::val::object();
  mask_nq_shape_options.set("label", node.Name() + "_/GQA/mask/new_query_shape");
  emscripten::val mask_nq_shape = model_builder.GetBuilder().call<emscripten::val>(
      "shape", new_query, mask_nq_shape_options);
  emscripten::val mask_tpv_shape_options = emscripten::val::object();
  mask_tpv_shape_options.set("label", node.Name() + "_/GQA/mask/true_present_value_shape");
  emscripten::val mask_tpv_shape = model_builder.GetBuilder().call<emscripten::val>(
      "shape", true_present_value, mask_tpv_shape_options);
  // Concat: [B,N,S] from new_query dims 0..2, [P] from true_present_value dim 2
  emscripten::val mask_shape_segments = emscripten::val::array();
  mask_shape_segments.call<void>("push", shape_utils::SliceShapeRange(
      model_builder.GetBuilder(), mask_nq_shape, 0, 3,
      node.Name() + "_/GQA/mask/slice_bns"));
  mask_shape_segments.call<void>("push", shape_utils::SliceShapeRange(
      model_builder.GetBuilder(), mask_tpv_shape, 2, 1,
      node.Name() + "_/GQA/mask/slice_p"));
  emscripten::val mask_shape_concat_options = emscripten::val::object();
  mask_shape_concat_options.set("label", node.Name() + "_/GQA/mask/shape_concat");
  emscripten::val mask_shape_ones_shape = model_builder.GetBuilder().call<emscripten::val>(
      "concat", mask_shape_segments, 0, mask_shape_concat_options);
  common_options.set("label", node.Name() + "_/GQA/GQA_mask_shape_ones/expand");
  emscripten::val mask_shape_ones_shape_constant = model_builder.GetBuilder().call<emscripten::val>(
      "dynamicExpand", value_int_one_constant, mask_shape_ones_shape, common_options);

  emscripten::val cumsum_options = emscripten::val::object();
  cumsum_options.set("label", node.Name() + "_range_of_mask_shape");
  cumsum_options.set("exclusive", true);
  cumsum_options.set("reversed", false);
  emscripten::val neq_left = model_builder.GetBuilder().call<emscripten::val>(
      "cumulativeSum", mask_shape_ones_shape_constant, gsl::narrow<uint32_t>(3), cumsum_options);

  // Build range [1..S] for the query token positions.
  // Get [S] from value_bsnh dim 1 via shape() → slice.
  emscripten::val mask_vbsnh_shape_options = emscripten::val::object();
  mask_vbsnh_shape_options.set("label", node.Name() + "_/GQA/mask/value_bsnh_shape");
  emscripten::val mask_vbsnh_shape = model_builder.GetBuilder().call<emscripten::val>(
      "shape", value_bsnh, mask_vbsnh_shape_options);
  emscripten::val range_s_ones_shape = shape_utils::SliceShapeRange(
      model_builder.GetBuilder(), mask_vbsnh_shape, 1, 1,
      node.Name() + "_/GQA/mask/slice_s");
  common_options.set("label", node.Name() + "_/GQA/mask/range_s_ones");
  emscripten::val range_s_plus_one = model_builder.GetBuilder().call<emscripten::val>(
      "dynamicExpand", value_int_one_constant, range_s_ones_shape, common_options);
  emscripten::val range_cumsum_options = emscripten::val::object();
  range_cumsum_options.set("label", node.Name() + "_/GQA/mask/range_s_cumsum");
  range_cumsum_options.set("exclusive", false);
  range_cumsum_options.set("reversed", false);
  range_s_plus_one = model_builder.GetBuilder().call<emscripten::val>(
      "cumulativeSum", range_s_plus_one, gsl::narrow<uint32_t>(0), range_cumsum_options);

  // Derive past_sequence_length from seqlens_k for the causal mask offset.
  // The model computes seqlens_k = reduceSum(attention_mask) - 1 = past_seq + (S - 1).
  // Subtracting (S-1) recovers past_seq:
  //   - Prefill (S = L):  past_seq = (L-1) - (L-1) = 0
  //   - Decode  (S = 1):  past_seq = seqlens_k - 0 = seqlens_k
  emscripten::val first_index_constant =
      model_builder.CreateOrGetConstant<int>(ONNX_NAMESPACE::TensorProto_DataType_INT32, 0, {1});
  emscripten::val gather_offset_options = emscripten::val::object();
  gather_offset_options.set("label", node.Name() + "_/GQA/attn_mask/scatter_pos_gather_first");
  gather_offset_options.set("axis", 0);
  emscripten::val scatter_pos_for_mask = model_builder.GetBuilder().call<emscripten::val>(
      "gather", seqlens_k_input, first_index_constant, gather_offset_options);

  common_options.set("label", node.Name() + "_/GQA/attn_mask/scatter_pos_fix");
  scatter_pos_for_mask = model_builder.GetBuilder().call<emscripten::val>(
      "sub", scatter_pos_for_mask, s_minus_1, common_options);

  // neq_right = range_s_plus_one + past_seq → [S] values: [past_seq+1, past_seq+2, ..., past_seq+S]
  common_options.set("label", node.Name() + "_/GQA/attn_mask/add");
  emscripten::val pre_neq_right = model_builder.GetBuilder().call<emscripten::val>(
      "add", range_s_plus_one, scatter_pos_for_mask, common_options);

  // Expand to [total_seq, S] then transpose to [S, total_seq] for broadcasting with neq_left.
  // Build [P, S] shape from two operands: P from true_present_value dim 2, S from value_bsnh dim 1.
  emscripten::val neq_expand_segments = emscripten::val::array();
  neq_expand_segments.call<void>("push", shape_utils::SliceShapeRange(
      model_builder.GetBuilder(), mask_tpv_shape, 2, 1,
      node.Name() + "_/GQA/neq_right/slice_p"));
  neq_expand_segments.call<void>("push", shape_utils::SliceShapeRange(
      model_builder.GetBuilder(), mask_vbsnh_shape, 1, 1,
      node.Name() + "_/GQA/neq_right/slice_s"));
  emscripten::val neq_expand_concat_options = emscripten::val::object();
  neq_expand_concat_options.set("label", node.Name() + "_/GQA/neq_right/expand_shape_concat");
  emscripten::val reshape_pre_neq_right = model_builder.GetBuilder().call<emscripten::val>(
      "concat", neq_expand_segments, 0, neq_expand_concat_options);

  common_options.set("label", node.Name() + "_/GQA/expand_neq_right");
  emscripten::val expanded_neq_right = model_builder.GetBuilder().call<emscripten::val>(
      "dynamicExpand", pre_neq_right, reshape_pre_neq_right, common_options);

  transpose_options.set("permutation", emscripten::val::array(std::vector<uint32_t>({1, 0})));
  transpose_options.set("label", node.Name() + "_/GQA/neq_right/transpose");
  emscripten::val neq_right =
      model_builder.GetBuilder().call<emscripten::val>("transpose", expanded_neq_right, transpose_options);

  common_options.set("label", node.Name() + "_/GQA/attn_mask/condition_1");
  emscripten::val condition_1 =
      model_builder.GetBuilder().call<emscripten::val>("lesser", neq_left, neq_right, common_options);

  emscripten::val condition = condition_1;
  // For local window size not equal to -1, new attention mask pattern for applying sliding window
  /*
     condition_1 (old attn_mask) ---> CumSum (axis=3, exclusive=true, reversed=true)
          |                             |
          |                           Lesser <--- local_window_size
          |                             |
      LogicalAnd <----------------- condition_2
          |
    new attn_mask
  */
  if (local_window_size != -1) {
    // Cast condition
    common_options.set("label", node.Name() + "_/GQA/attn_mask/condition_2/cast");
    emscripten::val casted_condition_1 =
        model_builder.GetBuilder().call<emscripten::val>("cast", condition_1, emscripten::val("int32"), common_options);

    cumsum_options = emscripten::val::object();
    cumsum_options.set("label", node.Name() + "_/GQA/attn_mask/condition_2/cumsum");
    cumsum_options.set("exclusive", true);
    cumsum_options.set("reversed", true);
    emscripten::val neq_left_2 = model_builder.GetBuilder().call<emscripten::val>(
        "cumulativeSum", casted_condition_1, gsl::narrow<uint32_t>(3), cumsum_options);

    emscripten::val local_window_size_constant =
        model_builder.CreateOrGetConstant<int>(ONNX_NAMESPACE::TensorProto_DataType_INT32, local_window_size, {1});

    common_options.set("label", node.Name() + "_/GQA/attn_mask/condition_2");
    emscripten::val condition_2 = model_builder.GetBuilder().call<emscripten::val>(
        "lesser", neq_left_2, local_window_size_constant, common_options);

    common_options.set("label", node.Name() + "_/GQA/attn_mask/condition/and");
    condition = model_builder.GetBuilder().call<emscripten::val>(
        "logicalAnd", condition_1, condition_2, common_options);
  }

  // For attended positions, use 0.0 (no change to attention scores)
  // For masked positions, use a very large negative number (softmax → 0)
  emscripten::val value_zero_constant_float = model_builder.CreateOrGetConstant<float>(q_type, 0, {1});

  // finfo_min: the minimum value of float32
  emscripten::val finfo_min_constant = model_builder.CreateOrGetConstant<float>(q_type, -3.4028234663852886e+38, {1});

  common_options.set("label", node.Name() + "_/GQA/attn_mask/where");
  emscripten::val attn_mask = model_builder.GetBuilder().call<emscripten::val>(
      "where", condition, value_zero_constant_float, finfo_min_constant, common_options);

  // Output shape: (B, S, N*H) — reshape from BNSH after attention.
  // new_query is [B, N, S, H]; after SDPA transpose output is [B, S, N, H].
  // ComputeShape on that with {0, 0, hidden} gives [B, S, hidden].
  std::vector<int64_t> reshape_output_target{0, 0, static_cast<int64_t>(qkv_hidden_size)};

  // Execute ScaledDotProductAttention
  emscripten::val output =
      ScaledDotProductAttention(model_builder, node, logger, new_query, true_present_key, true_present_value,
                                scale_constant, attn_mask, reshape_output_target,
                                HasDynamicShape(input_q_shape));

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  model_builder.AddOperand(node.OutputDefs()[1]->Name(), std::move(present_key));
  model_builder.AddOperand(node.OutputDefs()[2]->Name(), std::move(present_value));

  return Status::OK();
}

// Operator support related.

bool GroupQueryAttentionOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const Node& node,
                                                     const WebnnDeviceType /* device_type */,
                                                     const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  NodeAttrHelper helper(node);

  const int64_t do_rotary = helper.Get("do_rotary", static_cast<int64_t>(0));

  // When do_rotary is true, cos_cache and sin_cache must be provided
  if (do_rotary) {
    if (!TensorExists(input_defs, 7) || !TensorExists(input_defs, 8)) {
      LOGS(logger, VERBOSE) << op_type << " requires cos_cache and sin_cache when do_rotary is true";
      return false;
    }
  }

  const auto& total_sequence_length_name = input_defs[6]->Name();
  const auto* total_sequence_length_initializer = graph_viewer.GetConstantInitializer(total_sequence_length_name);
  emscripten::val total_sequence_length = emscripten::val::undefined();
  if (!total_sequence_length_initializer) {
    LOGS(logger, VERBOSE) << "total_sequence_length is not a constant";
  } else {
    const auto total_sequence_length_tensor = *total_sequence_length_initializer;
    if (!ReadScalarTensorData(total_sequence_length_tensor, total_sequence_length, graph_viewer, logger)) {
      return false;
    }
  }

  std::vector<int64_t> query_shape;
  if (!GetShape(*input_defs[0], query_shape, logger)) {
    LOGS(logger, VERBOSE) << "Cannot get query shape.";
    return false;
  }
  if (query_shape.size() != 3) {
    LOGS(logger, VERBOSE) << op_type << " query shape is not rank 3.";
    return false;
  }

  const int64_t sequence_length = query_shape[1];
  const bool known_sequence_length = sequence_length >= 0;

  // Check if past_key exists to determine past_sequence_length
  const bool has_past_key = TensorExists(input_defs, 3);
  int64_t past_sequence_length = 0;
  bool known_past_sequence_length = false;
  if (has_past_key) {
    std::vector<int64_t> past_key_shape;
    if (!GetShape(*input_defs[3], past_key_shape, logger)) {
      LOGS(logger, VERBOSE) << "Cannot get past_key shape.";
      return false;
    }
    past_sequence_length = past_key_shape[2];
    known_past_sequence_length = past_sequence_length >= 0;
  }

  // WebNN EP only supports past_sequence_length of past kv equals to present_sequence_length of present kv
  // According to CPU EP, present_sequence_length = max(past_sequence_length,total_sequence_length)
  // For prefilling stage (the first prompt), it requires sequence_length == total_sequence_length.
  // For dynamic shapes, sequence_length and/or past_sequence_length can be unknown at compile time.
  // In that case, defer these stage-specific checks to runtime behavior and keep the node supported.
  if (!total_sequence_length.isUndefined()) {
    if (known_sequence_length && sequence_length > 1) {
      if (sequence_length != total_sequence_length.as<int32_t>()) {
        LOGS(logger, VERBOSE) << op_type << " sequence_length != total_sequence_length. Not first prompt.";
        return false;
      }
    // For decoding stage, it requires past_sequence_length == total_sequence_length.
    } else if (known_sequence_length && sequence_length == 1) {
      if (has_past_key && known_past_sequence_length && past_sequence_length != total_sequence_length.as<int32_t>()) {
        LOGS(logger, VERBOSE) << op_type << " past_sequence_length != total_sequence_length.";
        return false;
      }
    }
  }

  const auto& output_defs = node.OutputDefs();
  if (output_defs.size() != 3) {
    LOGS(logger, VERBOSE) << op_type << " output count must be three.";
    return false;
  }

  return true;
}

bool GroupQueryAttentionOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                                          const emscripten::val& wnn_limits,
                                                          const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  NodeAttrHelper helper(node);

  const int64_t do_rotary = helper.Get("do_rotary", static_cast<int64_t>(0));

  // Validate required inputs: query(0), seqlens_k(5), total_sequence_length(6) are always required
  // key(1), value(2), past_key(3), past_value(4) are optional
  // cos_cache(7), sin_cache(8) are required when do_rotary is true
  // position_ids(9), attention_bias(10), head_sink(11) are optional

  // Check required inputs
  if (!TensorExists(input_defs, 0)) {
    LOGS(logger, VERBOSE) << op_type << " requires query input (index 0)";
    return false;
  }
  if (!TensorExists(input_defs, 5)) {
    LOGS(logger, VERBOSE) << op_type << " requires seqlens_k input (index 5)";
    return false;
  }
  if (!TensorExists(input_defs, 6)) {
    LOGS(logger, VERBOSE) << op_type << " requires total_sequence_length input (index 6)";
    return false;
  }

  // Check key/value pair consistency
  const bool has_key = TensorExists(input_defs, 1);
  const bool has_value = TensorExists(input_defs, 2);
  if (has_key != has_value) {
    LOGS(logger, VERBOSE) << op_type << " key and value must both be present or both be absent";
    return false;
  }

  // Check past_key/past_value pair consistency
  const bool has_past_key = TensorExists(input_defs, 3);
  const bool has_past_value = TensorExists(input_defs, 4);
  if (has_past_key != has_past_value) {
    LOGS(logger, VERBOSE) << op_type << " past_key and past_value must both be present or both be absent";
    return false;
  }

  // Check do_rotary requirements
  const bool has_cos_cache = TensorExists(input_defs, 7);
  const bool has_sin_cache = TensorExists(input_defs, 8);
  if (do_rotary) {
    if (!has_cos_cache || !has_sin_cache) {
      LOGS(logger, VERBOSE) << op_type << " requires cos_cache and sin_cache when do_rotary is true";
      return false;
    }
  }

  // Get query type (required)
  int32_t q_type = 0;
  if (!GetType(*input_defs[0], q_type, logger)) {
    return false;
  }

  // Check optional key/value types
  if (has_key) {
    int32_t k_type = 0;
    int32_t v_type = 0;
    if (!GetType(*input_defs[1], k_type, logger) || !GetType(*input_defs[2], v_type, logger)) {
      return false;
    }
    std::array<int32_t, 3> qkv_types{q_type, k_type, v_type};
    if (!AreDataTypesSame(op_type, qkv_types, logger)) {
      return false;
    }
  }

  // Check optional past_key/past_value types
  if (has_past_key) {
    int32_t past_k_type = 0;
    int32_t past_v_type = 0;
    if (!GetType(*input_defs[3], past_k_type, logger) || !GetType(*input_defs[4], past_v_type, logger)) {
      return false;
    }
    std::array<int32_t, 3> past_types{q_type, past_k_type, past_v_type};
    if (!AreDataTypesSame(op_type, past_types, logger)) {
      return false;
    }
  }

  // Check seqlens_k and total_sequence_length types
  int32_t seqlens_k_type = 0;
  int32_t total_sequence_length_type = 0;
  if (!GetType(*input_defs[5], seqlens_k_type, logger) ||
      !GetType(*input_defs[6], total_sequence_length_type, logger)) {
    return false;
  }

  if (q_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT && q_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    LOGS(logger, VERBOSE) << op_type << " query type must be float or float16";
    return false;
  }

  if (seqlens_k_type != ONNX_NAMESPACE::TensorProto_DataType_INT32 ||
      total_sequence_length_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    LOGS(logger, VERBOSE) << op_type << " seqlens_k and total_sequence_length must be int32";
    return false;
  }

  // Check cos_cache/sin_cache types when do_rotary is true
  if (do_rotary && has_cos_cache && has_sin_cache) {
    int32_t cos_cache_type = 0;
    int32_t sin_cache_type = 0;
    if (!GetType(*input_defs[7], cos_cache_type, logger) || !GetType(*input_defs[8], sin_cache_type, logger)) {
      return false;
    }
    std::array<int32_t, 3> cache_types{q_type, cos_cache_type, sin_cache_type};
    if (!AreDataTypesSame(op_type, cache_types, logger)) {
      return false;
    }
  }

  // Check shapes
  std::vector<int64_t> input_q_shape;
  if (!GetShape(*input_defs[0], input_q_shape, logger)) {
    return false;
  }
  const auto q_rank = input_q_shape.size();
  if (q_rank != 3) {  // The query shape should be BSW
    LOGS(logger, VERBOSE) << op_type << " query shape is not BSW.";
    return false;
  }

  if (has_key) {
    std::vector<int64_t> input_k_shape, input_v_shape;
    if (!GetShape(*input_defs[1], input_k_shape, logger) || !GetShape(*input_defs[2], input_v_shape, logger)) {
      return false;
    }
    const auto k_rank = input_k_shape.size();
    const auto v_rank = input_v_shape.size();
    if (k_rank != 3 || v_rank != 3) {  // The kv shape should be BSW
      LOGS(logger, VERBOSE) << op_type << " key/value shape is not BSW.";
      return false;
    }
  }

  if (has_past_key) {
    std::vector<int64_t> input_past_k_shape, input_past_v_shape;
    if (!GetShape(*input_defs[3], input_past_k_shape, logger) ||
        !GetShape(*input_defs[4], input_past_v_shape, logger)) {
      return false;
    }
    const auto past_k_rank = input_past_k_shape.size();
    const auto past_v_rank = input_past_v_shape.size();
    if (past_k_rank != 4 || past_v_rank != 4) {  // The past qkv shape should be BNSH
      LOGS(logger, VERBOSE) << op_type << " past qkv shape is not BNSH.";
      return false;
    }
  }

  return true;
}

bool GroupQueryAttentionOpBuilder::HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                                           const logging::Logger& logger) const {
  const auto& output_defs = node.OutputDefs();
  const std::string_view op_type = node.OpType();
  int32_t output_type = 0;
  int32_t present_k_type = 0;
  int32_t present_v_type = 0;
  if (!GetType(*output_defs[0], output_type, logger) || !GetType(*output_defs[1], present_k_type, logger) ||
      !GetType(*output_defs[2], present_v_type, logger)) {
    return false;
  }

  std::array<int32_t, 3> output_types{output_type, present_k_type, present_v_type};
  if (!AreDataTypesSame(op_type, output_types, logger)) {
    return false;
  }

  // GQA allows float16, bfloat16 and float32, but WebNN only supports float16 and float32.
  if (output_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      output_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    return false;
  }
  return true;
}

void CreateGroupQueryAttentionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GroupQueryAttentionOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
