// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include <cmath>

#include "base_op_builder.h"
#include "attention_helper.h"
#include "shape_utils.h"

namespace onnxruntime {
namespace webnn {

// com.microsoft.LinearAttention — the gated-DeltaNet "linear attention" recurrence used by
// Qwen3.5's linear-attention layers. Per (batch, kv-head) it maintains a state matrix
// S ∈ R[d_k, d_v] and, for each token t, applies the update selected by `update_rule`:
//   linear:      S = S + k⊗v;                       o = scale * qᵀS
//   gated:       S = exp(g)·S + k⊗v;                o = scale * qᵀS
//   delta:       S = S + β·k⊗(v − Sᵀk);            o = scale * qᵀS
//   gated_delta: S = exp(g)·S + β·k⊗(v − exp(g)Sᵀk); o = scale * qᵀS
//
// Sequence handling — the recurrence is unrolled over a STATIC sequence length:
//   The recurrence is inherently sequential: S_t depends on S_{t-1} through a data-dependent
//   transition (the `Sᵀk` retrieval term). It has no loop-free parallel form for a variable token
//   count, and WebNN is a declarative graph of predefined operators with no control flow (Loop/Scan)
//   and no linear-algebra solve/inverse — the WebGPU/CPU EPs handle T>1 with an imperative per-token
//   loop that WebNN cannot express.
//   However, the WebNN EP specializes the graph per concrete input shape, so the sequence length is
//   known at build time. This builder therefore UNROLLS the recurrence into `seq_len` fixed steps:
//   one step for decode (T=1) and T steps for prefill (T>1), threading the state S from step to step
//   and concatenating the per-token outputs. A static sequence length is required (guaranteed by the
//   per-shape specialization); a truly dynamic T would fall back to a single step (correct only for
//   runtime T=1).
//
// Each step is batched over kv_num_heads via WebNN batched matmul + broadcasting, mirroring
// OpenVINO's per-step fuse_gated_delta_net pattern (Exp/Mul/ReduceSum/Sub/Add/matmul).
class LinearAttentionOpBuilder : public BaseOpBuilder {
 public:
  // Optional inputs (past_state, decay, beta) may be absent depending on update_rule.
  LinearAttentionOpBuilder() : BaseOpBuilder(/*allow_empty_tensor_as_input=*/true) {}

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

namespace {

// Read dimension `idx` from a WebNN operand's build-time shape. Returns the concrete value when the
// dim is static (a JS number), or kDynamicDim when it is symbolic/unavailable. The WebNN EP
// specializes the graph per input shape, so derived operands carry concrete dims even when the ONNX
// NodeArg's dim is symbolic (e.g. 'sequence_length'), which GetShape would report as kDynamicDim.
int64_t ConcreteOperandDim(const emscripten::val& operand, uint32_t idx) {
  emscripten::val dims = operand["shape"];
  if (dims.isUndefined() || dims.isNull() || idx >= dims["length"].as<uint32_t>()) {
    return kDynamicDim;
  }
  emscripten::val d = dims[idx];
  if (d.typeOf().as<std::string>() == "number") {
    return static_cast<int64_t>(d.as<double>());
  }
  return kDynamicDim;
}

// Transpose the last two axes of a 4-D operand ([...,a,b] → [...,b,a]).
emscripten::val TransposeLast2(ModelBuilder& model_builder, const emscripten::val& x, const std::string& label) {
  emscripten::val options = emscripten::val::object();
  options.set("label", label);
  options.set("permutation", emscripten::val::array(std::vector<uint32_t>{0, 1, 3, 2}));
  return model_builder.GetBuilder().call<emscripten::val>("transpose", x, options);
}

}  // namespace

Status LinearAttentionOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                       const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();
  NodeAttrHelper helper(node);
  emscripten::val builder = model_builder.GetBuilder();

  const std::string update_rule = helper.Get("update_rule", "gated_delta");
  const bool needs_decay = (update_rule == "gated" || update_rule == "gated_delta");
  const bool needs_delta = (update_rule == "delta" || update_rule == "gated_delta");  // also needs retrieval+beta
  const int64_t q_num_heads = helper.Get("q_num_heads", static_cast<int64_t>(0));
  const int64_t kv_num_heads = helper.Get("kv_num_heads", static_cast<int64_t>(0));

  int32_t input_type;
  ORT_RETURN_IF_NOT(GetType(*input_defs[0], input_type, logger), "Cannot get query type");

  std::vector<int64_t> q_shape, k_shape, v_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], q_shape, logger), "Cannot get query shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[1], k_shape, logger), "Cannot get key shape");
  ORT_RETURN_IF_NOT(GetShape(*input_defs[2], v_shape, logger), "Cannot get value shape");

  const int64_t d_k = q_shape.back() / q_num_heads;
  const int64_t n_k_heads = k_shape.back() / d_k;
  const int64_t d_v = v_shape.back() / kv_num_heads;

  float scale = helper.Get("scale", 0.0f);
  if (scale == 0.0f) {
    scale = 1.0f / std::sqrt(static_cast<float>(d_k));
  }

  // The ONNX NodeArg seq dim is symbolic ('sequence_length'), so GetShape reports it as dynamic. But
  // the WebNN EP specializes the graph per input shape, so the query operand carries the concrete
  // sequence length. Read it from the operand and concretize the packed shapes' seq dim so the head
  // reshapes below take the static path and produce per-token sliceable operands.
  int64_t batch = ConcreteOperandDim(model_builder.GetOperand(input_defs[0]->Name()), 0);
  if (batch == kDynamicDim) batch = q_shape[0];
  int64_t seq_len = ConcreteOperandDim(model_builder.GetOperand(input_defs[0]->Name()), 1);
  if (seq_len == kDynamicDim) seq_len = q_shape[1];
  if (seq_len != kDynamicDim) {
    q_shape[1] = seq_len;
    k_shape[1] = seq_len;
    v_shape[1] = seq_len;
  }

  // Reshape a packed (B, T, H*D) operand to (B, H, T, D).
  auto to_heads = [&](const emscripten::val& x, const std::vector<int64_t>& shp,
                      int64_t heads, int64_t dim, const std::string& lbl) -> emscripten::val {
    emscripten::val x4 = shape_utils::Reshape(model_builder, x, shp, {0, 0, heads, dim}, lbl + "_reshape");
    emscripten::val opt = emscripten::val::object();
    opt.set("label", lbl + "_transpose");
    opt.set("permutation", emscripten::val::array(std::vector<uint32_t>{0, 2, 1, 3}));
    return builder.call<emscripten::val>("transpose", x4, opt);
  };

  // q: (B, Hq, T, d_k); k: (B, n_k, T, d_k); v: (B, Hkv, T, d_v).
  emscripten::val q = to_heads(model_builder.GetOperand(input_defs[0]->Name()), q_shape, q_num_heads, d_k, node.Name() + "_q");
  emscripten::val k = to_heads(model_builder.GetOperand(input_defs[1]->Name()), k_shape, n_k_heads, d_k, node.Name() + "_k");
  emscripten::val v = to_heads(model_builder.GetOperand(input_defs[2]->Name()), v_shape, kv_num_heads, d_v, node.Name() + "_v");

  // K is expanded from n_k_heads to kv_num_heads per step (below), i.e. after token slicing, so that
  // slicing always operates on the statically-shaped pre-expansion tensor.

  // Past state S: (B, Hkv, d_k, d_v). If absent, start from zeros.
  emscripten::val S = emscripten::val::undefined();
  if (TensorExists(input_defs, 3)) {
    S = model_builder.GetOperand(input_defs[3]->Name());
  } else {
    // zero state of shape [B, Hkv, d_k, d_v]; B taken from query's runtime shape.
    emscripten::val q_shape_op = builder.call<emscripten::val>("shape", q);
    emscripten::val batch = shape_utils::SliceShapeRange(builder, q_shape_op, 0, 1, node.Name() + "_state_batch");
    emscripten::val tail = model_builder.CreateOrGetConstant<uint32_t>(
        ONNX_NAMESPACE::TensorProto_DataType_UINT32, node.Name() + "_state_tail",
        std::vector<uint32_t>{SafeInt<uint32_t>(kv_num_heads), SafeInt<uint32_t>(d_k), SafeInt<uint32_t>(d_v)}, {3});
    emscripten::val seg = emscripten::val::array();
    seg.call<void>("push", batch);
    seg.call<void>("push", tail);
    emscripten::val concat_options = emscripten::val::object();
    concat_options.set("label", node.Name() + "_state_shape");
    emscripten::val state_shape = builder.call<emscripten::val>("concat", seg, 0, concat_options);
    emscripten::val zero = model_builder.CreateOrGetConstant<float>(input_type, 0);
    emscripten::val expand_options = emscripten::val::object();
    expand_options.set("label", node.Name() + "_zero_state");
    S = builder.call<emscripten::val>("expandDynamic", zero, state_shape, expand_options);
  }

  // The gated-delta recurrence is sequential over tokens. Using the concrete seq_len read above:
  // for prefill (seq_len > 1) unroll the recurrence into seq_len fixed steps; for decode (seq_len ==
  // 1, or a dynamic dim) a single step suffices. There is no loop-free parallel form for
  // delta/gated_delta, so a static seq length is required for T > 1.
  const bool unroll = seq_len != kDynamicDim && seq_len > 1;
  ORT_RETURN_IF(unroll && batch == kDynamicDim,
                "LinearAttention: prefill (sequence length > 1) requires a static batch size.");

  // Readout head mapping: inverse GQA (q < kv) expands Q to kv heads; standard GQA (q > kv) expands
  // the state to q heads. The recurrent state itself always has kv_num_heads.
  const bool expand_state = q_num_heads > kv_num_heads;
  const int64_t out_heads = q_num_heads > kv_num_heads ? q_num_heads : kv_num_heads;

  // decay/beta prepared at full sequence length (still packed by head); sliced per step below.
  emscripten::val exp_g_full = emscripten::val::undefined();
  bool decay_per_key = false;
  if (needs_decay) {
    ORT_RETURN_IF_NOT(TensorExists(input_defs, 4), "decay input required for update_rule=", update_rule);
    std::vector<int64_t> decay_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[4], decay_shape, logger), "Cannot get decay shape");
    if (seq_len != kDynamicDim) decay_shape[1] = seq_len;  // concretize seq dim for static reshape
    decay_per_key = decay_shape.back() == kv_num_heads * d_k;  // per-key-dim vs per-head scalar
    emscripten::val g = to_heads(model_builder.GetOperand(input_defs[4]->Name()), decay_shape,
                                 kv_num_heads, decay_per_key ? d_k : 1, node.Name() + "_decay");
    emscripten::val exp_options = emscripten::val::object();
    exp_options.set("label", node.Name() + "_exp_decay");
    exp_g_full = builder.call<emscripten::val>("exp", g, exp_options);  // (B,Hkv,T,1) or (B,Hkv,T,d_k)
  }
  emscripten::val beta_full = emscripten::val::undefined();
  int64_t beta_heads = 0;
  if (needs_delta) {
    ORT_RETURN_IF_NOT(TensorExists(input_defs, 5), "beta input required for update_rule=", update_rule);
    std::vector<int64_t> beta_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[5], beta_shape, logger), "Cannot get beta shape");
    if (seq_len != kDynamicDim) beta_shape[1] = seq_len;  // concretize seq dim for static reshape
    beta_heads = beta_shape.back();  // Hkv or 1
    beta_full = to_heads(model_builder.GetOperand(input_defs[5]->Name()), beta_shape,
                         beta_heads, 1, node.Name() + "_beta");  // (B,beta_heads,T,1)
  }

  // Slice token t (seq axis == 2) from a static (B, H, T, D) operand. Only used when unrolling.
  auto slice_token = [&](const emscripten::val& x, int64_t heads, int64_t dim, int64_t t,
                         const std::string& lbl) -> emscripten::val {
    const std::vector<uint32_t> starts{0, 0, SafeInt<uint32_t>(t), 0};
    const std::vector<uint32_t> sizes{SafeInt<uint32_t>(batch), SafeInt<uint32_t>(heads), 1,
                                      SafeInt<uint32_t>(dim)};
    emscripten::val opt = emscripten::val::object();
    opt.set("label", lbl);
    return builder.call<emscripten::val>("slice", x, emscripten::val::array(starts),
                                         emscripten::val::array(sizes), opt);
  };

  // One recurrence step for token t (t < 0 uses the full tensors — the single-token decode path).
  // Updates S in place and returns o_t of shape (B, out_heads, 1, d_v).
  auto run_step = [&](int64_t t, emscripten::val& S_ref) -> emscripten::val {
    const std::string lbl = (t < 0) ? node.Name() : node.Name() + "_t" + std::to_string(t);
    auto sl = [&](const emscripten::val& x, int64_t heads, int64_t dim, const char* suffix) {
      return (t < 0) ? x : slice_token(x, heads, dim, t, lbl + suffix);
    };

    // Slice per token first, then expand heads, so slicing stays on statically-shaped operands.
    emscripten::val k_t = sl(k, n_k_heads, d_k, "_k");
    if (n_k_heads != kv_num_heads) {
      k_t = BroadcastHeads(model_builder, k_t, SafeInt<uint32_t>(kv_num_heads / n_k_heads),
                           SafeInt<uint32_t>(kv_num_heads), lbl + "_k_expand");
    }
    emscripten::val v_t = sl(v, kv_num_heads, d_v, "_v");
    emscripten::val opt = emscripten::val::object();

    // decay: gated = S * exp(g_t)
    emscripten::val gated = S_ref;
    if (needs_decay) {
      emscripten::val exp_g_t = sl(exp_g_full, kv_num_heads, decay_per_key ? d_k : 1, "_g");
      if (decay_per_key) {
        exp_g_t = TransposeLast2(model_builder, exp_g_t, lbl + "_decay_t");  // (B,Hkv,d_k,1)
      }
      opt.set("label", lbl + "_gated_state");
      gated = builder.call<emscripten::val>("mul", S_ref, exp_g_t, opt);
    }

    // delta = beta_t * (v_t − k_tᵀ·S) for delta rules, else v_t.
    emscripten::val delta = v_t;
    if (needs_delta) {
      opt.set("label", lbl + "_retrieved");
      emscripten::val retrieved = builder.call<emscripten::val>("matmul", k_t, gated, opt);
      opt.set("label", lbl + "_v_minus_retrieved");
      emscripten::val diff = builder.call<emscripten::val>("sub", v_t, retrieved, opt);
      emscripten::val beta_t = sl(beta_full, beta_heads, 1, "_beta");
      opt.set("label", lbl + "_delta");
      delta = builder.call<emscripten::val>("mul", diff, beta_t, opt);
    }

    // state update: S = gated + k_tᵀ ⊗ delta
    emscripten::val k_tt = TransposeLast2(model_builder, k_t, lbl + "_kt");  // (B,Hkv,d_k,1)
    opt.set("label", lbl + "_outer_update");
    emscripten::val outer = builder.call<emscripten::val>("matmul", k_tt, delta, opt);
    opt.set("label", lbl + "_state_update");
    S_ref = builder.call<emscripten::val>("add", gated, outer, opt);

    // readout: o_t = scale * q_tᵀ · S
    emscripten::val q_t = sl(q, q_num_heads, d_k, "_q");
    if (q_num_heads < kv_num_heads) {
      q_t = BroadcastHeads(model_builder, q_t, SafeInt<uint32_t>(kv_num_heads / q_num_heads),
                           SafeInt<uint32_t>(kv_num_heads), lbl + "_q_expand");
    }
    emscripten::val s_ro = S_ref;
    if (expand_state) {
      s_ro = BroadcastHeads(model_builder, S_ref, SafeInt<uint32_t>(q_num_heads / kv_num_heads),
                            SafeInt<uint32_t>(q_num_heads), lbl + "_s_expand");
    }
    opt.set("label", lbl + "_readout");
    emscripten::val o_t = builder.call<emscripten::val>("matmul", q_t, s_ro, opt);  // (B,out_heads,1,d_v)
    if (scale != 1.0f) {
      emscripten::val scale_const = model_builder.CreateOrGetConstant<float>(input_type, scale);
      opt.set("label", lbl + "_scale");
      o_t = builder.call<emscripten::val>("mul", o_t, scale_const, opt);
    }
    return o_t;
  };

  emscripten::val o = emscripten::val::undefined();  // (B, out_heads, T, d_v)
  if (unroll) {
    emscripten::val o_list = emscripten::val::array();
    for (int64_t t = 0; t < seq_len; ++t) {
      o_list.call<void>("push", run_step(t, S));
    }
    emscripten::val concat_options = emscripten::val::object();
    concat_options.set("label", node.Name() + "_concat_output");
    o = builder.call<emscripten::val>("concat", o_list, static_cast<uint32_t>(2), concat_options);
  } else {
    o = run_step(-1, S);
  }

  // present_state output (B, Hkv, d_k, d_v) — the final recurrent state.
  if (TensorExists(output_defs, 1)) {
    emscripten::val present = S;
    model_builder.AddOperand(output_defs[1]->Name(), std::move(present));
  }

  // (B,out_heads,T,d_v) → (B,T,out_heads,d_v) → (B,T,out_heads*d_v).
  emscripten::val to_btd = emscripten::val::object();
  to_btd.set("label", node.Name() + "_out_transpose");
  to_btd.set("permutation", emscripten::val::array(std::vector<uint32_t>{0, 2, 1, 3}));
  o = builder.call<emscripten::val>("transpose", o, to_btd);
  std::vector<int64_t> o_shape{kDynamicDim, kDynamicDim, out_heads, d_v};
  emscripten::val output = shape_utils::Reshape(model_builder, o, o_shape, {0, 0, out_heads * d_v},
                                                node.Name() + "_out_reshape");
  model_builder.AddOperand(output_defs[0]->Name(), std::move(output));

  return Status::OK();
}

// Operator support related.

bool LinearAttentionOpBuilder::IsOpSupportedImpl(const GraphViewer&, const Node& node,
                                                 const WebnnDeviceType /* device_type */,
                                                 const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& name = node.Name();
  NodeAttrHelper helper(node);

  const std::string update_rule = helper.Get("update_rule", "gated_delta");
  if (update_rule != "linear" && update_rule != "gated" && update_rule != "delta" &&
      update_rule != "gated_delta") {
    LOGS(logger, VERBOSE) << "LinearAttention [" << name << "] unsupported update_rule: " << update_rule;
    return false;
  }

  const int64_t q_num_heads = helper.Get("q_num_heads", static_cast<int64_t>(0));
  const int64_t kv_num_heads = helper.Get("kv_num_heads", static_cast<int64_t>(0));
  if (q_num_heads <= 0 || kv_num_heads <= 0) {
    LOGS(logger, VERBOSE) << "LinearAttention [" << name << "] requires positive q_num_heads/kv_num_heads.";
    return false;
  }
  // Head groups must divide evenly (standard or inverse GQA).
  const bool heads_divisible = (q_num_heads >= kv_num_heads) ? (q_num_heads % kv_num_heads == 0)
                                                             : (kv_num_heads % q_num_heads == 0);
  if (!heads_divisible) {
    LOGS(logger, VERBOSE) << "LinearAttention [" << name << "] q/kv head counts must divide evenly.";
    return false;
  }

  std::vector<int64_t> q_shape;
  if (!GetShape(*input_defs[0], q_shape, logger) || q_shape.size() != 3) {
    LOGS(logger, VERBOSE) << "LinearAttention [" << name << "] query must be rank 3 (B, T, H*D).";
    return false;
  }

  // Note on sequence length: the gated_delta recurrence is sequential and has no loop-free parallel
  // form. The WebNN EP specializes the graph per concrete input shape, so AddToModelBuilderImpl knows
  // the sequence length and unrolls the recurrence into that many fixed steps (a single step for
  // decode, T for prefill). Partitioning happens on the symbolic graph (T is dynamic here), so we do
  // not gate on T; the concrete value is resolved at graph-build time.

  return true;
}

bool LinearAttentionOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
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

  for (const std::string_view decomposed_op_type : decomposed_op_map.at(op_type)) {
    const std::string_view webnn_op_type = GetWebNNOpType(decomposed_op_type);
    const std::string_view webnn_input_name = GetWebNNOpFirstInputName(decomposed_op_type);
    if (!IsDataTypeSupportedByWebNNOp(
            op_type, webnn_op_type, input_types[0], wnn_limits, webnn_input_name, "input", logger)) {
      return false;
    }
  }

  // The recurrence runs on rank-4 (B, H, T, D) operands; ensure matmul accepts that rank.
  return IsRankSupportedByWebNNOp(wnn_limits, "matmul", "a", 4, node.Name(), logger);
}

bool LinearAttentionOpBuilder::HasSupportedOutputsImpl(const Node& node, const emscripten::val& wnn_limits,
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

void CreateLinearAttentionOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<LinearAttentionOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
