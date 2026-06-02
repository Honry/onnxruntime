// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cctype>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/common.h"
#include "model.h"

namespace onnxruntime {
namespace webnn {

namespace {
bool IsLikelyCausalLMKVTensorName(const std::string& tensor_name) {
  std::string lower = tensor_name;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  const bool has_present_or_past = lower.find("present") != std::string::npos ||
                                   lower.find("past") != std::string::npos;
  const bool has_key_or_value = lower.find("key") != std::string::npos ||
                                lower.find("value") != std::string::npos ||
                                lower.find("kv") != std::string::npos ||
                                lower.find("key_values") != std::string::npos;

  return has_present_or_past && has_key_or_value;
}
}  // namespace

Model::Model(const emscripten::val& context, const emscripten::val& graph, const logging::Logger& logger,
             bool use_dispatch, bool enable_causal_lm)
    : wnn_context_(context),
      wnn_graph_(graph),
      logger_(logger),
      use_dispatch_(use_dispatch),
      enable_causal_lm_(enable_causal_lm) {}

Model::~Model() {
  if (!internal_dispatch_output_tensor_ids_.empty()) {
    auto webnnReleaseTensorId = emscripten::val::module_property("webnnReleaseTensorId");
    for (const auto& [name, tensor_id] : internal_dispatch_output_tensor_ids_) {
      ORT_UNUSED_PARAMETER(name);
      webnnReleaseTensorId(tensor_id);
    }
    internal_dispatch_output_tensor_ids_.clear();
  }
}

Status Model::Predict(const InlinedHashMap<std::string, OnnxTensorData>& inputs,
                      const InlinedHashMap<std::string, OnnxTensorData>& outputs) {
  if (use_dispatch_) {
    return Dispatch(inputs, outputs);
  } else {
    return Compute(inputs, outputs);
  }
}

onnxruntime::common::Status Model::Compute(const InlinedHashMap<std::string, OnnxTensorData>& inputs,
                                           const InlinedHashMap<std::string, OnnxTensorData>& outputs) {
  for (const auto& input : inputs) {
    const std::string& name = input.first;
    const struct OnnxTensorData tensor = input.second;
    auto num_elements = SafeInt<size_t>(Product(tensor.tensor_info.shape));
    emscripten::val view = emscripten::val::undefined();
    switch (tensor.tensor_info.data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      case ONNX_NAMESPACE::TensorProto_DataType_INT4:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT4:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const uint8_t*>(tensor.buffer))};
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const int8_t*>(tensor.buffer))};
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const uint16_t*>(tensor.buffer))};
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const float*>(tensor.buffer))};
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const int32_t*>(tensor.buffer))};
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const int64_t*>(tensor.buffer))};
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const uint32_t*>(tensor.buffer))};
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const uint64_t*>(tensor.buffer))};
        break;
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "The input of graph has unsupported type, name: ",
                               name, " type: ", tensor.tensor_info.data_type);
    }
    // Copy the inputs from Wasm ArrayBuffer to the WebNN inputs ArrayBuffer.
    // As Wasm ArrayBuffer is not detachable.
    wnn_inputs_[name].call<void>("set", view);
  }

  InlinedHashMap<std::string, emscripten::val> output_views;

  for (const auto& output : outputs) {
    const std::string& name = output.first;
    const struct OnnxTensorData tensor = output.second;
    auto num_elements = SafeInt<size_t>(Product(tensor.tensor_info.shape));
    emscripten::val view = emscripten::val::undefined();
    switch (tensor.tensor_info.data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      case ONNX_NAMESPACE::TensorProto_DataType_INT4:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT4:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const uint8_t*>(tensor.buffer))};
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const int8_t*>(tensor.buffer))};
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const uint16_t*>(tensor.buffer))};
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const float*>(tensor.buffer))};
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const int32_t*>(tensor.buffer))};
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const int64_t*>(tensor.buffer))};
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const uint32_t*>(tensor.buffer))};
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
        view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                             static_cast<const uint64_t*>(tensor.buffer))};
        break;
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "The output of graph has unsupported type, name: ",
                               name, " type: ", tensor.tensor_info.data_type);
    }

    output_views.insert({name, view});
  }
  emscripten::val results = wnn_context_.call<emscripten::val>(
                                            "compute", wnn_graph_, wnn_inputs_, wnn_outputs_)
                                .await();

  // Copy the outputs from pre-allocated ArrayBuffers back to the Wasm ArrayBuffer.
  for (const auto& output : outputs) {
    const std::string& name = output.first;
    emscripten::val view = output_views.at(name);
    view.call<void>("set", results["outputs"][name]);
  }
  // WebNN compute() method would return the input and output buffers via the promise
  // resolution. Reuse the buffers to avoid additional allocation.
  wnn_inputs_ = results["inputs"];
  wnn_outputs_ = results["outputs"];

  return Status::OK();
}

onnxruntime::common::Status Model::Dispatch(const InlinedHashMap<std::string, OnnxTensorData>& inputs,
                                            const InlinedHashMap<std::string, OnnxTensorData>& outputs) {
  auto object = emscripten::val::global("Object");
  InlinedHashSet<std::string> graph_inputs;
  InlinedHashSet<std::string> graph_outputs;

  {
    emscripten::val graph_input_descs = wnn_graph_.call<emscripten::val>("inputs");
    emscripten::val graph_input_names = object.call<emscripten::val>("keys", graph_input_descs);
    const uint32_t graph_input_count = graph_input_names["length"].as<uint32_t>();
    for (uint32_t i = 0; i < graph_input_count; ++i) {
      graph_inputs.insert(graph_input_names[i].as<std::string>());
    }
  }

  {
    emscripten::val graph_output_descs = wnn_graph_.call<emscripten::val>("outputs");
    emscripten::val graph_output_names = object.call<emscripten::val>("keys", graph_output_descs);
    const uint32_t graph_output_count = graph_output_names["length"].as<uint32_t>();
    for (uint32_t i = 0; i < graph_output_count; ++i) {
      graph_outputs.insert(graph_output_names[i].as<std::string>());
    }
  }

  for (const auto& graph_input : graph_inputs) {
    if (inputs.find(graph_input) == inputs.end()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Missing required WebNN dispatch input: ", graph_input);
    }
  }

  for (const auto& graph_output : graph_outputs) {
    if (outputs.find(graph_output) == outputs.end()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Missing required WebNN dispatch output metadata: ", graph_output);
    }
  }

  // Keep previously bound inputs across dispatches to enable CausalLM KV-input reuse.
  // Non-KV inputs are rebound below every run.
  wnn_outputs_ = emscripten::val::object();

  auto webnnEnsureTensor = emscripten::val::module_property("webnnEnsureTensor");
  auto webnnReserveTensorId = emscripten::val::module_property("webnnReserveTensorId");
  auto promises = emscripten::val::array();
  std::vector<std::string> bound_input_names;
  std::vector<std::string> bound_output_names;
  size_t provided_output_bindings = 0;
  size_t internal_output_bindings = 0;
  size_t reserved_internal_tensor_ids = 0;
  size_t reused_internal_tensor_ids = 0;
  size_t reused_causallm_kv_input_bindings = 0;
  size_t skipped_causallm_kv_outputs = 0;
  bool trace = emscripten::val::module_property("webnnEnableTraceEvent").as<bool>();
  emscripten::val console = emscripten::val::global("console");
  if (trace) {
    console.call<void>("time", emscripten::val("ORT::Dispatch::webnnEnsureTensor"));
  }
  for (const auto& [name, tensor] : inputs) {
    if (graph_inputs.find(name) == graph_inputs.end()) {
      // LOGS(logger_, VERBOSE) << "Skip extra dispatch input not used by WebNN graph: " << name;
      continue;
    }

    // Hard workaround: for CausalLM, if KV-like input has already been bound in a previous
    // dispatch, skip ensure/bind to avoid repeated KV input tensor handling.
    if (enable_causal_lm_ && IsLikelyCausalLMKVTensorName(name) &&
        wnn_inputs_.call<emscripten::val>("hasOwnProperty", name).as<bool>()) {
      ++reused_causallm_kv_input_bindings;
      continue;
    }

    emscripten::val shape = emscripten::val::array();
    for (const auto& dim : tensor.tensor_info.shape) {
      uint32_t dim_val = SafeInt<uint32_t>(dim);
      shape.call<void>("push", dim_val);
    }
    auto ml_tensor = webnnEnsureTensor(emscripten::val::undefined(), reinterpret_cast<intptr_t>(tensor.buffer), tensor.tensor_info.data_type, shape, true);
    promises.call<void>("push", ml_tensor);
    bound_input_names.push_back(name);
  }
  for (const auto& [name, tensor] : outputs) {
    if (graph_outputs.find(name) == graph_outputs.end()) {
      // LOGS(logger_, VERBOSE) << "Skip extra dispatch output not used by WebNN graph: " << name;
      continue;
    }

    // Hard workaround: for CausalLM decode with fetches-only logits, skip binding unrequested
    // KV-like outputs so dispatch does not create additional KV output tensors.
    if (enable_causal_lm_ && tensor.buffer == nullptr && IsLikelyCausalLMKVTensorName(name)) {
      ++skipped_causallm_kv_outputs;
      // LOGS(logger_, VERBOSE) << "[WebNN][Workaround] Skip CausalLM KV dispatch output binding: " << name;
      continue;
    }

    emscripten::val shape = emscripten::val::array();
    for (const auto& dim : tensor.tensor_info.shape) {
      uint32_t dim_val = SafeInt<uint32_t>(dim);
      shape.call<void>("push", dim_val);
    }
    intptr_t tensor_id = 0;
    if (tensor.buffer != nullptr) {
      tensor_id = reinterpret_cast<intptr_t>(tensor.buffer);
      ++provided_output_bindings;
    } else {
      ++internal_output_bindings;
      auto it = internal_dispatch_output_tensor_ids_.find(name);
      if (it == internal_dispatch_output_tensor_ids_.end()) {
        tensor_id = webnnReserveTensorId().as<intptr_t>();
        internal_dispatch_output_tensor_ids_.emplace(name, tensor_id);
        ++reserved_internal_tensor_ids;
      } else {
        tensor_id = it->second;
        ++reused_internal_tensor_ids;
      }
    }

    auto ml_tensor = webnnEnsureTensor(emscripten::val::undefined(), tensor_id, tensor.tensor_info.data_type, shape, false);
    promises.call<void>("push", ml_tensor);
    bound_output_names.push_back(name);
  }
  if (trace) {
    console.call<void>("timeEnd", emscripten::val("ORT::Dispatch::webnnEnsureTensor"));
  }
  auto ml_tensors = emscripten::val::global("Promise").call<emscripten::val>("all", promises).await();
  for (const auto& name : bound_input_names) {
    wnn_inputs_.set(name, ml_tensors.call<emscripten::val>("shift"));
  }
  for (const auto& name : bound_output_names) {
    wnn_outputs_.set(name, ml_tensors.call<emscripten::val>("shift"));
  }

  if (trace) {
    LOGS(logger_, INFO) << "[WebNN][Trace] DispatchBindings"
                        << " graph_inputs=" << graph_inputs.size()
                        << " graph_outputs=" << graph_outputs.size()
                        << " bound_inputs=" << bound_input_names.size()
                        << " bound_outputs=" << bound_output_names.size()
                        << " reused_causallm_kv_input_bindings=" << reused_causallm_kv_input_bindings
                        << " provided_output_bindings=" << provided_output_bindings
                        << " internal_output_bindings=" << internal_output_bindings
                        << " reserved_internal_tensor_ids=" << reserved_internal_tensor_ids
                        << " reused_internal_tensor_ids=" << reused_internal_tensor_ids
                        << " skipped_causallm_kv_outputs=" << skipped_causallm_kv_outputs;
  }

  if (trace) {
    console.call<void>("time", emscripten::val("ORT::Dispatch::webnn::dispatch"));
  }
  wnn_context_.call<void>("dispatch", wnn_graph_, wnn_inputs_, wnn_outputs_);
  if (trace) {
    console.call<void>("timeEnd", emscripten::val("ORT::Dispatch::webnn::dispatch"));
  }
  return Status::OK();
}

const OnnxTensorInfo& Model::GetInputOutputInfo(const std::string& name) const {
  return input_output_info_.at(name);
}

void Model::SetInputMap(InlinedHashMap<std::string, size_t>&& input_map) {
  input_map_ = std::move(input_map);
}

void Model::SetOutputMap(InlinedHashMap<std::string, size_t>&& output_map) {
  output_map_ = std::move(output_map);
}

// Pre-allocate the input and output buffers for the WebNN graph.
void Model::AllocateInputOutputBuffers() {
  // We don't need to allocate JS ArrayBuffers if the WebNN API supports MLTensor.
  if (use_dispatch_) {
    return;
  }
  for (const auto& input : inputs_) {
    const auto& input_info = input_output_info_.at(input);
    const auto input_shape = input_info.shape;
    const int32_t num_elements = SafeInt<int32_t>(Product(input_shape));
    const auto data_type = input_info.data_type;
    switch (data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      case ONNX_NAMESPACE::TensorProto_DataType_INT4:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT4:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        wnn_inputs_.set(input, emscripten::val::global("Uint8Array").new_(num_elements));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:
        wnn_inputs_.set(input, emscripten::val::global("Int8Array").new_(num_elements));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        wnn_inputs_.set(input, emscripten::val::global("Uint16Array").new_(num_elements));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        wnn_inputs_.set(input, emscripten::val::global("Float32Array").new_(num_elements));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        wnn_inputs_.set(input, emscripten::val::global("Int32Array").new_(num_elements));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        wnn_inputs_.set(input, emscripten::val::global("BigInt64Array").new_(num_elements));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
        wnn_inputs_.set(input, emscripten::val::global("Uint32Array").new_(num_elements));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
        wnn_inputs_.set(input, emscripten::val::global("BigUint64Array").new_(num_elements));
        break;
      default:
        break;
    }
  }
  for (const auto& output : outputs_) {
    const auto& output_info = input_output_info_.at(output);
    const auto output_shape = output_info.shape;
    const int32_t num_elements = SafeInt<int32_t>(Product(output_shape));
    const auto data_type = output_info.data_type;
    switch (data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      case ONNX_NAMESPACE::TensorProto_DataType_INT4:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT4:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        wnn_outputs_.set(output, emscripten::val::global("Uint8Array").new_(num_elements));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:
        wnn_outputs_.set(output, emscripten::val::global("Int8Array").new_(num_elements));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        wnn_outputs_.set(output, emscripten::val::global("Uint16Array").new_(num_elements));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        wnn_outputs_.set(output, emscripten::val::global("Float32Array").new_(num_elements));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        wnn_outputs_.set(output, emscripten::val::global("Int32Array").new_(num_elements));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        wnn_outputs_.set(output, emscripten::val::global("BigInt64Array").new_(num_elements));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
        wnn_outputs_.set(output, emscripten::val::global("Uint32Array").new_(num_elements));
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
        wnn_outputs_.set(output, emscripten::val::global("BigUint64Array").new_(num_elements));
        break;
      default:
        break;
    }
  }
}

size_t Model::GetMappedInputIdx(const std::string& name) const {
  return input_map_.at(name);
}

size_t Model::GetMappedOutputIdx(const std::string& name) const {
  return output_map_.at(name);
}

}  // namespace webnn
}  // namespace onnxruntime
