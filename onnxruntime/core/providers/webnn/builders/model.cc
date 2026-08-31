// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <vector>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/common.h"
#include "helper.h"
#include "model.h"

namespace onnxruntime {
namespace webnn {

Model::Model(const emscripten::val& context, const emscripten::val& graph, const logging::Logger& logger, bool use_dispatch)
    : wnn_context_(context),
      wnn_graph_(graph),
      logger_(logger),
      use_dispatch_(use_dispatch) {}

Model::~Model() {}

Status Model::ComputeShapes(
    const InlinedHashMap<std::string, std::vector<int64_t>>& input_shapes,
    const InlinedHashMap<std::string, std::vector<int64_t>>*& output_shapes) {
  if (!IsDynamicShapeSupported()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "MLGraph.computeShapes() is not available");
  }

  // Fast path: if the input shapes are identical to the last successful call, reuse the
  // cached output shapes and skip the JS computeShapes() round-trip. This is a pure-function
  // memoization keyed on input shapes, valid because the compiled MLGraph never changes.
  // Common in LLM decode loops that repeat the same input shapes across many runs.
  if (!cached_compute_input_shapes_.empty() &&
      cached_compute_input_shapes_.size() == input_shapes.size()) {
    bool same = true;
    for (const auto& [name, shape] : input_shapes) {
      auto it = cached_compute_input_shapes_.find(name);
      if (it == cached_compute_input_shapes_.end() || it->second != shape) {
        same = false;
        break;
      }
    }
    if (same) {
      output_shapes = &cached_compute_output_shapes_;
      return Status::OK();
    }
  }

  emscripten::val js_input_shapes = emscripten::val::object();
  for (const auto& [name, shape] : input_shapes) {
    emscripten::val js_shape = emscripten::val::array();
    for (const auto& dim : shape) {
      js_shape.call<void>("push", SafeInt<uint32_t>(dim).Ref());
    }
    js_input_shapes.set(name, js_shape);
  }

  // computeShapes() throws if the shapes cannot be inferred, e.g. when they depend on input values.
  emscripten::val js_output_shapes;
  ORT_RETURN_IF_ERROR(TryCallJs(wnn_graph_, "computeShapes", js_output_shapes, js_input_shapes));

  emscripten::val keys = emscripten::val::global("Object").call<emscripten::val>("keys", js_output_shapes);
  const size_t num_outputs = keys["length"].as<size_t>();
  InlinedHashMap<std::string, std::vector<int64_t>> computed_output_shapes;
  computed_output_shapes.reserve(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    std::string output_name = keys[i].as<std::string>();
    // WebNN dims are uint32. Don't read them as int64_t: that maps to a BigInt64Array (WASM_BIGINT), and
    // copying a JS array of Numbers into it throws a TypeError.
    const auto dims = emscripten::convertJSArrayToNumberVector<uint32_t>(js_output_shapes[output_name]);
    computed_output_shapes.emplace(output_name, std::vector<int64_t>(dims.begin(), dims.end()));
  }

  // Populate the memo only on the success path, so a failed call never poisons the cache.
  cached_compute_input_shapes_ = input_shapes;
  cached_compute_output_shapes_ = std::move(computed_output_shapes);
  output_shapes = &cached_compute_output_shapes_;

  return Status::OK();
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
  emscripten::val results;
  ORT_RETURN_IF_ERROR(TryCallJsAsync(wnn_context_, "compute", results, wnn_graph_, wnn_inputs_, wnn_outputs_));

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
  auto webnnEnsureTensor = emscripten::val::module_property("webnnEnsureTensor");
  auto promises = emscripten::val::array();
  bool trace = emscripten::val::module_property("webnnEnableTraceEvent").as<bool>();
  emscripten::val console = emscripten::val::global("console");
  if (trace) {
    console.call<void>("time", emscripten::val("ORT::Dispatch::webnnEnsureTensor"));
  }
  for (const auto& [_, tensor] : inputs) {
    emscripten::val shape = emscripten::val::array();
    for (const auto& dim : tensor.tensor_info.shape) {
      uint32_t dim_val = SafeInt<uint32_t>(dim);
      shape.call<void>("push", dim_val);
    }
    auto ml_tensor = webnnEnsureTensor(emscripten::val::undefined(), reinterpret_cast<intptr_t>(tensor.buffer), tensor.tensor_info.data_type, shape, true);
    promises.call<void>("push", ml_tensor);
  }
  for (const auto& [_, tensor] : outputs) {
    emscripten::val shape = emscripten::val::array();
    for (const auto& dim : tensor.tensor_info.shape) {
      uint32_t dim_val = SafeInt<uint32_t>(dim);
      shape.call<void>("push", dim_val);
    }
    auto ml_tensor = webnnEnsureTensor(emscripten::val::undefined(), reinterpret_cast<intptr_t>(tensor.buffer), tensor.tensor_info.data_type, shape, false);
    promises.call<void>("push", ml_tensor);
  }
  if (trace) {
    console.call<void>("timeEnd", emscripten::val("ORT::Dispatch::webnnEnsureTensor"));
  }
  auto ml_tensors = emscripten::val::global("Promise").call<emscripten::val>("all", promises).await();
  for (const auto& [name, _] : inputs) {
    wnn_inputs_.set(name, ml_tensors.call<emscripten::val>("shift"));
  }
  for (const auto& [name, _] : outputs) {
    wnn_outputs_.set(name, ml_tensors.call<emscripten::val>("shift"));
  }
  emscripten::val dispatch_result;
  ORT_RETURN_IF_ERROR(TryCallJs(wnn_context_, "dispatch", dispatch_result, wnn_graph_, wnn_inputs_, wnn_outputs_));

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
