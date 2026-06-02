// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include <cstdint>
#include <mutex>

#include <emscripten.h>
#include <emscripten/val.h>

namespace onnxruntime {
namespace webnn {

struct OnnxTensorInfo {
  const int32_t data_type;  // Uses TensorProto::DataType.
  const std::vector<int64_t> shape;
};

struct OnnxTensorData {
  OnnxTensorInfo tensor_info;
  void* buffer{nullptr};
};

class Model {
  friend class ModelBuilder;

 public:
  ~Model();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Model);

  onnxruntime::common::Status Predict(const InlinedHashMap<std::string, OnnxTensorData>& inputs,
                                      const InlinedHashMap<std::string, OnnxTensorData>& outputs);

  // Mutex for exclusive lock to this model object.
  std::mutex& GetMutex() { return mutex_; }

  // Input and output names in the onnx model's order.
  const std::vector<std::string>& GetInputs() const { return inputs_; }
  void SetInputs(std::vector<std::string>&& inputs) { inputs_ = std::move(inputs); }

  const std::vector<std::string>& GetOutputs() const { return outputs_; }
  void SetOutputs(std::vector<std::string>&& outputs) { outputs_ = std::move(outputs); }

  bool UseDispatch() const { return use_dispatch_; }
  bool IsCausalLMEnabled() const { return enable_causal_lm_; }

  const OnnxTensorInfo& GetInputOutputInfo(const std::string& name) const;

  // Call MLGraph.computeShapes() to infer output shapes from concrete input shapes.
  onnxruntime::common::Status ComputeShapes(
      const InlinedHashMap<std::string, std::vector<int64_t>>& input_shapes,
      InlinedHashMap<std::string, std::vector<int64_t>>& output_shapes);

  bool SupportsComputeShapes() const { return supports_compute_shapes_; }

  // Set the mapping between input/output name and ORT kernel context
  // input/output index, at execution time.
  void SetInputMap(InlinedHashMap<std::string, size_t>&& input_map);
  void SetOutputMap(InlinedHashMap<std::string, size_t>&& output_map);

  // Get the ORT kernel context input/output index with given name.
  size_t GetMappedInputIdx(const std::string& name) const;
  size_t GetMappedOutputIdx(const std::string& name) const;

 private:
  onnxruntime::common::Status Dispatch(const InlinedHashMap<std::string, OnnxTensorData>& inputs,
                                       const InlinedHashMap<std::string, OnnxTensorData>& outputs);

  onnxruntime::common::Status Compute(const InlinedHashMap<std::string, OnnxTensorData>& inputs,
                                      const InlinedHashMap<std::string, OnnxTensorData>& outputs);

  emscripten::val wnn_context_ = emscripten::val::object();
  emscripten::val wnn_graph_ = emscripten::val::object();
  const logging::Logger& logger_;

  emscripten::val wnn_inputs_ = emscripten::val::object();
  emscripten::val wnn_outputs_ = emscripten::val::object();

  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;

  InlinedHashMap<std::string, OnnxTensorInfo> input_output_info_;

  InlinedHashMap<std::string, size_t> input_map_;
  InlinedHashMap<std::string, size_t> output_map_;

  // Memoization for ComputeShapes(): the last input shapes seen and the output shapes
  // they produced. computeShapes() output is a pure function of the input shapes (the
  // compiled MLGraph is fixed), so identical input shapes yield identical output shapes.
  // Guarded by mutex_ (held by the caller during Predict/ComputeShapes).
  InlinedHashMap<std::string, std::vector<int64_t>> cached_compute_input_shapes_;
  InlinedHashMap<std::string, std::vector<int64_t>> cached_compute_output_shapes_;
  // Internal tensor IDs used by dispatch() for graph outputs not requested by fetches.
  // This keeps WebNN graph-output binding complete without allocating ORT output tensors.
  InlinedHashMap<std::string, intptr_t> internal_dispatch_output_tensor_ids_;

  std::mutex mutex_;

  bool use_dispatch_;
  bool supports_compute_shapes_;
  bool enable_causal_lm_;

  Model(const emscripten::val& context, const emscripten::val& path, const logging::Logger& logger,
      bool use_dispatch, bool enable_causal_lm);

  void SetInputOutputInfo(InlinedHashMap<std::string, OnnxTensorInfo>&& input_output_info) {
    input_output_info_ = std::move(input_output_info);
  }

  void AllocateInputOutputBuffers();
};

}  // namespace webnn
}  // namespace onnxruntime
