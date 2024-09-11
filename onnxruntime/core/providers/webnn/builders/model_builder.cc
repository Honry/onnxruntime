// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>

#include "model_builder.h"
#include "model.h"
#include "helper.h"
#include "op_builder_factory.h"

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensor_external_data_info.h"
#include <utility>

namespace onnxruntime {
namespace webnn {

ModelBuilder::ModelBuilder(const GraphViewer& graph_viewer, const logging::Logger& logger,
                           const emscripten::val& context, const DataLayout preferred_layout,
                           const WebnnDeviceType wnn_device_type)
    : graph_viewer_(graph_viewer),
      logger_(logger),
      wnn_context_(context),
      preferred_layout_(preferred_layout),
      wnn_device_type_(wnn_device_type) {
  // Create WebNN MLGraphBuilder for each ModelBuilder, because MLGraphBuilder.build()
  // is only allowed to be called once.
  wnn_builder_ = emscripten::val::global("MLGraphBuilder").new_(context);
  if (!wnn_builder_.as<bool>()) {
    ORT_THROW("Failed to create WebNN builder.");
  }
}

Status ModelBuilder::Initialize() {
  PreprocessInitializers();
  ORT_RETURN_IF_ERROR(RegisterInitializers());
  ORT_RETURN_IF_ERROR(RegisterModelInputs());
  ORT_RETURN_IF_ERROR(AddOperations());
  ORT_RETURN_IF_ERROR(RegisterModelOutputs());

  return Status::OK();
}

InitializedTensorSet ModelBuilder::GetInitializerTensors() {
  if (graph_viewer_.IsSubgraph()) {
    auto all_initializers = CollectAllInitializedTensors(graph_viewer_);
    const auto sub_graph_id = graph_viewer_.GetFilterInfo();
    const auto subgraph_initializer_names = sub_graph_id->GetMetaDef()->constant_initializers;
    InitializedTensorSet subgraph_initializers;

    for (const auto& name : subgraph_initializer_names) {
      auto it = all_initializers.find(name);
      if (it != all_initializers.end()) {
        subgraph_initializers.insert(*it);
      }
    }
    return subgraph_initializers;
  } else {
    return graph_viewer_.GetAllInitializedTensors();
  }
}

/* static */ const IOpBuilder* ModelBuilder::GetOpBuilder(const Node& node) {
  const auto& op_builders = GetOpBuilders();
  const auto it = op_builders.find(node.OpType());
  if (it != op_builders.cend())
    return it->second;

  return nullptr;
}

void ModelBuilder::PreprocessInitializers() {
  const auto& node_indices = graph_viewer_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_.GetNode(node_indices[i]));
    if (const auto* op_builder = GetOpBuilder(*node)) {
      op_builder->AddInitializersToSkip(*this, *node);
    }
  }
}

Status ModelBuilder::RegisterInitializers() {
  emscripten::val console = emscripten::val::global("console");
  console.call<void>("log", emscripten::val("model_builder.cc: RegisterInitializers"));
  for (const auto& pair : GetInitializerTensors()) {
    auto& tensor = *pair.second;
    const auto& name = tensor.name();
    // Optional tensors can be indicated by an empty name, just ignore it.
    if (name.empty() || Contains(skipped_initializers_, name))
      continue;

    const auto& shape = tensor.dims();
    std::vector<int32_t> dims;
    // When the shape is empty, it is scalar initializer that dims = {};
    std::transform(shape.cbegin(), shape.cend(),
                   std::back_inserter(dims),
                   [](int64_t dim) -> int32_t { return SafeInt<int32_t>(dim); });

    emscripten::val desc = emscripten::val::object();
    desc.set("dimensions", emscripten::val::array(dims));
    auto data_type = tensor.data_type();
    emscripten::val operand = emscripten::val::object();
    if (IsSupportedDataType(data_type, webnn_supported_data_types)) {
      ORT_RETURN_IF_NOT(SetWebnnDataType(desc, data_type), "Unsupported data type");
      auto num_elements = SafeInt<size_t>(Product(shape));
      emscripten::val view = emscripten::val::undefined();
      std::byte* tensor_ptr = nullptr;
      if (tensor.has_raw_data()) {
        tensor_ptr = reinterpret_cast<std::byte*>(const_cast<char*>(tensor.raw_data().c_str()));
      } else if (utils::HasExternalData(tensor)) {
        std::basic_string<ORTCHAR_T> external_file_path;
        onnxruntime::FileOffsetType data_offset;
        SafeInt<size_t> tensor_byte_size;
        // if (!model_path.empty()) {
        //  ORT_RETURN_IF_ERROR(GetDirNameFromFilePath(model_path, tensor_proto_dir));
        // }
        console.call<void>("log", emscripten::val("graph_viewer_.ModelPath: "));
        console.call<void>("log", emscripten::val(graph_viewer_.ModelPath().c_str()));
        ORT_RETURN_IF_ERROR(
            utils::GetExternalDataInfo(tensor, graph_viewer_.ModelPath(), external_file_path, data_offset, tensor_byte_size));
        // std::string external_file_absolute_path = std::filesystem::absolute(std::filesystem::path(external_file_path)).string();

        // auto err_code = EM_ASM_INT(({
        //                              // If available, "Module.MountedFiles" is a Map for all preloaded files.
        //                              if (typeof Module == 'undefined' || !Module.MountedFiles) {
        //                                return 0;  // "Module.MountedFiles" is not available.
        //                              }
        //                              let fileName = UTF8ToString($0 >>> 0);
        //                              console.log("model_builder.cc: file name: ", fileName);
        //                              if (fileName.startsWith('./')) {
        //                                fileName = fileName.substring(2);
        //                              }
        //                              const fileData = Module.MountedFiles.get(fileName);
        //                              if (!fileData) {
        //                                return 0;  // File not found in preloaded files.
        //                              }
        //                              const offset = $1 >>> 0;
        //                              const length = $2 >>> 0;
        //                              const constantId = Module.jsepReserveTensorId(1);
        //                              if (offset + length > fileData.byteLength) {
        //                                return 0;  // Out of bounds.
        //                              }

        //                              try {
        //                               //  Module.jsepSetCurrentBuilder();
        //                               //  const wnn_builder = Module['currentBuilder'];
        //                               //  console.log("model_builder.cc: wnn_builder 1: ", wnn_builder);
        //                                const data = fileData.subarray(offset, offset + length);
        //                                console.log("data: ", data);
        //                                console.log("dataIdOrBuffer: ", constantId);
        //                                // Load external data to WebNN memory.
        //                                Module.jsepUploadExternalBuffer(constantId, data);
        //                                return constantId;
        //                              } catch {
        //                                //  return 4;
        //                                console.log("model_builder.cc throws");
        //                                return 0;
        //                              }
        //                            }),
        //                            external_file_path.c_str(),
        //                            static_cast<int32_t>(data_offset),
        //                            static_cast<int32_t>(tensor_byte_size));
        auto jsepRegisterMLConstant = emscripten::val::module_property("jsepRegisterMLConstant");
        // operand = jsepRegisterMLConstant(err_code, wnn_builder_, desc);
        operand = jsepRegisterMLConstant(emscripten::val(external_file_path),
                                         static_cast<int32_t>(data_offset),
                                         static_cast<int32_t>(tensor_byte_size),
                                         wnn_builder_,
                                         desc).await();

        // Load external data from file.
        console.call<void>("log", emscripten::val("unpacked_external_tensor: " + name));
        console.call<void>("log", emscripten::val("unpacked_external_tensor done "), operand);
      } else {
        // Store temporary unpacked_tensor.
        unpacked_tensors_.push_back({});
        std::vector<uint8_t>& unpacked_tensor = unpacked_tensors_.back();
        ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(tensor, unpacked_tensor));
        tensor_ptr = reinterpret_cast<std::byte*>(unpacked_tensor.data());
      }
      if (!utils::HasExternalData(tensor)) {
        switch (data_type) {
          case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
          case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
            view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                                 reinterpret_cast<uint8_t*>(tensor_ptr))};
            break;
          case ONNX_NAMESPACE::TensorProto_DataType_INT8:
            view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                                 reinterpret_cast<int8_t*>(tensor_ptr))};
            break;
          case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
            view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                                 reinterpret_cast<uint16_t*>(tensor_ptr))};
            break;
          case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
            view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                                 reinterpret_cast<float*>(tensor_ptr))};
            break;
          case ONNX_NAMESPACE::TensorProto_DataType_INT32:
            view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                                 reinterpret_cast<int32_t*>(tensor_ptr))};
            break;
          case ONNX_NAMESPACE::TensorProto_DataType_INT64:
            view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                                 reinterpret_cast<int64_t*>(tensor_ptr))};
            break;
          case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
            view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                                 reinterpret_cast<uint32_t*>(tensor_ptr))};
            break;
          case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
            view = emscripten::val{emscripten::typed_memory_view(num_elements,
                                                                 reinterpret_cast<uint64_t*>(tensor_ptr))};
            break;
          default:
            break;
        }

        // Wasm memory grow will cause all array buffers reallocation, which will be treated as detached
        // buffers in JS side. Simply create a copy to fix it.
        operand = wnn_builder_.call<emscripten::val>("constant", desc, view.call<emscripten::val>("slice"));
      }
    } else {
      // TODO: support other type.
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The initializer of graph has unsupported type, name: ",
                             tensor.name(), " type: ", data_type);
    }
    wnn_operands_.insert(std::make_pair(name, operand));
  }

  return Status::OK();
}

Status ModelBuilder::RegisterModelInputOutput(const NodeArg& node_arg, bool is_input) {
  const auto& name = node_arg.Name();
  const std::string input_output_type = is_input ? "input" : "output";

  if (is_input) {
    // Input should not be an initializer.
    if (Contains(GetInitializerTensors(), name))
      return Status::OK();

    // This input will not be used.
    if (Contains(skipped_inputs_, name))
      return Status::OK();
  }

  std::vector<int32_t> dims;
  {  // input_output shape.
    const auto* shape_proto = node_arg.Shape();
    ORT_RETURN_IF(shape_proto == nullptr,
                  "shape_proto cannot be null for ", input_output_type, ": ", name);
    const auto& shape = shape_proto->dim();
    if (!shape.empty()) {
      dims.reserve(shape.size());
      for (const auto& dim : shape) {
        // dim_param free dimensions should have already been excluded by IsInputSupported().
        assert(dim.has_dim_value());
        dims.push_back(SafeInt<int32_t>(dim.dim_value()));
      }
    }
  }

  emscripten::val desc = emscripten::val::object();

  desc.set("dimensions", emscripten::val::array(dims));

  int32_t data_type;
  {  // type
    const auto* type_proto = node_arg.TypeAsProto();
    if (!type_proto || !type_proto->tensor_type().has_elem_type()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The ", input_output_type, " of graph doesn't have elem_type: ", name);
    }

    data_type = type_proto->tensor_type().elem_type();
    ORT_RETURN_IF_NOT(SetWebnnDataType(desc, data_type), "Unsupported data type");
  }

  if (is_input) {
    wnn_operands_.insert(std::make_pair(name, wnn_builder_.call<emscripten::val>("input", name, desc)));
    input_names_.push_back(name);
  } else {
    output_names_.push_back(name);
  }

  std::vector<int64_t> shape;
  std::transform(dims.cbegin(), dims.cend(),
                 std::back_inserter(shape),
                 [](int32_t dim) -> int64_t { return SafeInt<int64_t>(dim); });
  input_output_info_.emplace(name, OnnxTensorInfo{data_type, shape});

  return Status::OK();
}

Status ModelBuilder::RegisterModelInputs() {
  for (const auto* node_arg : graph_viewer_.GetInputs()) {
    ORT_RETURN_IF_ERROR(RegisterModelInputOutput(*node_arg, true /* is_input */));
  }

  return Status::OK();
}

Status ModelBuilder::AddOperations() {
  const auto& node_indices = graph_viewer_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_.GetNode(node_indices[i]));
    if (const auto* op_builder = GetOpBuilder(*node)) {
      ORT_RETURN_IF_ERROR(op_builder->AddToModelBuilder(*this, *node, logger_));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Node [", node->Name(), "], type [", node->OpType(), "] is not supported");
    }
  }

  return Status::OK();
}

Status ModelBuilder::AddOperandFromPersistMemoryBuffer(
    const std::string& name, const void* buffer, const size_t size,
    const std::vector<uint32_t> shape, const int32_t data_type) {
  auto persist_buffer = std::make_unique<uint8_t[]>(size);
  uint8_t* dest = persist_buffer.get();
  memcpy(dest, buffer, size);
  emscripten::val view = emscripten::val::undefined();
  emscripten::val desc = emscripten::val::object();
  ORT_RETURN_IF_NOT(SetWebnnDataType(desc, data_type), "Unsupported data type");
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      view = emscripten::val{emscripten::typed_memory_view(size / sizeof(uint8_t),
                                                           reinterpret_cast<const uint8_t*>(dest))};
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      view = emscripten::val{emscripten::typed_memory_view(size / sizeof(int8_t),
                                                           reinterpret_cast<const int8_t*>(dest))};
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      view = emscripten::val{emscripten::typed_memory_view(size / sizeof(uint16_t),
                                                           reinterpret_cast<const uint16_t*>(dest))};
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      view = emscripten::val{emscripten::typed_memory_view(size / sizeof(float),
                                                           reinterpret_cast<const float*>(dest))};
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      view = emscripten::val{emscripten::typed_memory_view(size / sizeof(int32_t),
                                                           reinterpret_cast<const int32_t*>(dest))};
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      view = emscripten::val{emscripten::typed_memory_view(size / sizeof(int64_t),
                                                           reinterpret_cast<const int64_t*>(dest))};
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      view = emscripten::val{emscripten::typed_memory_view(size / sizeof(uint32_t),
                                                           reinterpret_cast<const uint32_t*>(dest))};
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      view = emscripten::val{emscripten::typed_memory_view(size / sizeof(uint64_t),
                                                           reinterpret_cast<const uint64_t*>(dest))};
      break;
    default:
      break;
  }

  desc.set("dimensions", emscripten::val::array(shape));
  emscripten::val operand = emscripten::val::object();
  // Wasm memory grow will cause all array buffers reallocation, which will be treated as detached
  // buffers in JS side. Simply create a copy to fix it.
  operand = wnn_builder_.call<emscripten::val>("constant", desc, view.call<emscripten::val>("slice"));

  AddOperand(name, operand);
  mem_persist_buffers_.push_back(std::move(persist_buffer));
  return Status::OK();
}

Status ModelBuilder::RegisterModelOutputs() {
  for (const auto* node_arg : graph_viewer_.GetOutputs()) {
    ORT_RETURN_IF_ERROR(RegisterModelInputOutput(*node_arg, false /* is_input */));
  }

  return Status::OK();
}

Status ModelBuilder::Compile(std::unique_ptr<Model>& model) {
  ORT_RETURN_IF_ERROR(Initialize());
  emscripten::val named_operands = emscripten::val::object();
  for (auto& name : output_names_) {
    named_operands.set(name, wnn_operands_.at(name));
  }

  emscripten::val wnn_graph = wnn_builder_.call<emscripten::val>("build", named_operands).await();
  if (!wnn_graph.as<bool>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to build WebNN graph.");
  }
  // Explicitly release the WebNN builder to free memory.
  wnn_builder_ = emscripten::val::undefined();
  model.reset(new Model(std::move(wnn_context_), std::move(wnn_graph), logger_, IsMLTensorSupported()));
  model->SetInputs(std::move(input_names_));
  model->SetOutputs(std::move(output_names_));
  model->SetInputOutputInfo(std::move(input_output_info_));
  // Wasm heap is not transferrable, we have to pre-allocate the MLNamedArrayBufferViews
  // for inputs and outputs because they will be transferred after compute() done.
  // https://webmachinelearning.github.io/webnn/#api-mlcontext-async-execution
  model->AllocateInputOutputBuffers();
  return Status::OK();
}

void ModelBuilder::AddOperand(const std::string& name, const emscripten::val& operand) {
  wnn_operands_.insert(std::make_pair(name, operand));
}

// Get the zero scalar constant.
// Workaround for builer.constant(value, type) method since it has not been implemented now.
// https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-constant-value-type
// BTW, the spec is discussing if the builer.constant(value, type) should be dropped at
// https://github.com/webmachinelearning/webnn/issues/475. Fix me according to the spec decision.
const emscripten::val& ModelBuilder::GetZeroConstant(const std::string& data_type) {
  std::string name = "webnn_zero_constant_" + data_type;
  // If the operand does not exist, create it.
  if (wnn_operands_.find(name) == wnn_operands_.end()) {
    emscripten::val desc = emscripten::val::object();
    emscripten::val dims = emscripten::val::array();
    desc.set("dimensions", dims);
    emscripten::val zero_buffer = emscripten::val::undefined();
    if (data_type == "uint8") {
      if (!SetWebnnDataType(desc, ONNX_NAMESPACE::TensorProto_DataType_UINT8)) {
        ORT_THROW("Unsupported data type: " + data_type);
      }
      zero_buffer = emscripten::val::global("Uint8Array").new_(1);
    } else if (data_type == "float32") {
      if (!SetWebnnDataType(desc, ONNX_NAMESPACE::TensorProto_DataType_FLOAT)) {
        ORT_THROW("Unsupported data type: " + data_type);
      }
      zero_buffer = emscripten::val::global("Float32Array").new_(1);
    } else {
      ORT_THROW("Unsupported data type: " + data_type);
    }
    emscripten::val zero_constant = wnn_builder_.call<emscripten::val>("constant", desc, zero_buffer);
    wnn_operands_.insert(std::make_pair(name, zero_constant));
  }
  return wnn_operands_.at(name);
}

void ModelBuilder::AddInitializerToSkip(const std::string& tensor_name) {
  skipped_initializers_.insert(tensor_name);
}

void ModelBuilder::AddInputToSkip(const std::string& input_name) {
  skipped_inputs_.insert(input_name);
}

std::string ModelBuilder::GetUniqueName(const std::string& base_name) {
  std::string unique_name;
  do {
    std::ostringstream os;
    os << base_name << "_token_" << name_token_++;
    unique_name = os.str();
  } while (Contains(unique_names_, unique_name));

  return unique_name;
}

}  // namespace webnn
}  // namespace onnxruntime
