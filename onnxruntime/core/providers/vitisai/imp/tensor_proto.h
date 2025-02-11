// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "vaip/my_ort.h"
#include "vaip/vaip_gsl.h"
#include "vaip/dll_safe.h"

namespace vaip {
gsl::span<const char> tensor_proto_as_raw(const onnxruntime::Graph& graph, const ONNX_NAMESPACE::TensorProto& tensor);
vaip_core::DllSafe<std::vector<int64_t>> tensor_proto_get_shape(const ONNX_NAMESPACE::TensorProto& tensor);
const std::string& tensor_proto_get_name(const ONNX_NAMESPACE::TensorProto& tensor);
ONNX_NAMESPACE::TensorProto* tensor_proto_new_i4(const std::string& name, const std::vector<int64_t>& shape,
                                                 const std::vector<int8_t>& data);
ONNX_NAMESPACE::TensorProto* tensor_proto_new_u4(const std::string& name, const std::vector<int64_t>& shape,
                                                 const std::vector<uint8_t>& data);
ONNX_NAMESPACE::TensorProto* tensor_proto_new_i8(const std::string& name, const std::vector<int64_t>& shape,
                                                 const std::vector<int8_t>& data);
ONNX_NAMESPACE::TensorProto* tensor_proto_new_u8(const std::string& name, const std::vector<int64_t>& shape,
                                                 const std::vector<uint8_t>& data);
ONNX_NAMESPACE::TensorProto* tensor_proto_new_i16(const std::string& name, const std::vector<int64_t>& shape,
                                                  const std::vector<int16_t>& data);
ONNX_NAMESPACE::TensorProto* tensor_proto_new_u16(const std::string& name, const std::vector<int64_t>& shape,
                                                  const std::vector<uint16_t>& data);
ONNX_NAMESPACE::TensorProto* tensor_proto_new_u32(const std::string& name, const std::vector<int64_t>& shape,
                                                  const std::vector<uint32_t>& data);
ONNX_NAMESPACE::TensorProto* tensor_proto_new_i32(const std::string& name, const std::vector<int64_t>& shape,
                                                  const std::vector<int32_t>& data);
ONNX_NAMESPACE::TensorProto* tensor_proto_new_u64(const std::string& name, const std::vector<int64_t>& shape,
                                                  const std::vector<uint64_t>& data);
ONNX_NAMESPACE::TensorProto* tensor_proto_new_i64(const std::string& name, const std::vector<int64_t>& shape,
                                                  const std::vector<int64_t>& data);
ONNX_NAMESPACE::TensorProto* tensor_proto_new_floats(const std::string& name, const std::vector<int64_t>& shape,
                                                     const std::vector<float>& data);
ONNX_NAMESPACE::TensorProto* tensor_proto_new_bf16(const std::string& name, const std::vector<int64_t>& shape,
                                                   const std::vector<int16_t>& data);
ONNX_NAMESPACE::TensorProto* tensor_proto_new_fp16(const std::string& name, const std::vector<int64_t>& shape,
                                                   const std::vector<int16_t>& data);
ONNX_NAMESPACE::TensorProto* tensor_proto_new_doubles(const std::string& name, const std::vector<int64_t>& shape,
                                                      const std::vector<double>& data);
}  // namespace vaip
