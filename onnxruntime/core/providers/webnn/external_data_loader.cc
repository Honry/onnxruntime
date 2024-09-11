// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <emscripten.h>

#include "external_data_loader.h"

#include "core/framework/tensor.h"
#include <emscripten.h>
#include <emscripten/val.h>

namespace onnxruntime {
namespace webnn {

bool ExternalDataLoader::CanLoad(const OrtMemoryInfo& target_memory_info) const {
  emscripten::val console = emscripten::val::global("console");
  console.call<void>("log", emscripten::val("webnn CanLoad()... called"));
  return false;
//  return target_memory_info.device.Type() == OrtDevice::CPU
//#if defined(USE_WEBNN)
//         || (target_memory_info.device.Type() == OrtDevice::GPU && target_memory_info.name == WEBNN_BUFFER)
//#endif
//      ;
}

common::Status ExternalDataLoader::LoadTensor(const Env& env,
                                              const std::filesystem::path& data_file_path,
                                              FileOffsetType data_offset,
                                              SafeInt<size_t> data_length,
                                              Tensor& tensor) const {
  emscripten::val console = emscripten::val::global("console");
  console.call<void>("log", emscripten::val("webnn LoadTensor()... called"));
  ExternalDataLoadType load_type;
  if (tensor.Location().device.Type() == OrtDevice::CPU) {
    load_type = ExternalDataLoadType::CPU;
#if defined(USE_WEBNN)
  } else if (tensor.Location().device.Type() == OrtDevice::GPU &&
             tensor.Location().name == WEBNN_BUFFER) {
    load_type = ExternalDataLoadType::WEBNN_BUFFER;
#endif
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported tensor location: ", tensor.Location().ToString());
  }

  return LoadWebAssemblyExternalData(env, data_file_path, data_offset, data_length, load_type, tensor.MutableDataRaw());
}

}  // namespace webnn
}  // namespace onnxruntime
