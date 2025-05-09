// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "core/graph/constants.h"
#include "core/framework/session_options.h"

namespace onnxruntime {
namespace qnnctxgen {

struct MachineConfig {
  std::string provider_type_name{onnxruntime::kQnnExecutionProvider};
};

struct RunConfig {
  bool f_verbose{false};
  std::unordered_map<std::string, std::string> session_config_entries;
  std::unordered_map<std::string, std::string> provider_options;
};

struct TestConfig {
  std::vector<std::basic_string<ORTCHAR_T>> model_file_paths;
  RunConfig run_config;
  MachineConfig machine_config;
};

}  // namespace qnnctxgen
}  // namespace onnxruntime
