// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webnn/webnn_provider_factory_creator.h"
#include "webnn_execution_provider.h"

#include <string_view>

using namespace onnxruntime;

namespace onnxruntime {

struct WebNNProviderFactory : IExecutionProviderFactory {
  explicit WebNNProviderFactory(const std::string& webnn_device_flags,
                                bool enable_causal_lm)
      : webnn_device_flags_(webnn_device_flags),
        enable_causal_lm_(enable_causal_lm) {}
  ~WebNNProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger) override;

  std::string webnn_device_flags_;
  bool enable_causal_lm_;
};

std::unique_ptr<IExecutionProvider> WebNNProviderFactory::CreateProvider() {
  return std::make_unique<WebNNExecutionProvider>(webnn_device_flags_, enable_causal_lm_);
}

std::unique_ptr<IExecutionProvider> WebNNProviderFactory::CreateProvider(
    const OrtSessionOptions& /*session_options*/,
    const OrtLogger& /*session_logger*/) {
  return std::make_unique<WebNNExecutionProvider>(webnn_device_flags_, enable_causal_lm_);
}

std::shared_ptr<IExecutionProviderFactory> WebNNProviderFactoryCreator::Create(
    const ProviderOptions& provider_options) {
  const auto device_type_it = provider_options.find("deviceType");
  const std::string webnn_device_flags =
      (device_type_it != provider_options.end()) ? device_type_it->second : "cpu";

  // enableCausalLM: controls the KV-cache update strategy in GroupQueryAttention.
  //   "true"  → concat-based (stateful): present_kv = concat(past_kv, new_kv), cache grows each step.
  //   "false" (default) → ScatterND-based (stateless): new tokens scattered into fixed-size buffer.
  const auto enable_causal_lm_it = provider_options.find("enableCausalLM");
  const bool enable_causal_lm = (enable_causal_lm_it != provider_options.end() &&
                                  enable_causal_lm_it->second == "true");

  return std::make_shared<onnxruntime::WebNNProviderFactory>(webnn_device_flags, enable_causal_lm);
}

}  // namespace onnxruntime
