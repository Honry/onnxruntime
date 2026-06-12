// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/common.h>

#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"
#include "core/providers/webnn/builders/helper.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class RangeOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
};

// Add operator related.

Status RangeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                             const logging::Logger& /* logger */) const {
  emscripten::val start = model_builder.GetOperand(node.InputDefs()[0]->Name());
  emscripten::val limit = model_builder.GetOperand(node.InputDefs()[1]->Name());
  emscripten::val delta = model_builder.GetOperand(node.InputDefs()[2]->Name());
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>(
      "range", start, limit, delta, options);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

void CreateRangeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<RangeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
