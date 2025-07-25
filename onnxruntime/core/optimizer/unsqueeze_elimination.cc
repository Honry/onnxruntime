// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/unsqueeze_elimination.h"
#include "core/common/logging/logging.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph.h"
#include "core/optimizer/initializer.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {

Status UnsqueezeElimination::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const {
  NodeArg& input_def = *node.MutableInputDefs()[0];
  const auto& tensor_proto = *graph_utils::GetConstantInitializer(graph, input_def.Name());

  auto new_name = graph.GenerateNodeArgName("UnsqueezeElimination_" + input_def.Name());
  if (!graph_utils::CanReplaceNodeWithInitializer(graph, node, new_name, logger)) {
    LOGS(logger, WARNING) << "UnsqueezeElimination cannot remove node " << node.Name();
    return Status::OK();
  }

  InlinedVector<int64_t> axes;
  if (!graph_utils::GetRepeatedNodeAttributeValues(node, "axes", axes)) {
    // missing 'axes'. should have failed at model load but just in case...
    return Status::OK();
  }

  auto num_axes = axes.size();
  auto output_rank = num_axes + tensor_proto.dims().size();

  // handle any negative axis values
  for (auto& axis : axes) {
    if (axis < 0) {
      axis += output_rank;
    }
  }

  // Generate new dims.
  InlinedVector<int64_t> new_dims(output_rank, 0);
  for (int64_t axis : axes) {
    if (static_cast<size_t>(axis) >= new_dims.size()) {
      LOGS(logger, WARNING) << "UnsqueezeElimination cannot remove node due to invalid axes" << node.Name();
      return Status::OK();
    }
    new_dims[static_cast<size_t>(axis)] = 1;
  }

  auto begin = tensor_proto.dims().cbegin();
  for (auto& axis : new_dims) {
    if (axis == 0) {
      axis = *begin++;
    }
  }

  Initializer initializer(graph, tensor_proto, graph.ModelPath(), /*check_outer_scope=*/false);
  ONNX_NAMESPACE::TensorProto new_tensor_proto;
  OrtValue ort_value;
  initializer.ToProtoWithOrtValue(new_tensor_proto, ort_value);

  // Update shape of tensor proto.
  new_tensor_proto.set_name(new_name);
  new_tensor_proto.clear_dims();

  for (const auto& dim : new_dims) {
    new_tensor_proto.add_dims(dim);
  }

  if (utils::HasExternalDataInMemory(new_tensor_proto)) {
    ORT_ENFORCE(ort_value.IsAllocated());
    TensorShape new_shape(new_tensor_proto.dims());
    ort_value.GetMutable<Tensor>()->Reshape(new_shape);
  }

  auto& new_node_arg = graph_utils::AddInitializerWithExternalData(graph, new_tensor_proto, ort_value);
  graph_utils::ReplaceNodeWithInitializer(graph, node, new_node_arg);

  // Remove the Unsqueeze node and replace it with the initializer.
  rule_effect = RewriteRuleEffect::kRemovedCurrentNode;

  return Status::OK();
}

bool UnsqueezeElimination::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const {
  // Attempt to remove an Unsqueeze operator only if it gets a constant initializer as input.
  return graph_utils::IsConstantInitializer(graph, node.InputDefs()[0]->Name());
}

}  // namespace onnxruntime
