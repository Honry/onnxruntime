// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shape_subgraph_folder.h"

#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"

#include <queue>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace onnxruntime {
namespace webnn {

ShapeSubgraphFolder::ShapeSubgraphFolder(const GraphViewer& graph_viewer,
                                         const FreeDimensionBounds& free_dimension_bounds,
                                         const logging::Logger& logger)
    : graph_viewer_(graph_viewer),
      free_dimension_bounds_(free_dimension_bounds),
      logger_(logger) {
}

bool ShapeSubgraphFolder::IsSupportedShapeOp(const Node& node) {
  static const InlinedHashSet<std::string_view> supported_ops = {
      "Shape", "Gather", "Concat", "Unsqueeze", "Squeeze", "Slice",
      "Cast", "Add", "Sub", "Mul", "Div", "Equal", "Where",
      "ConstantOfShape", "Range", "Reshape", "Expand",
      "Neg", "Abs", "Floor", "Ceil",
  };
  // Only standard ONNX ops (empty domain or "onnx")
  if (!node.Domain().empty() && node.Domain() != "onnx" && node.Domain() != kOnnxDomain) {
    return false;
  }
  return supported_ops.count(node.OpType()) > 0;
}

bool ShapeSubgraphFolder::GetResolvedShape(const NodeArg* arg, std::vector<int64_t>& shape) const {
  const auto* shape_proto = arg->Shape();
  if (!shape_proto) return false;

  shape.clear();
  for (int i = 0; i < shape_proto->dim_size(); i++) {
    const auto& dim = shape_proto->dim(i);
    if (dim.has_dim_value()) {
      shape.push_back(dim.dim_value());
    } else if (dim.has_dim_param()) {
      // Try to resolve from free_dimension_bounds
      const auto& dim_param = dim.dim_param();
      auto it = free_dimension_bounds_.find(dim_param);
      if (it != free_dimension_bounds_.end()) {
        // Use maxSize as the resolved value (consistent with WebNN EP behavior)
        shape.push_back(static_cast<int64_t>(it->second.max_size));
      } else {
        return false;  // Can't resolve this symbolic dim
      }
    } else {
      return false;  // Unknown dim
    }
  }
  return true;
}

// Check if shape-consuming inputs (Reshape[1], Expand[1], etc.) are candidates for folding.
static bool IsShapeConsumingSlot(const Node& consumer, size_t input_index) {
  const auto& op = consumer.OpType();
  if ((op == "Reshape" || op == "Expand") && input_index == 1) return true;
  if (op == "ConstantOfShape" && input_index == 0) return true;
  if (op == "Tile" && input_index == 1) return true;
  // Slice has starts[1], ends[2], axes[3], steps[4]
  if (op == "Slice" && input_index >= 1 && input_index <= 4) return true;
  return false;
}

bool ShapeSubgraphFolder::TryFoldShapeSubgraph(const NodeArg* shape_arg) {
  const std::string& target_name = shape_arg->Name();

  // Already folded?
  if (folded_shapes_.count(target_name)) return true;

  // Check if it's already a constant initializer
  if (graph_viewer_.GetConstantInitializer(target_name)) return false;  // Already handled normally

  // BFS backward to find the producer subgraph
  std::unordered_map<std::string, std::vector<int64_t>> known_values;
  std::vector<const Node*> topo_order;  // nodes in forward eval order
  InlinedHashSet<NodeIndex> visited_nodes;
  std::queue<const NodeArg*> worklist;
  worklist.push(shape_arg);

  InlinedHashSet<std::string> visited_args;
  visited_args.insert(target_name);

  bool can_fold = true;

  while (!worklist.empty() && can_fold) {
    const NodeArg* current = worklist.front();
    worklist.pop();
    const std::string& name = current->Name();

    // Skip if already known
    if (known_values.count(name)) continue;

    // Check if it's a constant initializer
    const auto* init = graph_viewer_.GetConstantInitializer(name);
    if (init) {
      // Read int64 values from the initializer
      std::vector<int64_t> values;
      if (init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
        if (!init->int64_data().empty()) {
          values.assign(init->int64_data().begin(), init->int64_data().end());
        } else if (!init->raw_data().empty()) {
          const int64_t* data = reinterpret_cast<const int64_t*>(init->raw_data().data());
          size_t count = init->raw_data().size() / sizeof(int64_t);
          values.assign(data, data + count);
        }
      } else if (init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
        if (!init->int32_data().empty()) {
          for (auto v : init->int32_data()) values.push_back(static_cast<int64_t>(v));
        } else if (!init->raw_data().empty()) {
          const int32_t* data = reinterpret_cast<const int32_t*>(init->raw_data().data());
          size_t count = init->raw_data().size() / sizeof(int32_t);
          for (size_t i = 0; i < count; i++) values.push_back(static_cast<int64_t>(data[i]));
        }
      } else if (init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        if (!init->float_data().empty()) {
          for (auto v : init->float_data()) values.push_back(static_cast<int64_t>(v));
        } else if (!init->raw_data().empty()) {
          const float* data = reinterpret_cast<const float*>(init->raw_data().data());
          size_t count = init->raw_data().size() / sizeof(float);
          for (size_t i = 0; i < count; i++) values.push_back(static_cast<int64_t>(data[i]));
        }
      } else if (init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_BOOL) {
        if (!init->int32_data().empty()) {
          for (auto v : init->int32_data()) values.push_back(static_cast<int64_t>(v));
        } else if (!init->raw_data().empty()) {
          const uint8_t* data = reinterpret_cast<const uint8_t*>(init->raw_data().data());
          size_t count = init->raw_data().size();
          for (size_t i = 0; i < count; i++) values.push_back(static_cast<int64_t>(data[i]));
        }
      } else {
        can_fold = false;
        break;
      }
      // Handle scalar initializers (0-dim tensors with no data entries but have raw_data)
      if (values.empty() && init->dims_size() == 0) {
        // Try scalar
        if (init->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
          values.push_back(0);
        }
      }
      known_values[name] = std::move(values);
      continue;
    }

    // Check if it's a graph input (cannot fold if depends on runtime input, unless it's just shape)
    const auto* producer = graph_viewer_.GetProducerNode(name);
    if (!producer) {
      // It's a graph input - can't fold unless we can get its shape via Shape op
      can_fold = false;
      break;
    }

    // Check if the producer is a supported shape op
    if (!IsSupportedShapeOp(*producer)) {
      can_fold = false;
      break;
    }

    // Add producer to visit list
    if (visited_nodes.insert(producer->Index()).second) {
      topo_order.push_back(producer);

      // Add all inputs of this producer to the worklist
      for (const auto* input_def : producer->InputDefs()) {
        if (input_def && input_def->Exists()) {
          if (visited_args.insert(input_def->Name()).second) {
            worklist.push(input_def);
          }
        }
      }
    }
  }

  if (!can_fold) return false;

  // Sort nodes in topological order (reverse of discovery = forward eval order)
  std::reverse(topo_order.begin(), topo_order.end());

  // Evaluate nodes in topological order
  for (const Node* node : topo_order) {
    std::vector<int64_t> result;
    if (!EvaluateNode(*node, known_values, result)) {
      return false;
    }
    // Store results for all outputs
    for (const auto* output_def : node->OutputDefs()) {
      if (output_def && output_def->Exists()) {
        known_values[output_def->Name()] = result;
      }
    }
  }

  // The target should now be in known_values
  auto it = known_values.find(target_name);
  if (it == known_values.end()) return false;

  // Store the folded result
  folded_shapes_[target_name] = it->second;

  // Mark nodes in the subgraph as folded, but only if ALL their outputs are consumed
  // exclusively by other folded nodes or shape-consuming slots. If any output feeds
  // a non-shape consumer outside the subgraph, we cannot skip that node.
  for (const Node* node : topo_order) {
    bool can_skip = true;
    for (auto it2 = node->OutputEdgesBegin(); it2 != node->OutputEdgesEnd(); ++it2) {
      const Node& consumer = it2->GetNode();
      if (visited_nodes.count(consumer.Index())) continue;  // consumer is in our subgraph
      // Check if the consumer uses this output only in a shape-consuming slot
      if (!IsShapeConsumingSlot(consumer, it2->GetDstArgIndex())) {
        can_skip = false;
        break;
      }
    }
    if (can_skip) {
      folded_nodes_.insert(node->Index());
    }
  }

  return true;
}

bool ShapeSubgraphFolder::EvaluateNode(
    const Node& node,
    const std::unordered_map<std::string, std::vector<int64_t>>& known_values,
    std::vector<int64_t>& result) {
  const auto& op = node.OpType();
  const auto& inputs = node.InputDefs();

  // Helper to get input values
  auto get_input = [&](size_t idx) -> const std::vector<int64_t>* {
    if (idx >= inputs.size() || !inputs[idx] || !inputs[idx]->Exists()) return nullptr;
    auto it = known_values.find(inputs[idx]->Name());
    return (it != known_values.end()) ? &it->second : nullptr;
  };

  if (op == "Shape") {
    // Shape op: return the resolved shape of input[0]
    if (!GetResolvedShape(inputs[0], result)) return false;

    // Handle start/end attributes (Shape opset 15+)
    const auto& attrs = node.GetAttributes();
    int64_t start = 0, end = static_cast<int64_t>(result.size());
    if (attrs.count("start")) start = attrs.at("start").i();
    if (attrs.count("end")) end = attrs.at("end").i();
    if (start < 0) start += static_cast<int64_t>(result.size());
    if (end < 0) end += static_cast<int64_t>(result.size());
    start = std::max(int64_t(0), std::min(start, static_cast<int64_t>(result.size())));
    end = std::max(int64_t(0), std::min(end, static_cast<int64_t>(result.size())));
    result = std::vector<int64_t>(result.begin() + static_cast<ptrdiff_t>(start), result.begin() + static_cast<ptrdiff_t>(end));
    return true;
  }

  if (op == "Gather") {
    const auto* data = get_input(0);
    const auto* indices = get_input(1);
    if (!data || !indices) return false;

    result.clear();
    for (int64_t idx : *indices) {
      if (idx < 0) idx += static_cast<int64_t>(data->size());
      if (idx < 0 || idx >= static_cast<int64_t>(data->size())) return false;
      result.push_back((*data)[static_cast<size_t>(idx)]);
    }
    // If indices is scalar (0-dim), result should also be scalar-like
    if (indices->empty()) {
      // scalar index case - not handled here
      return false;
    }
    return true;
  }

  if (op == "Concat") {
    // For 1-D shape vectors, axis is always 0 — just concatenate all inputs.
    result.clear();
    for (size_t i = 0; i < inputs.size(); i++) {
      const auto* inp = get_input(i);
      if (!inp) return false;
      result.insert(result.end(), inp->begin(), inp->end());
    }
    return true;
  }

  if (op == "Unsqueeze") {
    const auto* data = get_input(0);
    if (!data) return false;
    // For shape subgraphs, unsqueeze typically wraps a scalar into [1] shape
    result = *data;
    // If axes input exists (opset 13+), handle it
    if (inputs.size() > 1) {
      const auto* axes = get_input(1);
      if (!axes) return false;
      // Insert dimensions of size 1 at specified axes
      // For shape vectors this is typically making a scalar into a 1-element vector
    }
    return true;
  }

  if (op == "Squeeze") {
    const auto* data = get_input(0);
    if (!data) return false;
    result = *data;
    return true;
  }

  if (op == "Cast") {
    const auto* data = get_input(0);
    if (!data) return false;
    // Cast just passes through for int64 purposes
    result = *data;
    return true;
  }

  if (op == "Neg") {
    const auto* data = get_input(0);
    if (!data) return false;
    result.resize(data->size());
    for (size_t i = 0; i < data->size(); i++) result[i] = -(*data)[i];
    return true;
  }

  if (op == "Abs") {
    const auto* data = get_input(0);
    if (!data) return false;
    result.resize(data->size());
    for (size_t i = 0; i < data->size(); i++) result[i] = std::abs((*data)[i]);
    return true;
  }

  // Binary element-wise ops: Add, Sub, Mul, Div
  if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div") {
    const auto* a = get_input(0);
    const auto* b = get_input(1);
    if (!a || !b) return false;

    // Broadcasting: if one is scalar (size 1), broadcast to the other's size
    size_t size = std::max(a->size(), b->size());
    result.resize(size);
    for (size_t i = 0; i < size; i++) {
      int64_t va = (*a)[a->size() == 1 ? 0 : i];
      int64_t vb = (*b)[b->size() == 1 ? 0 : i];
      if (op == "Add") result[i] = va + vb;
      else if (op == "Sub") result[i] = va - vb;
      else if (op == "Mul") result[i] = va * vb;
      else if (op == "Div") {
        if (vb == 0) return false;
        result[i] = va / vb;
      }
    }
    return true;
  }

  if (op == "Equal") {
    const auto* a = get_input(0);
    const auto* b = get_input(1);
    if (!a || !b) return false;

    size_t size = std::max(a->size(), b->size());
    result.resize(size);
    for (size_t i = 0; i < size; i++) {
      int64_t va = (*a)[a->size() == 1 ? 0 : i];
      int64_t vb = (*b)[b->size() == 1 ? 0 : i];
      result[i] = (va == vb) ? 1 : 0;
    }
    return true;
  }

  if (op == "Where") {
    const auto* cond = get_input(0);
    const auto* x = get_input(1);
    const auto* y = get_input(2);
    if (!cond || !x || !y) return false;

    size_t size = std::max({cond->size(), x->size(), y->size()});
    result.resize(size);
    for (size_t i = 0; i < size; i++) {
      int64_t c = (*cond)[cond->size() == 1 ? 0 : i];
      int64_t vx = (*x)[x->size() == 1 ? 0 : i];
      int64_t vy = (*y)[y->size() == 1 ? 0 : i];
      result[i] = c ? vx : vy;
    }
    return true;
  }

  if (op == "Range") {
    const auto* start_v = get_input(0);
    const auto* limit_v = get_input(1);
    const auto* delta_v = get_input(2);
    if (!start_v || !limit_v || !delta_v) return false;
    if (start_v->empty() || limit_v->empty() || delta_v->empty()) return false;

    int64_t start = (*start_v)[0];
    int64_t limit = (*limit_v)[0];
    int64_t delta = (*delta_v)[0];
    if (delta == 0) return false;

    result.clear();
    if (delta > 0) {
      for (int64_t v = start; v < limit; v += delta) result.push_back(v);
    } else {
      for (int64_t v = start; v > limit; v += delta) result.push_back(v);
    }
    return true;
  }

  if (op == "ConstantOfShape") {
    const auto* shape_input = get_input(0);
    if (!shape_input) return false;

    // Get the fill value from attribute
    int64_t fill_value = 0;
    const auto& attrs = node.GetAttributes();
    if (attrs.count("value")) {
      const auto& tensor = attrs.at("value").t();
      if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
        if (!tensor.int64_data().empty()) fill_value = tensor.int64_data(0);
        else if (!tensor.raw_data().empty())
          fill_value = *reinterpret_cast<const int64_t*>(tensor.raw_data().data());
      } else if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
        if (!tensor.int32_data().empty()) fill_value = tensor.int32_data(0);
        else if (!tensor.raw_data().empty())
          fill_value = *reinterpret_cast<const int32_t*>(tensor.raw_data().data());
      } else if (tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        float fv = 0.0f;
        if (!tensor.float_data().empty()) fv = tensor.float_data(0);
        else if (!tensor.raw_data().empty())
          fv = *reinterpret_cast<const float*>(tensor.raw_data().data());
        fill_value = static_cast<int64_t>(fv);
      }
    }

    // Compute total size from shape
    int64_t total = 1;
    for (int64_t d : *shape_input) total *= d;
    if (total < 0 || total > 1000000) return false;  // Safety limit

    result.assign(static_cast<size_t>(total), fill_value);
    return true;
  }

  if (op == "Slice") {
    const auto* data = get_input(0);
    const auto* starts = get_input(1);
    const auto* ends = get_input(2);
    if (!data || !starts || !ends) return false;

    // For 1-D shape tensors
    int64_t start = (*starts)[0];
    int64_t end = (*ends)[0];
    int64_t dim_size = static_cast<int64_t>(data->size());

    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;
    start = std::max(int64_t(0), std::min(start, dim_size));
    end = std::max(int64_t(0), std::min(end, dim_size));

    int64_t step = 1;
    if (inputs.size() > 4) {
      const auto* steps_v = get_input(4);
      if (steps_v && !steps_v->empty()) step = (*steps_v)[0];
    }

    result.clear();
    if (step > 0) {
      for (int64_t i = start; i < end; i += step) result.push_back((*data)[static_cast<size_t>(i)]);
    } else if (step < 0) {
      for (int64_t i = start; i > end; i += step) result.push_back((*data)[static_cast<size_t>(i)]);
    }
    return true;
  }

  if (op == "Reshape") {
    const auto* data = get_input(0);
    if (!data) return false;
    // For shape subgraphs, Reshape just passes data through (reshaping a 1-D vector)
    result = *data;
    return true;
  }

  if (op == "Expand") {
    const auto* data = get_input(0);
    const auto* shape = get_input(1);
    if (!data || !shape) return false;
    // For shape subgraphs, Expand broadcasts scalar/small tensor
    if (data->size() == 1 && !shape->empty()) {
      int64_t total = 1;
      for (int64_t d : *shape) total *= d;
      if (total < 0 || total > 1000000) return false;
      result.assign(static_cast<size_t>(total), (*data)[0]);
    } else {
      result = *data;
    }
    return true;
  }

  // Unsupported op
  return false;
}

Status ShapeSubgraphFolder::Run() {
  // Find all shape-consuming input slots and try to fold them
  const auto& nodes = graph_viewer_.GetNodesInTopologicalOrder();

  for (auto node_idx : nodes) {
    const auto* node = graph_viewer_.GetNode(node_idx);
    if (!node) continue;

    const auto& input_defs = node->InputDefs();
    for (size_t i = 0; i < input_defs.size(); i++) {
      if (!IsShapeConsumingSlot(*node, i)) continue;

      const auto* shape_arg = input_defs[i];
      if (!shape_arg || !shape_arg->Exists()) continue;

      // Skip if already a constant initializer (handled normally)
      if (graph_viewer_.GetConstantInitializer(shape_arg->Name())) continue;

      // Try to fold this shape input
      if (TryFoldShapeSubgraph(shape_arg)) {
        LOGS(logger_, VERBOSE) << "ShapeSubgraphFolder: Folded shape input '"
                               << shape_arg->Name() << "' for "
                               << node->OpType() << " node '" << node->Name() << "'"
                               << " -> [" << [&]() {
                                    std::string s;
                                    for (auto v : folded_shapes_[shape_arg->Name()]) {
                                      if (!s.empty()) s += ", ";
                                      s += std::to_string(v);
                                    }
                                    return s;
                                  }()
                               << "]";
      }
    }
  }

  LOGS(logger_, VERBOSE) << "ShapeSubgraphFolder: Folded " << folded_shapes_.size()
                      << " shape subgraphs, " << folded_nodes_.size() << " nodes eliminated.";

  return Status::OK();
}

bool ShapeSubgraphFolder::IsFoldedShape(const std::string& name) const {
  return folded_shapes_.count(name) > 0;
}

const std::vector<int64_t>* ShapeSubgraphFolder::GetFoldedShape(const std::string& name) const {
  auto it = folded_shapes_.find(name);
  return (it != folded_shapes_.end()) ? &it->second : nullptr;
}

bool ShapeSubgraphFolder::IsFoldedNode(NodeIndex node_index) const {
  return folded_nodes_.count(node_index) > 0;
}

}  // namespace webnn
}  // namespace onnxruntime
