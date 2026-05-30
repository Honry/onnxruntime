// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

#include "core/common/inlined_containers.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/webnn/builders/helper.h"

namespace onnxruntime {
namespace webnn {

// ShapeSubgraphFolder: Pre-evaluates shape-computing subgraphs in the ONNX graph.
//
// In unfused (HuggingFace-Optimum-style) models, Reshape/Expand/ConstantOfShape ops
// have shape inputs produced by chains of shape-domain ops (Shape, Gather, Concat,
// Where, Equal, Range, ConstantOfShape, etc.). Chromium's WebNN ShapeFoldingInterpreter
// can't handle all of these, causing "Graph has been destroyed" errors.
//
// This folder:
// 1. Identifies "shape-consuming" input slots (Reshape[1], Expand[1], etc.)
// 2. Traces each shape input's producer subgraph backward
// 3. If the entire subgraph can be evaluated with known constants + free_dimension_bounds,
//    evaluates it to produce a concrete int64 shape tensor
// 4. Makes these folded shapes available as synthetic constant initializers
//
// Runs once at session creation → zero per-inference cost.
class ShapeSubgraphFolder {
 public:
  ShapeSubgraphFolder(const GraphViewer& graph_viewer,
                      const FreeDimensionBounds& free_dimension_bounds,
                      const logging::Logger& logger);

  // Run the folding pass. After this, GetFoldedShape() and IsFoldedNode() are valid.
  Status Run();

  // Check if a NodeArg name has been folded to a constant shape.
  bool IsFoldedShape(const std::string& name) const;

  // Get the folded int64 tensor data for a shape NodeArg.
  // Returns nullptr if not folded.
  const std::vector<int64_t>* GetFoldedShape(const std::string& name) const;

  // Check if a node is part of a folded shape subgraph (should be skipped in AddOperations).
  bool IsFoldedNode(NodeIndex node_index) const;

  // Get the set of node indices that were folded (for skipping).
  const InlinedHashSet<NodeIndex>& GetFoldedNodes() const { return folded_nodes_; }

 private:
  // Evaluate a shape-producing subgraph rooted at the given NodeArg.
  // Returns true if successfully folded, with result stored in folded_shapes_.
  bool TryFoldShapeSubgraph(const NodeArg* shape_arg);

  // Mini-interpreter: evaluate a single node given its input values.
  // Returns true if the node can be evaluated.
  bool EvaluateNode(const Node& node,
                    const std::unordered_map<std::string, std::vector<int64_t>>& known_values,
                    std::vector<int64_t>& result);

  // Get the resolved shape of a NodeArg (using free_dimension_bounds for symbolic dims).
  bool GetResolvedShape(const NodeArg* arg, std::vector<int64_t>& shape) const;

  // Check if a node is a supported shape-domain op for the mini-interpreter.
  static bool IsSupportedShapeOp(const Node& node);

  const GraphViewer& graph_viewer_;
  const FreeDimensionBounds& free_dimension_bounds_;
  const logging::Logger& logger_;

  // Maps NodeArg name → folded int64 shape values.
  std::unordered_map<std::string, std::vector<int64_t>> folded_shapes_;

  // Set of node indices that are part of folded subgraphs (to be skipped).
  InlinedHashSet<NodeIndex> folded_nodes_;
};

}  // namespace webnn
}  // namespace onnxruntime
