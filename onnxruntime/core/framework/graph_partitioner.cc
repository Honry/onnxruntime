// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/graph_partitioner.h"

#include <cassert>
#include <functional>

#include "core/common/inlined_containers.h"
#include "core/common/string_utils.h"
#include "core/framework/compute_capability.h"
#include "core/framework/execution_providers.h"
#include "core/framework/func_kernel.h"
#include "core/framework/kernel_lookup.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/resource_accountant.h"
#include "core/graph/function.h"
#include "core/graph/function_utils.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/model_saving_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

// uncomment this line to count non-CUDA ops in ONNX domain
// #define COUNT_NON_CUDA_OPS

#ifdef COUNT_NON_CUDA_OPS
class NonCudaOps {
 public:
  ~NonCudaOps() {
    printf("Non-CUDA ops:\n");
    for (auto i : map_) {
      printf("%s: %d\n", i.first.c_str(), i.second);
    }
  }

  void AddOp(const std::string& name) {
    if (map_.count(name))
      map_.at(name)++;
    else
      map_.insert({name, 1});
  }

 private:
  std::map<std::string, int> map_;
};

NonCudaOps non_cuda;
#endif

namespace onnxruntime {

namespace {

// contains some common parameters used by the partitioning helper functions
struct PartitionParams {
  std::reference_wrapper<Graph> graph;
  std::reference_wrapper<const CheckLoadCancellationFn> check_load_cancellation_fn;
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  std::reference_wrapper<FuncManager> func_mgr;
  std::reference_wrapper<KernelRegistry> fused_kernel_registry;
  std::reference_wrapper<int> fused_node_unique_id;
  std::reference_wrapper<const layout_transformation::TransformLayoutFunction> transform_layout_function;
  std::reference_wrapper<const layout_transformation::DebugGraphFn> debug_graph_fn;
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
};
}  // namespace

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

// minimal KernelDef based on MetaDef instead of a Function based node
static void BuildFusedKernelDef(KernelDefBuilder& builder, const IndexedSubGraph::MetaDef& metadef,
                                const std::string& provider_type) {
  builder.SetName(metadef.name)
      .SetDomain(metadef.domain)
      .SinceVersion(metadef.since_version)
      .Provider(provider_type);
}

/// <summary>
/// Check if a node can be placed on a specific provider. If yes, then set the nodes execution provider.
/// Do nothing if the node is already assigned.
/// </summary>
/// <param name="graph">Graph in question.</param>
/// <param name="capability">Indexed subgraph which needs to be assigned</param>
/// <param name="provider_type">The EP to assign the Indexed subgraph to</param>
static bool TryAssignNodes(Graph& graph, const IndexedSubGraph& capability,
                           const std::string& provider_type) {
  // Before assigning the ep to any node, first walk through all the nodes and ensure
  // none of the nodes have already been assigned. If a node is assigned, simply return.
  for (auto node_index : capability.nodes) {
    const auto* node = graph.GetNode(node_index);
    if ((nullptr == node) ||
        (!node->GetExecutionProviderType().empty() && node->GetExecutionProviderType() != provider_type)) {
      return false;
    }
  }

  const bool acc_enabled = capability.IsAccountingEnabled();
  for (size_t i = 0, limit = capability.nodes.size(); i < limit; ++i) {
    auto* node = graph.GetNode(capability.nodes[i]);
    node->SetExecutionProviderType(provider_type);
    if (acc_enabled) {
      capability.AccountForNode(i);
    }
  }
  return true;
}

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

static bool TryAssignSingleNode(Graph& graph,
                                const IndexedSubGraph& indexed_sub_graph,
                                const std::string& provider_type) {
  // The provider can run a single node in the <graph> if not using meta-defs.
  // A fused kernel is not supported in this case.
  assert(indexed_sub_graph.GetMetaDef() == nullptr && indexed_sub_graph.nodes.size() == 1);

  auto* node = graph.GetNode(indexed_sub_graph.nodes[0]);
  if (nullptr != node && node->GetExecutionProviderType().empty()) {
    // The node was not fused or assigned. Assign it to <provider_type>.
    node->SetExecutionProviderType(provider_type);
    if (indexed_sub_graph.IsAccountingEnabled()) {
      indexed_sub_graph.AccountForNode(0);
    }
    return true;
  }

  return false;
}

namespace {
struct GetCapabilityForEPParams {
  std::reference_wrapper<Graph> graph;
  std::reference_wrapper<const KernelRegistryManager> kernel_registry_mgr;
  std::reference_wrapper<IExecutionProvider> current_ep;
  std::reference_wrapper<std::vector<std::unique_ptr<ComputeCapability>>> capabilities;

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  GraphPartitioner::Mode mode;
  std::reference_wrapper<const layout_transformation::TransformLayoutFunction> transform_layout;
  std::reference_wrapper<const layout_transformation::DebugGraphFn> debug_graph_fn;
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  IResourceAccountant* resource_accountant;
  std::reference_wrapper<const GraphOptimizerRegistry> graph_optimizer_registry;
  std::reference_wrapper<const CheckLoadCancellationFn> check_load_cancellation_fn;
};

auto get_capabilities = [](const IExecutionProvider& ep,
                           const GraphViewer& graph_viewer,
                           const IExecutionProvider::IKernelLookup& kernel_lookup,
                           IResourceAccountant* resource_accountant,
                           const GraphOptimizerRegistry& graph_optimizer_registry) {
  auto capabilities = ep.GetCapability(graph_viewer, kernel_lookup, graph_optimizer_registry, resource_accountant);

  // In theory an EP could return an empty capability. Remove those.
  capabilities.erase(std::remove_if(capabilities.begin(), capabilities.end(),
                                    [](const std::unique_ptr<ComputeCapability>& capability) {
                                      return !capability || !capability->sub_graph;
                                    }),
                     capabilities.end());

  return capabilities;
};
}  // namespace

static Status GetCapabilityForEP(const GetCapabilityForEPParams& params, const logging::Logger& logger) {
  auto& current_ep = params.current_ep.get();
  const auto& ep_type = current_ep.Type();

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  if (current_ep.GetPreferredLayout() == DataLayout::NHWC && !params.transform_layout.get()) {
    LOGS(logger, WARNING) << ep_type << " cannot be used with this model due to its ONNX opset not being supported by "
                                        "the layout transformer.";
    return Status::OK();
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

  const auto& kernel_registry_mgr = params.kernel_registry_mgr.get();
  const auto kernel_registries_for_ep = kernel_registry_mgr.GetKernelRegistriesByProviderType(ep_type);
  const KernelLookup kernel_lookup{ep_type,
                                   kernel_registries_for_ep,
                                   kernel_registry_mgr.GetKernelTypeStrResolver(),
                                   logger};

  auto& graph = params.graph.get();
  auto& capabilities = params.capabilities.get();
  const auto& graph_optimizer_registry = params.graph_optimizer_registry.get();

  {
    const GraphViewer graph_viewer(graph);
    capabilities = get_capabilities(current_ep, graph_viewer, kernel_lookup, params.resource_accountant,
                                    graph_optimizer_registry);
    if (params.check_load_cancellation_fn()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED,
                             "Graph partitioning was canceled by user request");
    }

    if (capabilities.empty()) {
      return Status::OK();
    }
  }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  // Run layout transformer for EPs with preferred layout of NHWC
  // CPU EP layout transformation happens later when level 3 transformers are run.
  if (params.mode != GraphPartitioner::Mode::kAssignOnly && params.transform_layout.get() &&
      current_ep.GetPreferredLayout() == DataLayout::NHWC) {
    for (auto& capability : capabilities) {
      TryAssignNodes(graph, *capability->sub_graph, ep_type);
    }

    const NodeIndex first_new_node = graph.MaxNodeIndex();

    // Perform layout transformation on the specific EP assigned graph
    bool modified = false;
    ORT_RETURN_IF_ERROR(params.transform_layout(graph, modified, current_ep, params.debug_graph_fn));
    if (params.check_load_cancellation_fn()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED,
                             "GetCapabilities was canceled by user request");
    }

    // It is possible some new nodes are introduced during transformation. These nodes can be either existing nodes
    // which are reconstructed to update domain or completely new nodes which are necessary for layout transformation.
    // we always give GetCapability the second call as long as capabilities is not empty. GetCapability have different
    // behaviors for first/second call, the first call only tag those nodes supported by this EP and then
    // assigned by `AssignNodes`, the second call will do some node processing and
    // node fusion whenever ops were layout-sensitive or not.
    // So we are calling GetCapability twice here to make things simple and finish the following procedures;
    // 1. To process new nodes introduced by transform_layout function.
    // 2. To do Op-fusion and graph optimization
    // 3. QDQ node-group fusion

    const NodeIndex end_node = graph.MaxNodeIndex();

    capabilities.clear();

    const GraphViewer graph_viewer(graph);
    capabilities = get_capabilities(current_ep, graph_viewer, kernel_lookup, params.resource_accountant,
                                    graph_optimizer_registry);
    if (params.check_load_cancellation_fn()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED,
                             "GetCapabilities was canceled by user request");
    }

    // all nodes with an index >= first_new_node with domain of kMSInternalNHWCDomain should be in the capabilities
    InlinedHashSet<NodeIndex> new_nodes_in_capabilities;
    for (const auto& capability : capabilities) {
      for (auto node_index : capability->sub_graph->nodes) {
        if (node_index >= first_new_node) {
          new_nodes_in_capabilities.insert(node_index);
        }
      }
    }

    for (NodeIndex idx = first_new_node; idx < end_node; ++idx) {
      const Node* node = graph.GetNode(idx);
      if (node != nullptr && node->Domain() == kMSInternalNHWCDomain) {
        if (new_nodes_in_capabilities.count(node->Index()) == 0) {
          return ORT_MAKE_STATUS(
              ONNXRUNTIME, FAIL,
              "Node '", node->Name(), "' OpType:", node->OpType(), " with domain:", kMSInternalNHWCDomain,
              " was inserted using the NHWC format as requested by ", ep_type, ", but was not selected",
              " by that EP. This means the graph is now invalid as there will not be an EP able to run the node."
              " This could be a bug in layout transformer, or in the GetCapability implementation of the EP.");
        }
      }
    }
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD)

// This function queries the capabilities for a given EP, but it does not assign the nodes.
// It also does not perform layout transformation. This will be done during normal partitioning.
static Status GetCapabilityForEPForAotInlining(const GraphViewer& graph_viewer,
                                               const KernelRegistryManager& kernel_registry_mgr,
                                               const IExecutionProvider& current_ep,
                                               const GraphOptimizerRegistry& graph_optimizer_registry,
                                               const logging::Logger& logger,
                                               std::vector<std::unique_ptr<ComputeCapability>>& capabilities) {
  const auto& ep_type = current_ep.Type();

  const auto kernel_registries_for_ep = kernel_registry_mgr.GetKernelRegistriesByProviderType(ep_type);
  const KernelLookup kernel_lookup{ep_type,
                                   kernel_registries_for_ep,
                                   kernel_registry_mgr.GetKernelTypeStrResolver(),
                                   logger};

  // TODO: Provide EP with a capability to look inside the functions.
  capabilities = get_capabilities(current_ep, graph_viewer, kernel_lookup, nullptr, graph_optimizer_registry);

  return Status::OK();
}

/**
 * Check whether the given IndexedSubGraph is available for assigning to a specific provider.
 *
 */
static bool IsIndexedSubGraphAvailableForAssignment(Graph& graph,
                                                    const IndexedSubGraph& capability,
                                                    GraphPartitioner::Mode mode,
                                                    const std::string& provider_type) {
  // The provider can run a single node in the <graph> if not using meta-defs.
  if (capability.GetMetaDef() == nullptr && capability.nodes.size() == 1) {
    auto* node = graph.GetNode(capability.nodes[0]);
    if (nullptr != node && node->GetExecutionProviderType().empty()) {
      // The node was not fused or assigned.
      return true;
    }
    return false;
  }

  // if mode is kAssignOnly we want all nodes that can _potentially_ be taken by compiling EPs to be assigned,
  // so that we aggregate the nodes covered and ensure the original nodes remain in the ORT format model by
  // preventing level 2 and 3 optimizers from changing them. optimizers check the EP the node is assigned to
  // and only make changes if the EP is on the optimizer's list of supported EPs. an EP that compiles nodes
  // should never be on those lists.
  //
  // when the ORT format model is loaded we will process it normally with EP priority being applied for
  // whichever EPs are enabled at the time.
  //
  // e.g. an Android NNAPI EP may take different/overlapping nodes to a iOS CoreML EP.
  // We want the ORT format model to be able to be run as efficiently as possible on either platform,
  // so we want all the nodes that either may take to be preserved. If we did not do this we would
  // need to create one ORT format model for Android and one for iOS.
  if (mode == GraphPartitioner::Mode::kAssignOnly) {
    return true;
  }

  for (auto node_index : capability.nodes) {
    const auto* node = graph.GetNode(node_index);
    if ((nullptr == node) ||
        (!node->GetExecutionProviderType().empty() && node->GetExecutionProviderType() != provider_type)) {
      // The node was fused or assigned, so that the whole sub-graph will not be assigned to this <provider>
      // The assumption is that this <provider> can only run the sub-graph as a whole unit.
      return false;
    }
  }

  return true;
}

/**
 * Return a fused node or assign the nodes in the indexed subgraph to the current EP.
 *
 * \param graph
 * \param capability
 * \param kernel_registry_mgr
 * \param provider_type name of the provider to test
 * \param mode
 * \param fused_node_unique_id A counter for generating fused node names. Unique across the entire model.
 * \return Fused node. Return nullptr if there is no fuse
 */
static Node* PlaceNode(Graph& graph, const IndexedSubGraph& capability,
                       IExecutionProvider::FusionStyle fusion_style,
                       const std::string& provider_type,
                       GraphPartitioner::Mode mode,
                       int& fused_node_unique_id) {
  Node* result = nullptr;

  if (nullptr == capability.GetMetaDef()) {
    TryAssignSingleNode(graph, capability, provider_type);
  } else {
    const bool acc_enabled = capability.IsAccountingEnabled();
    if (mode == GraphPartitioner::Mode::kNormal) {
      std::ostringstream oss;
      oss << provider_type << "_" << capability.GetMetaDef()->name << "_" << fused_node_unique_id++;
      std::string node_name = oss.str();

      Node* fused_node = nullptr;
      if (fusion_style == IExecutionProvider::FusionStyle::Function) {
        fused_node = &graph.FuseSubGraph(capability, node_name);
      } else {
        // create a fused node without copying everything to a Function body. The IndexedSubGraph will be passed
        // through to Compile via a filtered GraphViewer.
        fused_node = &graph.BeginFuseSubGraph(capability, node_name);
      }

      fused_node->SetExecutionProviderType(provider_type);
      if (acc_enabled) {
        // We account for the fused node. We operate under assumption
        // that the fused node would use no more memory when the nodes we are fusing.
        // and potentially less than that, and therefore, no threshold check is needed here.
        // All threshold checks are done within the EP.
        capability.ComputeAndAccountForNode(*fused_node);
      }

      result = fused_node;
    } else {
      // assign the nodes in the indexed subgraph to the current EP so that level 2+ optimizers will not change them.
      // This is used when exporting an ORT format model to maintain the original nodes and re-do the fusion
      // at runtime. The original nodes provide a fallback if fewer nodes can be fused at runtime due to device
      // capabilities.
      for (size_t i = 0, limit = capability.nodes.size(); i < limit; ++i) {
        auto* node = graph.GetNode(capability.nodes[i]);
        if (node != nullptr) {
          node->SetExecutionProviderType(provider_type);
          if (acc_enabled) {
            capability.AccountForNode(i);
          }
        }
      }
    }
  }

  return result;
}

// for the current EP, recursively iterate through the Graph and any nested subgraphs (recursion is bottom-up).
// assign any nodes to the EP that are currently unassigned, and that the EP can handle.
static Status PartitionOnnxFormatModelImpl(Graph& graph, FuncManager& func_mgr,
                                           KernelRegistryManager& kernel_registry_mgr,
                                           KernelRegistry& fused_kernel_registry,
                                           IExecutionProvider& current_ep,
                                           GraphPartitioner::Mode mode,
                                           int& fused_node_unique_id,
                                           const layout_transformation::TransformLayoutFunction& transform_layout_fn,
                                           const layout_transformation::DebugGraphFn& debug_graph_fn,
                                           const CheckLoadCancellationFn& check_load_cancellation_fn,
                                           const logging::Logger& logger, IResourceAccountant* resource_accountant,
                                           const GraphOptimizerRegistry& graph_optimizer_registry,
                                           bool disable_model_compile) {
  // handle testing edge case where optimizers or constant lifting results in graph with no nodes.
  // doing it here saves all providers checking for this in GetCapability
  if (graph.NumberOfNodes() == 0) {
    return Status::OK();
  }

  // recurse into nested graphs first to partition bottom up.
  for (auto& node : graph.Nodes()) {
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      Graph* subgraph = entry.second;
      // we pass through the FuncManager from the top level graph
      ORT_RETURN_IF_ERROR(PartitionOnnxFormatModelImpl(*subgraph, func_mgr, kernel_registry_mgr,
                                                       fused_kernel_registry, current_ep, mode, fused_node_unique_id,
                                                       transform_layout_fn, debug_graph_fn,
                                                       check_load_cancellation_fn,
                                                       logger, resource_accountant,
                                                       graph_optimizer_registry, disable_model_compile));
    }
  }

  // If an execution provider returns the capability that it can run a sub-graph,
  // onnxruntime will fuse the sub-graph into a function node. For compilation
  // based execution providers (one which needs to compile graph at runtime.
  // Indicated by need_compile flag), onnxruntime will invoke the "Compile" method
  // to get compiled binary. There are two mode of compile, one is return the entry
  // point to the compiled binary directly, another is export the compiled binary to
  // shared library for future reuse.

  // TODO: when the graph contains a function node, and user passes in the dll which could
  // run the function by SessionOption, we should create a function kernel for it and
  // delegate the compute to the functions inside the dlls.
  std::vector<std::unique_ptr<ComputeCapability>> capabilities;
  const auto get_capability_params = GetCapabilityForEPParams{
      std::ref(graph),
      std::cref(kernel_registry_mgr),
      std::ref(current_ep),
      std::ref(capabilities),
      mode,
      std::cref(transform_layout_fn),
      std::cref(debug_graph_fn),
      resource_accountant,
      std::ref(graph_optimizer_registry),
      std::cref(check_load_cancellation_fn)};

  ORT_RETURN_IF_ERROR(GetCapabilityForEP(get_capability_params, logger));
  if (capabilities.empty()) {
    return Status::OK();
  }

  const std::string& type = current_ep.Type();
  auto fusion_style = current_ep.GetFusionStyle();
  std::vector<Node*> nodes_to_compile;

  // The fused node may map to an existing kernel, so it is fused but doesn't need to be compiled
  // But we still need to finalize the graph fusion for those nodes.
  std::vector<Node*> nodes_to_complete_fuse;
  std::vector<std::unique_ptr<ComputeCapability>> capabilities_to_complete_fuse;

  // filter out the ComputeCapability instances that do not need compiling so we have a std::vector that's 1:1 with
  // nodes_to_compile.
  std::vector<std::unique_ptr<ComputeCapability>> capabilities_to_compile;
  capabilities_to_compile.reserve(std::count_if(capabilities.cbegin(), capabilities.cend(),
                                                [](const std::unique_ptr<ComputeCapability>& entry) {
                                                  return entry != nullptr &&
                                                         entry->sub_graph != nullptr &&
                                                         entry->sub_graph->GetMetaDef() != nullptr;
                                                }));
  for (auto& capability : capabilities) {
    // The <provider> can run a fused <sub_graph> in the <graph>.
    // Check whether any node in the <sub_graph> was already assigned. If so it cannot be stolen as assignment is done
    // in order of EP priority
    bool sub_graph_available_for_assignment = IsIndexedSubGraphAvailableForAssignment(graph, *capability->sub_graph, mode, type);

    // If the <sub_graph> is available to be assigned to the EP and the ComputeCapability has nodes_to_optimize,
    // run EP related optimizations and update ComputeCapability.
    if (sub_graph_available_for_assignment && !capability->nodes_to_optimize.empty()) {
      for (auto& optimization_cc : capability->nodes_to_optimize) {
        if (optimization_cc->optimization_func) {
          auto status = optimization_cc->optimization_func(graph, *optimization_cc, *capability, graph_optimizer_registry);
          if (status != Status::OK()) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, type, "The optimization function failed to finish.");
          }
          // #TODO: Handle nested optimization ComputeCapability
        }
      }
    }

    Node* n = nullptr;
    if (sub_graph_available_for_assignment) {
      n = PlaceNode(graph, *capability->sub_graph, fusion_style, type, mode, fused_node_unique_id);
    }

    if (n != nullptr) {
      // searching in kernel registries, if no kernel registered for the fused_node, use compile approach
      if (!KernelRegistryManager::HasImplementationOf(kernel_registry_mgr, *n, type, logger)) {
        nodes_to_compile.push_back(n);
        capabilities_to_compile.push_back(std::move(capability));
      } else {
        // there is a predefined kernel for the fused node. doesn't need compile, but need to complete the fusing.
        nodes_to_complete_fuse.push_back(n);
        capabilities_to_complete_fuse.push_back(std::move(capability));
      }
    }
  }

  // Helper function that returns true if any of the nodes assigned to a compiling EP is not already compiled.
  auto graph_viewer_has_non_compiled_node = [](const GraphViewer& graph_viewer) -> bool {
    for (const auto& node : graph_viewer.Nodes()) {
      if (node.OpType() != "EPContext") {
        return true;
      }
    }
    return false;
  };

  // NOTE: if mode_ is kAssignOnly, nodes_to_compile will be empty at this point due to logic in PlaceNode
  // even with single node, EP might still want to compile it.
  // for example, it want to JIT an optimized kernel for LSTM with a given shape.
  if (!nodes_to_compile.empty()) {
    std::vector<NodeComputeInfo> node_compute_funcs;
    // !!! The Function style fusion is deprecated.
    if (fusion_style == IExecutionProvider::FusionStyle::Function) {
      // Create a Function based node where the fused nodes have a new Graph instance.
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, type, "The Function Style fusion is deprecated.");
    } else {
      // temporary storage for the GraphViewer for each IndexedSubGraph
      std::vector<std::unique_ptr<GraphViewer>> viewers;
      viewers.reserve(nodes_to_compile.size());
      std::vector<IExecutionProvider::FusedNodeAndGraph> nodes_and_viewers;
      nodes_and_viewers.reserve(nodes_to_compile.size());

      for (size_t j = 0, end = nodes_to_compile.size(); j < end; j++) {
        auto* node = nodes_to_compile[j];
        const auto& cur_capability = *capabilities_to_compile[j];
        auto graph_viewer = std::make_unique<GraphViewer>(graph, *cur_capability.sub_graph);

        if (disable_model_compile && graph_viewer_has_non_compiled_node(*graph_viewer)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_REQUIRES_COMPILATION, "User disabled EP compilation but EP '",
                                 type, "' needs to compile one or more nodes.");
        }

        viewers.push_back(std::move(graph_viewer));
        nodes_and_viewers.push_back(IExecutionProvider::FusedNodeAndGraph{*node, *viewers.back()});
      }

      ORT_RETURN_IF_ERROR(current_ep.Compile(nodes_and_viewers, node_compute_funcs));
      ORT_RETURN_IF(check_load_cancellation_fn(),
                    "Graph partitioning is canceled due to user request.");

      if (node_compute_funcs.size() != nodes_to_compile.size()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, type, " did not return correct number of compiled functions");
      }

      for (size_t j = 0, end = nodes_to_compile.size(); j < end; j++) {
        auto* node = nodes_to_compile[j];

        ORT_RETURN_IF_ERROR(func_mgr.AddFuncInfo(node->Name(), std::move(node_compute_funcs[j])));

        const auto& cur_capability = capabilities_to_compile[j];
        const IndexedSubGraph& indexed_sub_graph = *cur_capability->sub_graph;
        const IndexedSubGraph::MetaDef& metadef = *indexed_sub_graph.GetMetaDef();

        // create the func kernel for the name in the MetaDef. this is also the node name and that name that will
        // used as the key in the FuncManager entry. We need the registry to own the KernelCreateInfo that is
        // used by SessionState
        KernelDefBuilder builder;
        BuildFusedKernelDef(builder, metadef, type);
        ORT_RETURN_IF_ERROR(fused_kernel_registry.Register(
            builder,
            [](FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
              return FunctionKernel::Create(func_mgr, info, out);
            }));

        // now that we're done compiling we can remove the original nodes from the Graph and wire in the new one
        graph.FinalizeFuseSubGraph(indexed_sub_graph, *node);
      }
    }
  }

  if (!nodes_to_complete_fuse.empty()) {
    for (size_t j = 0, end = nodes_to_complete_fuse.size(); j < end; j++) {
      auto* node = nodes_to_complete_fuse[j];

      const auto& cur_capability = capabilities_to_complete_fuse[j];
      const IndexedSubGraph& indexed_sub_graph = *cur_capability->sub_graph;

      // now that we're done compiling we can remove the original nodes from the Graph and wire in the new one
      graph.FinalizeFuseSubGraph(indexed_sub_graph, *node);
    }
  }

  // if this is the main graph call Resolve to put the Graph back into a guaranteed good state
  // TODO: Graph::FuseSubGraph and Graph::FinalizeFuseSubGraph should now create valid edges so this call to
  // Graph::Resolve should not be required. Need to test to validate that, especially if node being fused
  // was a control flow node with its own subgraph as more than just the edges may need updating.
  if (!graph.IsSubgraph()) {
    ORT_RETURN_IF_ERROR(graph.Resolve());
  }

  // For some cases, like fp16 on cpu, right now we don't have any kernel support that.
  // But we will insert cast op to run the model, so skip the error checking here.
  // If after graph transform phase, the node still not assigned, we will report error
  // during kernel creation phase.
#ifdef COUNT_NON_CUDA_OPS
  for (auto& node : graph.Nodes()) {
    if (node.GetExecutionProviderType() != kCudaExecutionProvider &&
        node.Domain() != kMLDomain &&
        node.Domain() != kMSDomain)
      non_cuda.AddOp(node.OpType());
  }
#endif

  return Status::OK();
}

// expand any nodes that have an ONNX function definition but no matching ORT kernel
static Status InlineNodes(Graph& graph, bool& modified_graph) {
  // recurse into nested graphs first so we process from bottom up
  for (auto& node : graph.Nodes()) {
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      Graph* subgraph = entry.second;
      ORT_RETURN_IF_ERROR(InlineNodes(*subgraph, modified_graph));
    }
  }

  // See if the node with no provider can be inlined. If one such nodes can be
  // successfully inlined, we re-run the partitioner on the modified graph.
  // NOTE: Inlining the function will change the nodes in the Graph instance, so we can't do that while iterating
  // using graph.Nodes().
  InlinedVector<Node*> nodes_to_inline;
  for (auto& node : graph.Nodes()) {
    if (node.GetExecutionProviderType().empty() && node.CanBeInlined()) {
      nodes_to_inline.push_back(&node);
    }
  }

  for (auto* node : nodes_to_inline) {
    ORT_RETURN_IF_ERROR(graph.InlineFunction(*node));
    modified_graph = true;
  }

  return Status::OK();
}

static Status InlineFunctionsAOTImpl(const ExecutionProviders& execution_providers,
                                     const KernelRegistryManager& kernel_registry_mgr,
                                     Graph& graph,
                                     const GraphOptimizerRegistry& graph_optimizer_registry,
                                     const logging::Logger& logger,
                                     const CheckLoadCancellationFn& check_load_cancellation_fn,
                                     InlinedHashSet<std::string>& not_inlined,
                                     size_t& inlined_count) {
  // handle testing edge case where optimizers or constant lifting results in graph with no nodes.
  // doing it here saves all providers checking for this in GetCapability
  if (graph.NumberOfNodes() == 0) {
    return Status::OK();
  }

  for (auto& node : graph.Nodes()) {
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      Graph* subgraph = entry.second;
      // we pass through the FuncManager from the top level graph
      ORT_RETURN_IF_ERROR(InlineFunctionsAOTImpl(execution_providers,
                                                 kernel_registry_mgr,
                                                 *subgraph,
                                                 graph_optimizer_registry,
                                                 logger,
                                                 check_load_cancellation_fn,
                                                 not_inlined,
                                                 inlined_count));
    }
  }

  // Gather the candidates
  InlinedVector<NodeIndex> inline_candidates;
  for (auto& node : graph.Nodes()) {
    if (node.CanBeInlined()) {
      inline_candidates.push_back(node.Index());
    }
  }

  if (inline_candidates.empty()) {
    return Status::OK();
  }

  // Find out all the nodes that are already taken
  const GraphViewer graph_viewer(graph);

  InlinedHashSet<NodeIndex> claimed_by_ep;
  for (const auto& ep : execution_providers) {
    std::vector<std::unique_ptr<ComputeCapability>> capabilities;
    ORT_RETURN_IF_ERROR(GetCapabilityForEPForAotInlining(graph_viewer, kernel_registry_mgr, *ep,
                                                         graph_optimizer_registry, logger,
                                                         capabilities));
    if (check_load_cancellation_fn()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED, "AOT inlining is canceled due to user request.");
    }

    for (auto& capability : capabilities) {
      const auto& nodes = capability->sub_graph->nodes;
      if (nodes.size() == 1) {
        // Single node capability.
        ORT_IGNORE_RETURN_VALUE(claimed_by_ep.insert(nodes[0]));
      } else {
        // Make sure none is claimed by other EPs mirroring the logic in PartitionOnnxFormatModelImpl.
        if (std::all_of(nodes.cbegin(), nodes.cend(), [&claimed_by_ep](NodeIndex node_index) {
              return claimed_by_ep.count(node_index) == 0;
            })) {
          claimed_by_ep.insert(nodes.cbegin(), nodes.cend());
        }
      }
    }
  }

  // TODO: Insert version check. We need to collect all the versions
  // that imported by the model. If the version is not supported by
  // the model, we can not inline it.

  for (auto node_index : inline_candidates) {
    auto* node = graph.GetNode(node_index);
    if (node != nullptr) {
      if (claimed_by_ep.count(node_index) == 0) {
        ORT_RETURN_IF_ERROR(graph.InlineFunction(*node));
        ++inlined_count;
      } else {
        // OpType is the same as function name.
        auto function_id = function_utils::GetFunctionIdentifier(node->Domain(), node->OpType());
        ORT_IGNORE_RETURN_VALUE(not_inlined.insert(std::move(function_id)));
      }
    }
    if (check_load_cancellation_fn()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED, "AOT inlining is canceled due to user request.");
    }
  }

  return Status::OK();
}

// Validate the ep_context_path to make sure it is file path and check whether the file exist already
static Status GetValidatedEpContextPath(const std::filesystem::path& ep_context_path,
                                        const std::filesystem::path& model_path,
                                        std::filesystem::path& context_cache_path,
                                        bool error_if_output_file_exists = true) {
  if (!ep_context_path.empty()) {
    context_cache_path = ep_context_path;
    if (!(context_cache_path.has_filename() && context_cache_path.extension() != "")) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "context_file_path should not point to a folder.");
    }
  } else if (!model_path.empty()) {
    auto pos = model_path.native().find_last_of(ORT_TSTR("."));
    if (pos != std::string::npos) {
      context_cache_path = model_path.native().substr(0, pos) + ORT_TSTR("_ctx.onnx");
    } else {
      context_cache_path = model_path.native() + ORT_TSTR("_ctx.onnx");
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Both ep_context_path and model_path are empty.");
  }

  if (std::filesystem::exists(context_cache_path) && error_if_output_file_exists) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to generate EP context model since the file '",
                           context_cache_path, "' exists already. Please remove the EP context model if you want to re-generate it.");
  }

  return Status::OK();
}

static Status CreateEpContextModel(const ExecutionProviders& execution_providers,
                                   const Graph& graph,
                                   const EpContextModelGenerationOptions& ep_context_gen_options,
                                   const logging::Logger& logger) {
  InlinedVector<const Node*> all_ep_context_nodes;
  for (const auto& ep : execution_providers) {
    const InlinedVector<const Node*> ep_context_nodes = ep->GetEpContextNodes();
    all_ep_context_nodes.insert(all_ep_context_nodes.begin(), ep_context_nodes.begin(), ep_context_nodes.end());
  }

  if (all_ep_context_nodes.size() < 1) {
    auto action_if_no_compiled_nodes = ep_context_gen_options.action_if_no_compiled_nodes;

    ORT_RETURN_IF(action_if_no_compiled_nodes == EpContextModelGenerationOptions::ActionIfNoCompiledNodes::kReturnError,
                  "Unable to compile any nodes. Check that the session EPs support compilation and can execute "
                  "at least one subgraph in the model.");

    if (action_if_no_compiled_nodes == EpContextModelGenerationOptions::ActionIfNoCompiledNodes::kDontGenerateModel) {
      LOGS(logger, WARNING) << "Unable to compile any nodes. ONNX Runtime will not generate a compiled model. "
                               "Either the session EPs do not support compilation or the model is already compiled.";
      // Note: this path is only taken if a model is compiled with the original compilation approach that uses
      // session options configs only. The explicit compile API instead only chooses between
      // kReturnError and kGenerateModel.
      return Status::OK();
    }

    // Assert so that this is caught in a test in DEBUG builds (in case a new enum value is added)
    assert(action_if_no_compiled_nodes == EpContextModelGenerationOptions::ActionIfNoCompiledNodes::kGenerateModel);
    LOGS(logger, INFO) << "Unable to compile any nodes but will still generate an output model. "
                          "Either the session EPs do not support compilation or the model is already compiled.";
  }

  auto get_ep_context_node = [&all_ep_context_nodes](const std::string& node_name) -> std::pair<bool, const Node*> {
    for (auto& node : all_ep_context_nodes) {
      if (node_name == node->Name()) {
        return std::make_pair(true, node);
      }
    }
    return std::make_pair(false, static_cast<const Node*>(nullptr));
  };

  bool saving_to_buffer = ep_context_gen_options.output_model_buffer_ptr != nullptr &&
                          ep_context_gen_options.output_model_buffer_size_ptr != nullptr &&
                          ep_context_gen_options.output_model_buffer_allocator != nullptr;

  std::filesystem::path context_cache_path;
  if (!saving_to_buffer || !graph.ModelPath().empty()) {
    ORT_RETURN_IF_ERROR(GetValidatedEpContextPath(ep_context_gen_options.output_model_file_path,
                                                  graph.ModelPath(),
                                                  context_cache_path,
                                                  ep_context_gen_options.error_if_output_file_exists));
  }

  // Utility function to detect a fused node with an unsupported domain.
  // Ex: when compiling an already compiled model, an EPContext node in the input model would be wrapped
  // into a fused node with a domain like "QNN". Such fused nodes do not pass ONNX correctness checks, so
  // we should detect them here and return a better error message. Otherwise, an ORT_INVALID_GRAPH error is raised
  // with a confusing error message *after* the invalid model has been saved/generated.
  // Note: This only applies to the explicit compile API. The original compilation approach (via session options),
  // early exits above (without error) if the model is already compiled.
  auto is_invalid_fused_node = [&graph](const Node& node) {
    const std::unordered_map<std::string, int>& supported_domains = graph.DomainToVersionMap();
    return (node.NodeType() == Node::Type::Fused) && (supported_domains.find(node.Domain()) == supported_domains.end());
  };

  Model ep_context_model(graph.Name(), false, graph.GetModel().MetaData(),
                         graph.GetModel().ModelPath(),  // use source model path so that external initializers can find the data file path
                         IOnnxRuntimeOpSchemaRegistryList{graph.GetSchemaRegistry()},
                         graph.DomainToVersionMap(), {}, logger);
  auto& ep_graph = ep_context_model.MainGraph();
  ep_graph.SetDescription(graph.Description());

  // Set inputs outputs explicitly to make sure the order is same as the user model.
  auto inputs = graph.GetInputs();
  auto outputs = graph.GetOutputs();

  InlinedVector<const NodeArg*> ep_graph_inputs;
  ep_graph_inputs.reserve(inputs.size());
  for (auto& input : inputs) {
    auto input_arg = graph.GetNodeArg(input->Name());
    auto& ep_graph_input_arg = ep_graph.GetOrCreateNodeArg(input_arg->Name(), input_arg->TypeAsProto());
    ep_graph_inputs.push_back(&ep_graph_input_arg);
  }

  InlinedVector<const NodeArg*> ep_graph_outputs;
  ep_graph_outputs.reserve(outputs.size());
  for (auto& output : outputs) {
    auto output_arg = graph.GetNodeArg(output->Name());
    auto& ep_graph_output_arg = ep_graph.GetOrCreateNodeArg(output_arg->Name(), output_arg->TypeAsProto());
    ep_graph_outputs.push_back(&ep_graph_output_arg);
  }

  ep_graph.SetInputs(ep_graph_inputs);
  ep_graph.SetOutputs(ep_graph_outputs);

  for (const auto& node : graph.Nodes()) {
    // the fused node and EPContext node has same node name
    auto ep_context_node = get_ep_context_node(node.Name());
    // Use EpContext node created by the EPs if name matched, otherwise use node from original model
    if (ep_context_node.first) {
      ep_graph.AddNode(*ep_context_node.second);
    } else if (is_invalid_fused_node(node)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH, "Encountered an invalid node while compiling a model. ",
                             "Please ensure the input model is not already compiled.");
    } else {
      ep_graph.AddNode(node);
    }
  }

  // handle initializers
  for (const auto& [name, _] : graph.GetAllInitializedTensors()) {
    if (ep_graph.GetNodeArg(name) != nullptr) {
      graph_utils::MakeInitializerCopyIfNotExist(graph, ep_graph, name);
    }
  }

  size_t ini_size_threshold = ep_context_gen_options.output_external_initializer_size_threshold;
  std::filesystem::path external_ini_path = ep_context_gen_options.output_external_initializers_file_path;
  bool force_embed_external_ini = false;
  if (external_ini_path.empty()) {
    // if no external ini file specified, set force_embed_external_ini to true to avoid intermedia file creation
    // and force all initializers embed into the Onnx file
    ini_size_threshold = SIZE_MAX;
    force_embed_external_ini = true;
  }

  ModelSavingOptions model_saving_options{ini_size_threshold};
  model_saving_options.force_embed_external_ini = force_embed_external_ini;

  if (saving_to_buffer) {
    ORT_RETURN_IF_ERROR(ep_context_model.MainGraph().Resolve());
    // TODO(adrianlizarraga): Investigate if we can make this more memory efficient.
    // May be able to use allocator to directly allocate the ModelProto to avoid a copy.
    ONNX_NAMESPACE::ModelProto model_proto = ep_context_model.ToGraphProtoWithExternalInitializers(external_ini_path,
                                                                                                   context_cache_path,
                                                                                                   model_saving_options);
    size_t buffer_size = model_proto.ByteSizeLong();
    ORT_RETURN_IF(buffer_size > static_cast<size_t>(std::numeric_limits<int>::max()),
                  "Cannot serialize ONNX ModelProto larger than 2GB");

    AllocatorPtr allocator = ep_context_gen_options.output_model_buffer_allocator;
    IAllocatorUniquePtr<void> buffer = IAllocator::MakeUniquePtr<void>(allocator, buffer_size);
    model_proto.SerializeToArray(buffer.get(), static_cast<int>(buffer_size));

    *ep_context_gen_options.output_model_buffer_size_ptr = buffer_size;
    *ep_context_gen_options.output_model_buffer_ptr = buffer.release();
  } else {
    ORT_RETURN_IF_ERROR(Model::SaveWithExternalInitializers(ep_context_model, context_cache_path,
                                                            external_ini_path, model_saving_options));
  }

  return Status::OK();
}

static Status PartitionOnnxFormatModel(const PartitionParams& partition_params, GraphPartitioner::Mode mode,
                                       const ExecutionProviders& execution_providers,
                                       KernelRegistryManager& kernel_registry_manager,
                                       const std::optional<ResourceAccountantMap>& acc_map,
                                       const GraphOptimizerRegistry& graph_optimizer_registry,
                                       const logging::Logger& logger, bool disable_model_compile) {
  bool modified_graph = false;

  auto& graph = partition_params.graph.get();
  auto& func_mgr = partition_params.func_mgr.get();
  auto& fused_kernel_registry = partition_params.fused_kernel_registry.get();
  auto& fused_node_unique_id = partition_params.fused_node_unique_id.get();
  const auto& transform_layout_function = partition_params.transform_layout_function;
  const CheckLoadCancellationFn& check_load_cancellation_fn = partition_params.check_load_cancellation_fn;

  do {
    // process full graph with each EP
    for (const auto& ep : execution_providers) {
      IResourceAccountant* resource_accountant = nullptr;
      if (acc_map.has_value()) {
        auto hit = acc_map->find(ep->Type());
        if (hit != acc_map->end()) {
          resource_accountant = hit->second.get();
        }
      }
      ORT_RETURN_IF_ERROR(PartitionOnnxFormatModelImpl(graph, func_mgr, kernel_registry_manager,
                                                       fused_kernel_registry, *ep, mode, fused_node_unique_id,
                                                       transform_layout_function,
                                                       partition_params.debug_graph_fn,
                                                       check_load_cancellation_fn,
                                                       logger, resource_accountant, graph_optimizer_registry,
                                                       disable_model_compile));
    }

    // expand any nodes that have an ONNX function definition but no matching ORT kernel.
    modified_graph = false;
    ORT_RETURN_IF_ERROR(InlineNodes(graph, modified_graph));

    // Resolve and rerun graph partitioning and inlining if there was a change
    if (modified_graph) {
      ORT_RETURN_IF_ERROR(graph.Resolve());
    }
  } while (modified_graph);

  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

static Status PartitionOrtFormatModelImpl(const PartitionParams& partition_params,
                                          KernelRegistryManager& kernel_registry_mgr,
                                          IExecutionProvider& current_ep,
                                          const GraphOptimizerRegistry& graph_optimizer_registry,
                                          const logging::Logger& logger) {
  // handle testing edge case where optimizers or constant lifting results in graph with no nodes.
  // doing it here saves all providers checking for this in GetCapability
  auto& graph = partition_params.graph.get();
  if (graph.NumberOfNodes() == 0) {
    return Status::OK();
  }

  // recurse into nested graphs first to partition bottom up.
  for (auto& node : graph.Nodes()) {
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      auto& subgraph = *entry.second;
      PartitionParams subgraph_partition_params = partition_params;
      subgraph_partition_params.graph = std::ref(subgraph);
      ORT_RETURN_IF_ERROR(PartitionOrtFormatModelImpl(subgraph_partition_params, kernel_registry_mgr,
                                                      current_ep, graph_optimizer_registry, logger));
    }
  }

  std::vector<std::unique_ptr<ComputeCapability>> capabilities;
  // clang-format off
  const auto get_capability_params = GetCapabilityForEPParams{
      std::ref(graph),
      std::cref(kernel_registry_mgr),
      std::ref(current_ep),
      std::ref(capabilities),
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
      GraphPartitioner::Mode::kOrtFormatLoad,
      std::cref(partition_params.transform_layout_function),
      std::cref(partition_params.debug_graph_fn),
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
      nullptr,
      std::ref(graph_optimizer_registry),
      partition_params.check_load_cancellation_fn
  };
  // clang-format on

  ORT_RETURN_IF_ERROR(GetCapabilityForEP(get_capability_params, logger));
  if (capabilities.empty()) {
    return Status::OK();
  }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  struct CompilationEntry {
    std::unique_ptr<GraphViewer> viewer;
    std::reference_wrapper<Node> fused_node;
    std::reference_wrapper<const ComputeCapability> capability;
  };
  std::vector<CompilationEntry> compilation_entries;
  compilation_entries.reserve(capabilities.size());
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

  const std::string& type = current_ep.Type();
  for (const auto& capability : capabilities) {
    const IndexedSubGraph& indexed_sub_graph = *capability->sub_graph;
    const IndexedSubGraph::MetaDef* metadef = indexed_sub_graph.GetMetaDef();
    if (!metadef) {
      TryAssignSingleNode(graph, indexed_sub_graph, type);
    } else {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
      std::ostringstream oss;
      oss << type << "_" << metadef->name << "_" << partition_params.fused_node_unique_id++;
      const std::string node_name = oss.str();

      Node& fused_node = graph.BeginFuseSubGraph(indexed_sub_graph, node_name);
      fused_node.SetExecutionProviderType(type);
      if (indexed_sub_graph.IsAccountingEnabled()) {
        indexed_sub_graph.ComputeAndAccountForNode(fused_node);
      }

      // create filtered graph viewer for this set of nodes
      //
      // TODO: Could avoid the topological sort in the GraphViewer ctor by constructing from an existing
      // GraphViewer instance instead of the Graph (copying the topological order instead of recalculating).
      auto viewer = std::make_unique<GraphViewer>(graph, indexed_sub_graph);
      compilation_entries.push_back(CompilationEntry{std::move(viewer), fused_node, *capability});
#else   // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Compiling capabilities is not supported in this build.");
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
    }
  }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  // We will compile the fused nodes one by one, and fuse the subgraph if successful.
  for (const auto& compilation_entry : compilation_entries) {
    const bool acc_enabled = compilation_entry.capability.get().sub_graph->IsAccountingEnabled();
    Node& node = compilation_entry.fused_node;
    std::vector<NodeComputeInfo> single_node_compute_func;
    ORT_RETURN_IF_ERROR(current_ep.Compile({IExecutionProvider::FusedNodeAndGraph{node, *compilation_entry.viewer}},
                                           single_node_compute_func));
    if (partition_params.check_load_cancellation_fn()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED, "Graph partitioning is canceled due to user request.");
    }

    ORT_RETURN_IF(single_node_compute_func.empty(), "single_node_compute_func should have 1 element.");
    auto& func_mgr = partition_params.func_mgr.get();
    ORT_RETURN_IF_ERROR(func_mgr.AddFuncInfo(node.Name(), std::move(single_node_compute_func[0])));

    const ComputeCapability& cur_capability = compilation_entry.capability;
    const IndexedSubGraph& indexed_sub_graph = *cur_capability.sub_graph;
    const IndexedSubGraph::MetaDef& metadef = *indexed_sub_graph.GetMetaDef();

    KernelDefBuilder builder;
    BuildFusedKernelDef(builder, metadef, type);
    auto kernel_def = builder.Build();

    auto& fused_kernel_registry = partition_params.fused_kernel_registry.get();
    ORT_RETURN_IF_ERROR(fused_kernel_registry.Register(
        KernelCreateInfo(
            std::move(kernel_def),
            [](FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
              return FunctionKernel::Create(func_mgr, info, out);
            })));

    // now that we're done compiling we can remove the original nodes from the Graph and wire in the new one
    graph.FinalizeFuseSubGraph(indexed_sub_graph, node);
    if (acc_enabled) {
      compilation_entry.capability.get().sub_graph->ComputeAndAccountForNode(node);
    }
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

  return Status::OK();
}

// Simplified partitioning where custom EPs may produce compiled nodes.
static Status PartitionOrtFormatModel(const PartitionParams& partition_params,
                                      const ExecutionProviders& execution_providers,
                                      KernelRegistryManager& kernel_registry_manager,
                                      const GraphOptimizerRegistry& graph_optimizer_registry,
                                      const logging::Logger& logger) {
  // process full graph with each EP
  for (const auto& ep : execution_providers) {
    ORT_RETURN_IF_ERROR(PartitionOrtFormatModelImpl(partition_params, kernel_registry_manager, *ep, graph_optimizer_registry, logger));
  }

  return Status::OK();
}

#ifndef ORT_MINIMAL_BUILD

Status GraphPartitioner::InlineFunctionsAOT(Model& model,
                                            const ExecutionProviders& execution_providers,
                                            const KernelRegistryManager& kernel_registry_manager,
                                            const logging::Logger& logger) const {
  const auto local_functions_num = model.GetModelLocalFunctionTemplates().size();
  const bool is_there_local_functions = local_functions_num > 0;

  if (!is_there_local_functions) {
    LOGS(logger, INFO) << "This model does not have any local functions defined. AOT Inlining is not performed";
    return Status::OK();
  }

  auto check_load_cancellation_fn = [this]() -> bool { return IsLoadCancellationFlagSet(); };

  auto& graph = model.MainGraph();
  InlinedHashSet<std::string> not_inlined;
  do {
    size_t inlined_count = 0;
    ORT_RETURN_IF_ERROR(InlineFunctionsAOTImpl(execution_providers,
                                               kernel_registry_manager,
                                               graph,
                                               *graph_optimizer_registry_,
                                               logger,
                                               check_load_cancellation_fn,
                                               not_inlined,
                                               inlined_count));

    if (inlined_count == 0) {
      break;
    }
    ORT_RETURN_IF_ERROR(graph.Resolve());
  } while (true);

  model.RemoveLocalFunctionsProtos(not_inlined);

  LOGS(logger, INFO)
      << "AOT inlining completed. (" << (local_functions_num - model.GetModelLocalFunctionTemplates().size())
      << ") functions of ("
      << local_functions_num
      << ") pruned.";

  return Status::OK();
}

#endif

Status GraphPartitioner::Partition(Graph& graph, FuncManager& func_mgr,
                                   const layout_transformation::TransformLayoutFunction& transform_layout_function,
                                   const ConfigOptions& config_options,
                                   const logging::Logger& logger,
                                   Mode mode,
                                   const EpContextModelGenerationOptions& ep_context_gen_options,
                                   const layout_transformation::DebugGraphFn& debug_graph_fn) const {
  // It is a greedy partitioning algorithm per provider preferences user provided when calling ONNX RUNTIME right now.
  // 1. Execution providers' capabilities are checked one by one.
  // 2. All sub-graphs that an execution provider returns will be assigned to it if it's not assigned yet.
  //    NOTE: A 'sub-graph' is a subset of nodes within the current Graph instance.
  //          The control flow nodes have nested Graph instance/s which are also called subgraphs,
  //          but are completely separate Graph instances and not a subset of nodes within a single Graph instance.
  // 3. CPU execution provider is expected to be able to run any node and is the last one in execution provider
  //    preference.
  if (providers_.Empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "No provider specified.");
  }

  CheckLoadCancellationFn check_load_cancellation_fn = [this]() -> bool { return IsLoadCancellationFlagSet(); };

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  // fused_kernel_registry is preparing the kernels created on the fly for fused sub graph.
  // It is only visible for current session.
  std::shared_ptr<KernelRegistry> fused_kernel_registry = std::make_shared<KernelRegistry>();

  // we make sure each fused node name is unique across the entire model for clarity
  int fused_node_unique_id = 0;

  PartitionParams partition_params{
      std::ref(graph),
      std::cref(check_load_cancellation_fn),
      std::ref(func_mgr),
      std::ref(*fused_kernel_registry),
      std::ref(fused_node_unique_id),
      std::cref(transform_layout_function),
      std::cref(debug_graph_fn)};

#else  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

  ORT_UNUSED_PARAMETER(func_mgr);
  ORT_UNUSED_PARAMETER(transform_layout_function);
  ORT_UNUSED_PARAMETER(debug_graph_fn);
  PartitionParams partition_params{
      std::ref(graph),
      std::cref(check_load_cancellation_fn),
  };

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

  if (mode == Mode::kNormal || mode == Mode::kAssignOnly) {
#if !defined(ORT_MINIMAL_BUILD)
    if (ep_context_gen_options.enable && ep_context_gen_options.output_model_buffer_ptr == nullptr) {
      // Check before EP compile graphs
      std::filesystem::path context_cache_path;
      ORT_RETURN_IF_ERROR(GetValidatedEpContextPath(ep_context_gen_options.output_model_file_path, graph.ModelPath(),
                                                    context_cache_path,
                                                    ep_context_gen_options.error_if_output_file_exists));
    }

    // We use this only if Resource Aware Partitioning is enabled for any of the EPs
    // The map is empty if not created if not enabled
    std::optional<ResourceAccountantMap> ep_acc_map;
    ORT_RETURN_IF_ERROR(NodeStatsRecorder::CreateAccountants(config_options, graph.ModelPath(), ep_acc_map));

    bool disable_model_compile = config_options.GetConfigOrDefault(kOrtSessionOptionsDisableModelCompile, "0") == "1";
    ORT_RETURN_IF_ERROR(PartitionOnnxFormatModel(partition_params, mode, providers_, kernel_registry_mgr_,
                                                 ep_acc_map, *graph_optimizer_registry_, logger,
                                                 disable_model_compile));

    if (ep_context_gen_options.enable) {
      ORT_RETURN_IF_ERROR(CreateEpContextModel(providers_, graph, ep_context_gen_options, logger));
    }
#else
    ORT_UNUSED_PARAMETER(config_options);
    ORT_UNUSED_PARAMETER(logger);
    ORT_UNUSED_PARAMETER(ep_context_gen_options);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ONNX models are not supported in this build.");
#endif  //! defined(ORT_MINIMAL_BUILD)
  } else {
    ORT_RETURN_IF_ERROR(PartitionOrtFormatModel(partition_params, providers_, kernel_registry_mgr_, *graph_optimizer_registry_, logger));
  }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  if (!fused_kernel_registry->IsEmpty()) {
    kernel_registry_mgr_.RegisterKernelRegistry(fused_kernel_registry);
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

  return Status::OK();
}

}  // namespace onnxruntime
