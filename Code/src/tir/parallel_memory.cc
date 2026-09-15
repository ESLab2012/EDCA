/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tir/parallel_memory.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <climits>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace neutvm {
namespace tir {
namespace cpu {

namespace {

using tvm::tir::AllocateConstNode;
using tvm::tir::AllocateNode;
using tvm::tir::BufferLoadNode;
using tvm::tir::BufferStoreNode;
using tvm::tir::Call;
using tvm::tir::CallNode;
using tvm::tir::PrimFunc;
using tvm::tir::Stmt;
using tvm::tir::StmtExprVisitor;
using tvm::tir::StringImm;
using tvm::tir::Var;
using tvm::tir::VarNode;
using tvm::tir::usmp::BufferInfo;

struct CallBufferInfo {
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

struct WorkspaceInfo {
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<std::string> workspaces;
};

using WorkspaceNameMap = std::unordered_map<const VarNode*, std::string>;

class RWVisitor : public StmtExprVisitor {
 public:
  RWVisitor(const Map<Var, tvm::tir::Buffer>& buffer_map, const Array<Var>& params) {
    for (size_t i = 0; i < params.size(); ++i) {
      auto it = buffer_map.find(params[i]);
      if (it == buffer_map.end()) {
        continue;
      }
      const VarNode* data = (*it).second->data.get();
      data_to_param_index_[data] = i;
      param_indices_.push_back(i);
    }
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    auto it = data_to_param_index_.find(op->buffer->data.get());
    if (it != data_to_param_index_.end() && output_indices_.count(it->second) == 0) {
      input_indices_.insert(it->second);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    auto it = data_to_param_index_.find(op->buffer->data.get());
    if (it != data_to_param_index_.end()) {
      output_indices_.insert(it->second);
      input_indices_.erase(it->second);
    } else if (temp_output_nodes_.count(op->buffer->data.get()) == 0) {
      temp_output_nodes_.insert(op->buffer->data.get());
      temp_output_nodes_ordered_.push_back(op->buffer->data.get());
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  std::vector<size_t> InputIndices() const { return OrderedIndices(input_indices_); }

  std::vector<size_t> OutputIndices() const { return OrderedIndices(output_indices_); }

  const std::vector<const VarNode*>& TempOutputNodes() const { return temp_output_nodes_ordered_; }

 private:
  std::vector<size_t> OrderedIndices(const std::unordered_set<size_t>& indices) const {
    std::vector<size_t> ordered;
    for (size_t param_index : param_indices_) {
      if (indices.count(param_index) != 0) {
        ordered.push_back(param_index);
      }
    }
    return ordered;
  }

  std::unordered_map<const VarNode*, size_t> data_to_param_index_;
  std::vector<size_t> param_indices_;
  std::unordered_set<size_t> input_indices_;
  std::unordered_set<size_t> output_indices_;
  std::unordered_set<const VarNode*> temp_output_nodes_;
  std::vector<const VarNode*> temp_output_nodes_ordered_;
};

class CallBufferCollector : public StmtExprVisitor {
 public:
  explicit CallBufferCollector(Map<String, PrimFunc> functions,
                               WorkspaceNameMap managed_workspace_names)
      : functions_(std::move(functions)),
        managed_workspace_names_(std::move(managed_workspace_names)) {}

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(tvm::tir::builtin::call_extern()) ||
        op->op.same_as(tvm::tir::builtin::tvm_call_cpacked())) {
      StringImm func_name = Downcast<StringImm>(op->args[0]);
      auto it = functions_.find(func_name->value);
      if (it != functions_.end()) {
        auto actual_args = Array<PrimExpr>(op->args.begin() + 1, op->args.end());
        RecordCall((*it).second, actual_args);
      }
      return;
    }

    if (op->op->IsInstance<tvm::tir::PrimFuncNode>()) {
      RecordCall(Downcast<PrimFunc>(op->op), op->args);
      return;
    }

    StmtExprVisitor::VisitExpr_(op);
  }

  std::vector<CallBufferInfo> call_buffers;
  std::vector<WorkspaceInfo> output_to_workspace;

 private:
  std::optional<std::string> ArgNameAt(const Array<PrimExpr>& args, size_t index) const {
    if (index >= args.size()) {
      return std::nullopt;
    }
    if (const auto* var = args[index].as<VarNode>()) {
      return var->name_hint;
    }
    return std::nullopt;
  }

  void RecordCall(const PrimFunc& func, const Array<PrimExpr>& actual_args) {
    RWVisitor rw(func->buffer_map, func->params);
    rw(func->body);

    CallBufferInfo call_info;
    for (size_t index : rw.InputIndices()) {
      if (auto name = ArgNameAt(actual_args, index)) {
        call_info.inputs.push_back(name.value());
      }
    }
    for (size_t index : rw.OutputIndices()) {
      if (auto name = ArgNameAt(actual_args, index)) {
        call_info.outputs.push_back(name.value());
      }
    }

    WorkspaceInfo workspace_info;
    workspace_info.inputs = call_info.inputs;
    workspace_info.outputs = call_info.outputs;
    for (const VarNode* node : rw.TempOutputNodes()) {
      auto it = managed_workspace_names_.find(node);
      if (it != managed_workspace_names_.end()) {
        workspace_info.workspaces.push_back(it->second);
      }
    }

    if (!call_info.inputs.empty() || !call_info.outputs.empty()) {
      call_buffers.push_back(std::move(call_info));
      output_to_workspace.push_back(std::move(workspace_info));
    }
  }

  Map<String, PrimFunc> functions_;
  WorkspaceNameMap managed_workspace_names_;
};

WorkspaceNameMap CollectManagedWorkspaceNames(const Map<BufferInfo, Stmt>& buffer_info_map) {
  WorkspaceNameMap names;
  for (const auto& kv : buffer_info_map) {
    // PrimFunc-local names are not globally unique. Use the Allocate Var shared
    // by BufferStore and BufferInfo instead of reconstructing names by count.
    if (const auto* allocate = kv.second.as<AllocateNode>()) {
      names[allocate->buffer_var.get()] = kv.first->name_hint;
    }
  }
  return names;
}

std::unordered_map<std::string, std::unordered_set<int>> AssignBufferDepths(
    const std::vector<CallBufferInfo>& call_buffers) {
  std::unordered_map<std::string, int> temp_buffer_depths;
  std::queue<std::string> to_process;

  for (const CallBufferInfo& call : call_buffers) {
    for (const std::string& input : call.inputs) {
      if (temp_buffer_depths.find(input) == temp_buffer_depths.end()) {
        temp_buffer_depths[input] = -1;
        to_process.push(input);
      }
    }
  }

  while (!to_process.empty()) {
    std::string current = to_process.front();
    to_process.pop();

    for (const CallBufferInfo& call : call_buffers) {
      if (std::find(call.inputs.begin(), call.inputs.end(), current) == call.inputs.end()) {
        continue;
      }
      int new_depth = temp_buffer_depths[current] + 1;
      for (const std::string& output : call.outputs) {
        temp_buffer_depths[output] = std::max(new_depth, temp_buffer_depths[output]);
      }
    }
  }

  std::unordered_map<std::string, std::unordered_set<int>> buffer_depths;
  for (const CallBufferInfo& call : call_buffers) {
    for (const std::string& input : call.inputs) {
      if (buffer_depths.find(input) == buffer_depths.end()) {
        buffer_depths[input].insert(temp_buffer_depths[input]);
      }
      for (const std::string& output : call.outputs) {
        buffer_depths[input].insert(temp_buffer_depths[output] - 1);
      }
    }
  }

  for (auto& kv : buffer_depths) {
    if (kv.second.empty()) {
      continue;
    }
    auto range = std::minmax_element(kv.second.begin(), kv.second.end());
    for (int depth = *range.first; depth <= *range.second; ++depth) {
      kv.second.insert(depth);
    }
  }

  return buffer_depths;
}

std::unordered_map<std::string, std::unordered_set<int>> AssignWorkspaceDepths(
    const std::unordered_map<std::string, std::unordered_set<int>>& buffer_depths,
    const std::vector<WorkspaceInfo>& output_to_workspace) {
  std::unordered_map<std::string, std::unordered_set<int>> workspace_depths;

  for (const WorkspaceInfo& info : output_to_workspace) {
    for (const std::string& workspace : info.workspaces) {
      int min_output_depth = INT_MAX;
      for (const std::string& output : info.outputs) {
        auto it = buffer_depths.find(output);
        if (it == buffer_depths.end() || it->second.empty()) {
          continue;
        }
        min_output_depth = std::min(min_output_depth, *std::min_element(it->second.begin(),
                                                                        it->second.end()));
      }
      if (min_output_depth != INT_MAX) {
        // A PrimFunc-local workspace can be referenced by more than one call
        // site when TVM reuses the same PrimFunc. Each call keeps its own
        // execution depth, so retain every observed depth.
        workspace_depths[workspace].insert(min_output_depth);
        continue;
      }

      int max_input_depth = INT_MIN;
      for (const std::string& input : info.inputs) {
        auto it = buffer_depths.find(input);
        if (it == buffer_depths.end() || it->second.empty()) {
          continue;
        }
        max_input_depth =
            std::max(max_input_depth, *std::max_element(it->second.begin(), it->second.end()));
      }
      if (max_input_depth != INT_MIN) {
        workspace_depths[workspace].insert(max_input_depth + 1);
      }
    }
  }

  return workspace_depths;
}

std::unordered_map<int, std::vector<std::string>> InvertBufferDepths(
    const std::unordered_map<std::string, std::unordered_set<int>>& depths) {
  std::unordered_map<int, std::vector<std::string>> inverted;
  for (const auto& kv : depths) {
    for (int depth : kv.second) {
      auto& names = inverted[depth];
      if (std::find(names.begin(), names.end(), kv.first) == names.end()) {
        names.push_back(kv.first);
      }
    }
  }
  return inverted;
}

std::unordered_map<int, std::vector<std::string>> InvertWorkspaceDepths(
    const std::unordered_map<std::string, std::unordered_set<int>>& depths) {
  std::unordered_map<int, std::vector<std::string>> inverted;
  for (const auto& kv : depths) {
    for (int depth : kv.second) {
      auto& names = inverted[depth];
      if (std::find(names.begin(), names.end(), kv.first) == names.end()) {
        names.push_back(kv.first);
      }
    }
  }
  return inverted;
}

bool HasConflict(const BufferInfo& lhs, const BufferInfo& rhs) {
  return std::find_if(lhs->conflicts.begin(), lhs->conflicts.end(), [&](const ObjectRef& conflict) {
           return conflict.same_as(rhs);
         }) != lhs->conflicts.end();
}

void AddDirectedConflict(const BufferInfo& lhs, const BufferInfo& rhs) {
  if (!lhs.defined() || !rhs.defined() || lhs.same_as(rhs) || HasConflict(lhs, rhs)) {
    return;
  }
  lhs->conflicts.push_back(rhs);
}

void AddConflict(const BufferInfo& lhs, const BufferInfo& rhs) {
  AddDirectedConflict(lhs, rhs);
  AddDirectedConflict(rhs, lhs);
}

void AddConflictsByNames(const std::string& lhs_name, const std::vector<std::string>& rhs_names,
                         const std::unordered_map<std::string, BufferInfo>& name_to_buffer) {
  auto lhs_it = name_to_buffer.find(lhs_name);
  if (lhs_it == name_to_buffer.end()) {
    return;
  }
  for (const std::string& rhs_name : rhs_names) {
    auto rhs_it = name_to_buffer.find(rhs_name);
    if (rhs_it != name_to_buffer.end()) {
      AddConflict(lhs_it->second, rhs_it->second);
    }
  }
}

void ApplyParallelConflicts(
    const std::unordered_map<std::string, std::unordered_set<int>>& buffer_depths,
    const std::unordered_map<std::string, std::unordered_set<int>>& workspace_depths,
    const std::unordered_map<std::string, BufferInfo>& name_to_buffer) {
  auto depth_to_buffer = InvertBufferDepths(buffer_depths);
  auto depth_to_workspace = InvertWorkspaceDepths(workspace_depths);

  for (const auto& kv : buffer_depths) {
    for (int depth : kv.second) {
      AddConflictsByNames(kv.first, depth_to_buffer[depth - 1], name_to_buffer);
      AddConflictsByNames(kv.first, depth_to_buffer[depth], name_to_buffer);
      AddConflictsByNames(kv.first, depth_to_workspace[depth], name_to_buffer);
      AddConflictsByNames(kv.first, depth_to_buffer[depth + 1], name_to_buffer);
      AddConflictsByNames(kv.first, depth_to_workspace[depth + 1], name_to_buffer);
    }
  }

  for (const auto& kv : workspace_depths) {
    for (int depth : kv.second) {
      AddConflictsByNames(kv.first, depth_to_workspace[depth], name_to_buffer);
      AddConflictsByNames(kv.first, depth_to_buffer[depth - 1], name_to_buffer);
      AddConflictsByNames(kv.first, depth_to_buffer[depth], name_to_buffer);
    }
  }
}

void FinalizeConstantConflicts(Map<BufferInfo, Stmt>* buffer_info_map) {
  Array<BufferInfo> vars;
  Array<BufferInfo> constants;
  for (const auto& kv : *buffer_info_map) {
    const Stmt& stmt = kv.second;
    if (stmt->IsInstance<AllocateConstNode>()) {
      constants.push_back(kv.first);
    } else {
      vars.push_back(kv.first);
    }
  }

  Map<ObjectRef, ObjectRef> constant_lookup;
  for (const BufferInfo& buffer : constants) {
    constant_lookup.Set(buffer, buffer);
    Array<ObjectRef> conflicts;
    for (const BufferInfo& other : constants) {
      if (!buffer.same_as(other)) {
        conflicts.push_back(other);
      }
    }
    buffer->conflicts.Assign(conflicts.begin(), conflicts.end());
  }

  for (const BufferInfo& buffer : vars) {
    Array<ObjectRef> conflicts;
    std::copy_if(buffer->conflicts.begin(), buffer->conflicts.end(), std::back_inserter(conflicts),
                 [&constant_lookup](const ObjectRef& conflict) {
                   return constant_lookup.find(conflict) == constant_lookup.end();
                 });
    buffer->conflicts.Assign(conflicts.begin(), conflicts.end());
  }
}

Map<String, PrimFunc> CollectPrimFunctions(const IRModule& mod) {
  Map<String, PrimFunc> functions;
  for (const auto& gv_func : mod->functions) {
    if (gv_func.second->IsInstance<tvm::tir::PrimFuncNode>()) {
      functions.Set(gv_func.first->name_hint, Downcast<PrimFunc>(gv_func.second));
    }
  }
  return functions;
}

Map<String, Array<Integer>> ConvertDepthSetMap(
    const std::unordered_map<std::string, std::unordered_set<int>>& depths) {
  Map<String, Array<Integer>> result;
  for (const auto& kv : depths) {
    std::vector<int> ordered(kv.second.begin(), kv.second.end());
    std::sort(ordered.begin(), ordered.end());
    Array<Integer> values;
    for (int value : ordered) {
      values.push_back(Integer(value));
    }
    result.Set(String(kv.first), values);
  }
  return result;
}

Map<String, ObjectRef> DebugAnalyzeParallelMemory(const PrimFunc& main_func, const IRModule& mod,
                                                  const Map<BufferInfo, Stmt>& buffer_info_map) {
  CallBufferCollector collector(CollectPrimFunctions(mod),
                                CollectManagedWorkspaceNames(buffer_info_map));
  collector(main_func->body);
  auto buffer_depths = AssignBufferDepths(collector.call_buffers);
  auto workspace_depths = AssignWorkspaceDepths(buffer_depths, collector.output_to_workspace);

  Map<String, ObjectRef> result;
  result.Set("buffer_depths", ConvertDepthSetMap(buffer_depths));
  result.Set("workspace_depths", ConvertDepthSetMap(workspace_depths));
  result.Set("call_count", Integer(static_cast<int>(collector.call_buffers.size())));
  return result;
}

}  // namespace

void ApplyParallelMemoryAnalysis(ParallelMemoryAnalysisContext* ctx) {
  ICHECK(ctx != nullptr);
  ICHECK(ctx->main_func.defined());
  ICHECK(ctx->buffer_info_map != nullptr);

  if (ctx->clear_existing_conflicts) {
    for (const auto& kv : *ctx->buffer_info_map) {
      kv.first->conflicts = Array<ObjectRef>();
    }
  }

  CallBufferCollector collector(ctx->functions,
                                CollectManagedWorkspaceNames(*ctx->buffer_info_map));
  collector(ctx->main_func->body);
  auto buffer_depths = AssignBufferDepths(collector.call_buffers);
  auto workspace_depths = AssignWorkspaceDepths(buffer_depths, collector.output_to_workspace);

  std::unordered_map<std::string, BufferInfo> name_to_buffer;
  for (const auto& kv : *ctx->buffer_info_map) {
    name_to_buffer[kv.first->name_hint] = kv.first;
  }

  ApplyParallelConflicts(buffer_depths, workspace_depths, name_to_buffer);
  FinalizeConstantConflicts(ctx->buffer_info_map);

  if (ctx->memory_pressure_bytes != nullptr) {
    *ctx->memory_pressure_bytes = 0;
  }
}

TVM_REGISTER_GLOBAL("tir.usmp.analysis.neutvm.cpu.debug_analyze_parallel_memory")
    .set_body_typed(DebugAnalyzeParallelMemory);

}  // namespace cpu
}  // namespace tir
}  // namespace neutvm
}  // namespace tvm
