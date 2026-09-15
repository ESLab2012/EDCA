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

#include <tvm/ir/module.h>
#include <tvm/neutvm/config.h>
#include <relay/conv2d.h>
#include <relay/split_attrs.h>
#include <relay/split_rewrite.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace neutvm {
namespace relay {
namespace cpu {

namespace {

using ::tvm::relay::Call;
using ::tvm::relay::CallNode;
using ::tvm::relay::Expr;
using ::tvm::relay::Function;
using ::tvm::relay::FunctionNode;
using ::tvm::relay::Let;
using ::tvm::relay::LetNode;
using ::tvm::relay::OpNode;
using ::tvm::relay::TensorTypeNode;
using ::tvm::relay::Var;
using ::tvm::relay::VarNode;

struct CandidateFunction {
  Function func;
  SplitAttrs attrs;
  LetBinding binding;
  bool emitted{false};
  bool consumed{false};
};

struct DirectCallInfo {
  Call call;
  const VarNode* source_var{nullptr};
};

struct Conv2DFilterStats {
  int64_t checks{0};
  int64_t accepted{0};
  int64_t below_threshold{0};
  int64_t unknown_shape{0};
};

bool RecordConv2DFilterDecision(const CallNode* call_node, Conv2DFilterStats* stats) {
  int64_t macs = 0;
  Conv2DComputeScaleDecision decision = CheckConv2DComputeScale(call_node, &macs);
  ++stats->checks;
  if (decision == Conv2DComputeScaleDecision::kAccept) {
    ++stats->accepted;
    return true;
  }
  if (decision == Conv2DComputeScaleDecision::kBelowThreshold) {
    ++stats->below_threshold;
  } else {
    ++stats->unknown_shape;
  }
  return false;
}

class VarUseCounter : public ::tvm::relay::ExprVisitor {
 public:
  void VisitExpr_(const VarNode* var_node) final { ++counts_[var_node]; }

  int GetUseCount(const Var& var) const {
    auto it = counts_.find(var.get());
    return it == counts_.end() ? 0 : it->second;
  }

 private:
  std::unordered_map<const VarNode*, int> counts_;
};

void FlattenLetChain(const Expr& expr, std::vector<LetBinding>* bindings, Expr* tail) {
  Expr current = expr;
  while (const auto* let_node = current.as<LetNode>()) {
    bindings->push_back({let_node->var, let_node->value});
    current = let_node->body;
  }
  *tail = current;
}

Expr RebuildLetChain(const std::vector<LetBinding>& bindings, Expr tail) {
  Expr expr = tail;
  for (int i = static_cast<int>(bindings.size()) - 1; i >= 0; --i) {
    expr = Let(bindings[i].var, bindings[i].value, expr);
  }
  return expr;
}

Expr IgnoreOnDeviceCallValue(const Expr& value) {
  const auto* call_node = value.as<CallNode>();
  if (call_node == nullptr) {
    return value;
  }
  const auto* op_node = call_node->op.as<OpNode>();
  if (op_node != nullptr && op_node->name == "on_device" && call_node->args.size() == 1) {
    return call_node->args[0];
  }
  return value;
}

bool IsCallOp(const CallNode* call_node, const std::string& op_name) {
  if (call_node == nullptr) {
    return false;
  }
  const auto* op_node = call_node->op.as<OpNode>();
  return op_node != nullptr && op_node->name == op_name;
}

std::optional<int64_t> GetStaticConv2DOutputChannels(const CallNode* conv2d_call) {
  if (conv2d_call == nullptr || conv2d_call->args.size() != 2) {
    return std::nullopt;
  }
  const auto* kernel_type = conv2d_call->args[1]->checked_type_.as<TensorTypeNode>();
  if (kernel_type == nullptr || kernel_type->shape.size() != 4) {
    return std::nullopt;
  }
  const auto* out_channels = kernel_type->shape[0].as<IntImmNode>();
  if (out_channels == nullptr) {
    return std::nullopt;
  }
  return out_channels->value;
}

std::optional<SplitAttrs> MakeDirectConv2DSplitAttrs(const CallNode* conv2d_call,
                                                     const std::string& pattern,
                                                     Conv2DFilterStats* stats) {
  if (!IsCallOp(conv2d_call, "nn.conv2d")) {
    return std::nullopt;
  }

  const auto* conv2d_attrs = conv2d_call->attrs.as<::tvm::relay::Conv2DAttrs>();
  if (conv2d_attrs == nullptr || conv2d_attrs->groups != 1) {
    return std::nullopt;
  }

  int split_num = GetCPUConv2DSplitNum();
  if (split_num <= 1) {
    return std::nullopt;
  }

  std::optional<int64_t> out_channels = GetStaticConv2DOutputChannels(conv2d_call);
  if (!out_channels.has_value() || split_num > out_channels.value()) {
    return std::nullopt;
  }
  if (!RecordConv2DFilterDecision(conv2d_call, stats)) {
    return std::nullopt;
  }

  SplitAttrs attrs;
  attrs.enabled = true;
  attrs.root_op = "nn.conv2d";
  attrs.pattern = pattern;
  attrs.axis = "output_channel";
  attrs.split_num = split_num;
  attrs.level = GetCPUOpSplitLevel();
  return attrs;
}

std::optional<DirectCallInfo> ResolveDirectCall(
    const Expr& value, const std::unordered_map<const VarNode*, Expr>& bound_values) {
  Expr inner = IgnoreOnDeviceCallValue(value);
  if (const auto* call_node = inner.as<CallNode>()) {
    return DirectCallInfo{GetRef<Call>(call_node), nullptr};
  }
  if (const auto* var_node = inner.as<VarNode>()) {
    auto it = bound_values.find(var_node);
    if (it == bound_values.end()) {
      return std::nullopt;
    }
    Expr resolved = IgnoreOnDeviceCallValue(it->second);
    if (const auto* call_node = resolved.as<CallNode>()) {
      return DirectCallInfo{GetRef<Call>(call_node), var_node};
    }
  }
  return std::nullopt;
}

std::optional<Call> GetCandidateCall(const Expr& value) {
  const auto* call_node = value.as<CallNode>();
  if (call_node == nullptr) {
    return std::nullopt;
  }

  if (call_node->op.as<VarNode>() != nullptr) {
    return GetRef<Call>(call_node);
  }

  const auto* op_node = call_node->op.as<OpNode>();
  if (op_node != nullptr && op_node->name == "on_device" && call_node->args.size() == 1) {
    const auto* inner_call_node = call_node->args[0].as<CallNode>();
    if (inner_call_node != nullptr && inner_call_node->op.as<VarNode>() != nullptr) {
      return GetRef<Call>(inner_call_node);
    }
  }

  return std::nullopt;
}

std::optional<SplitResult> TrySplitDirectConv2D(
    const LetBinding& binding, const std::unordered_map<const VarNode*, Expr>& bound_values,
    const VarUseCounter& use_counter, std::vector<LetBinding>* new_bindings,
    Conv2DFilterStats* filter_stats) {
  std::optional<DirectCallInfo> maybe_call_info = ResolveDirectCall(binding.value, bound_values);
  if (!maybe_call_info.has_value()) {
    return std::nullopt;
  }
  const DirectCallInfo& call_info = maybe_call_info.value();
  const Call& call = call_info.call;
  std::optional<SplitAttrs> attrs =
      MakeDirectConv2DSplitAttrs(call.get(), "conv2d", filter_stats);
  if (!attrs.has_value()) {
    return std::nullopt;
  }
  if (call_info.source_var != nullptr) {
    Var source_var = GetRef<Var>(call_info.source_var);
    if (use_counter.GetUseCount(source_var) != 1 || new_bindings->empty() ||
        new_bindings->back().var.get() != call_info.source_var) {
      return std::nullopt;
    }
  }

  Function synthetic_func({}, call, call->checked_type_, {});
  SplitCandidate candidate;
  candidate.func = synthetic_func;
  candidate.call_var = binding.var;
  candidate.call_value = binding.value;
  candidate.call = call;
  candidate.attrs = attrs.value();
  std::optional<SplitResult> result = SplitConv2D(candidate);
  if (result.has_value() && result.value().changed && call_info.source_var != nullptr) {
    new_bindings->pop_back();
  }
  return result;
}

std::optional<SplitResult> TrySplitDirectConv2DBiasAdd(
    const LetBinding& binding, const std::unordered_map<const VarNode*, Expr>& bound_values,
    Conv2DFilterStats* filter_stats) {
  std::optional<DirectCallInfo> maybe_bias_call_info = ResolveDirectCall(binding.value, bound_values);
  if (!maybe_bias_call_info.has_value()) {
    return std::nullopt;
  }
  if (maybe_bias_call_info.value().source_var != nullptr) {
    return std::nullopt;
  }
  const Call& bias_call = maybe_bias_call_info.value().call;
  if (!IsCallOp(bias_call.get(), "nn.bias_add") || bias_call->args.size() != 2) {
    return std::nullopt;
  }

  const auto* bias_add_attrs = bias_call->attrs.as<::tvm::relay::BiasAddAttrs>();
  if (bias_add_attrs == nullptr || bias_add_attrs->axis != 1) {
    return std::nullopt;
  }

  const auto* conv_var = bias_call->args[0].as<VarNode>();
  if (conv_var == nullptr) {
    return std::nullopt;
  }
  auto bound_it = bound_values.find(conv_var);
  if (bound_it == bound_values.end()) {
    return std::nullopt;
  }

  std::optional<DirectCallInfo> maybe_conv_call_info = ResolveDirectCall(bound_it->second, bound_values);
  if (!maybe_conv_call_info.has_value()) {
    return std::nullopt;
  }
  if (maybe_conv_call_info.value().source_var != nullptr) {
    return std::nullopt;
  }
  const Call& conv_call = maybe_conv_call_info.value().call;
  std::optional<SplitAttrs> attrs =
      MakeDirectConv2DSplitAttrs(conv_call.get(), "conv2d_bias_add", filter_stats);
  if (!attrs.has_value()) {
    return std::nullopt;
  }

  Expr nested_body = Call(bias_call->op, {conv_call, bias_call->args[1]}, bias_call->attrs,
                          bias_call->type_args, bias_call->span);
  nested_body->checked_type_ = bias_call->checked_type_;
  Function synthetic_func({}, nested_body, bias_call->checked_type_, {});

  SplitCandidate candidate;
  candidate.func = synthetic_func;
  candidate.call_var = binding.var;
  candidate.call_value = binding.value;
  candidate.call = bias_call;
  candidate.attrs = attrs.value();
  return SplitConv2D(candidate);
}

std::optional<SplitResult> DispatchSplit(const SplitCandidate& candidate) {
  if (candidate.attrs.root_op == "nn.conv2d") {
    return SplitConv2D(candidate);
  }
  return std::nullopt;
}

Expr ApplyOpSplitToMainBody(const Expr& body, bool* changed, Conv2DFilterStats* filter_stats) {
  std::vector<LetBinding> bindings;
  Expr tail;
  FlattenLetChain(body, &bindings, &tail);

  VarUseCounter use_counter;
  for (const LetBinding& binding : bindings) {
    use_counter.VisitExpr(binding.value);
  }
  use_counter.VisitExpr(tail);

  std::vector<LetBinding> new_bindings;
  std::unordered_map<const VarNode*, CandidateFunction> candidates;
  std::unordered_map<const VarNode*, Expr> bound_values;
  std::unordered_set<const VarNode*> direct_split_outputs;

  for (const LetBinding& binding : bindings) {
    std::optional<SplitResult> direct_bias_add_result =
        TrySplitDirectConv2DBiasAdd(binding, bound_values, filter_stats);
    if (direct_bias_add_result.has_value() && direct_bias_add_result.value().changed) {
      std::optional<DirectCallInfo> maybe_bias_call_info =
          ResolveDirectCall(binding.value, bound_values);
      const auto* conv_var = maybe_bias_call_info.value().call->args[0].as<VarNode>();
      if (conv_var != nullptr && use_counter.GetUseCount(GetRef<Var>(conv_var)) == 1 &&
          direct_split_outputs.count(conv_var) == 0 && !new_bindings.empty() &&
          new_bindings.back().var.get() == conv_var) {
        new_bindings.pop_back();
        new_bindings.insert(new_bindings.end(), direct_bias_add_result.value().bindings.begin(),
                            direct_bias_add_result.value().bindings.end());
        bound_values.emplace(binding.var.get(), binding.value);
        *changed = true;
        continue;
      }
    }

    std::optional<SplitResult> direct_conv2d_result =
        TrySplitDirectConv2D(binding, bound_values, use_counter, &new_bindings, filter_stats);
    if (direct_conv2d_result.has_value() && direct_conv2d_result.value().changed) {
      new_bindings.insert(new_bindings.end(), direct_conv2d_result.value().bindings.begin(),
                          direct_conv2d_result.value().bindings.end());
      bound_values.emplace(binding.var.get(), binding.value);
      direct_split_outputs.insert(binding.var.get());
      *changed = true;
      continue;
    }

    if (const auto* func_node = binding.value.as<FunctionNode>()) {
      Function func = GetRef<Function>(func_node);
      if (!IsGeneratedByOpSplit(func) && HasSplitAttrs(func) && IsSplitEnabled(func) &&
          GetSplitLevel(func) == "relay") {
        candidates.emplace(binding.var.get(),
                           CandidateFunction{func, GetSplitAttrs(func), binding, false, false});
        continue;
      }
      new_bindings.push_back(binding);
      continue;
    }

    std::optional<Call> maybe_call = GetCandidateCall(binding.value);
    if (maybe_call.has_value()) {
      const Call& call = maybe_call.value();
      if (const auto* op_var = call->op.as<VarNode>()) {
        auto it = candidates.find(op_var);
        if (it != candidates.end()) {
          SplitCandidate candidate;
          candidate.func_var = GetRef<Var>(op_var);
          candidate.func = it->second.func;
          candidate.call_var = binding.var;
          candidate.call_value = binding.value;
          candidate.call = call;
          candidate.attrs = it->second.attrs;

          std::optional<SplitResult> result = DispatchSplit(candidate);
          if (result.has_value() && result.value().changed) {
            new_bindings.insert(new_bindings.end(), result.value().bindings.begin(),
                                result.value().bindings.end());
            it->second.consumed = true;
            *changed = true;
            continue;
          }

          if (!it->second.emitted) {
            new_bindings.push_back(it->second.binding);
            it->second.emitted = true;
          }
        }
      }
    }

    new_bindings.push_back(binding);
    bound_values.emplace(binding.var.get(), binding.value);
  }

  for (auto& kv : candidates) {
    CandidateFunction& candidate = kv.second;
    if (!candidate.consumed && !candidate.emitted) {
      new_bindings.push_back(candidate.binding);
    }
  }

  if (!*changed) {
    return body;
  }
  return RebuildLetChain(new_bindings, tail);
}

}  // namespace

tvm::transform::Pass ApplyOpSplit() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext ctx) {
    if (!IsCPURelayOpSplitEnabled()) {
      return mod;
    }
    if (!mod->ContainGlobalVar("main")) {
      return mod;
    }

    GlobalVar main_gv = mod->GetGlobalVar("main");
    BaseFunc base_func = mod->Lookup(main_gv);
    const auto* main_func_node = base_func.as<FunctionNode>();
    if (main_func_node == nullptr) {
      return mod;
    }

    bool changed = false;
    Conv2DFilterStats filter_stats;
    Function main_func = GetRef<Function>(main_func_node);
    Expr new_body = ApplyOpSplitToMainBody(main_func->body, &changed, &filter_stats);
    // if (filter_stats.checks > 0) {
    //   LOG(INFO) << "[neuTVM][CPU][Conv2DFilter] phase=direct_rewrite threshold_macs="
    //             << GetCPUMinConv2DMACs() << " checks=" << filter_stats.checks
    //             << " accepted=" << filter_stats.accepted
    //             << " below_threshold=" << filter_stats.below_threshold
    //             << " unknown_shape=" << filter_stats.unknown_shape;
    // }
    if (!changed) {
      return mod;
    }

    Function new_main = Function(main_func->params, new_body, main_func->ret_type,
                                 main_func->type_params, main_func->attrs, main_func->span);
    IRModule updated = mod->ShallowCopy();
    updated->Update(main_gv, new_main);
    return updated;
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "neutvm.relay.cpu.ApplyOpSplit", {});
}

TVM_REGISTER_GLOBAL("relay._transform.neutvm.cpu.ApplyOpSplit").set_body_typed(ApplyOpSplit);

}  // namespace cpu
}  // namespace relay
}  // namespace neutvm
}  // namespace tvm
