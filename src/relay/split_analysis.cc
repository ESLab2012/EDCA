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
#include <tvm/neutvm/relay/cpu/op_splitters/conv2d.h>
#include <tvm/neutvm/relay/cpu/split_analysis.h>
#include <tvm/neutvm/relay/cpu/split_attrs.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace tvm {
namespace neutvm {
namespace relay {
namespace cpu {

namespace {

using ::tvm::relay::CallNode;
using ::tvm::relay::Expr;
using ::tvm::relay::Function;
using ::tvm::relay::FunctionNode;
using ::tvm::relay::Let;
using ::tvm::relay::LetNode;
using ::tvm::relay::OpNode;
using ::tvm::relay::TensorTypeNode;

struct Conv2DFilterStats {
  int64_t checks{0};
  int64_t accepted{0};
  int64_t below_threshold{0};
  int64_t unknown_shape{0};
};

using Analyzer =
    std::function<std::optional<SplitAttrs>(const Function&, Conv2DFilterStats*)>;

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

bool IsCallOp(const CallNode* call_node, const std::string& op_name) {
  if (call_node == nullptr) {
    return false;
  }
  const auto* op_node = call_node->op.as<OpNode>();
  return op_node != nullptr && op_node->name == op_name;
}

std::optional<int64_t> GetStaticOutputChannels(const CallNode* call_node) {
  if (call_node->args.size() != 2) {
    return std::nullopt;
  }

  const auto* kernel_type = call_node->args[1]->checked_type_.as<TensorTypeNode>();
  if (kernel_type == nullptr || kernel_type->shape.size() != 4) {
    return std::nullopt;
  }

  const auto* out_channels = kernel_type->shape[0].as<IntImmNode>();
  if (out_channels == nullptr) {
    return std::nullopt;
  }
  return out_channels->value;
}

std::optional<SplitAttrs> MakeConv2DSplitAttrs(const CallNode* call_node,
                                               const std::string& pattern,
                                               Conv2DFilterStats* stats) {
  const auto* op_node = call_node->op.as<OpNode>();
  if (op_node == nullptr || op_node->name != "nn.conv2d") {
    return std::nullopt;
  }

  const auto* conv2d_attrs = call_node->attrs.as<::tvm::relay::Conv2DAttrs>();
  if (conv2d_attrs == nullptr || conv2d_attrs->groups != 1) {
    return std::nullopt;
  }

  int split_num = GetCPUConv2DSplitNum();
  if (split_num <= 1) {
    return std::nullopt;
  }

  std::optional<int64_t> out_channels = GetStaticOutputChannels(call_node);
  if (!out_channels.has_value() || split_num > out_channels.value()) {
    return std::nullopt;
  }
  if (!RecordConv2DFilterDecision(call_node, stats)) {
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

const CallNode* GetConv2DFromBiasAdd(const Expr& expr) {
  const auto* bias_add_call = expr.as<CallNode>();
  if (!IsCallOp(bias_add_call, "nn.bias_add") || bias_add_call->args.size() != 2) {
    return nullptr;
  }

  const auto* bias_add_attrs = bias_add_call->attrs.as<::tvm::relay::BiasAddAttrs>();
  if (bias_add_attrs == nullptr || bias_add_attrs->axis != 1) {
    return nullptr;
  }

  const auto* bias_type = bias_add_call->args[1]->checked_type_.as<TensorTypeNode>();
  if (bias_type == nullptr || bias_type->shape.size() != 1) {
    return nullptr;
  }

  const auto* conv2d_call = bias_add_call->args[0].as<CallNode>();
  if (!IsCallOp(conv2d_call, "nn.conv2d")) {
    return nullptr;
  }
  return conv2d_call;
}

const CallNode* GetConv2DFromBiasAddSiLU(const Expr& expr) {
  const auto* multiply_call = expr.as<CallNode>();
  if (!IsCallOp(multiply_call, "multiply") || multiply_call->args.size() != 2) {
    return nullptr;
  }

  for (int bias_arg_index = 0; bias_arg_index < 2; ++bias_arg_index) {
    int sigmoid_arg_index = 1 - bias_arg_index;
    const Expr& bias_expr = multiply_call->args[bias_arg_index];
    const Expr& sigmoid_expr = multiply_call->args[sigmoid_arg_index];

    const auto* sigmoid_call = sigmoid_expr.as<CallNode>();
    if (!IsCallOp(sigmoid_call, "sigmoid") || sigmoid_call->args.size() != 1) {
      continue;
    }
    if (!sigmoid_call->args[0].same_as(bias_expr)) {
      continue;
    }

    const CallNode* conv2d_call = GetConv2DFromBiasAdd(bias_expr);
    if (conv2d_call != nullptr) {
      return conv2d_call;
    }
  }

  return nullptr;
}

const CallNode* GetConv2DFromBiasAddLeakyReLU(const Expr& expr) {
  const auto* leaky_relu_call = expr.as<CallNode>();
  if (!IsCallOp(leaky_relu_call, "nn.leaky_relu") || leaky_relu_call->args.size() != 1) {
    return nullptr;
  }

  const auto* leaky_relu_attrs = leaky_relu_call->attrs.as<::tvm::relay::LeakyReluAttrs>();
  if (leaky_relu_attrs == nullptr) {
    return nullptr;
  }

  return GetConv2DFromBiasAdd(leaky_relu_call->args[0]);
}

std::optional<SplitAttrs> AnalyzeConv2D(const Function& func, Conv2DFilterStats* stats) {
  const auto* call_node = func->body.as<CallNode>();
  if (call_node == nullptr) {
    return std::nullopt;
  }

  return MakeConv2DSplitAttrs(call_node, "conv2d", stats);
}

std::optional<SplitAttrs> AnalyzeConv2DBiasAdd(const Function& func, Conv2DFilterStats* stats) {
  const auto* bias_add_call = func->body.as<CallNode>();
  if (bias_add_call == nullptr || bias_add_call->args.size() != 2) {
    return std::nullopt;
  }

  const auto* bias_add_op = bias_add_call->op.as<OpNode>();
  if (bias_add_op == nullptr || bias_add_op->name != "nn.bias_add") {
    return std::nullopt;
  }

  const auto* bias_add_attrs = bias_add_call->attrs.as<::tvm::relay::BiasAddAttrs>();
  if (bias_add_attrs == nullptr || bias_add_attrs->axis != 1) {
    return std::nullopt;
  }

  const auto* conv2d_call = bias_add_call->args[0].as<CallNode>();
  if (conv2d_call == nullptr) {
    return std::nullopt;
  }

  const auto* bias_type = bias_add_call->args[1]->checked_type_.as<TensorTypeNode>();
  if (bias_type == nullptr || bias_type->shape.size() != 1) {
    return std::nullopt;
  }

  return MakeConv2DSplitAttrs(conv2d_call, "conv2d_bias_add", stats);
}

std::optional<SplitAttrs> AnalyzeConv2DBiasAddSiLU(const Function& func,
                                                   Conv2DFilterStats* stats) {
  const CallNode* conv2d_call = GetConv2DFromBiasAddSiLU(func->body);
  if (conv2d_call == nullptr) {
    return std::nullopt;
  }

  return MakeConv2DSplitAttrs(conv2d_call, "conv2d_bias_add_silu", stats);
}

std::optional<SplitAttrs> AnalyzeConv2DBiasAddLeakyReLU(const Function& func,
                                                        Conv2DFilterStats* stats) {
  const CallNode* conv2d_call = GetConv2DFromBiasAddLeakyReLU(func->body);
  if (conv2d_call == nullptr) {
    return std::nullopt;
  }

  return MakeConv2DSplitAttrs(conv2d_call, "conv2d_bias_add_leaky_relu", stats);
}

std::optional<SplitAttrs> AnalyzeConv2DBiasAddSiLUResidualAdd(const Function& func,
                                                              Conv2DFilterStats* stats) {
  const auto* add_call = func->body.as<CallNode>();
  if (!IsCallOp(add_call, "add") || add_call->args.size() != 2) {
    return std::nullopt;
  }

  for (int silu_arg_index = 0; silu_arg_index < 2; ++silu_arg_index) {
    int residual_arg_index = 1 - silu_arg_index;
    const Expr& silu_expr = add_call->args[silu_arg_index];
    const Expr& residual_expr = add_call->args[residual_arg_index];

    const auto* residual_type = residual_expr->checked_type_.as<TensorTypeNode>();
    if (residual_type == nullptr || residual_type->shape.size() != 4) {
      continue;
    }

    const CallNode* conv2d_call = GetConv2DFromBiasAddSiLU(silu_expr);
    if (conv2d_call == nullptr) {
      continue;
    }
    return MakeConv2DSplitAttrs(conv2d_call, "conv2d_bias_add_silu_residual_add", stats);
  }

  return std::nullopt;
}

const std::vector<Analyzer>& GetAnalyzerRegistry() {
  static const std::vector<Analyzer> analyzers = {
      AnalyzeConv2D,
      AnalyzeConv2DBiasAdd,
      AnalyzeConv2DBiasAddSiLU,
      AnalyzeConv2DBiasAddLeakyReLU,
      AnalyzeConv2DBiasAddSiLUResidualAdd,
  };
  return analyzers;
}

std::optional<SplitAttrs> AnalyzeByRegistry(const Function& func, Conv2DFilterStats* stats) {
  for (const Analyzer& analyzer : GetAnalyzerRegistry()) {
    std::optional<SplitAttrs> attrs = analyzer(func, stats);
    if (attrs.has_value()) {
      return attrs;
    }
  }
  return std::nullopt;
}

Function AnnotateFunctionIfSplittable(Function func, bool* changed, Conv2DFilterStats* stats) {
  if (!func->HasNonzeroAttr(::tvm::relay::attr::kPrimitive)) {
    return func;
  }
  if (IsGeneratedByOpSplit(func)) {
    return func;
  }

  std::optional<SplitAttrs> split_attrs = AnalyzeByRegistry(func, stats);
  if (!split_attrs.has_value()) {
    return func;
  }

  *changed = true;
  return WithSplitAttrs(std::move(func), split_attrs.value());
}

Expr AnnotateLetBoundFunctions(const Expr& body, bool* changed, Conv2DFilterStats* stats) {
  const auto* let_node = body.as<LetNode>();
  if (let_node == nullptr) {
    return body;
  }

  Expr new_value = let_node->value;
  if (const auto* func_node = let_node->value.as<FunctionNode>()) {
    new_value = AnnotateFunctionIfSplittable(GetRef<Function>(func_node), changed, stats);
  }

  Expr new_body = AnnotateLetBoundFunctions(let_node->body, changed, stats);
  if (new_value.same_as(let_node->value) && new_body.same_as(let_node->body)) {
    return body;
  }
  return Let(let_node->var, new_value, new_body, let_node->span);
}

}  // namespace

tvm::transform::Pass AnalyzeSplittableOps() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext ctx) {
    if (!IsCPUOpSplitEnabled()) {
      return mod;
    }

    IRModule updated = mod->ShallowCopy();
    Conv2DFilterStats filter_stats;
    for (const auto& kv : mod->functions) {
      const GlobalVar& global_var = kv.first;
      const BaseFunc& base_func = kv.second;

      const auto* func_node = base_func.as<FunctionNode>();
      if (func_node == nullptr) {
        continue;
      }

      bool changed = false;
      Function func =
          AnnotateFunctionIfSplittable(GetRef<Function>(func_node), &changed, &filter_stats);
      Expr new_body = AnnotateLetBoundFunctions(func->body, &changed, &filter_stats);
      if (!new_body.same_as(func->body)) {
        func = Function(func->params, new_body, func->ret_type, func->type_params, func->attrs,
                        func->span);
      }

      if (changed) {
        updated->Update(global_var, func);
      }
    }
    // if (filter_stats.checks > 0) {
    //   LOG(INFO) << "[neuTVM][CPU][Conv2DFilter] phase=analysis threshold_macs="
    //             << GetCPUMinConv2DMACs() << " checks=" << filter_stats.checks
    //             << " accepted=" << filter_stats.accepted
    //             << " below_threshold=" << filter_stats.below_threshold
    //             << " unknown_shape=" << filter_stats.unknown_shape;
    // }
    return updated;
  };

  return tvm::transform::CreateModulePass(pass_func, 0,
                                          "neutvm.relay.cpu.AnalyzeSplittableOps", {});
}

TVM_REGISTER_GLOBAL("relay._transform.neutvm.cpu.AnalyzeSplittableOps")
    .set_body_typed(AnalyzeSplittableOps);

}  // namespace cpu
}  // namespace relay
}  // namespace neutvm
}  // namespace tvm
