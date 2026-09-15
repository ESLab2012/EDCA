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

#include <relay/conv2d.h>
#include <tvm/neutvm/config.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/expr.h>

#include <cstdlib>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
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
using ::tvm::relay::OpNode;
using ::tvm::relay::TensorType;
using ::tvm::relay::TensorTypeNode;
using ::tvm::relay::Tuple;
using ::tvm::relay::TupleType;
using ::tvm::relay::TypeVar;
using ::tvm::relay::Var;

struct Conv2DPatternInfo {
  const CallNode* conv2d_call{nullptr};
  const CallNode* bias_add_call{nullptr};
  const CallNode* leaky_relu_call{nullptr};
  int residual_arg_index{-1};
};

struct ReusablePartFunction {
  Var func_var;
};

std::string ObjectRefKey(const ObjectRef& value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

std::string TensorTypeKey(const TensorType& type) {
  std::ostringstream os;
  os << "dtype=" << type->dtype << ",shape=[";
  for (size_t i = 0; i < type->shape.size(); ++i) {
    if (i != 0) {
      os << ",";
    }
    os << ObjectRefKey(type->shape[i]);
  }
  os << "]";
  return os.str();
}

std::string OptionalTensorTypeKey(const TensorType& type) {
  return type.defined() ? TensorTypeKey(type) : "none";
}

std::string Conv2DAttrsKey(const ::tvm::relay::Conv2DAttrs* attrs, int part_out_channels) {
  std::ostringstream os;
  os << "strides=" << ObjectRefKey(attrs->strides);
  os << ";padding=" << ObjectRefKey(attrs->padding);
  os << ";dilation=" << ObjectRefKey(attrs->dilation);
  os << ";groups=" << attrs->groups;
  os << ";channels=" << part_out_channels;
  os << ";kernel_size=" << ObjectRefKey(attrs->kernel_size);
  os << ";data_layout=" << attrs->data_layout;
  os << ";kernel_layout=" << attrs->kernel_layout;
  os << ";out_layout=" << attrs->out_layout;
  os << ";out_dtype=" << attrs->out_dtype;
  os << ";auto_scheduler_rewritten_layout=" << attrs->auto_scheduler_rewritten_layout;
  os << ";meta_schedule_original_shape=" << ObjectRefKey(attrs->meta_schedule_original_shape);
  return os.str();
}

std::string BiasAddAttrsKey(const ::tvm::relay::BiasAddAttrs* attrs) {
  if (attrs == nullptr) {
    return "none";
  }
  return "axis=" + std::to_string(attrs->axis);
}

std::string LeakyReluAttrsKey(const ::tvm::relay::LeakyReluAttrs* attrs) {
  if (attrs == nullptr) {
    return "none";
  }
  return "alpha=" + std::to_string(attrs->alpha);
}

bool IsCallOp(const CallNode* call_node, const std::string& op_name) {
  if (call_node == nullptr) {
    return false;
  }
  const auto* op_node = call_node->op.as<OpNode>();
  return op_node != nullptr && op_node->name == op_name;
}

const CallNode* GetConv2DCallFromPrimFunc(const Function& func) {
  const auto* call_node = func->body.as<CallNode>();
  if (call_node == nullptr) {
    return nullptr;
  }
  if (!IsCallOp(call_node, "nn.conv2d") || call_node->args.size() != 2) {
    return nullptr;
  }
  return call_node;
}

bool FillBiasAddConv2DInfo(const Expr& expr, Conv2DPatternInfo* info) {
  const auto* bias_add_call = expr.as<CallNode>();
  if (!IsCallOp(bias_add_call, "nn.bias_add") || bias_add_call->args.size() != 2) {
    return false;
  }

  const auto* bias_add_attrs = bias_add_call->attrs.as<::tvm::relay::BiasAddAttrs>();
  if (bias_add_attrs == nullptr || bias_add_attrs->axis != 1) {
    return false;
  }

  const auto* conv2d_call = bias_add_call->args[0].as<CallNode>();
  if (!IsCallOp(conv2d_call, "nn.conv2d") || conv2d_call->args.size() != 2) {
    return false;
  }

  info->conv2d_call = conv2d_call;
  info->bias_add_call = bias_add_call;
  return true;
}

bool FillBiasAddSiLUConv2DInfo(const Expr& expr, Conv2DPatternInfo* info) {
  const auto* multiply_call = expr.as<CallNode>();
  if (!IsCallOp(multiply_call, "multiply") || multiply_call->args.size() != 2) {
    return false;
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
    if (FillBiasAddConv2DInfo(bias_expr, info)) {
      return true;
    }
  }

  return false;
}

bool FillBiasAddLeakyReLUConv2DInfo(const Expr& expr, Conv2DPatternInfo* info) {
  const auto* leaky_relu_call = expr.as<CallNode>();
  if (!IsCallOp(leaky_relu_call, "nn.leaky_relu") || leaky_relu_call->args.size() != 1) {
    return false;
  }

  const auto* leaky_relu_attrs = leaky_relu_call->attrs.as<::tvm::relay::LeakyReluAttrs>();
  if (leaky_relu_attrs == nullptr) {
    return false;
  }

  if (!FillBiasAddConv2DInfo(leaky_relu_call->args[0], info)) {
    return false;
  }
  info->leaky_relu_call = leaky_relu_call;
  return true;
}

int FindFunctionParamIndex(const Function& func, const Expr& expr) {
  for (size_t i = 0; i < func->params.size(); ++i) {
    if (func->params[i].same_as(expr)) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

std::optional<Expr> ResolveFunctionArg(const Function& func, const Call& call, const Expr& expr) {
  int param_index = FindFunctionParamIndex(func, expr);
  if (param_index < 0) {
    return expr;
  }
  if (param_index >= static_cast<int>(call->args.size())) {
    return std::nullopt;
  }
  return call->args[param_index];
}

Conv2DPatternInfo GetConv2DPatternInfo(const Function& func, const String& pattern) {
  Conv2DPatternInfo info;
  if (pattern == "conv2d") {
    info.conv2d_call = GetConv2DCallFromPrimFunc(func);
    return info;
  }

  if (pattern == "conv2d_bias_add") {
    FillBiasAddConv2DInfo(func->body, &info);
    return info;
  }

  if (pattern == "conv2d_bias_add_silu") {
    FillBiasAddSiLUConv2DInfo(func->body, &info);
    return info;
  }

  if (pattern == "conv2d_bias_add_leaky_relu") {
    FillBiasAddLeakyReLUConv2DInfo(func->body, &info);
    return info;
  }

  if (pattern != "conv2d_bias_add_silu_residual_add") {
    return info;
  }

  const auto* add_call = func->body.as<CallNode>();
  if (!IsCallOp(add_call, "add") || add_call->args.size() != 2) {
    return info;
  }

  for (int silu_arg_index = 0; silu_arg_index < 2; ++silu_arg_index) {
    int residual_arg_index = 1 - silu_arg_index;
    const Expr& silu_expr = add_call->args[silu_arg_index];
    const Expr& residual_expr = add_call->args[residual_arg_index];

    Conv2DPatternInfo maybe_info;
    if (!FillBiasAddSiLUConv2DInfo(silu_expr, &maybe_info)) {
      continue;
    }
    int param_index = FindFunctionParamIndex(func, residual_expr);
    if (param_index < 0) {
      continue;
    }
    maybe_info.residual_arg_index = param_index;
    return maybe_info;
  }

  return info;
}

const TensorTypeNode* AsTensorType(const Expr& expr) {
  return expr->checked_type_.as<TensorTypeNode>();
}

std::optional<int64_t> StaticDim(const TensorTypeNode* type, size_t axis) {
  if (type == nullptr || axis >= type->shape.size()) {
    return std::nullopt;
  }
  const auto* dim = type->shape[axis].as<IntImmNode>();
  if (dim == nullptr) {
    return std::nullopt;
  }
  return dim->value;
}

TensorType TensorTypeRef(const TensorTypeNode* type) { return GetRef<TensorType>(type); }

ObjectPtr<::tvm::relay::Conv2DAttrs> MakePartConv2DAttrs(const ::tvm::relay::Conv2DAttrs* attrs,
                                                         int part_out_channels) {
  auto new_attrs = make_object<::tvm::relay::Conv2DAttrs>();
  new_attrs->strides = attrs->strides;
  new_attrs->padding = attrs->padding;
  new_attrs->dilation = attrs->dilation;
  new_attrs->groups = attrs->groups;
  new_attrs->channels = Integer(part_out_channels);
  new_attrs->kernel_size = attrs->kernel_size;
  new_attrs->data_layout = attrs->data_layout;
  new_attrs->kernel_layout = attrs->kernel_layout;
  new_attrs->out_layout = attrs->out_layout;
  new_attrs->auto_scheduler_rewritten_layout = attrs->auto_scheduler_rewritten_layout;
  new_attrs->meta_schedule_original_shape = attrs->meta_schedule_original_shape;
  new_attrs->out_dtype = attrs->out_dtype;
  return new_attrs;
}

Expr MakeKernelSliceCall(const Expr& weight, int begin_oc, int end_oc, int in_channels, int kh,
                         int kw) {
  auto slice_attrs = make_object<::tvm::relay::StridedSliceAttrs>();
  slice_attrs->begin = Array<Integer>{Integer(begin_oc), Integer(0), Integer(0), Integer(0)};
  slice_attrs->end =
      Array<Integer>{Integer(end_oc), Integer(in_channels), Integer(kh), Integer(kw)};
  slice_attrs->strides = Array<Integer>{Integer(1), Integer(1), Integer(1), Integer(1)};
  slice_attrs->slice_mode = "end";
  static const Op& strided_slice_op = Op::Get("strided_slice");
  return Call(strided_slice_op, {weight}, Attrs(slice_attrs), {});
}

Expr MakeBiasSliceCall(const Expr& bias, int begin_oc, int end_oc) {
  auto slice_attrs = make_object<::tvm::relay::StridedSliceAttrs>();
  slice_attrs->begin = Array<Integer>{Integer(begin_oc)};
  slice_attrs->end = Array<Integer>{Integer(end_oc)};
  slice_attrs->strides = Array<Integer>{Integer(1)};
  slice_attrs->slice_mode = "end";
  static const Op& strided_slice_op = Op::Get("strided_slice");
  return Call(strided_slice_op, {bias}, Attrs(slice_attrs), {});
}

Expr MakeResidualSliceCall(const Expr& residual, int begin_oc, int end_oc, int n, int h, int w) {
  auto slice_attrs = make_object<::tvm::relay::StridedSliceAttrs>();
  slice_attrs->begin = Array<Integer>{Integer(0), Integer(begin_oc), Integer(0), Integer(0)};
  slice_attrs->end = Array<Integer>{Integer(n), Integer(end_oc), Integer(h), Integer(w)};
  slice_attrs->strides = Array<Integer>{Integer(1), Integer(1), Integer(1), Integer(1)};
  slice_attrs->slice_mode = "end";
  static const Op& strided_slice_op = Op::Get("strided_slice");
  return Call(strided_slice_op, {residual}, Attrs(slice_attrs), {});
}

Function MakeKernelSliceFunction(const std::string& tag, int part_idx,
                                 const TensorType& full_weight_type,
                                 const TensorType& part_weight_type, int begin_oc, int end_oc,
                                 int in_channels, int kh, int kw) {
  Var weight("w_full", full_weight_type);
  Expr body = MakeKernelSliceCall(weight, begin_oc, end_oc, in_channels, kh, kw);
  Function func({weight}, body, part_weight_type, Array<TypeVar>());
  func = WithAttr(std::move(func), ::tvm::relay::attr::kPrimitive, Integer(1));
  func = WithAttr(std::move(func), "hash",
                  String("neutvm_conv2d_kernel_slice_" + tag + "_" + std::to_string(part_idx)));
  return MarkGeneratedByOpSplit(std::move(func));
}

Function MakeBiasSliceFunction(const std::string& tag, int part_idx,
                               const TensorType& full_bias_type, const TensorType& part_bias_type,
                               int begin_oc, int end_oc) {
  Var bias("b_full", full_bias_type);
  Expr body = MakeBiasSliceCall(bias, begin_oc, end_oc);
  Function func({bias}, body, part_bias_type, Array<TypeVar>());
  func = WithAttr(std::move(func), ::tvm::relay::attr::kPrimitive, Integer(1));
  func = WithAttr(std::move(func), "hash",
                  String("neutvm_conv2d_bias_slice_" + tag + "_" + std::to_string(part_idx)));
  return MarkGeneratedByOpSplit(std::move(func));
}

Function MakeResidualSliceFunction(const std::string& tag, int part_idx,
                                   const TensorType& full_residual_type,
                                   const TensorType& part_residual_type, int begin_oc, int end_oc,
                                   int n, int h, int w) {
  Var residual("r_full", full_residual_type);
  Expr body = MakeResidualSliceCall(residual, begin_oc, end_oc, n, h, w);
  Function func({residual}, body, part_residual_type, Array<TypeVar>());
  func = WithAttr(std::move(func), ::tvm::relay::attr::kPrimitive, Integer(1));
  func = WithAttr(std::move(func), "hash",
                  String("neutvm_conv2d_residual_slice_" + tag + "_" + std::to_string(part_idx)));
  return MarkGeneratedByOpSplit(std::move(func));
}

Function MakeConv2DPartFunction(const std::string& tag, int part_idx, const TensorType& data_type,
                                const TensorType& part_weight_type, const TensorType& part_out_type,
                                const ::tvm::relay::Conv2DAttrs* attrs, int part_out_channels) {
  Var data("x", data_type);
  Var weight("w_part", part_weight_type);
  auto new_attrs = MakePartConv2DAttrs(attrs, part_out_channels);
  static const Op& conv2d_op = Op::Get("nn.conv2d");
  Expr body = Call(conv2d_op, {data, weight}, Attrs(new_attrs), {});
  Function func({data, weight}, body, part_out_type, Array<TypeVar>());
  func = WithAttr(std::move(func), ::tvm::relay::attr::kPrimitive, Integer(1));
  func = WithAttr(std::move(func), "hash",
                  String("neutvm_conv2d_part_" + tag + "_" + std::to_string(part_idx)));
  return MarkGeneratedByOpSplit(std::move(func));
}

Function MakeConv2DBiasAddPartFunction(
    const std::string& tag, int part_idx, const TensorType& data_type,
    const TensorType& part_weight_type, const TensorType& part_bias_type,
    const TensorType& part_out_type, const ::tvm::relay::Conv2DAttrs* conv2d_attrs,
    const ::tvm::relay::BiasAddAttrs* bias_add_attrs, int part_out_channels) {
  Var data("x", data_type);
  Var weight("w_part", part_weight_type);
  Var bias("b_part", part_bias_type);
  auto new_conv2d_attrs = MakePartConv2DAttrs(conv2d_attrs, part_out_channels);
  auto new_bias_add_attrs = make_object<::tvm::relay::BiasAddAttrs>();
  new_bias_add_attrs->axis = bias_add_attrs->axis;

  static const Op& conv2d_op = Op::Get("nn.conv2d");
  static const Op& bias_add_op = Op::Get("nn.bias_add");
  Expr conv = Call(conv2d_op, {data, weight}, Attrs(new_conv2d_attrs), {});
  Expr body = Call(bias_add_op, {conv, bias}, Attrs(new_bias_add_attrs), {});
  Function func({data, weight, bias}, body, part_out_type, Array<TypeVar>());
  func = WithAttr(std::move(func), ::tvm::relay::attr::kPrimitive, Integer(1));
  func = WithAttr(std::move(func), "hash",
                  String("neutvm_conv2d_bias_add_part_" + tag + "_" + std::to_string(part_idx)));
  return MarkGeneratedByOpSplit(std::move(func));
}

Function MakeConv2DBiasAddSiLUPartFunction(
    const std::string& tag, int part_idx, const TensorType& data_type,
    const TensorType& part_weight_type, const TensorType& part_bias_type,
    const TensorType& part_out_type, const ::tvm::relay::Conv2DAttrs* conv2d_attrs,
    const ::tvm::relay::BiasAddAttrs* bias_add_attrs, int part_out_channels) {
  Var data("x", data_type);
  Var weight("w_part", part_weight_type);
  Var bias("b_part", part_bias_type);
  auto new_conv2d_attrs = MakePartConv2DAttrs(conv2d_attrs, part_out_channels);
  auto new_bias_add_attrs = make_object<::tvm::relay::BiasAddAttrs>();
  new_bias_add_attrs->axis = bias_add_attrs->axis;

  static const Op& conv2d_op = Op::Get("nn.conv2d");
  static const Op& bias_add_op = Op::Get("nn.bias_add");
  static const Op& sigmoid_op = Op::Get("sigmoid");
  static const Op& multiply_op = Op::Get("multiply");
  Expr conv = Call(conv2d_op, {data, weight}, Attrs(new_conv2d_attrs), {});
  Expr biased = Call(bias_add_op, {conv, bias}, Attrs(new_bias_add_attrs), {});
  Expr sigmoid = Call(sigmoid_op, {biased}, Attrs(), {});
  Expr body = Call(multiply_op, {biased, sigmoid}, Attrs(), {});
  Function func({data, weight, bias}, body, part_out_type, Array<TypeVar>());
  func = WithAttr(std::move(func), ::tvm::relay::attr::kPrimitive, Integer(1));
  func =
      WithAttr(std::move(func), "hash",
               String("neutvm_conv2d_bias_add_silu_part_" + tag + "_" + std::to_string(part_idx)));
  return MarkGeneratedByOpSplit(std::move(func));
}

Function MakeConv2DBiasAddLeakyReLUPartFunction(
    const std::string& tag, int part_idx, const TensorType& data_type,
    const TensorType& part_weight_type, const TensorType& part_bias_type,
    const TensorType& part_out_type, const ::tvm::relay::Conv2DAttrs* conv2d_attrs,
    const ::tvm::relay::BiasAddAttrs* bias_add_attrs,
    const ::tvm::relay::LeakyReluAttrs* leaky_relu_attrs, int part_out_channels) {
  Var data("x", data_type);
  Var weight("w_part", part_weight_type);
  Var bias("b_part", part_bias_type);
  auto new_conv2d_attrs = MakePartConv2DAttrs(conv2d_attrs, part_out_channels);
  auto new_bias_add_attrs = make_object<::tvm::relay::BiasAddAttrs>();
  new_bias_add_attrs->axis = bias_add_attrs->axis;
  auto new_leaky_relu_attrs = make_object<::tvm::relay::LeakyReluAttrs>();
  new_leaky_relu_attrs->alpha = leaky_relu_attrs->alpha;

  static const Op& conv2d_op = Op::Get("nn.conv2d");
  static const Op& bias_add_op = Op::Get("nn.bias_add");
  static const Op& leaky_relu_op = Op::Get("nn.leaky_relu");
  Expr conv = Call(conv2d_op, {data, weight}, Attrs(new_conv2d_attrs), {});
  Expr biased = Call(bias_add_op, {conv, bias}, Attrs(new_bias_add_attrs), {});
  Expr body = Call(leaky_relu_op, {biased}, Attrs(new_leaky_relu_attrs), {});
  Function func({data, weight, bias}, body, part_out_type, Array<TypeVar>());
  func = WithAttr(std::move(func), ::tvm::relay::attr::kPrimitive, Integer(1));
  func = WithAttr(
      std::move(func), "hash",
      String("neutvm_conv2d_bias_add_leaky_relu_part_" + tag + "_" + std::to_string(part_idx)));
  return MarkGeneratedByOpSplit(std::move(func));
}

Function MakeConv2DBiasAddSiLUResidualAddPartFunction(
    const std::string& tag, int part_idx, const TensorType& data_type,
    const TensorType& part_weight_type, const TensorType& part_bias_type,
    const TensorType& part_residual_type, const TensorType& part_out_type,
    const ::tvm::relay::Conv2DAttrs* conv2d_attrs, const ::tvm::relay::BiasAddAttrs* bias_add_attrs,
    int part_out_channels) {
  Var data("x", data_type);
  Var weight("w_part", part_weight_type);
  Var bias("b_part", part_bias_type);
  Var residual("r_part", part_residual_type);
  auto new_conv2d_attrs = MakePartConv2DAttrs(conv2d_attrs, part_out_channels);
  auto new_bias_add_attrs = make_object<::tvm::relay::BiasAddAttrs>();
  new_bias_add_attrs->axis = bias_add_attrs->axis;

  static const Op& conv2d_op = Op::Get("nn.conv2d");
  static const Op& bias_add_op = Op::Get("nn.bias_add");
  static const Op& sigmoid_op = Op::Get("sigmoid");
  static const Op& multiply_op = Op::Get("multiply");
  static const Op& add_op = Op::Get("add");
  Expr conv = Call(conv2d_op, {data, weight}, Attrs(new_conv2d_attrs), {});
  Expr biased = Call(bias_add_op, {conv, bias}, Attrs(new_bias_add_attrs), {});
  Expr sigmoid = Call(sigmoid_op, {biased}, Attrs(), {});
  Expr silu = Call(multiply_op, {biased, sigmoid}, Attrs(), {});
  Expr body = Call(add_op, {residual, silu}, Attrs(), {});
  Function func({data, weight, bias, residual}, body, part_out_type, Array<TypeVar>());
  func = WithAttr(std::move(func), ::tvm::relay::attr::kPrimitive, Integer(1));
  func = WithAttr(std::move(func), "hash",
                  String("neutvm_conv2d_bias_add_silu_residual_add_part_" + tag + "_" +
                         std::to_string(part_idx)));
  return MarkGeneratedByOpSplit(std::move(func));
}

Function MakeConcatenateFunction(const std::string& tag, const Array<Type>& input_types,
                                 const TensorType& output_type, int axis) {
  TupleType tuple_type(input_types);
  Var tuple_var("p", tuple_type);
  auto concat_attrs = make_object<::tvm::relay::ConcatenateAttrs>();
  concat_attrs->axis = axis;
  static const Op& concat_op = Op::Get("concatenate");
  Expr body = Call(concat_op, {tuple_var}, Attrs(concat_attrs), {});
  Function func({tuple_var}, body, output_type, Array<TypeVar>());
  func = WithAttr(std::move(func), ::tvm::relay::attr::kPrimitive, Integer(1));
  func = WithAttr(std::move(func), "hash", String("neutvm_conv2d_concatenate_" + tag));
  return MarkGeneratedByOpSplit(std::move(func));
}

Expr WrapLikeOriginalCallValue(const Expr& original_value, const Expr& new_inner_value) {
  const auto* outer_call = original_value.as<CallNode>();
  if (outer_call == nullptr) {
    return new_inner_value;
  }
  const auto* op_node = outer_call->op.as<OpNode>();
  if (op_node != nullptr && op_node->name == "on_device" && outer_call->args.size() == 1) {
    return Call(outer_call->op, {new_inner_value}, outer_call->attrs, outer_call->type_args,
                outer_call->span);
  }
  return new_inner_value;
}

std::string MakeConv2DPartReuseKey(
    const String& pattern, const TensorType& data_type, const TensorType& part_weight_type,
    const TensorType& part_bias_type, const TensorType& part_residual_type,
    const TensorType& part_output_type, const ::tvm::relay::Conv2DAttrs* conv2d_attrs,
    const ::tvm::relay::BiasAddAttrs* bias_add_attrs,
    const ::tvm::relay::LeakyReluAttrs* leaky_relu_attrs, int part_out_channels) {
  std::ostringstream os;
  os << "pattern=" << pattern;
  os << ";data=" << TensorTypeKey(data_type);
  os << ";weight=" << TensorTypeKey(part_weight_type);
  os << ";bias=" << OptionalTensorTypeKey(part_bias_type);
  os << ";residual=" << OptionalTensorTypeKey(part_residual_type);
  os << ";output=" << TensorTypeKey(part_output_type);
  os << ";conv2d_attrs=" << Conv2DAttrsKey(conv2d_attrs, part_out_channels);
  os << ";bias_add_attrs=" << BiasAddAttrsKey(bias_add_attrs);
  os << ";leaky_relu_attrs=" << LeakyReluAttrsKey(leaky_relu_attrs);
  return os.str();
}

bool IsConv2DSplitPatternAllowed(const String& pattern) {
  const char* env = std::getenv("NEUTVM_CPU_SPLIT_PATTERNS");
  if (env == nullptr || std::string(env).empty()) {
    return true;
  }

  std::unordered_set<std::string> allowed;
  std::stringstream ss(env);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      allowed.insert(item);
    }
  }
  return allowed.count(pattern) != 0;
}

}  // namespace

Conv2DComputeScaleDecision CheckConv2DComputeScale(const CallNode* conv2d_call, int64_t* macs) {
  ICHECK(macs != nullptr);
  *macs = 0;

  const int64_t threshold = GetCPUMinConv2DMACs();
  if (threshold == 0) {
    return Conv2DComputeScaleDecision::kAccept;
  }
  if (!IsCallOp(conv2d_call, "nn.conv2d") || conv2d_call->args.size() != 2) {
    return Conv2DComputeScaleDecision::kUnknownShape;
  }

  const auto* attrs = conv2d_call->attrs.as<::tvm::relay::Conv2DAttrs>();
  const auto* kernel_type = conv2d_call->args[1]->checked_type_.as<TensorTypeNode>();
  const auto* output_type = conv2d_call->checked_type_.as<TensorTypeNode>();
  if (attrs == nullptr || attrs->data_layout != "NCHW" || attrs->kernel_layout != "OIHW" ||
      !(attrs->out_layout == "" || attrs->out_layout == "NCHW") || kernel_type == nullptr ||
      kernel_type->shape.size() != 4 || output_type == nullptr || output_type->shape.size() != 4) {
    return Conv2DComputeScaleDecision::kUnknownShape;
  }

  auto get_static_positive_dim = [](const PrimExpr& dim) -> std::optional<int64_t> {
    const auto* imm = dim.as<IntImmNode>();
    if (imm == nullptr || imm->value <= 0) {
      return std::nullopt;
    }
    return imm->value;
  };

  std::vector<std::optional<int64_t>> dimensions = {
      get_static_positive_dim(output_type->shape[0]),
      get_static_positive_dim(output_type->shape[2]),
      get_static_positive_dim(output_type->shape[3]),
      get_static_positive_dim(output_type->shape[1]),
      get_static_positive_dim(kernel_type->shape[1]),
      get_static_positive_dim(kernel_type->shape[2]),
      get_static_positive_dim(kernel_type->shape[3]),
  };

  int64_t total_macs = 1;
  for (const std::optional<int64_t>& dimension : dimensions) {
    if (!dimension.has_value() ||
        total_macs > std::numeric_limits<int64_t>::max() / dimension.value()) {
      return Conv2DComputeScaleDecision::kUnknownShape;
    }
    total_macs *= dimension.value();
  }

  *macs = total_macs;
  return total_macs >= threshold ? Conv2DComputeScaleDecision::kAccept
                                 : Conv2DComputeScaleDecision::kBelowThreshold;
}

SplitResult SplitConv2D(const SplitCandidate& candidate) {
  SplitResult result;

  if ((candidate.attrs.pattern != "conv2d" && candidate.attrs.pattern != "conv2d_bias_add" &&
       candidate.attrs.pattern != "conv2d_bias_add_silu" &&
       candidate.attrs.pattern != "conv2d_bias_add_leaky_relu" &&
       candidate.attrs.pattern != "conv2d_bias_add_silu_residual_add") ||
      candidate.attrs.axis != "output_channel") {
    return result;
  }
  if (!IsConv2DSplitPatternAllowed(candidate.attrs.pattern)) {
    return result;
  }

  Conv2DPatternInfo pattern_info = GetConv2DPatternInfo(candidate.func, candidate.attrs.pattern);
  const CallNode* conv2d_call = pattern_info.conv2d_call;
  if (conv2d_call == nullptr) {
    return result;
  }
  int64_t macs = 0;
  if (CheckConv2DComputeScale(conv2d_call, &macs) !=
      Conv2DComputeScaleDecision::kAccept) {
    return result;
  }
  const bool has_bias_add = pattern_info.bias_add_call != nullptr;
  const bool has_silu = candidate.attrs.pattern == "conv2d_bias_add_silu";
  const bool has_leaky_relu = candidate.attrs.pattern == "conv2d_bias_add_leaky_relu";
  const bool has_residual_add = candidate.attrs.pattern == "conv2d_bias_add_silu_residual_add";

  const auto* attrs = conv2d_call->attrs.as<::tvm::relay::Conv2DAttrs>();
  if (attrs == nullptr || attrs->groups != 1 || attrs->data_layout != "NCHW" ||
      attrs->kernel_layout != "OIHW" || !(attrs->out_layout == "" || attrs->out_layout == "NCHW")) {
    return result;
  }

  std::optional<Expr> maybe_data =
      ResolveFunctionArg(candidate.func, candidate.call, conv2d_call->args[0]);
  std::optional<Expr> maybe_weight =
      ResolveFunctionArg(candidate.func, candidate.call, conv2d_call->args[1]);
  if (!maybe_data.has_value() || !maybe_weight.has_value()) {
    return result;
  }
  Expr actual_data = maybe_data.value();
  Expr actual_weight = maybe_weight.value();
  Expr actual_bias;
  if (has_bias_add) {
    std::optional<Expr> maybe_bias =
        ResolveFunctionArg(candidate.func, candidate.call, pattern_info.bias_add_call->args[1]);
    if (!maybe_bias.has_value()) {
      return result;
    }
    actual_bias = maybe_bias.value();
  }
  Expr actual_residual;
  if (has_residual_add) {
    if (pattern_info.residual_arg_index < 0 ||
        pattern_info.residual_arg_index >= static_cast<int>(candidate.call->args.size())) {
      return result;
    }
    actual_residual = candidate.call->args[pattern_info.residual_arg_index];
  }

  const auto* data_type_node = AsTensorType(actual_data);
  const auto* full_weight_type_node = AsTensorType(actual_weight);
  const auto* full_bias_type_node = has_bias_add ? AsTensorType(actual_bias) : nullptr;
  const auto* full_residual_type_node = has_residual_add ? AsTensorType(actual_residual) : nullptr;
  const auto* output_type_node = candidate.call->checked_type_.as<TensorTypeNode>();
  if (data_type_node == nullptr || full_weight_type_node == nullptr ||
      output_type_node == nullptr || full_weight_type_node->shape.size() != 4 ||
      output_type_node->shape.size() != 4) {
    return result;
  }
  if (has_bias_add &&
      (full_bias_type_node == nullptr || full_bias_type_node->shape.size() != 1 ||
       pattern_info.bias_add_call->attrs.as<::tvm::relay::BiasAddAttrs>() == nullptr)) {
    return result;
  }
  if (has_residual_add &&
      (full_residual_type_node == nullptr || full_residual_type_node->shape.size() != 4)) {
    return result;
  }

  std::optional<int64_t> out_channels = StaticDim(full_weight_type_node, 0);
  std::optional<int64_t> in_channels = StaticDim(full_weight_type_node, 1);
  std::optional<int64_t> kh = StaticDim(full_weight_type_node, 2);
  std::optional<int64_t> kw = StaticDim(full_weight_type_node, 3);
  if (!out_channels.has_value() || !in_channels.has_value() || !kh.has_value() || !kw.has_value()) {
    return result;
  }
  if (has_bias_add) {
    std::optional<int64_t> bias_channels = StaticDim(full_bias_type_node, 0);
    if (!bias_channels.has_value() || bias_channels.value() != out_channels.value()) {
      return result;
    }
  }
  std::optional<int64_t> residual_n;
  std::optional<int64_t> residual_channels;
  std::optional<int64_t> residual_h;
  std::optional<int64_t> residual_w;
  if (has_residual_add) {
    residual_n = StaticDim(full_residual_type_node, 0);
    residual_channels = StaticDim(full_residual_type_node, 1);
    residual_h = StaticDim(full_residual_type_node, 2);
    residual_w = StaticDim(full_residual_type_node, 3);
    if (!residual_n.has_value() || !residual_channels.has_value() || !residual_h.has_value() ||
        !residual_w.has_value() || residual_channels.value() != out_channels.value()) {
      return result;
    }
  }

  int split_num = candidate.attrs.split_num;
  if (split_num <= 1 || split_num > out_channels.value()) {
    return result;
  }

  TensorType data_type = TensorTypeRef(data_type_node);
  TensorType full_weight_type = TensorTypeRef(full_weight_type_node);
  TensorType full_bias_type = has_bias_add ? TensorTypeRef(full_bias_type_node) : TensorType();
  TensorType full_residual_type =
      has_residual_add ? TensorTypeRef(full_residual_type_node) : TensorType();
  TensorType full_output_type = TensorTypeRef(output_type_node);
  const auto* bias_add_attrs =
      has_bias_add ? pattern_info.bias_add_call->attrs.as<::tvm::relay::BiasAddAttrs>() : nullptr;
  const auto* leaky_relu_attrs = has_leaky_relu
                                     ? pattern_info.leaky_relu_call->attrs.as<::tvm::relay::LeakyReluAttrs>()
                                     : nullptr;

  Array<Expr> concat_inputs;
  Array<Type> concat_input_types;
  std::string tag = candidate.call_var->name_hint();
  // Reused Conv2D part functions still contain function-local temporary workspace accesses after
  // lowering.  The current parallel memory planner assigns those temporary buffers per PrimFunc,
  // not per call site, so enabling both features can make different split parts alias the same
  // scratch space.  Keep reuse disabled in parallel-memory mode until the scratch workspace is
  // passed as an explicit per-call parameter.
  const bool enable_part_reuse =
      ::tvm::neutvm::IsCPUConv2DPartReuseEnabled() &&
      !::tvm::neutvm::IsCPUParallelMemoryEnabled();
  std::unordered_map<std::string, ReusablePartFunction> reusable_part_funcs;

  for (int part = 0; part < split_num; ++part) {
    int begin_oc = static_cast<int>((out_channels.value() * part) / split_num);
    int end_oc = static_cast<int>((out_channels.value() * (part + 1)) / split_num);
    int part_oc = end_oc - begin_oc;
    if (part_oc <= 0) {
      return SplitResult();
    }

    Array<PrimExpr> part_weight_shape{Integer(part_oc), Integer(in_channels.value()),
                                      Integer(kh.value()), Integer(kw.value())};
    TensorType part_weight_type(part_weight_shape, full_weight_type_node->dtype);
    TensorType part_bias_type;
    if (has_bias_add) {
      Array<PrimExpr> part_bias_shape{Integer(part_oc)};
      part_bias_type = TensorType(part_bias_shape, full_bias_type_node->dtype);
    }

    Array<PrimExpr> part_output_shape;
    for (const PrimExpr& dim : output_type_node->shape) {
      part_output_shape.push_back(dim);
    }
    part_output_shape.Set(1, Integer(part_oc));
    TensorType part_output_type(part_output_shape, output_type_node->dtype);
    TensorType part_residual_type;
    if (has_residual_add) {
      part_residual_type = part_output_type;
    }

    Function slice_func =
        MakeKernelSliceFunction(tag, part, full_weight_type, part_weight_type, begin_oc, end_oc,
                                static_cast<int>(in_channels.value()), static_cast<int>(kh.value()),
                                static_cast<int>(kw.value()));
    FuncType slice_func_type({full_weight_type}, part_weight_type, Array<TypeVar>(), {});
    Var slice_func_var("neutvm_conv2d_slice_fn" + std::to_string(part), slice_func_type);
    result.bindings.push_back({slice_func_var, slice_func});

    Var sliced_weight_var("neutvm_conv2d_weight" + std::to_string(part), part_weight_type);
    result.bindings.push_back({sliced_weight_var, Call(slice_func_var, {actual_weight})});

    Var sliced_bias_var;
    if (has_bias_add) {
      Function bias_slice_func =
          MakeBiasSliceFunction(tag, part, full_bias_type, part_bias_type, begin_oc, end_oc);
      FuncType bias_slice_func_type({full_bias_type}, part_bias_type, Array<TypeVar>(), {});
      Var bias_slice_func_var("neutvm_conv2d_bias_slice_fn" + std::to_string(part),
                              bias_slice_func_type);
      result.bindings.push_back({bias_slice_func_var, bias_slice_func});

      sliced_bias_var = Var("neutvm_conv2d_bias" + std::to_string(part), part_bias_type);
      result.bindings.push_back({sliced_bias_var, Call(bias_slice_func_var, {actual_bias})});
    }

    Var sliced_residual_var;
    if (has_residual_add) {
      Function residual_slice_func = MakeResidualSliceFunction(
          tag, part, full_residual_type, part_residual_type, begin_oc, end_oc,
          static_cast<int>(residual_n.value()), static_cast<int>(residual_h.value()),
          static_cast<int>(residual_w.value()));
      FuncType residual_slice_func_type({full_residual_type}, part_residual_type, Array<TypeVar>(),
                                        {});
      Var residual_slice_func_var("neutvm_conv2d_residual_slice_fn" + std::to_string(part),
                                  residual_slice_func_type);
      result.bindings.push_back({residual_slice_func_var, residual_slice_func});

      sliced_residual_var =
          Var("neutvm_conv2d_residual" + std::to_string(part), part_residual_type);
      result.bindings.push_back(
          {sliced_residual_var, Call(residual_slice_func_var, {actual_residual})});
    }

    std::string part_reuse_key;
    auto reusable_it = reusable_part_funcs.end();
    if (enable_part_reuse) {
      part_reuse_key = MakeConv2DPartReuseKey(
          candidate.attrs.pattern, data_type, part_weight_type, part_bias_type, part_residual_type,
          part_output_type, attrs, bias_add_attrs, leaky_relu_attrs, part_oc);
      reusable_it = reusable_part_funcs.find(part_reuse_key);
    }
    Var conv_func_var;
    if (enable_part_reuse && reusable_it != reusable_part_funcs.end()) {
      conv_func_var = reusable_it->second.func_var;
    } else {
      Function conv_func;
      FuncType conv_func_type;
      if (has_residual_add) {
        conv_func = MakeConv2DBiasAddSiLUResidualAddPartFunction(
            tag, part, data_type, part_weight_type, part_bias_type, part_residual_type,
            part_output_type, attrs, bias_add_attrs, part_oc);
        conv_func_type = FuncType({data_type, part_weight_type, part_bias_type, part_residual_type},
                                  part_output_type, Array<TypeVar>(), {});
      } else if (has_bias_add && has_silu) {
        conv_func = MakeConv2DBiasAddSiLUPartFunction(tag, part, data_type, part_weight_type,
                                                      part_bias_type, part_output_type, attrs,
                                                      bias_add_attrs, part_oc);
        conv_func_type = FuncType({data_type, part_weight_type, part_bias_type}, part_output_type,
                                  Array<TypeVar>(), {});
      } else if (has_bias_add && has_leaky_relu) {
        conv_func = MakeConv2DBiasAddLeakyReLUPartFunction(
            tag, part, data_type, part_weight_type, part_bias_type, part_output_type, attrs,
            bias_add_attrs, leaky_relu_attrs, part_oc);
        conv_func_type = FuncType({data_type, part_weight_type, part_bias_type}, part_output_type,
                                  Array<TypeVar>(), {});
      } else if (has_bias_add) {
        conv_func =
            MakeConv2DBiasAddPartFunction(tag, part, data_type, part_weight_type, part_bias_type,
                                          part_output_type, attrs, bias_add_attrs, part_oc);
        conv_func_type = FuncType({data_type, part_weight_type, part_bias_type}, part_output_type,
                                  Array<TypeVar>(), {});
      } else {
        conv_func = MakeConv2DPartFunction(tag, part, data_type, part_weight_type, part_output_type,
                                           attrs, part_oc);
        conv_func_type =
            FuncType({data_type, part_weight_type}, part_output_type, Array<TypeVar>(), {});
      }

      size_t conv_func_index = enable_part_reuse ? reusable_part_funcs.size() : part;
      conv_func_var = Var("neutvm_conv2d_part_fn" + std::to_string(conv_func_index),
                          conv_func_type);
      result.bindings.push_back({conv_func_var, conv_func});
      if (enable_part_reuse) {
        reusable_part_funcs.emplace(part_reuse_key, ReusablePartFunction{conv_func_var});
      }
    }

    Var part_output_var("neutvm_conv2d_output" + std::to_string(part), part_output_type);
    if (has_residual_add) {
      result.bindings.push_back(
          {part_output_var, Call(conv_func_var, {actual_data, sliced_weight_var, sliced_bias_var,
                                                 sliced_residual_var})});
    } else if (has_bias_add) {
      result.bindings.push_back(
          {part_output_var,
           Call(conv_func_var, {actual_data, sliced_weight_var, sliced_bias_var})});
    } else {
      result.bindings.push_back(
          {part_output_var, Call(conv_func_var, {actual_data, sliced_weight_var})});
    }

    concat_inputs.push_back(part_output_var);
    concat_input_types.push_back(part_output_type);
  }

  Function concat_func = MakeConcatenateFunction(tag, concat_input_types, full_output_type, 1);
  TupleType concat_tuple_type(concat_input_types);
  FuncType concat_func_type({concat_tuple_type}, full_output_type, Array<TypeVar>(), {});
  Var concat_func_var("neutvm_conv2d_concat_fn", concat_func_type);
  result.bindings.push_back({concat_func_var, concat_func});

  Expr concat_call = Call(concat_func_var, {Tuple(concat_inputs)});
  Expr final_value = WrapLikeOriginalCallValue(candidate.call_value, concat_call);
  result.bindings.push_back({candidate.call_var, final_value});

  result.changed = true;
  return result;
}

}  // namespace cpu
}  // namespace relay
}  // namespace neutvm
}  // namespace tvm
