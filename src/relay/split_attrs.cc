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

#include <tvm/neutvm/relay/cpu/split_attrs.h>
#include <tvm/runtime/logging.h>

namespace tvm {
namespace neutvm {
namespace relay {
namespace cpu {

namespace {

String GetStringAttrOrDefault(const ::tvm::relay::Function& func, const char* key,
                              const String& default_value) {
  return func->GetAttr<String>(key, default_value).value();
}

int GetIntegerAttrOrDefault(const ::tvm::relay::Function& func, const char* key,
                            Integer default_value) {
  int value = func->GetAttr<Integer>(key, default_value).value()->value;
  ICHECK_GE(value, 1) << key << " must be >= 1, but got " << value;
  return value;
}

}  // namespace

::tvm::relay::Function WithSplitAttrs(::tvm::relay::Function func,
                                      const SplitAttrs& split_attrs) {
  ICHECK(split_attrs.root_op.size() != 0) << attr::kSplitRootOp << " must not be empty";
  ICHECK(split_attrs.axis.size() != 0) << attr::kSplitAxis << " must not be empty";
  ICHECK_GE(split_attrs.split_num, 1) << attr::kSplitNum << " must be >= 1";
  ICHECK(split_attrs.level == "relay" || split_attrs.level == "tir")
      << attr::kSplitLevel << " must be \"relay\" or \"tir\", but got \"" << split_attrs.level
      << "\"";

  func = WithAttr(std::move(func), attr::kSplitEnabled, Bool(split_attrs.enabled));
  func = WithAttr(std::move(func), attr::kSplitRootOp, split_attrs.root_op);
  func = WithAttr(std::move(func), attr::kSplitPattern, split_attrs.pattern);
  func = WithAttr(std::move(func), attr::kSplitAxis, split_attrs.axis);
  func = WithAttr(std::move(func), attr::kSplitNum, Integer(split_attrs.split_num));
  func = WithAttr(std::move(func), attr::kSplitLevel, split_attrs.level);
  return func;
}

::tvm::relay::Function MarkGeneratedByOpSplit(::tvm::relay::Function func) {
  return WithAttr(std::move(func), attr::kGeneratedByOpSplit, Bool(true));
}

bool HasSplitAttrs(const ::tvm::relay::Function& func) {
  return func->GetAttr<Bool>(attr::kSplitEnabled).defined();
}

bool IsGeneratedByOpSplit(const ::tvm::relay::Function& func) {
  return func->GetAttr<Bool>(attr::kGeneratedByOpSplit, Bool(false)).value()->value;
}

bool IsSplitEnabled(const ::tvm::relay::Function& func) {
  return func->GetAttr<Bool>(attr::kSplitEnabled, Bool(false)).value()->value;
}

String GetSplitRootOp(const ::tvm::relay::Function& func) {
  return GetStringAttrOrDefault(func, attr::kSplitRootOp, String(""));
}

String GetSplitPattern(const ::tvm::relay::Function& func) {
  return GetStringAttrOrDefault(func, attr::kSplitPattern, String(""));
}

String GetSplitAxis(const ::tvm::relay::Function& func) {
  return GetStringAttrOrDefault(func, attr::kSplitAxis, String(""));
}

int GetSplitNum(const ::tvm::relay::Function& func) {
  return GetIntegerAttrOrDefault(func, attr::kSplitNum, Integer(1));
}

String GetSplitLevel(const ::tvm::relay::Function& func) {
  return GetStringAttrOrDefault(func, attr::kSplitLevel, String("relay"));
}

SplitAttrs GetSplitAttrs(const ::tvm::relay::Function& func) {
  SplitAttrs attrs;
  attrs.enabled = IsSplitEnabled(func);
  attrs.root_op = GetSplitRootOp(func);
  attrs.pattern = GetSplitPattern(func);
  attrs.axis = GetSplitAxis(func);
  attrs.split_num = GetSplitNum(func);
  attrs.level = GetSplitLevel(func);
  return attrs;
}

}  // namespace cpu
}  // namespace relay
}  // namespace neutvm
}  // namespace tvm
