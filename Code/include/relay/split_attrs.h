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

/*!
 * \file tvm/neutvm/relay/cpu/split_attrs.h
 * \brief Relay Function attrs used by neuTVM CPU op splitting passes.
 */
#ifndef TVM_NEUTVM_RELAY_CPU_SPLIT_ATTRS_H_
#define TVM_NEUTVM_RELAY_CPU_SPLIT_ATTRS_H_

#include <tvm/relay/function.h>
#include <tvm/runtime/container/string.h>

namespace tvm {
namespace neutvm {
namespace relay {
namespace cpu {

namespace attr {

constexpr const char* kSplitEnabled = "neutvm.split.enabled";
constexpr const char* kSplitRootOp = "neutvm.split.root_op";
constexpr const char* kSplitPattern = "neutvm.split.pattern";
constexpr const char* kSplitAxis = "neutvm.split.axis";
constexpr const char* kSplitNum = "neutvm.split.num";
constexpr const char* kSplitLevel = "neutvm.split.level";
constexpr const char* kGeneratedByOpSplit = "neutvm.split.generated";

}  // namespace attr

struct SplitAttrs {
  bool enabled{false};
  String root_op;
  String pattern;
  String axis;
  int split_num{1};
  String level;
};

TVM_DLL ::tvm::relay::Function WithSplitAttrs(::tvm::relay::Function func,
                                              const SplitAttrs& split_attrs);
TVM_DLL ::tvm::relay::Function MarkGeneratedByOpSplit(::tvm::relay::Function func);
TVM_DLL bool HasSplitAttrs(const ::tvm::relay::Function& func);
TVM_DLL bool IsGeneratedByOpSplit(const ::tvm::relay::Function& func);
TVM_DLL bool IsSplitEnabled(const ::tvm::relay::Function& func);
TVM_DLL String GetSplitRootOp(const ::tvm::relay::Function& func);
TVM_DLL String GetSplitPattern(const ::tvm::relay::Function& func);
TVM_DLL String GetSplitAxis(const ::tvm::relay::Function& func);
TVM_DLL int GetSplitNum(const ::tvm::relay::Function& func);
TVM_DLL String GetSplitLevel(const ::tvm::relay::Function& func);
TVM_DLL SplitAttrs GetSplitAttrs(const ::tvm::relay::Function& func);

}  // namespace cpu
}  // namespace relay
}  // namespace neutvm
}  // namespace tvm

#endif  // TVM_NEUTVM_RELAY_CPU_SPLIT_ATTRS_H_
