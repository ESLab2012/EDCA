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
 * \file tvm/neutvm/relay/cpu/split_rewrite.h
 * \brief Apply neuTVM CPU Relay op splitting according to split attrs.
 */
#ifndef TVM_NEUTVM_RELAY_CPU_SPLIT_REWRITE_H_
#define TVM_NEUTVM_RELAY_CPU_SPLIT_REWRITE_H_

#include <tvm/ir/transform.h>
#include <tvm/neutvm/relay/cpu/split_attrs.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>

#include <vector>

namespace tvm {
namespace neutvm {
namespace relay {
namespace cpu {

struct LetBinding {
  ::tvm::relay::Var var;
  ::tvm::relay::Expr value;
};

struct SplitCandidate {
  ::tvm::relay::Var func_var;
  ::tvm::relay::Function func;
  ::tvm::relay::Var call_var;
  ::tvm::relay::Expr call_value;
  ::tvm::relay::Call call;
  SplitAttrs attrs;
};

struct SplitResult {
  bool changed{false};
  std::vector<LetBinding> bindings;
};

TVM_DLL tvm::transform::Pass ApplyOpSplit();

}  // namespace cpu
}  // namespace relay
}  // namespace neutvm
}  // namespace tvm

#endif  // TVM_NEUTVM_RELAY_CPU_SPLIT_REWRITE_H_
