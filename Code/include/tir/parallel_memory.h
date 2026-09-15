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
 * \file tvm/neutvm/tir/cpu/parallel_memory.h
 * \brief CPU parallel memory conflict analysis for TIR/USMP.
 */
#ifndef TVM_NEUTVM_TIR_CPU_PARALLEL_MEMORY_H_
#define TVM_NEUTVM_TIR_CPU_PARALLEL_MEMORY_H_

#include <tvm/ir/module.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/string.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/usmp/utils.h>

namespace tvm {
namespace neutvm {
namespace tir {
namespace cpu {

/*!
 * \brief State supplied by TVM's USMP ExtractBufferInfo stage.
 *
 * The neuTVM analysis updates `buffer_info_map` in place by changing the conflicts field of
 * BufferInfo objects. It does not rewrite the PrimFunc or allocate memory by itself.
 */
struct ParallelMemoryAnalysisContext {
  /*! \brief Main TIR function whose call sequence is analyzed. */
  tvm::tir::PrimFunc main_func;
  /*! \brief TIR PrimFuncs addressable from calls inside `main_func`. */
  Map<String, tvm::tir::PrimFunc> functions;
  /*! \brief BufferInfo objects already collected by TVM USMP ExtractBufferInfo. */
  Map<tvm::tir::usmp::BufferInfo, tvm::tir::Stmt>* buffer_info_map{nullptr};
  /*! \brief Optional memory pressure output. The first neuTVM version keeps it at zero. */
  int* memory_pressure_bytes{nullptr};
  /*! \brief Clear pre-existing conflicts before applying neuTVM's replacement strategy. */
  bool clear_existing_conflicts{true};
};

/*!
 * \brief Apply neuTVM CPU parallel memory conflict analysis.
 *
 * This function is intended to be called from TVM's USMP ExtractBufferInfo hook after all
 * BufferInfo objects have been collected.
 */
TVM_DLL void ApplyParallelMemoryAnalysis(ParallelMemoryAnalysisContext* ctx);

}  // namespace cpu
}  // namespace tir
}  // namespace neutvm
}  // namespace tvm

#endif  // TVM_NEUTVM_TIR_CPU_PARALLEL_MEMORY_H_
