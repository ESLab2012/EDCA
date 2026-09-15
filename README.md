# Overview

DNN inference on embedded multicore CPUs must balance latency requirements with limited memory budgets. For level-scheduled ahead-of-time (AOT) inference, static memory planning must account for concurrent task execution to safely reuse buffer storage.

EDCA (Execution-Depth-Aware Conflict Analysis) is a compile-time conflict-analysis module integrated into TVM's AOT static memory-planning pipeline. It models intermediate activation buffers and task-local workspaces using conservative stage-residency intervals, then constructs conflict constraints through a depth sweep that enumerates only overlapping pairs. These constraints guide TVM's existing static allocator, enabling parallel-safe memory reuse without runtime conflict analysis.

# Repository Structure

```text
EDCA/
├── Code/
│   ├── include/
│   │   ├── relay/
│   │   │   ├── conv2d.h
│   │   │   ├── split_analysis.h
│   │   │   ├── split_attrs.h
│   │   │   └── split_rewrite.h
│   │   └── tir/
│   │       └── parallel_memory.h
│   └── src/
│       ├── relay/
│       │   ├── conv2d.cc
│       │   ├── split_analysis.cc
│       │   ├── split_attrs.cc
│       │   └── split_rewrite.cc
│       └── tir/
│           └── parallel_memory.cc
├── Models/
│   ├── yolov4_tiny.onnx
│   ├── yolov5n.onnx
│   ├── yolov8n.onnx
│   ├── yolov8s.onnx
│   ├── resnet18.onnx
│   └── resnet50.onnx
└── README.md
```

`Code/` contains source extensions to TVM 0.16.0, organized into two functional modules, with interface declarations in `include/` and implementations in `src/`:

- **Operator partitioning (`relay/`):** Identifies eligible Conv2D operators and statically partitions them along the output-channel dimension, concatenating the subtask outputs to reconstruct the original output.
- **Parallel memory planning (`tir/`):** Constructs buffer-conflict constraints for parallel-safe memory reuse and supplies them to TVM's existing USMP allocator for memory placement.

`Models/` contains six test models in ONNX format, prepared following the official tutorials for their respective models. These files serve as inputs to the project's model compilation and execution workflow.

# Quick Start

The following steps use Ubuntu 20.04 and TVM **v0.16.0**. Run the shell commands in Bash on the compilation host. This repository provides compiler extensions; users supply the TVM configuration and pipeline hooks, the level-scheduled runtime, and the target-platform integration described below.

## 1. Prepare the TVM environment

Install the host compiler, Python development tools, and LLVM 10:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential git python3 python3-dev python3-venv python3-pip \
  libtinfo-dev zlib1g-dev libedit-dev libxml2-dev llvm-10-dev

mkdir -p "$HOME/edca-work"
cd "$HOME/edca-work"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip<25"
python -m pip install "cmake==3.28.4"
```

TVM v0.16.0 requires CMake 3.18 or newer; Ubuntu 20.04's default CMake package may be too old. The commands above use Python 3.8 from Ubuntu 20.04 and install CMake in the virtual environment. See the [TVM v0.16.0 build instructions](https://github.com/apache/tvm/blob/v0.16.0/docs/install/from_source.rst) and [CMake configuration](https://github.com/apache/tvm/blob/v0.16.0/CMakeLists.txt).

Download TVM and its submodules, then install the Python dependencies:

```bash
git clone --branch v0.16.0 --recursive \
  https://github.com/apache/tvm.git tvm-v0.16.0
export TVM_HOME="$PWD/tvm-v0.16.0"
git -C "$TVM_HOME" submodule update --init --recursive

python -m pip install "numpy<2" "onnx==1.16.2" \
  -r "$TVM_HOME/python/requirements/core.txt"

cmake --version
llvm-config-10 --version
```

The NumPy and ONNX constraints are compatibility choices for this older TVM release. Set `EDCA_ROOT` to this repository's location on the Ubuntu host:

```bash
export EDCA_ROOT=/absolute/path/to/EDCA
```

## 2. Integrate the extensions and build TVM

Merge the extensions into the TVM source tree while preserving their header layout. One explicit arrangement is:

```bash
mkdir -p "$TVM_HOME/src/neutvm/edca"
cp -a "$EDCA_ROOT/Code/." "$TVM_HOME/src/neutvm/edca/"
```

This creates `src/neutvm/edca/include/{relay,tir}` and `src/neutvm/edca/src/{relay,tir}` inside TVM. The extension's `conv2d.cc` is not a replacement for TVM's upstream Conv2D operator implementation.

In TVM's top-level `CMakeLists.txt`, add the following after the existing `add_library(tvm_objs OBJECT ${COMPILER_SRCS})` statement:

```cmake
set(EDCA_CODE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src/neutvm/edca")

target_sources(tvm_objs PRIVATE
  "${EDCA_CODE_ROOT}/src/relay/conv2d.cc"
  "${EDCA_CODE_ROOT}/src/relay/split_attrs.cc"
  "${EDCA_CODE_ROOT}/src/relay/split_analysis.cc"
  "${EDCA_CODE_ROOT}/src/relay/split_rewrite.cc"
  "${EDCA_CODE_ROOT}/src/tir/parallel_memory.cc"
)
target_include_directories(tvm_objs PRIVATE
  "${EDCA_CODE_ROOT}/include"
)
```

Before building, complete these integration points in your TVM checkout:

- **Configuration:** Provide `tvm/neutvm/config.h` and the implementations of the configuration getters used by `Code/`, and include their implementation sources in the build. These control operator partitioning, partition count, eligibility thresholds, and parallel memory analysis. For four-way partitioning, configure a split count of four and enable the corresponding Relay partitioning and parallel-memory options. Their configuration interface is supplied by the user; they are not upstream TVM options.
- **Relay pipeline:** Invoke `AnalyzeSplittableOps` and `ApplyOpSplit` at the appropriate typed Relay/ANF stage before lowering to TIR. Their registered names are `relay._transform.neutvm.cpu.AnalyzeSplittableOps` and `relay._transform.neutvm.cpu.ApplyOpSplit`. Check that eligible Conv2D calls have actually been partitioned in the transformed graph.
- **USMP pipeline:** Call `tvm::neutvm::tir::cpu::ApplyParallelMemoryAnalysis` after collecting USMP `BufferInfo` objects and before address placement. Populate `ParallelMemoryAnalysisContext` with the entry TIR function, resolved callees, and buffer-information map. The analysis updates `BufferInfo::conflicts`; TVM's allocator and offset-rewriting passes can then consume those constraints.
- **Scheduling metadata:** Preserve task-buffer mappings and dependency information before USMP offset rewriting for subsequent schedule generation.

Copying the files or enabling USMP alone does not activate these custom passes. Keep the task graph used by conflict analysis consistent with the parallel runtime in Step 4.

Configure and build the host compiler:

```bash
cmake -S "$TVM_HOME" -B "$TVM_HOME/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_LLVM=/usr/bin/llvm-config-10 \
  -DUSE_MICRO=ON \
  -DUSE_CUDA=OFF \
  -DUSE_GTEST=OFF \
  -DUSE_LIBBACKTRACE=OFF
cmake --build "$TVM_HOME/build" --parallel 4
```

This configuration uses LLVM for host-side compilation support and enables microTVM resources for the AOT C-runtime export below. Start with a fresh build directory; if you already have a `config.cmake`, check that its settings do not override these options.

Expose the locally built TVM Python package and shared library:

```bash
export PYTHONPATH="$TVM_HOME/python${PYTHONPATH:+:$PYTHONPATH}"
export TVM_LIBRARY_PATH="$TVM_HOME/build"

python -c "import tvm; print(tvm.__version__); print(tvm.__file__)"
python -c "import tvm; print(tvm.get_global_func('relay._transform.neutvm.cpu.ApplyOpSplit'))"
```

Activate the same virtual environment and restore these variables in subsequent terminals. The registration check confirms that the extension was loaded, not that the compiler pipeline invoked it.

## 3. Compile an ONNX model to AOT source code

Save the following example as `compile_aot.py` in your working directory. It reads concrete input shapes from an ONNX model and exports the generated C sources using TVM's Model Library Format (MLF). Resolve any dynamic input dimensions before compilation.

The example assumes that the custom configuration and pipeline hooks from Step 2 are already implemented and enabled. The standard AOT/USMP parameters below select source generation and static placement; they do not replace those hooks.

```python
import sys
from pathlib import Path

import onnx
import tvm
from tvm import relay
from tvm.micro import export_model_library_format
from tvm.relay.backend import Executor, Runtime

model_path = Path(sys.argv[1])
model = onnx.load(str(model_path))
initializers = {tensor.name for tensor in model.graph.initializer}
shape_dict = {}
for value in model.graph.input:
    if value.name in initializers:
        continue
    dims = value.type.tensor_type.shape.dim
    if any(not d.HasField("dim_value") or d.dim_value <= 0 for d in dims):
        raise ValueError(f"Specify a static shape for input {value.name!r}")
    shape_dict[value.name] = tuple(int(d.dim_value) for d in dims)

mod, params = relay.frontend.from_onnx(
    model, shape=shape_dict, freeze_params=True
)
executor = Executor("aot", {"interface-api": "c", "unpacked-api": True})
runtime = Runtime("crt", {"system-lib": True})

with tvm.transform.PassContext(
    opt_level=0,
    config={
        "tir.disable_vectorize": True,
        "tir.usmp.enable": True,
        "tir.usmp.algorithm": "greedy_by_size",
        "relay.FuseOps.max_depth": 1,
    },
):
    factory = relay.build(
        mod,
        target=tvm.target.Target("c"),
        params=params,
        executor=executor,
        runtime=runtime,
        mod_name="model",
    )

output = Path("build/model.tar")
output.parent.mkdir(parents=True, exist_ok=True)
export_model_library_format(factory, output)
print(f"Exported AOT sources to {output}")
```

Here, `Executor("aot", ...)` selects AOT execution, `target="c"` selects C-source generation, and `Runtime("crt", ...)` selects the C runtime. USMP uses TVM's existing `greedy_by_size` allocator. The fusion setting limits fusion depth to preserve individual operator boundaries, following TVM's [AOT test helper](https://github.com/apache/tvm/blob/v0.16.0/python/tvm/testing/aot.py).

Run the example and unpack the source archive:

```bash
python compile_aot.py "$EDCA_ROOT/Models/resnet18.onnx"
mkdir -p build/model
tar -xf build/model.tar -C build/model
```

The archive includes generated C files under `codegen/host/src/`, the C interface under `codegen/host/include/`, and model metadata, parameter artifacts, and C-runtime resources. Exporting the complete MLF archive retains imported operator modules as well as the entry function. This is a source package, not an executable for the target board. See TVM's [MLF exporter](https://github.com/apache/tvm/blob/v0.16.0/python/tvm/micro/model_library_format.py).

## 4. Provide a level-scheduled parallel runtime

The generated AOT entry function does not itself provide the required level scheduler. Users implement a source-driven integration tool and runtime to execute the compiled operator tasks in parallel:

1. **Recover the task graph.** Use the generated call structure together with task-buffer metadata preserved during compilation. Treat each operator call site as a task and derive producer-consumer dependencies from logical buffers. Preserve this information before USMP rewrites buffers to shared pool offsets: reused physical addresses alone cannot reliably identify the original dependencies.
2. **Form execution levels.** Assign source tasks depth zero and every other task one plus the maximum depth of its predecessors. Group tasks by depth, using the same task boundaries and depths as the memory analysis, including tasks introduced by partitioning.
3. **Dispatch tasks to workers.** Generate task descriptors or dispatch code that binds each compiled function to its arguments and planned buffers. At each level, distribute ready tasks across the workers and wait until all tasks in that level complete before releasing the next level. For a four-worker configuration, create four workers and configure their core affinity as appropriate for the target.
4. **Preserve memory and call contracts.** Allocate and bind the planned pools with the required sizes and alignment, retain input/output and parameter ownership, and preserve the generated call ABI. Concurrent tasks must not share scratch storage unless the memory plan explicitly permits it. Propagate task failures before advancing to another level.

Perform graph extraction and schedule generation on the host; the target runtime executes compiled functions and task descriptors rather than parsing C text on every inference. Do not parallelize a sequentially planned AOT program merely by changing its dispatch order: the memory plan must first be constructed for the same level-scheduled execution contract.

## 5. Cross-compile and deploy on Phytium D2000

The host TVM build generates source code. Deploying that code on D2000 additionally requires a toolchain, board support package (BSP), and runtime libraries for the target operating system.

For RTEMS deployment, follow the [Phytium RTEMS SDK](https://gitee.com/phytium_embedded/phytium-rtems-sdk) and the official RTEMS instructions for [building a BSP](https://docs.rtems.org/docs/6.2/user/start/bsp-build.html) and [building an application](https://docs.rtems.org/docs/6.2/user/start/app.html). Select a D2000 BSP with SMP support and a compatible AArch64 RTEMS toolchain, such as the `aarch64-rtems6` toolchain family. The public Phytium SDK documents RTEMS 6.1 support; using RTEMS 6.2 requires a verified D2000 BSP/port for that version.

Add the generated model sources and parameter definitions, the user-written scheduler, the required TVM CRT support, and the RTEMS application initialization/configuration to your BSP application project. Let that project's build system supply the target compilation flags, startup code, linker script, and RTEMS libraries. Build and load the resulting image using the instructions for your board and BSP.

An AArch64 Linux executable is not an RTEMS executable. Cross-compile the exported C sources with the toolchain and ABI matching the installed BSP; the host compiler and the generated source archive alone are insufficient for deployment.
