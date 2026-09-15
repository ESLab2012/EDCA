# Overview

DNN inference on embedded multicore CPUs must balance latency requirements with limited memory budgets. For level-scheduled ahead-of-time (AOT) inference, static memory planning must account for concurrent task execution to safely reuse buffer storage.

EDCA (Execution-Depth-Aware Conflict Analysis) is a compile-time conflict-analysis module integrated into TVM's AOT static memory-planning pipeline. It models intermediate activation buffers and task-local workspaces using conservative stage-residency intervals, then constructs conflict constraints through a depth sweep that enumerates only overlapping pairs. These constraints guide TVM's existing static allocator, enabling parallel-safe memory reuse without runtime conflict analysis.

# Repository Structure



# Quick Start