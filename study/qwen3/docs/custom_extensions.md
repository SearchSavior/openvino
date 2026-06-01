# OpenVINO custom op extensions — practical guide

A from-zero guide to writing and using custom operations in OpenVINO,
covering the three integration surfaces: Python (`openvino.Op` subclass),
native C++ (`.so` loaded via `core.add_extension`), and OpenVINO GenAI
(`ov_genai.LLMPipeline(..., extensions=[so])`).

Every snippet here came out of the `study/qwen3` work and is a
working pattern in this repo — file references point to the actual
sources you can copy from.

---

## TL;DR

| You want to... | Use this path | See |
|---|---|---|
| Prototype a custom op in 30 lines | Python `Op` subclass | [Python Op](#1-python-custom-op) |
| Ship a fast custom op | C++ extension `.so` | [C++ Op](#2-c-custom-op) |
| Use a custom op with OV GenAI | Build `.so`, pass `extensions=[...]` | [GenAI](#3-openvino-genai-extensions) |
| Replace a subgraph with your op | Graph rewrite | [Rewrites](#4-graph-rewrites) |
| Make sure your fast C++ path actually runs | Serialize/reload trick | [Gotchas](#serialize--reload-pattern) |

---

## 0. Mental model

A **custom op** is an OV `Node` subclass that:
- declares its output shape and element type from its inputs
  (`validate_and_infer_types`),
- gets cloned during plugin transformations
  (`clone_with_new_inputs`),
- runs at inference (`evaluate`).

Once an op is registered with an `ov::Core`, you can build it into an IR
and the plugin will dispatch to your `evaluate` when it encounters that
op type during inference.

There are two flavours of registration:

- **Python flavour** — `core.add_extension(YourPyOpClass)`. Convenient
  for prototyping; `evaluate` runs in Python (with numpy / ctypes).
- **C++ flavour** — `core.add_extension("/path/to/libfoo.so")`. The `.so`
  exports a factory via `OPENVINO_CREATE_EXTENSIONS`. `evaluate` is
  native C++. Required for OV GenAI to use the op.

Both flavours register an op by its **type name** (the `OPENVINO_OP("Foo")`
string). When the same name is registered both ways, the Python one wins
unless you use the [serialize/reload trick](#serialize--reload-pattern).

---

## 1. Python custom op

### Minimum viable op

```python
import numpy as np
import openvino as ov
from openvino import Op


class MyOp(Op):
    """y = x + bias_scalar."""

    # Required: forward inputs to the base Op.
    def __init__(self, inputs=None):
        super().__init__(self, inputs)

    # Required: declare output element type and shape.
    def validate_and_infer_types(self):
        self.set_output_type(
            0,
            self.get_input_element_type(0),
            self.get_input_partial_shape(0),  # same shape as input
        )

    # Required: clone during plugin passes.
    def clone_with_new_inputs(self, new_inputs):
        return MyOp(list(new_inputs))

    # Required: visit attributes for serialization. Return True if no attrs.
    def visit_attributes(self, visitor):
        return True

    # Required: tell OV the op has a Python evaluate.
    def has_evaluate(self):
        return True

    # Required: the actual compute.
    def evaluate(self, outputs, inputs):
        x = np.asarray(inputs[0].data)
        outputs[0].shape = x.shape
        np.asarray(outputs[0].data)[...] = x + 1.0
        return True
```

### Wiring it into a graph

```python
from openvino import opset15 as ops

# Register with a Core.
core = ov.Core()
core.add_extension(MyOp)

# Build a tiny model that uses MyOp.
param = ops.parameter([1, 4], ov.Type.f32, name="x")
my = MyOp([param.output(0)])
result = ops.result(my.output(0))
model = ov.Model([result], [param])

# Compile and infer.
compiled = core.compile_model(model, "CPU")
req = compiled.create_infer_request()
out = req.infer({0: np.zeros((1, 4), dtype=np.float32)})
print(out[result.output(0)])   # [[1, 1, 1, 1]]
```

### Multiple outputs

`validate_and_infer_types` calls `set_output_type(idx, ...)` once per
output. `evaluate` writes to each `outputs[idx]`:

```python
def validate_and_infer_types(self):
    self.set_output_type(0, ov.Type.f32, self.get_input_partial_shape(0))
    self.set_output_type(1, ov.Type.i8,  self.get_input_partial_shape(0))

def evaluate(self, outputs, inputs):
    x = np.asarray(inputs[0].data)
    outputs[0].shape = x.shape
    outputs[1].shape = x.shape
    np.asarray(outputs[0].data)[...] = x.astype(np.float32)
    np.asarray(outputs[1].data)[...] = np.clip(x.round(), -128, 127).astype(np.int8)
    return True
```

### Dispatching to a C kernel via ctypes

For real performance you want the Python op to be a thin wrapper around
a C kernel. The pattern used throughout this repo:

```python
# kernels.py wraps a libqwen3_kernels.so via ctypes
import ctypes
from pathlib import Path

_SO = ctypes.CDLL(str(Path(__file__).parent / "libqwen3_kernels.so"))
_SO.my_kernel.restype = None
_SO.my_kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

def my_kernel(x, out):
    assert x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    assert out.dtype == np.float32 and out.flags["C_CONTIGUOUS"]
    _SO.my_kernel(x.ctypes.data, out.ctypes.data, x.size)


# In the Op evaluate:
def evaluate(self, outputs, inputs):
    x = np.asarray(inputs[0].data)
    outputs[0].shape = x.shape
    out = np.asarray(outputs[0].data)
    x_c = np.ascontiguousarray(x, dtype=np.float32)
    out_c = np.empty_like(x_c)
    my_kernel(x_c, out_c)
    out[...] = out_c
    return True
```

Build the C kernel as a separate `.so` and load it from Python. This is
**not** the same `.so` you load via `core.add_extension(...)` — that one
is a C++ OV extension.

A real example: [`kernels/kernels.py`](../kernels/kernels.py) loads
`libqwen3_kernels.so` built from [`kernels/kernels.c`](../kernels/kernels.c)
by [`kernels/build_kernels.sh`](../kernels/build_kernels.sh).

### Examples in this repo

- [`kernels/fused_linear_attn.py`](../kernels/fused_linear_attn.py) —
  `GatedDeltaRule` / V2 / V3 with two signatures and a `evaluate`
  dispatching to ctypes.
- [`kernels/quantized_kv.py`](../kernels/quantized_kv.py) —
  `QuantizedKVCache` (3 inputs, 3 outputs).
- [`kernels/quantized_matmul.py`](../kernels/quantized_matmul.py) —
  `QuantizedMatMul` with both numpy fallback and ctypes-to-C path.

---

## 2. C++ custom op

You build a `.so` exporting a factory that creates `ov::OpExtension`s
for each of your ops. Loading the `.so` via `core.add_extension(path)`
makes those ops available to the plugin and to OV GenAI.

### Header

```cpp
// my_op.hpp
#pragma once
#include <openvino/op/op.hpp>

namespace MyExt {

class MyOp : public ov::op::Op {
public:
    OPENVINO_OP("MyOp");                  // must match the IR type name
    MyOp() = default;
    MyOp(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace MyExt
```

### Implementation

```cpp
// my_op.cpp
#include "my_op.hpp"
#include <cstring>

using namespace MyExt;

MyOp::MyOp(const ov::OutputVector& args) : Op(args) {
    constructor_validate_and_infer_types();   // calls validate_and_infer_types
}

void MyOp::validate_and_infer_types() {
    const auto et = get_input_element_type(0);
    set_output_type(0, et, get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> MyOp::clone_with_new_inputs(
        const ov::OutputVector& new_args) const {
    return std::make_shared<MyOp>(new_args);
}

bool MyOp::visit_attributes(ov::AttributeVisitor&) { return true; }
bool MyOp::has_evaluate() const { return true; }

bool MyOp::evaluate(ov::TensorVector& outputs,
                    const ov::TensorVector& inputs) const {
    const auto& in = inputs[0];
    auto& out = outputs[0];
    out.set_shape(in.get_shape());

    const float* xp = static_cast<const float*>(in.data());
    float* yp = static_cast<float*>(out.data());
    const size_t n = in.get_size();
    for (size_t i = 0; i < n; ++i) yp[i] = xp[i] + 1.0f;
    return true;
}
```

### Registration entry point

`OPENVINO_CREATE_EXTENSIONS` is the macro the dynamic loader looks for:

```cpp
// extension.cpp
#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include "my_op.hpp"

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        std::make_shared<ov::OpExtension<MyExt::MyOp>>(),
    }));
```

Add one `OpExtension<>` per op class you want to expose.

### CMake

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_ov_ext LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate the OpenVINO installed in the active Python interpreter.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')"
    OUTPUT_VARIABLE OpenVINO_DIR_PY)
find_package(OpenVINO REQUIRED PATHS "${OpenVINO_DIR_PY}")

add_library(my_ov_ext MODULE
    my_op.cpp
    extension.cpp)

set_target_properties(my_ov_ext PROPERTIES PREFIX "lib")
target_compile_options(my_ov_ext PRIVATE -O3 -march=native -ffast-math -Wall)
target_link_libraries(my_ov_ext PRIVATE openvino::runtime)

# CRITICAL: do NOT link libgomp. See "Gotchas: libgomp vs libtbb" below.
```

Build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# Produces build/libmy_ov_ext.so
```

### Using the .so from Python

```python
import openvino as ov

core = ov.Core()
core.add_extension("/path/to/libmy_ov_ext.so")
compiled = core.compile_model("/path/to/model_with_my_op.xml", "CPU")
```

### Examples in this repo

[`cpp_ext/`](../cpp_ext/) has seven C++ ops:

- `GatedDeltaRule`, `GatedDeltaRuleV2`, `GatedDeltaRuleV3` —
  linear-attention recurrence, with progressively more absorbed work.
- `FusedCausalConv1d`.
- `QuantizedKVCache`, `QuantizedKVCacheUpdate`.
- `QuantizedInt8SDPA`.

[`cpp_ext/ov_extension.cpp`](../cpp_ext/ov_extension.cpp) is the
canonical `OPENVINO_CREATE_EXTENSIONS` block;
[`cpp_ext/CMakeLists.txt`](../cpp_ext/CMakeLists.txt) is the canonical
build setup (note the explicit "do not link libgomp" comment).

---

## 3. OpenVINO GenAI extensions

OV GenAI's `LLMPipeline` and `VLMPipeline` accept an `extensions=[...]`
kwarg that's forwarded to the underlying `ov.Core`:

```python
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline(
    "/path/to/model_dir",
    "CPU",
    extensions=["/path/to/libmy_ov_ext.so"],
)
cfg = ov_genai.GenerationConfig(); cfg.max_new_tokens = 32
print(pipe.generate("Hello", generation_config=cfg))
```

Requirements:

- `openvino_genai >= 2026.3` (the `extensions=` property landed in
  2026.3).
- The model dir must contain a rewritten IR that uses your custom op
  types. The simplest way:
    1. Read the original IR with a `Core` that has your Python op
       classes registered.
    2. Apply your graph rewrites.
    3. `ov.serialize(model, .../openvino_language_model.xml, .../openvino_language_model.bin)`.
    4. Symlink the other files (tokenizer, embeddings, config) so the
       genai pipeline can find them.

A worked example: [`genai/genai_vlm_pipeline.py`](../genai/genai_vlm_pipeline.py).
The pattern is:

```python
def make_fused_dir():
    # symlink every original file except the LM .xml/.bin
    FUSED.mkdir(parents=True, exist_ok=True)
    for f in ORIG.iterdir():
        dst = FUSED / f.name
        if dst.exists(): dst.unlink()
        if f.name in {"openvino_language_model.xml",
                      "openvino_language_model.bin"}:
            continue
        dst.symlink_to(f, target_is_directory=f.is_dir())

def rewrite_and_save():
    model = ov.Core().read_model(str(ORIG / "openvino_language_model.xml"))
    apply_my_rewrites(model)
    ov.serialize(model, str(FUSED / "openvino_language_model.xml"),
                        str(FUSED / "openvino_language_model.bin"))

make_fused_dir()
rewrite_and_save()
vlm = ov_genai.VLMPipeline(str(FUSED), "CPU",
                           extensions=[str(SO_PATH)])
```

Because the genai process loads only the `.so` (no Python custom op
classes are registered in its `Core`), the C++ `evaluate` is what runs.

---

## 4. Graph rewrites

A graph rewrite is just Python that walks `model.get_ops()`, finds a
subgraph it wants to replace, and rewires the outputs.

### The minimum

```python
def replace_my_pattern(model):
    n = 0
    for op in model.get_ops():
        if op.get_type_name() != "Multiply":
            continue
        # ... check it's the Multiply you want ...
        a = op.input(0).get_source_output()
        b = op.input(1).get_source_output()
        fused = MyOp([a, b])
        op.output(0).replace(fused.output(0))    # rewire all consumers
        n += 1
    return n
```

### Walking up an input chain

```python
def trace_back(start_node, max_hops=10):
    n = start_node
    for hop in range(max_hops):
        yield hop, n
        if n.get_input_size() == 0: return
        n = n.input(0).get_source_output().get_node()
```

Useful for finding the producer of a value. Real example:
[`kernels/fused_linear_attn.py::_find_mixed_qkv_for_q`](../kernels/fused_linear_attn.py).

### Replacing only some consumers

`output.replace(new_output)` rewires **all** consumers. To rewire
selectively, iterate `get_target_inputs()`:

```python
for consumer in list(old_node.output(0).get_target_inputs()):
    if consumer.get_node() is keep_this_consumer:
        continue
    consumer.replace_source_output(new_node.output(0))
```

### Stateful variables

```python
from openvino.op.util import Variable, VariableInfo
from openvino import opset15 as ops

# Create a new Variable.
info = VariableInfo()
info.variable_id = "my_state.i8"
info.data_type = ov.Type.i8
info.data_shape = ov.PartialShape([-1, 16, -1, 128])
var = Variable(info)
model.add_variables([var])

# Initialiser + ReadValue.
init = ops.constant(np.zeros((1, 16, 0, 128), dtype=np.int8))
rv = ops.read_value(init, var)

# Assign back at the end.
new_value = ...  # some Node producing data of the right type
asg = ops.assign(new_value, var)
model.add_sinks([asg])
```

Remove an old Variable: `model.remove_variable(var)`. Both `ReadValue`
and `Assign` referencing it must be gone first. Real example:
[`kernels/quantized_kv.py::replace_kv_with_int8`](../kernels/quantized_kv.py).

### Python API limits

There is **no `model.remove_node(op)`**. You can only remove parameters,
results, sinks (Assigns), and Variables. Other ops become "dead" when
nothing references them and the plugin's DCE elides them at compile —
but only if they're truly unreachable from any sink / result /
parameter.

If you need to delete arbitrary ops (e.g. an orphan `ReadValue`), the
workaround is to **redirect the orphan's consumers to a stand-in
Constant**, then `model.remove_variable(...)`. See the dead-chain
cleanup at the end of `replace_kv_with_int8_sdpa`.

---

## Serialize / reload pattern

**The single most important gotcha.** When you have:

- a Python `Op` subclass registered via `core.add_extension(YourPyOp)`,
  and
- a `.so` registered via `core.add_extension("/path/to/libfoo.so")`,

both exposing an op with the same `OPENVINO_OP("YourOp")` name — **the
Python `evaluate()` always wins**. Verified empirically by patching the
Python `evaluate` to print and observing it called on every step, even
with the `.so` loaded.

The order of `add_extension` calls doesn't matter. Even importing the
Python class is enough — subclasses of `Op` auto-register globally.

To make the C++ `evaluate` actually run, **serialize the IR and
re-load it with a fresh `ov.Core()` that has ONLY the `.so` registered**:

```python
# Step 1: build the IR using the Python class (needed for IR construction).
core_build = ov.Core()
core_build.add_extension(MyPyOp)
model = core_build.read_model("/path/to/orig.xml")
apply_my_rewrites(model)             # uses MyPyOp internally
ov.serialize(model, "/tmp/rewritten.xml", "/tmp/rewritten.bin")

# Step 2: fresh Core with ONLY the .so registered.
core_run = ov.Core()                 # NB: do not import the Python class here
core_run.add_extension("/path/to/libmy_ov_ext.so")
model2 = core_run.read_model("/tmp/rewritten.xml")
compiled = core_run.compile_model(model2, "CPU")
```

The second `Core` resolves "MyOp" to the C++ implementation. Verified:
Python `evaluate` is called 0 times, C++ `evaluate` runs.

`scripts/working/bench_v2.py` and `scripts/working/bench_v3.py` both
follow this exact pattern.

---

## Gotchas

### libgomp vs libtbb

**Do not link your C++ extension `.so` against libgomp.** OV uses
libtbb internally. When both threading runtimes coexist in the same
process they fight each other.

In this repo, just loading a libgomp-linked `.so` via
`core.add_extension(SO)` dropped baseline decode throughput by 5×
(10.43 → 1.82 tok/s) even when no custom op was invoked.

Fix in CMake:

```cmake
target_link_libraries(my_ov_ext PRIVATE openvino::runtime)
# NB: NO target_link_libraries(... OpenMP::OpenMP_CXX) here.
```

`#pragma omp parallel for` becomes a no-op without `-fopenmp`. To parallelise
inside the C++ Op anyway, use `std::thread`:

```cpp
#include <thread>
#include <vector>

const int n_threads = std::min(B * H, (int)std::thread::hardware_concurrency());
std::vector<std::thread> workers;
const int chunk = (B * H + n_threads - 1) / n_threads;
for (int t = 0; t < n_threads; ++t) {
    int s = t * chunk, e = std::min(s + chunk, B * H);
    workers.emplace_back([=]() {
        my_kernel_slice(..., s, e);
    });
}
for (auto& th : workers) th.join();
```

Real example:
[`cpp_ext/quantized_int8_sdpa.cpp`](../cpp_ext/quantized_int8_sdpa.cpp).
The Python ctypes path keeps libgomp in its own `libqwen3_kernels.so`
(loaded outside the OV plugin process, so no conflict).

### `-D_GNU_SOURCE` for `sched_setaffinity` and friends

If your C code uses `cpu_set_t` / `CPU_SET` / `sched_setaffinity`, the
glibc headers gate those behind `_GNU_SOURCE`. Define it on the compile
line:

```bash
gcc ... -D_GNU_SOURCE kernels.c
```

```cmake
target_compile_definitions(my_ov_ext PRIVATE _GNU_SOURCE)
```

Without this, the `.so` builds but fails to load with
`undefined symbol: CPU_SET`.

### TBB pins the calling thread

When you `evaluate` from inside the OV plugin, you're called from a TBB
worker pinned to one CPU. Any `#pragma omp parallel` inherits that
affinity and ends up running 4 threads on 1 core. Reset affinity before
parallelising:

```c
#include <sched.h>
static void unpin_self(void) {
    cpu_set_t cs; CPU_ZERO(&cs);
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 0) n = 4;
    for (long i = 0; i < n; ++i) CPU_SET(i, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
}
```

We verified the slowdown is real: `taskset -c 0` on the standalone
kernel reproduced the in-OV slowdown exactly (60 ms pinned vs 14 ms
unpinned for `qmm_kernel` at lm_head size).

### Op `output.replace(...)` rewires everything

`a_output.replace(b_output)` redirects **every consumer** of `a_output`
to `b_output`. That includes ops you didn't realise consumed it
(Assigns, ShapeOfs, etc.). When you only want to rewire some consumers,
iterate `get_target_inputs()` and call `replace_source_output` per
consumer. See [Replacing only some consumers](#replacing-only-some-consumers).

### Dynamic shapes in `validate_and_infer_types`

When an input dim is dynamic, propagate it as dynamic. Don't substitute
a guess (`T_q=128`) — you'll break re-compilation if the model is
reshaped:

```python
def validate_and_infer_types(self):
    mqkv = self.get_input_partial_shape(0)         # [B, T, qkv_dim]
    gps  = self.get_input_partial_shape(1)         # [B, T, H]
    sps  = self.get_input_partial_shape(3)         # [B, H, D, D]
    out_shape = ov.PartialShape([mqkv[0], mqkv[1], gps[2], sps[3]])
    self.set_output_type(0, self.get_input_element_type(0), out_shape)
```

To get static shapes for analysis (not for runtime), use
`model.reshape({input: [B, T, ...]})` before compile_model — see
[`scripts/working/attribute_mem.py`](../scripts/working/attribute_mem.py).

### `opset.assign` only accepts a single-output Node

```python
# Won't work — qkv.output(1) is an Output, not a Node:
asg = ops.assign(qkv.output(1), var)   # RuntimeError

# Wrap through an identity Convert (single-output Node):
data_out = ops.convert(qkv.output(1), ov.Type.i8)
asg = ops.assign(data_out, var)
```

Affects all multi-output custom ops you want to write back into state.

---

## Debugging and measurement

### Inspecting the post-compile graph

```python
compiled = core.compile_model(model, "CPU", {"PERF_COUNT": True})
rt = compiled.get_runtime_model()
for op in rt.get_ops():
    if op.get_type_name() == "Constant": continue
    print(op.get_type_name(), op.get_friendly_name(),
          op.get_output_partial_shape(0),
          op.get_output_element_type(0).get_type_name())
```

Use `rt.get_ops()` to walk what the plugin actually executes (different
from `model.get_ops()` — the plugin's transformations have run).

### Per-node profiling

```python
req.infer({...})
prof = req.get_profiling_info()
prof.sort(key=lambda p: p.real_time, reverse=True)
for p in prof[:10]:
    print(f"{p.real_time.total_seconds()*1e6:>10.0f} us "
          f"{p.node_type:<22s} {p.exec_type:<22s} {p.node_name}")
```

`exec_type` tells you which plugin kernel handled the node (oneDNN,
JIT, Reference, etc.). Custom Python/C++ ops show as `Reference`.

### Memory attribution

[`scripts/working/attribute_mem.py`](../scripts/working/attribute_mem.py)
is the canonical tool for understanding *where* the bytes go. It:

- walks Constants in the source IR to bucket weights by role,
- binds dynamic dims via `model.reshape({input: [B, T_q, ...]})`,
- walks `get_runtime_model()` and sums per-tensor bytes,
- buckets activations by friendly-name prefix
  (linear_attn / self_attn / mlp / other).

Run it:

```bash
python scripts/working/attribute_mem.py --config baseline \
    --T_q 128 --T_full 2048 --top 20
```

### oneDNN verbose

The fastest way to see what oneDNN primitive ran for each op, with full
memory descriptors:

```bash
ONEDNN_VERBOSE=all python my_inference.py 2> /tmp/onednn.log
```

Each line includes the op type, input shapes / layouts / element types,
and time in microseconds.

### Peak RSS sampler

`smaps_rollup` is cheap and gives both anonymous and file-backed bytes:

```python
import threading, time

def rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024

class PeakSampler:
    def __init__(self, period=0.005):
        self.period, self.peak = period, 0
        self._stop = threading.Event()
    def __enter__(self):
        self.peak = rss_mb()
        def loop():
            while not self._stop.is_set():
                self.peak = max(self.peak, rss_mb())
                time.sleep(self.period)
        self.t = threading.Thread(target=loop, daemon=True); self.t.start()
        return self
    def __exit__(self, *exc):
        self._stop.set(); self.t.join()

with PeakSampler() as p:
    req.infer({...})
print(f"peak RSS during infer: {p.peak:.0f} MiB")
```

Run each config in a **fresh subprocess** when comparing peak RSS —
weights and the plugin's compile-time scratch from earlier configs
accumulate in the same Python process and inflate later peaks.

---

## End-to-end skeleton

Putting it all together, a minimal "build a custom op, use it via OV
GenAI" project looks like:

```
my_ext/
├── kernels.c kernels.h            # C kernels
├── build_kernels.sh               # builds libmy_kernels.so for ctypes
├── my_op.py                       # Python Op subclass + rewrite
├── cpp/                           # OV C++ extension
│   ├── my_op.{hpp,cpp}
│   ├── extension.cpp              # OPENVINO_CREATE_EXTENSIONS
│   └── CMakeLists.txt
└── run.py                         # rewrite + serialize + use via genai
```

```python
# run.py
import openvino as ov, openvino_genai as ov_genai
from pathlib import Path
from my_op import MyOp, replace_my_pattern

ORIG = Path("/path/to/original/model")
FUSED = Path("/tmp/fused")
SO = Path("cpp/build/libmy_ov_ext.so").resolve()

# 1. rewrite IR with the Python class
core = ov.Core(); core.add_extension(MyOp)
m = core.read_model(str(ORIG / "openvino_language_model.xml"))
replace_my_pattern(m)

# 2. serialize into a directory genai can load
FUSED.mkdir(exist_ok=True)
for f in ORIG.iterdir():
    if f.suffix in {".xml", ".bin"}: continue
    dst = FUSED / f.name
    if dst.exists(): dst.unlink()
    dst.symlink_to(f)
ov.serialize(m, str(FUSED / "openvino_language_model.xml"),
                str(FUSED / "openvino_language_model.bin"))

# 3. drive via OV GenAI with the C++ extension
pipe = ov_genai.LLMPipeline(str(FUSED), "CPU", extensions=[str(SO)])
cfg = ov_genai.GenerationConfig(); cfg.max_new_tokens = 32
print(pipe.generate("Hello", generation_config=cfg))
```

---

## Reference index

In this repo:

- Build scripts: [`kernels/build_kernels.sh`](../kernels/build_kernels.sh),
  [`cpp_ext/CMakeLists.txt`](../cpp_ext/CMakeLists.txt)
- Python op + rewrite: [`kernels/fused_linear_attn.py`](../kernels/fused_linear_attn.py)
  (3 versions, 6/4/6 inputs, with rewrites)
- C++ op:
  [`cpp_ext/gated_delta_rule_v3.{hpp,cpp}`](../cpp_ext/gated_delta_rule_v3.cpp)
- Registration: [`cpp_ext/ov_extension.cpp`](../cpp_ext/ov_extension.cpp)
- GenAI integration: [`genai/genai_vlm_pipeline.py`](../genai/genai_vlm_pipeline.py)
- Serialize/reload bench pattern:
  [`scripts/working/bench_v3.py`](../scripts/working/bench_v3.py)
- Memory attribution: [`scripts/working/attribute_mem.py`](../scripts/working/attribute_mem.py)
- Stateful variable surgery: [`kernels/quantized_kv.py`](../kernels/quantized_kv.py)
- libgomp-vs-libtbb workaround: [`cpp_ext/quantized_int8_sdpa.cpp`](../cpp_ext/quantized_int8_sdpa.cpp)
- TBB-affinity reset: [`kernels/kernels.c::qmm_unpin_self`](../kernels/kernels.c)

In the OV repo (`/home/user/openvino/src`):

- Plugin custom-op extension API: `src/inference/include/openvino/runtime/core.hpp`
  (`Core::add_extension` overloads).
- `OpExtension` template: `src/core/include/openvino/core/op_extension.hpp`.
- `OPENVINO_CREATE_EXTENSIONS` macro: `src/core/include/openvino/core/extension.hpp`.
