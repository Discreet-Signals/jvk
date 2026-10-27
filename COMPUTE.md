# jvk compute

General GPU work, beside the rendering: kernels (GLSL compute shaders) over
buffers. It is not about graphics or any one use. A plugin uses it for what
should run wide on the GPU: training, analysis, offline renders.

`include/jvk/compute/Compute.h` is the whole interface. `shaders/include/jvk/`
is a small GLSL library (`compute.glsl`, `scan.glsl`). `jvk_compute_shaders()`
in `CMakeLists.txt` builds a project's kernels.

## In five lines

```cpp
auto context = jvk::compute::Context::create (jvk::Device::acquire(), &why);   // nullptr: no GPU, and why
auto data = context->buffer (n * sizeof (float), jvk::compute::Usage::Shared); // mapped: write it from the CPU
auto scale = context->kernel (mykernels::scale, "scale");                      // bindings and push constants read from its SPIR-V
auto stream = context->stream();
stream->dispatch (scale, jvk::compute::Groups::cover (n, 64, context->limits()), { data.range() }, Push { n, 2.0f });
stream->submit();
while (stream->wait (0.01) == jvk::compute::Stream::Result::pending) { /* a stop asked for? */ }
```

with `scale.comp` beside the code that uses it:

```glsl
#version 450
#extension GL_GOOGLE_include_directive : require
layout(local_size_x = 64) in;
#include "jvk/compute.glsl"
layout(std430, binding = 0) buffer Data { float data[]; };
layout(push_constant) uniform Push { uint n; float by; } pc;
void main()
{
    const uint i = jvkThread();
    if (i < pc.n)
        data[i] *= pc.by;
}
```

and in the project's CMake, once jvk has been added:

```cmake
jvk_compute_shaders(MyTarget HEADER MyKernels.h NAMESPACE mykernels SOURCES scale.comp)
```

## The pieces

- **Context**: the GPU, for compute. `create()` returns nullptr with a reason
  when there is no usable GPU (no Vulkan, no driver, a lost device), so a
  caller can fall back to the CPU. `limits()` gives the device's sizes, which
  work should be sized by and never assumed: workgroup sizes, shared memory,
  subgroup size, storage alignment, whether memory is unified, timestamps.
  Safe to use from any thread.
- **Buffer**: VMA-allocated memory, freed when it goes. `Usage::Device` is
  the GPU's alone and fastest. `Upload` and `Readback` are mapped for one
  direction. `Shared` is mapped both ways: on unified memory (Apple silicon,
  integrated GPUs) it is the GPU's own memory and costs nothing. Call
  `written()` after the CPU writes mapped memory and `toRead()` before it
  reads; both are no-ops on coherent memory.
- **Kernel**: a compute pipeline made from SPIR-V. Its interface is read from
  the SPIR-V: storage and uniform buffers in set 0, one push-constant block,
  and its workgroup size. Specialization constants are set when it is made.
  It goes through the device's pipeline cache, so on MoltenVK the translation
  to Metal is cached on disk too.
- **Stream**: commands recorded, submitted, waited for, then recorded again.
  The commands are `dispatch`, `barrier`, `fill`, `copy`, `upload`,
  `download` and `time`. Each stream has its own command buffer, descriptor
  pools, staging memory and timestamps. A stream is one thread's at a time;
  many may be in flight. `wait(seconds)` returns `pending` on a timeout, so a
  long job can check whether it was asked to stop. It returns `lost` if the
  GPU was lost.

**Order.** A stream's commands start in order but may overlap. Put
`barrier()` between a command that writes memory and any later command that
reads or writes it. That includes kernels, copies, fills, uploads and
downloads.

**Bindings.** A dispatch binds buffer ranges in binding order. Each offset
must be a multiple of `limits().storageAlignment`. Several arrays in one big
buffer is the cheap way to keep many things on the GPU: bind the buffer once
and pass element offsets in push constants.

## The queue

`Device` gives compute its own queue where it can:
- a compute family without graphics where the GPU has one (NVIDIA's and
  AMD's async compute runs beside the editors' frames);
- otherwise, another family with compute (MoltenVK offers four families with
  a queue each, every one a Metal command queue of its own);
- otherwise, a second queue of the graphics family;
- otherwise, the graphics queue itself.

The last case is reported as `limits().ownQueue == false`, and then compute
submits under the renderer's lock. The compute queue has the lower priority.
A kernel is not interrupted once it runs, so keep each submit short (a few
milliseconds) when an editor is open.

## The GLSL library

- `jvk/compute.glsl`: `jvkThread()` and `jvkGroup()`, the indices of a
  one-dimensional dispatch sized by `Groups::cover()`, which spreads its
  workgroups over y past the limit of one dimension.
- `jvk/scan.glsl`: scans of any associative operation in a workgroup.
  - Order is kept, so the operation need not commute: matrix products and
    affine maps work as well as sums.
  - `jvkScanExclusive()`: each thread passes one element.
  - `jvkScanSequence()`: one workgroup scans a sequence of any length, read
    and written through macros.
  - It uses no subgroup operations, so it runs on Vulkan 1.0.
  - It never waits on another workgroup. Vulkan does not promise that one
    workgroup makes progress while another waits for it, and single-pass
    scans that rely on that can hang.

A **linear recursion** (`x[t] = A[t] x[t-1] + b[t]`: IIR filters, phase
accumulators, one-pole smoothers) runs in parallel over time with a scan:
1. Each chunk of the sequence is one affine map of its starting state.
2. A scan of the chunks' maps gives every chunk its starting state.
3. Each chunk runs from there.

Spider's state-variable filter is the worked example: `source/gpu/` in
SpiderSynth.

## Building kernels

`jvk_compute_shaders(<target> HEADER <h> NAMESPACE <ns> SOURCES <.comp>...
[INCLUDE_DIRS <dir>...] [TARGET_ENV vulkan1.0|vulkan1.1|...])`:
- compiles each kernel with the Vulkan SDK's glslc (optimized; Vulkan 1.0's
  SPIR-V by default, so it loads anywhere);
- embeds them all in one header as `ns::<file name>`, a
  `std::span<const uint32_t>`;
- rebuilds on any change to a kernel or anything it includes (glslc's
  dependency files).

The header is generated in the build tree, on the target's include path.

`jvk_runtime(<target>)` copies the shipped MoltenVK beside a plain executable
(a test runner, a tool) on macOS. A plugin's bundle carries its own in
`Contents/Frameworks`. `loadVulkan` looks in the bundle first, then beside
the executable.

## Version

The instance asks for Vulkan 1.2 where the loader has 1.1 or later, and 1.0
otherwise. `Device::apiVersion()` is the lower of that and the GPU's version.
Rendering makes only 1.0's calls either way. Compute reads subgroup
properties when the version is 1.1 or later (`limits().subgroupSize`; 0 when
unknown).
