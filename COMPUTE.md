# jvk compute

General GPU work: kernels (GLSL compute shaders) over buffers, on every backend.
Kernels run in two places:
- **Streams** run beside the rendering, on a queue of their own where the GPU
  has one: training, analysis, offline renders.
- **Frames** run a kernel in paint order (`jvk::Graphics::dispatch`): results
  that the same frame draws, with no copy back to the CPU.

The interface backends implement is `include/jvk/core/gpu/Compute.h`. The user API
is `include/jvk/core/compute/Compute.h` (`jvk::compute`). `shaders/include/jvk/` is
a small GLSL library (`compute.glsl`, `scan.glsl`). `jvk_compute_shaders()` in
`CMakeLists.txt` builds a project's kernels for the backends the build has.

This replaces the `compute-shaders` branch's `jvk::compute`. That version was
Vulkan only (`VkBuffer` in its ranges, `jvk::Device` in its constructor). See
"From the compute-shaders branch" for what changed.

## In five lines

```cpp
auto context = jvk::compute::Context::create (&why);                           // nullptr: no GPU, and why
auto data = context->buffer (n * sizeof (float), jvk::compute::Usage::Shared); // mapped: write it from the CPU
auto scale = context->kernel (mykernels::scale);                               // its interface was reflected at build time
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

- **Context**: the GPU, for compute.
  - `create()` uses the backend the editors draw with (`jvk::defaultBackend()`).
    `create(backend)` takes a given one.
  - Either returns nullptr with a reason when there is no usable GPU, so a caller
    can fall back to the CPU.
  - `limits()` gives the device's sizes. Size work by these; never assume them.
  - Safe to use from any thread.
- **Buffer**: GPU memory, freed when it goes. It may go at any time: the backend
  keeps the memory until every submitted stream and frame that used it has
  finished.
  - `Usage::Device`: the GPU's alone, and the fastest.
  - `Upload` and `Readback`: mapped for one direction.
  - `Shared`: mapped both ways. On unified memory (Apple silicon, integrated
    GPUs) it is the GPU's own memory and costs nothing.
  - Call `written()` after the CPU writes mapped memory, and `toRead()` before it
    reads. Both are no-ops on coherent memory (Metal, most Vulkan host memory).
- **Kernel**: a compute pipeline made from a `KernelSource`. Its interface was
  reflected from its SPIR-V at build time: buffers in set 0, one push-constant
  block, its workgroup size, its specialization constants.
  - Specialization constants are set when it is made.
  - Its workgroup size may be one of them (`local_size_x_id`).
  - Its GPU memory goes with it, once nothing submitted still runs it. So a
    program may make kernels as it runs (code it writes, specialized to its
    data, as a `KernelSource` of its own) and drop them, and keep only the
    memory of those it still has:
    - Metal: kernels of one source share its library while any of them lives,
      and the last one to go takes it out. A pipeline state needs no library
      once made, and Metal's compiler cache makes a source made again cheap.
      A `jvk::Shader`'s functions go the same way, with the last program
      that uses them.
    - Vulkan: kernels stay out of the device's pipeline cache. That cache keeps
      everything put in it for the device's life and is saved to disk for every
      later run to load. Drivers keep compiled shaders on disk themselves
      (MoltenVK through Metal's compiler cache), so a kernel made again is
      cheap anyway.
- **Stream**: commands recorded, submitted, then waited for.
  - The commands are `dispatch`, `barrier`, `fill`, `copy`, `upload`,
    `download` and `time`.
  - Recording is CPU work alone, so the next batch can be recorded while the last
    runs. `submit()` waits for the last first: one batch in flight a stream.
  - `wait(seconds)` returns `pending` on a timeout, so a long job can check
    whether it was asked to stop. It returns `lost` if the GPU was lost.
  - A stream is one thread's at a time; many may be in flight.

**Order.** A stream's commands start in order but may overlap. Put `barrier()`
between a command that writes memory and any later command that reads or writes
it. That includes kernels, copies, fills, uploads and downloads. (Metal orders them
regardless; Vulkan does not.)

**Bindings.** A dispatch binds buffer ranges in binding order. Each offset must be a
multiple of `limits().storageAlignment`. Several arrays in one big buffer is the
cheap way to keep many things on the GPU: bind the buffer once and pass element
offsets in push constants.

## Kernels in frames

```cpp
void paint (juce::Graphics& g) override
{
    auto* jg = jvk::Graphics::create (g);   // null when jvk isn't painting it
    if (jg == nullptr) return;
    jg->dispatch (decimate, Groups::cover (columns, 64, limits), { audio.range(), peaks.range() }, Push { columns });
    peaksShader.set ("Peaks", peaks.range());
    jg->drawShader (peaksShader, getLocalBounds().toFloat());
}
```

- `Graphics::dispatch` records a kernel where paint is.
- The render graph runs it as a compute pass of its own between scene passes, on
  the frame's queue, the way an effect runs.
- Each dispatch sees every draw and dispatch before it, and everything after it
  sees its writes. No barriers to place: the backend orders them (Vulkan:
  barriers around and inside the compute pass; Metal: hazard tracking).
- Clips, transforms and targets don't apply; a dispatch draws nothing.
- **A draw reads a buffer**: `Shader::set(name, range)` binds a buffer range to a
  `jvk::Shader`'s storage or uniform block, in place of the bytes `set()` copies.

A stream's writes are seen by the frames submitted after its `wait()` said `done`.
Nothing orders a stream against a frame on the GPU: double-buffer what a stream
writes while frames read it.

The kernel and buffers must come from a Context on the editor's backend:
`Context::create()` with no backend gives that.

## The interface (`core/gpu/Compute.h`)

What a backend implements, beside `Backend.h`:

- **Buffers**: `createBuffer` (size, memory kind), `destroyBuffer` (deferred like a
  texture), `bufferData` (the persistent CPU mapping, null for device memory), and
  `flushBuffer` / `invalidateBuffer`.
- **Kernels**: `createKernel` from a `KernelDesc`, which holds:
  - the code in the backend's shader format;
  - the bindings by binding number;
  - the constants block's size;
  - the final workgroup size;
  - the specialization constants' values and types.
- **Streams**: `createComputeStream` returns a `ComputeStream`, which runs a
  `ComputePacket` (commands, dispatches, constants and upload bytes, all valid for
  the call alone) and waits for it.
- **Frames**: a `Pass` of kind `Compute` is a run of `Dispatch`es in
  `FramePacket::dispatches`, and a `Draw`'s storage slot can name a `BufferId`
  (`Draw::storageBuffer`).

`Caps::compute` says what the backend can do. A backend without compute leaves
`supported` false and the defaults in place.

### Binding convention

| | Buffer binding b (set 0) | Push constants | Buffer lengths (`.length()`) |
|---|---|---|---|
| Vulkan | set 0, binding b | push constants | — |
| Metal | `[[buffer(b)]]` | `[[buffer(24)]]` | `[[buffer(25)]]`, a `uint` per binding, by b |

- At most 16 bindings (`gpu::kMaxKernelBindings`), and the device's own limit
  (`limits().maxBindings`).
- Buffers only for now (storage and uniform). `jvk_compute_shaders` refuses a
  kernel that binds a texture or a set other than 0.
- Textures in kernels are the next step. In frames they are straightforward (the
  frame's queue tracks texture state). In streams they need queue ownership
  transfers on Vulkan.

### Specialization constants

- Vulkan: `VkSpecializationInfo`.
- Metal: SPIRV-Cross turns each scalar constant into a function constant,
  `[[function_constant(id)]]`, which the backend sets when it makes the function.
- A constant that sizes an array can't be a function constant. SPIRV-Cross makes it
  a macro, `SPIRV_CROSS_CONSTANT_ID_<id>`. The backend defines that ahead of the
  source when the source uses it.
- The workgroup size follows its constants on both backends. Metal is given it at
  each dispatch (`threadsPerThreadgroup`). A kernel whose size passes the
  pipeline's `maxTotalThreadsPerThreadgroup` is refused when it is made.

## The queues

- **Vulkan.** `Device` gives compute its own queue where it can. In order of
  preference:
  1. a compute family without graphics, where the GPU has one (NVIDIA's and AMD's
     async compute runs beside the editors' frames);
  2. another family with compute (MoltenVK offers four families with a queue
     each, every one a Metal command queue of its own);
  3. a second queue of the graphics family;
  4. the graphics queue itself.

  The last case is reported as `limits().ownQueue == false`, and streams then
  submit under the frames' lock. The compute queue has the lower priority.
  Buffers are shared between the two families (`VK_SHARING_MODE_CONCURRENT`)
  when they differ.
- **Metal.** Streams have a command queue of their own beside the frames' queue.
- A kernel is not interrupted once it runs. Keep each submit short (a few
  milliseconds) when an editor is open.

## Timing

`time(label)` records when everything before it has finished. `times()` gives each
label's time in ms since the batch began.
- Vulkan: timestamp queries, where the compute family has them.
- Metal: a command buffer per timed segment, and each one's `GPUEndTime`. That
  times stages on Apple GPUs too, where MoltenVK could not.

## The GLSL library

- `jvk/compute.glsl`: `jvkThread()` and `jvkGroup()`, the indices of a
  one-dimensional dispatch sized by `Groups::cover()`, which spreads its workgroups
  over y past the limit of one dimension.
- `jvk/scan.glsl`: scans of any associative operation in a workgroup.
  - Order is kept, so the operation need not commute: matrix products and affine
    maps work as well as sums.
  - `jvkScanExclusive()`: each thread passes one element.
  - `jvkScanSequence()`: one workgroup scans a sequence of any length, read and
    written through macros.
  - It uses no subgroup operations, so it runs on Vulkan 1.0 and Metal 2.1.
  - It never waits on another workgroup. Neither API promises that one workgroup
    makes progress while another waits for it, and single-pass scans that rely on
    that can hang.

A **linear recursion** (`x[t] = A[t] x[t-1] + b[t]`: IIR filters, phase
accumulators, one-pole smoothers) runs in parallel over time with a scan:
1. Each chunk of the sequence is one affine map of its starting state.
2. A scan of the chunks' maps gives every chunk its starting state.
3. Each chunk runs from there.

## Building kernels

```cmake
jvk_compute_shaders(<target> HEADER <h> NAMESPACE <ns> SOURCES <.comp>...
                    [INCLUDE_DIRS <dir>...] [TARGET_ENV vulkan1.0|vulkan1.1|...])
```

- Compiles each kernel with the Vulkan SDK's glslc (optimized; Vulkan 1.0's SPIR-V
  by default, so it loads anywhere). It rebuilds on any change to a kernel or
  anything it includes (glslc's dependency files).
- `jvk_kernel` (`tools/jvk_kernel.cpp`, built for the build machine with the
  vendored SPIRV-Cross) then writes one header with, for each kernel:
  - its SPIR-V;
  - its MSL;
  - its interface, reflected;
  - and `ns::<file name>`, a `jvk::compute::KernelSource` over all of it.
- A build includes only the forms its backends take (`JVK_VULKAN`, `JVK_METAL`),
  as the built-in shaders do. So a Metal build ships no SPIR-V, and nothing
  translates or reflects at run time.
- The header is generated in the build tree, on the target's include path. It is
  rewritten only when its text changes.

`jvk_runtime(<target>)` copies the shipped MoltenVK beside a plain executable (a
test runner, a tool) on macOS builds with the Vulkan backend. A plugin's bundle
carries its own (`jvk_bundle`). A Metal build needs nothing beside it.

## From the compute-shaders branch

The API keeps its shape. What a project changes:

| Branch | Now |
|---|---|
| `Context::create (jvk::Device::acquire(), &why)` | `Context::create (&why)` (or `create (backend, &why)`) |
| `VkDeviceSize` sizes and offsets | `uint64_t` |
| `Range { VkBuffer, offset, bytes }` | `Range { gpu::BufferId, offset, bytes }` |
| `VK_WHOLE_SIZE` | `jvk::compute::kWhole` |
| `ns::kernel`, a `std::span<const uint32_t>` | a `jvk::compute::KernelSource` (`ns::kernel.spirv` is the span, where the build has Vulkan) |
| `context->kernel (spirv, "name", constants)` | `context->kernel (ns::kernel, constants)`; the name defaults to the kernel's |
| `Limits::apiVersion` | gone: `limits().device` names the GPU |
| a Stream records only after `wait()` | records any time; `submit()` waits for the last batch first |
| Buffers, Kernels and Streams must go before their Context | any order: each holds what it needs |

## Version

The Vulkan instance asks for Vulkan 1.2 where the loader has 1.1 or later, and 1.0
otherwise. Rendering makes only 1.0's calls either way. Compute reads subgroup
properties when the version is 1.1 or later (`limits().subgroupSize`; 0 when
unknown). Metal reports 32 on Apple GPUs and 0 elsewhere (a pipeline's
`threadExecutionWidth` varies on AMD and Intel).
