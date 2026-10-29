# jvk Translation Layer

jvk is becoming a GPU translation layer for `juce::Graphics`: one core that owns
JUCE compliance and every shader, and a set of backends that run it. Vulkan is the
first backend. Metal, WebGPU, D3D12 and an eacp backend should each be a backend
and nothing more.

This document is the target architecture and the migration plan. The interface it
describes is in `include/jvk/core/gpu/` (namespace `jvk::gpu`).

---

## Layers

```
          juce::Graphics / paint()                    jvk extensions (effects, targets, shaders)
                     │                                              │
┌────────────────────▼──────────────────────────────────────────────▼─────────────┐
│ CORE   include/jvk/core/          JUCE semantics, GPU-agnostic                  │
│   record  — LowLevelGraphicsContext, state stack, clip semantics, captures      │
│   lower   — CPU geometry, pass planning, pipeline interning → gpu::FramePacket  │
│   caches  — glyph / gradient / image policy and keys, atlas packing             │
│   shaders — the bundle: sources, compiled forms, layouts                        │
└────────────────────┬────────────────────────────────────────────────────────────┘
                     │ jvk::gpu  (core/gpu/: Types, PipelineState, FramePacket, Backend)
┌────────────────────▼──────────────┐  ┌───────────────────────────────────────────┐
│ BACKEND  include/jvk/backends/*/  │  │ HOST  window + frame loop                 │
│   memory, native objects,         │◄─┤   jvk::AudioProcessorEditor (JUCE peer)   │
│   barriers, submission, present   │  │   offscreen (ImageType, ShaderImage,      │
│   vulkan/  [metal/ webgpu/ eacp/] │  │   compliance tests), eacp EmbeddedView    │
└───────────────────────────────────┘  └───────────────────────────────────────────┘
```

**Core.** Depends on JUCE, never on a GPU API. CMake enforces it at configure time:
a Vulkan, Metal, D3D12 or WebGPU name anywhere under `include/jvk/core/` fails the
configure.

**Backend.** Depends on its GPU API and on `core/gpu/`, never on JUCE. It implements
`gpu::Backend` (one per process per GPU, shared by every editor) and `gpu::Surface`
(one per swapchain or offscreen target).

**Host.** Ties a backend to somewhere frames go: it creates the window surface from
the platform view (that needs backend-specific types), drives the frame loop, and
calls into the core. jvk's `AudioProcessorEditor` is the JUCE host. An eacp host is
an `eacp::EmbeddedView` parented to a JUCE peer's native view.

---

## Who owns what

**The rule:** the core owns anything that changes pixels or encodes JUCE semantics,
plus any policy that holds on every API. The backend owns anything whose only effect
is validity or speed on one API.

| Core | Backend |
|---|---|
| `LowLevelGraphicsContext`, the state stack, all clip semantics (rect lists, exclude, path, image alpha) | Device, queues, surfaces/swapchains, present, frame pacing |
| CPU geometry: path flattening, stroking, tile binning, rect/ellipse SDF parameters, glyph layout | **All GPU memory**: allocation, sub-allocation, staging |
| MSDF generation, gradient LUT rows, atlas packing | Native textures behind `TextureId`s, deferred destruction until the GPU is done |
| Cache keys and eviction (`juce::Image` identity, `ColourGradient`, glyph keys) | Descriptor sets / bind groups / argument buffers, sampler objects |
| Every shader, and pipeline state as data (`PipelineDesc`: blend, stencil, formats) | Native pipeline objects built from `PipelineDesc`, and their cache |
| Pass planning: effect splits, ping-pong versioning, target liveness, region bounds, stencil reference per draw | Barriers, layout transitions, render passes / encoders, which thread submits |
| The extended API: effects, `setTarget`, `drawShader`, frame values | Transient memory the packet's streams are copied into |

Grey zones, decided:

- **Atlases.** The core packs and generates glyph and gradient atlases on the CPU and
  sends dirty rectangles as `Upload`s. The backend holds the textures.
- **Lifetimes.** Packets refer to handles, not pointers. `destroyTexture` is always
  safe: the backend defers it. This retires the core-side `FrameRetained` pins,
  per-slot retire buckets and staging recycling (much of CLEANUP_PLAN Phase 1). That
  synchronisation is API-specific, so it lives in each backend.
- **`jvk::ImageType`** (GPU-resident `juce::Image`) is core. Rendering into one is an
  offscreen surface; `BitmapData` reads go through `Backend::readTexture`.
- **Pass planning** is core: where passes split changes what a shader reads. A backend
  may later expose a capability (framebuffer fetch / programmable blending on Apple
  GPUs) that the core plans around, but it never restructures passes on its own.

---

## The frame

```
record (message thread, during paint)       lower (the submitting thread)             backend
──────────────────────────────────────      ──────────────────────────────────        ─────────────────
Graphics → Recorder: DrawCommands +    ──►  Lowering: params → vertices, draws,  ──►  Surface::submit:
Arena, captures, path segments,             constants; PassPlanner: passes;           uploads, passes,
clip ops, target reads                      intern PipelineDescs → PipelineIds        draws, present
```

1. **Record.** `jvk::Graphics` pushes commands into the core's `Recorder`. No GPU work;
   the only backend calls are creating and writing textures (a cache miss, a glyph
   page, a dirty `jvk::ImageType`).
2. **Lower.** A pure function of the recorded frame and `FrameInfo`. It runs on
   whichever thread submits (jvk's worker, eacp's main thread), so vertex generation
   stays off the message thread as it is today. It merges consecutive draws that share
   a pipeline, bindings, scissor and stencil reference into one draw; today every quad
   is its own `vkCmdDraw`.
3. **Submit.** `Surface::submit(packet)`. The packet's memory stays valid until
   `Surface::busy()` is false: jvk's current one-frame-in-flight contract, unchanged.

A frame costs a handful of virtual calls, not one per draw. The packet's spans point
into core arenas that reset each frame, so the zero-allocation steady state survives.
The backend copies the streams into its own transient memory; that is one `memcpy` of
roughly 0.1–1 MB per frame, and on WebGPU it would happen anyway (no persistent mapping).

### The packet

`core/gpu/FramePacket.h`. Passes run in order, each a run of draws. Texture writes
are not in the packet: `Backend::writeTexture` copies the bytes at once and lands them
before the next frame submitted on ANY surface, which is what makes the image cache
safe to share between windows. A pass names its colour attachments and depth/stencil attachment with
load/store ops and a render area. A draw names a `PipelineId`, a scissor, a stencil
reference, texture bindings with sampler state, storage ranges, a constants range and
a vertex range.

The backend never reorders, merges or drops anything. It works out barriers and
layout transitions from each texture's use in consecutive passes. The core guarantees
that no texture is both sampled and attached in one pass.

---

## Pass planning (the core's render graph)

This is today's `Renderer::execute`, with every `vkCmd*` replaced by appending a
`gpu::Pass` or `gpu::Draw`. The semantics port unchanged:

- **Main colour, versioned.** Two textures (halves A/B) plus one depth/stencil texture.
  A scene pass draws into the current half. An effect that reads the scene ends the
  pass, then:
  - draws from the current half into the other (`effectPassAndSwap`);
  - for a partial region, draws a region-bounded copy back;
  - for a clipped region, pre-copies the outside-clip pixels.

  The stencil texture is attached, `Load`ed, to every pass that tests the clip, in
  either half. That is how clips persist across effects.
- **Render targets, versioned** through a scratch texture of the same size and format
  when read and written in one pass.
- **Liveness.** A target is live up to its last reader. Draws into a dead target are
  dropped, and its attachments `Store` → `DontCare`. Scaled targets are cleared lazily
  by a pass with `Clear` and no draws.
- **Region of interest.** Each effect pass's render area and scissor are the next
  pass's input region plus that pass's kernel apron, clamped.
- **No blits.** The frame ends with a fullscreen draw of the copy program, from the
  current half into `FrameInfo::output`. eacp and WebGPU copies need matching formats
  and can't scale, and the swap format (BGRA8) needn't match the scene's.

---

## Resources

- **Handles.** `TextureId`, `ProgramId`, `PipelineId`: opaque 32-bit values the
  backend allocates. A generation in the high bits catches stale ones.
- **Textures.** `TextureDesc` says what a texture is for (`Sampled`, `RenderTarget`,
  `Upload`, `Readback`). Depth/stencil textures name no format; the backend picks one
  with 8 stencil bits.
- **Samplers** are per binding (`SamplerDesc` in `TextureBinding`), not baked into
  textures as `jvk::Image` does today.
- **Texture writes** are backend calls, not packet entries: `writeTexture` (tightly
  packed rows into a region of one layer and mip, the rest kept) and `clearTexture`.
  The backend copies the bytes immediately and records the copies at the start of the
  next frame submitted on any surface, after every frame submitted before the call.
  They cover the old `upload`, `uploadRect`, `uploadRegion`, `uploadDynamic` and
  `initializeImage`.
- **Lifetimes** are record serials (`core/Context`). Each recorded frame takes the
  next serial; a resource a frame uses is stamped with it. `Context::retire` destroys
  a resource once no renderer has an unsubmitted frame at or before its stamp, and the
  backend then defers the native object until the GPU is done. No pins, no waits:
  `FrameRetained`, the retire buckets and `cancelUploads` are gone.
- **Caches** stay in the core with their current policy: serial-based LRU, the same
  ages, shared across editors through the `Context` of the shared `Backend`.

---

## Shaders

The core owns every shader. The bundle is generated at build time:

```
shaders/*.vert|frag (GLSL)
   └─ glslc ──► SPIR-V (canonical) ──┬─ Vulkan: as is
                                     ├─ SPIRV-Cross ──► MSL, HLSL (SM 5), GLSL for eacp-on-Vulkan
                                     ├─ Tint / Naga ──► WGSL
                                     └─ spirv-reflect ──► ProgramLayout + binding tables (generated C++)
```

A backend consumes the format named by `Caps::shaderFormat`. Reflection happens at
build time, so no backend needs spirv-reflect at run time. User shaders go through the
same CMake function. A user shader supplied only as SPIR-V at run time (today's
`vk->shader(spirv)`) keeps working on the Vulkan backend, and on others wherever the
host ships a runtime translator: the Metal backend does (`Caps::spirvPrograms`), with a
vendored SPIRV-Cross (`backends/metal/vendor/spirv_cross`, its own translation unit
`jvk_spirv_cross.cpp`) that maps each native binding to the slot `jvk::Shader` reflected.

Slang is worth evaluating later as the source language, since it emits all four
formats itself. Moving to it would mean migrating the sources.

### Binding convention

Programs bind by slot:
- textures `0..n-1`, each with a separate sampler;
- read-only storage buffers `0..m-1`;
- one per-draw constants block of at most `Caps::maxDrawConstantsBytes` (128).

The bundle generator emits each format bound the way its backend expects:

| | Textures | Samplers | Storage | Constants |
|---|---|---|---|---|
| Vulkan | set 0, binding i | set 0, binding 8 + i | set 0, binding 16 + j | push constants |
| Metal | `[[texture(i)]]` | `[[sampler(8 + i)]]` | `[[buffer(16 + j)]]` | `[[buffer(24)]]`, the same bytes to both stages |
| WGSL | `@group(0) @binding(i)` | `@binding(8 + i)` | `@binding(16 + j)` | `@binding(24)`, dynamic uniform offset, 256-aligned |
| eacp | its slot map (Metal `texture(n)`, D3D12 `t0`–`t7`, Vulkan combined samplers at 8 + n), emitted by SPIRV-Cross with remapped bindings | | | per-draw bytes |

Each stage is compiled twice from one GLSL source (`shaders/jvk_portable.glsl`):
with `-DJVK_VULKAN` the constants block is `push_constant` (the Vulkan SPIR-V), without
it a `std140` uniform block at binding 24, which SPIRV-Cross translates
(`--msl-decoration-binding`, `--flip-vert-y`; MSL entry point `main0`). The Metal limits
fit: textures 0–7, samplers 8–15, buffers 16–24, vertex buffers above.

A program from elsewhere — a user's `jvk::Shader` or `ShaderImage`, SPIR-V at run
time — names its native bindings instead (`gpu::BindingDesc`: set, binding and kind per
slot, combined samplers and uniform buffers included). Only a backend that consumes
SPIR-V takes those.

### Portable profile

These are rules for every built-in shader. Each one also works on Vulkan, so they
land before any second backend:

1. **No specialisation constants.** `ui2d.frag` and `path_sdf.frag` used 22 (`paintsMain`
   plus 7 target clear colours). `paintsMain` is now a constant, and the clear colours
   are per-frame storage (`FrameData { vec4 targetClear[7]; }`, always bound). D3D12
   via SPIRV-Cross and eacp have no specialisation constants.
2. **Separate textures and samplers.** WGSL has no combined image samplers.
3. **One constants block, at most 128 bytes.** It is declared through a bundle macro
   that expands to `push_constant` for the Vulkan build and to a uniform block at the
   constants slot for the translated builds. WebGPU has no push constants. Scalars,
   vectors and vec4 arrays only, so std140 and std430 agree (`compile.sh` checks), and
   both stages declare byte-compatible views. The bundle rounds a block up to 16 bytes.
4. **No blits** (see Pass planning).
5. **MRT budget.** The main colour plus scale-1 targets must fit
   `Caps::maxColorBytesPerSample` (32 on WebGPU). Beyond it, `setTargets` refuses
   targets on that backend.
6. **Derivatives in uniform control flow.** WGSL's uniformity analysis rejects
   `textureSample` and `fwidth` under non-uniform branches, such as `ui2d.frag`'s
   shape-type switch. Sample with explicit LOD or gradients inside the branches.

---

## Hosts

- **`jvk::AudioProcessorEditor`** (`hosts/`) is generic over a `WindowHost`
  (`hosts/WindowHost.h`): the backend, plus a native child view with a window
  `Surface`. The render loop, `paintEntireComponent`, the software fallback, resize
  policy and the timer are the editor's. `createWindowHost()` (`hosts/WindowHost.cpp`)
  picks the host: Metal on macOS (`JVK_BACKEND=vulkan` runs MoltenVK instead, to compare
  them), Vulkan elsewhere. The Vulkan host (`backends/vulkan/VulkanHost`) keeps
  `NSViewGenerator` / `HWNDGenerator` and the swapchain; the Metal host
  (`backends/metal/MetalHost`) draws the same `NSViewGenerator` view's `CAMetalLayer`.
- **Offscreen host.** `ShaderImage` (`hosts/`) is one pass of one draw into a texture
  on an offscreen `Surface` of `defaultBackend()`, read back a frame late.
  `jvk::ImageType` keeps a CPU mirror as its source of truth and its own texture,
  written when dirty.
- **eacp host.** An `eacp::EmbeddedView` under the editor's peer, with an eacp
  `GPUView` driving frames.

---

## Directory layout

```
include/jvk/
  core/                the translation layer (no GPU API; CMake checks)
    gpu/               the backend interface: Types, PipelineState, FramePacket, Backend
    shaders/           Program.h (the bundle's shape), Bundle.h (generated: SPIR-V + MSL)
    Recorder           P2: DrawOp, DrawCommand, Arena, captures, record target, liveness, segments
    Params, Targets    command payloads; render targets and their blend rules
    Context            per backend: built-in programs, interned pipelines, default textures,
                       the image cache, record serials and the retire list
    Resources          GradientAtlas, ImageSource, ImageCache
    GlyphAtlas, MSDF   text
    FrameBuilder       the packet being lowered (draw merging) and the Emitter (2D draw state)
    Renderer           per surface: frame textures, the worker, lower() — the render graph
    ops/               lowering of the 2D draws (ColorDraw) and the programs' constants
    Graphics, Shader, ImageType
                       the JUCE-facing API
  hosts/               WindowHost (interface), AudioProcessorEditor, ShaderImage
  backends/SlotTable   the handle tables both backends share
  backends/vulkan/     VulkanBackend, VulkanSurface (packet executor, swapchain, offscreen),
                       VulkanHost; Device, Memory, Resource, Formats, macos/, windows/, vendor/
  backends/metal/      MetalBackend (textures, programs, pipelines, staged and in-place
                       writes), MetalSurface (packet executor, CAMetalLayer, offscreen),
                       MetalHost; SpirvToMsl + vendor/spirv_cross (jvk_spirv_cross.cpp)
  reflect/             spirv-reflect (jvk::Shader and ShaderImage reflect user SPIR-V)
shaders/               GLSL sources, jvk_portable.glsl, compile.sh → core/shaders/Bundle.h
```

---

## Migration

The old plan kept the old replay behind `JVK_LEGACY_REPLAY` for A/B. That could not
work: P4 takes the `VkDescriptorSet`s out of `Params`, which the old replay reads. The
port was done in place instead, and A/B is against `main` (compliance captures and
benchmark numbers from a `main` build).

| Phase | What | State |
|---|---|---|
| **P0 Layout** | `backends/vulkan/`, `core/`, configure-time core check; `toVkFormat` split out of `PixelFormat` | done |
| **P1 Interface** | `core/gpu/`: handles, `TextureDesc`, `PipelineDesc`, `FramePacket`, `Backend`, `Surface` | done |
| **P2 Recorder** | `core/Recorder`: the record half of the old `Renderer` | done, unbuilt |
| **P3 Vulkan backend** | `vulkan::Backend` + `Surface`: handle tables, `PipelineDesc` → `VkPipeline`, render pass / framebuffer caches, packet executor with derived barriers, deferred destruction | done, unbuilt |
| **P4 Resources** | Image cache, gradient atlas, glyph atlas, `ImageType` on `TextureId`s and `writeTexture`; `Context` lifetimes | done, unbuilt |
| **P5 Lowering** | Every op lowered in the core; the render graph is `Renderer::lower`; record → lower → submit. The old renderer and pipelines are deleted | done, unbuilt |
| **P6 Shaders** | Portable profile, build-time bundle (SPIR-V + MSL via SPIRV-Cross, reflected layouts). WGSL and HLSL not generated yet (no Tint/Naga in the toolchain) | done |
| **P7 Hosts** | `WindowHost`, generic `AudioProcessorEditor`, Vulkan host, `ShaderImage` on an offscreen surface | done, unbuilt |
| **P8 Second backend** | Metal (`backends/metal/`): the macOS default. Pixel-identical to Vulkan on 12 of the 13 compliance scenes (the 13th: 0.01% of pixels); draws the packet straight into Metal command buffers (no descriptor sets, no pipeline-cache file at start-up), unified-memory textures written and read in place | done |

---

## Open questions

1. **Second backend: eacp or raw Metal first?** Raw Metal (P8): it removed MoltenVK
   from the macOS path. eacp remains the candidate for D3D12 and WebGPU.
2. **Vulkan pass execution.** The backend caches `VkRenderPass` / `VkFramebuffer`
   objects per attachment set (works everywhere). `VK_KHR_dynamic_rendering` would
   avoid the caches; check it against the pinned MoltenVK 1.4.0 and the macOS 10.15
   floor before switching.
3. **Public API that names Vulkan.** Done for `Target::clear` (`gpu::ClearColor`),
   `Shader::set` for a GPU-resident texture (a `gpu::TextureId`), and the editor
   (`getBackend()` replaces `getDevice()`). The user GLSL contract still uses
   `push_constant` and Vulkan-style sets, so `jvk::Shader` and `ShaderImage` draw only
   on a backend that takes SPIR-V; portable user shaders need the bundle's CMake
   function (not built yet).
4. **For eacp (Eyal):**
   - a depth/stencil texture that isn't a plane of one colour target, or a pass that
     can attach another texture's stencil. Clipped effects draw into the other half
     while testing the scene's clip stencil;
   - precompiled shader input (metallib / DXIL / SPIR-V) instead of source strings;
   - recording on a thread other than the one that made the `Device`.
