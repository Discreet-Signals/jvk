# JVK Target Architecture

Translated from GAVINS_ARCHITECTURE.md — that file is the source of truth.
This document adds Vulkan specifics, concrete interfaces, and implementation details.

---

## Ownership Graph

```
Device (shared_ptr, ref-counted, one per process)
 ├── VkInstance, VkPhysicalDevice, VkDevice, VkQueue, VkCommandPool
 ├── Memory::L1 (GPU pool — persistent device-local, sub-allocated)
 ├── Memory::L2 (staging — host-visible, CPU→GPU transfer)
 ├── Memory::M  (bindings — descriptor set pool)
 ├── upload()   (transfer operation: L2 → L1)
 └── ResourceCaches (shared across all plugin instances)
      ├── Cache<K, CachedImage>  (GPU-resident: gradients, textures)
      ├── Cache<K, CachedBuffer> (GPU-resident: static meshes)
      ├── Cache<K, V>            (CPU-resident: flattened paths, intermediaries)
      └── GlyphAtlas

Renderer (one per plugin instance or offscreen target, refs Device)
 ├── RenderTarget (swapchain, offscreen, etc.)
 ├── Memory::V (vertices — per-frame scratch, reset each frame)
 ├── Pipeline modules (registered, lazily built)
 ├── DrawCommand vector + Arena (reset each frame)
 ├── State (dirty-tracked GPU state + clip stack, internal)
 └── execute() → sort + upload + replay + submit + present

RenderTarget
 ├── SwapchainTarget (VkSwapchainKHR, framebuffers, MSAA, sync primitives)
 ├── OffscreenTarget (single Image, single framebuffer)
 └── [future: BufferTarget, etc.]

jvk::AudioProcessorEditor (user-facing — JUCE plugin entrypoint)
 ├── shared_ptr<Device>
 ├── Renderer (owns SwapchainTarget)
 ├── Registers built-in pipeline modules (color, stencil, blend, resolve)
 └── Drives render loop: Graphics → record → Renderer::execute()

jvk::ShaderImage (user-facing — offscreen rendering)
 ├── shared_ptr<Device> (same Device, shared caches)
 ├── Renderer (owns OffscreenTarget)
 └── Result is a texture usable by the main renderer

jvk::Graphics (JUCE adapter — implements LowLevelGraphicsContext, pushes into Renderer)
 ├── Recording state stack (transform, fill, clip, z-order)
 ├── Effects API (blur, darken, tint, drawShader)
 └── ref → Renderer

Resource (base class — owns L1 allocation, RAII)
 ├── Image : Resource  — VkImage + VkImageView + VkSampler
 └── Buffer : Resource — VkBuffer (static data only, NOT per-frame uniforms)

Pipeline (base class — subclassed per pipeline module)
 ├── config(), clipConfig(), supportedOps(), execute()
 ├── Owns VkPipeline (normal) + VkPipeline (clip variant) + VkPipelineLayout
 ├── State auto-selects clip variant when stencilDepth > 0
 └── Self-contained modules in graphics/pipelines/
      ├── color/    (fillRect, drawImage, drawGlyphs — has clip variant)
      ├── stencil/  (pushClipPath, popClip — writes/reads stencil buffer)
      ├── blend/    (darken, tint, brighten — has clip variant)
      ├── resolve/  (blur, hue, saturation — no clip variant, separate framebuffer)
      └── [future: scene/, particles/, ...]
```

---

## Memory

```cpp
namespace jvk::Memory {

// ============================================================================
// L1 — GPU Pool (persistent, device-local)
// ============================================================================
// Deep storage. Where cached textures, atlas pages, gradient LUTs, and effect
// images live. Written once (via L2 staging), read thousands of times per frame.
//
// Data structure: HashMap<memoryTypeIndex, Vec<Block>>
// Block: { VkDeviceMemory, size, freeList: Vec<{offset, size}> }
//
// Allocates large blocks (64 MB), sub-allocates with first-fit + coalescing
// free-list. Handles alignment and bufferImageGranularity automatically.
// In steady state: 2-3 blocks total, alloc = free-list scan, free = insert + merge.

class L1 {
public:
    L1(VkPhysicalDevice physDevice, VkDevice device);
    ~L1();  // Frees all blocks.

    struct Allocation {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize   offset = 0;
        VkDeviceSize   size   = 0;
        uint32_t       blockIndex = 0;
        uint32_t       memoryType = 0;
    };

    // Sub-allocate from a pooled block. Returns (memory, offset) pair for
    // vkBindBufferMemory / vkBindImageMemory.
    Allocation alloc(VkMemoryRequirements req, VkMemoryPropertyFlags properties);

    // Return sub-allocation to pool. Coalesces with adjacent free ranges.
    void free(Allocation alloc);

private:
    static constexpr VkDeviceSize BLOCK_SIZE = 64 * 1024 * 1024;

    struct FreeRange { VkDeviceSize offset, size; };
    struct Block {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize   size = 0;
        std::vector<FreeRange> freeList;  // sorted by offset
    };

    std::unordered_map<uint32_t, std::vector<Block>> pools;
    VkPhysicalDevice physDevice;
    VkDevice device;
    VkDeviceSize bufferImageGranularity;
};


// ============================================================================
// L2 — Staging (per-frame, host-visible, CPU→GPU transfer)
// ============================================================================
// Temporary taxi for pixel data. CPU writes pixels here, then a GPU copy command
// moves them into L1. After the copy, the region is dead — freed at frame reset.
//
// Data structure: Vec<Block> active, Vec<Block> free
// Block: { VkBuffer(TRANSFER_SRC), mappedPtr, writeHead }
//
// Linear bump allocator. Blocks recycled between frames via a free list.
// In steady state: blocks exist from warmup, alloc = pointer bump. Zero allocation.

class L2 {
public:
    L2(VkPhysicalDevice physDevice, VkDevice device);
    ~L2();

    struct Allocation {
        void*        mappedPtr;   // CPU-writable pointer — memcpy pixels here
        VkBuffer     buffer;      // source buffer for vkCmdCopyBufferToImage
        VkDeviceSize offset;
    };

    // Write data into staging memory. Returns mapped pointer + buffer handle.
    Allocation alloc(VkDeviceSize size);

    // Move active blocks to per-frame used list (called after recording uploads).
    void moveActiveTo(std::vector<Block>& usedBlocks);

    // Return completed blocks to free list (called after fence signals).
    void recycle(std::vector<Block>& usedBlocks);

    // Reset for next frame. Active blocks move to used, write heads reset.
    void reset();

private:
    static constexpr VkDeviceSize BLOCK_SIZE = 4 * 1024 * 1024;

    struct Block {
        VkBuffer buffer = VK_NULL_HANDLE;
        void*    mapped = nullptr;
        VkDeviceSize capacity = 0;
        VkDeviceSize writeHead = 0;
    };

    std::vector<Block> activeBlocks;
    std::vector<Block> freeBlocks;
    VkPhysicalDevice physDevice;
    VkDevice device;
};


// ============================================================================
// M — Bindings (descriptor set pool)
// ============================================================================
// Metadata that tells shaders where to find resources. A descriptor set is a
// small GPU-side struct: "binding 0 → this image view + this sampler."
// Allocated from VkDescriptorPools. Grows by adding new pools on exhaustion.
//
// Data structure: Vec<VkDescriptorPool>, Vec<VkDescriptorSetLayout>
//
// Cache-friendly: descriptor sets are allocated once per resource, cached
// alongside the resource, and freed when the resource is evicted.
// NOT reset per-frame — that would defeat caching.

class M {
public:
    M(VkDevice device);
    ~M();  // Destroys all pools and layouts.

    using LayoutID = uint32_t;
    static constexpr LayoutID IMAGE_SAMPLER = 0;  // default: 1 combined image sampler at binding 0

    // Register a descriptor set layout. Returns an ID for future allocations.
    LayoutID registerLayout(std::span<const VkDescriptorSetLayoutBinding> bindings);
    VkDescriptorSetLayout getLayout(LayoutID id) const;

    // Allocate a descriptor set from the given layout. Grows pool on exhaustion.
    VkDescriptorSet alloc(LayoutID id = IMAGE_SAMPLER);

    // Return a descriptor set to the pool for reuse.
    void free(VkDescriptorSet set);

    // Write resource bindings into a descriptor set.
    static void writeImage(VkDevice device, VkDescriptorSet set, uint32_t binding,
                           VkImageView view, VkSampler sampler);
    static void writeBuffer(VkDevice device, VkDescriptorSet set, uint32_t binding,
                            VkDescriptorType type, VkBuffer buffer,
                            VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE);

private:
    VkDevice device;
    struct LayoutEntry {
        VkDescriptorSetLayout layout;
        std::vector<VkDescriptorPoolSize> poolSizes;
    };
    std::vector<LayoutEntry> layouts;
    std::vector<VkDescriptorPool> pools;
};


// ============================================================================
// V — Vertices (per-frame, host-visible, linear bump allocator)
// ============================================================================
// Hot per-frame writes. Vertices generated during command replay are written
// here via memcpy + pointer bump. Two slots alternating per frame-in-flight.
// Doubles on overflow, never shrinks. In steady state: zero allocations.
//
// Data structure: Slot[2], each { VkBuffer(VERTEX_BUFFER), mappedPtr, capacity, writeHead }
//
// Owned by Renderer, not Device, because each Renderer has its own
// frame timing and frame-in-flight alternation.

class V {
public:
    V(VkPhysicalDevice physDevice, VkDevice device);
    ~V();

    // Reset write head to 0. Call at frame start after waiting on this slot's fence.
    void beginFrame(int frameSlot);

    // Write vertex data. Returns byte offset for vkCmdBindVertexBuffers.
    // Doubles the buffer if too small (old buffer deferred for safe destruction).
    VkDeviceSize write(const void* data, VkDeviceSize byteCount);

    VkBuffer getBuffer() const;
    bool isValid() const;

private:
    static constexpr VkDeviceSize INITIAL_CAPACITY = 8 * 1024 * 1024;
    static constexpr int MAX_SLOTS = 2;

    struct Slot {
        VkBuffer     buffer = VK_NULL_HANDLE;
        void*        mapped = nullptr;
        VkDeviceSize capacity = 0;
        VkDeviceSize writeHead = 0;
    };

    Slot slots[MAX_SLOTS];
    int  currentSlot = 0;
    VkPhysicalDevice physDevice;
    VkDevice device;

    void grow(Slot& slot, VkDeviceSize newCapacity);
};


// ============================================================================
// CPU — CPU-side cache memory
// ============================================================================
// For data that lives on the CPU between frames: flattened path vertex lists,
// rasterized glyph intermediaries, computed gradient color stops, any
// pre-processed data between what the user submits and what gets uploaded to GPU.
//
// Data structure: arena allocator or pool, frame-based or reference-counted lifetime.
// No Vulkan involvement. No descriptor sets. Just bytes on the heap.
//
// Used by Cache<K, CPUResource> for CPU-resident cached intermediaries.

class CPU {
public:
    CPU() = default;

    struct Allocation {
        void*  ptr  = nullptr;
        size_t size = 0;
    };

    // Allocate a block of CPU memory.
    Allocation alloc(size_t size);

    // Free a block.
    void free(Allocation alloc);

private:
    // Internal: pool allocator, arena, or simple new/delete wrapper.
    // The key value is that Cache<K,V> can use the same interface to manage
    // CPU-resident resources as it does for GPU-resident ones.
};

} // jvk::Memory
```

---

## Device

```cpp
namespace jvk {

// The global Vulkan instance. One per process, shared across all plugin instances.
// Reference-counted — first plugin creates it, last plugin destroys it.
// Selects the best discrete GPU. Handles hardware capabilities and extensions.
class Device {
public:
    static std::shared_ptr<Device> acquire();

    VkInstance       instance()       const;
    VkPhysicalDevice physicalDevice() const;
    VkDevice         device()         const;
    VkQueue          graphicsQueue()  const;
    uint32_t         graphicsFamily() const;
    VkQueue          presentQueue()   const;
    uint32_t         presentFamily()  const;
    VkCommandPool    commandPool()    const;

    // Memory tiers (shared across all Applications)
    Memory::L1& pool();      // persistent GPU memory
    Memory::L2& staging();   // CPU→GPU transfer
    Memory::M&  bindings();  // descriptor set allocation

    // Transfer operation: stage data in L2, copy to L1.
    // Records vkCmdCopyBufferToImage with layout transitions.
    void upload(Memory::L2::Allocation src, Image& dst,
                uint32_t width, uint32_t height);

    // Flush all pending L2→L1 copies. Called once per frame before render pass.
    void flushUploads(VkCommandBuffer cmd);

    // One-shot command submission (for immediate uploads, layout transitions).
    void submitImmediate(std::function<void(VkCommandBuffer)> fn);

    // Deferred destruction — resources retired here stay alive until their
    // frame-in-flight fence signals.
    void retire(core::Buffer&& buffer);
    void retire(core::Image&& image);
    void flushRetired(int frameSlot);  // called after fence wait

    // Shared caches
    ResourceCaches& caches();

    ~Device();

private:
    Device();
    void selectPhysicalDevice();
    void createLogicalDevice();

    // Memory tiers
    Memory::L1 pool_;
    Memory::L2 staging_;
    Memory::M  bindings_;

    // Deferred destruction (two slots, one per frame-in-flight)
    struct RetiredResources {
        std::vector<core::Buffer> buffers;
        std::vector<core::Image>  images;
    };
    RetiredResources retired_[2];

    // Pending uploads (accumulated between frames, flushed before render pass)
    struct PendingUpload {
        VkImage dstImage;
        uint32_t width, height;
        VkBuffer srcBuffer;
        VkDeviceSize srcOffset;
    };
    std::vector<PendingUpload> pendingUploads_;

    std::unique_ptr<ResourceCaches> caches_;
};

} // jvk
```

**Vulkan mapping:** Your "Device" maps to two Vulkan concepts — `VkInstance` (the Vulkan library
connection) and `VkDevice` (the logical device handle for a specific GPU). Both live here because
they share a lifetime. The `VkInstance` is created first to enumerate physical devices, then the
best `VkPhysicalDevice` is selected, then a `VkDevice` is created from it. All three are destroyed
together when the last reference drops.

The three memory tiers (L1, L2, M) live on the Device because they're shared across all plugin
instances. Memory::V lives on each Application because it's per-frame and per-instance.

---

## Renderer

```cpp
namespace jvk {

// ============================================================================
// The Renderer is the core execution engine. It lives in core/ — no JUCE dependency.
// It owns a command buffer (vector of DrawCommands + flat Arena), sorts them,
// and replays them against Vulkan via registered Pipeline modules.
//
// Anyone can use a Renderer: jvk::Graphics pushes commands during JUCE paint(),
// but a game engine, a ShaderImage, or a custom app can push commands directly.
// Single-threaded: push() and execute() must be called from the same thread.
// ============================================================================

class Renderer {
public:
    Renderer(Device& device, RenderTarget& target);
    ~Renderer();

    // ===== Command recording (single-threaded) =====

    // Push a draw command. Params are copied into the arena.
    // The command vector doubles if it fills up — no other allocation.
    template <typename Params>
    void push(DrawOp op, float zOrder, const juce::Rectangle<int>& clip,
              uint8_t stencilDepth, uint32_t scopeDepth, const Params& params);

    // ===== Pipeline registration =====

    // Pipeline modules register themselves on construction.
    // Renderer builds a DrawOp → Pipeline* lookup table from supportedOps().
    void registerPipeline(Pipeline& pipeline);

    // ===== Execute (single-threaded, same thread as push) =====

    // 1. Sort commands: stable sort by z, then by pipeline within z.
    //    StateOp scopes preserved — only sort within scope boundaries.
    // 2. target.beginFrame() → get Frame (cmd buffer, frame slot).
    //    V.beginFrame(frameSlot), device.flushRetired(frameSlot).
    // 3. Flush L2→L1 uploads into the command buffer (before render pass).
    // 4. Begin render pass (vkCmdBeginRenderPass).
    // 5. Replay sorted commands: dispatch each DrawOp to its Pipeline's execute().
    // 6. End render pass (vkCmdEndRenderPass).
    // 7. target.endFrame() → submit command buffer, present.
    void execute();

    // Reset for next frame — rewind arena and command vector. No deallocation.
    void reset();

    // ===== Access for Pipeline draw functions during replay =====

    Device&         device();
    RenderTarget&   target();
    ResourceCaches& caches();
    State&          state();
    Memory::V&      vertices();
    const Arena&    arena() const;

private:
    Device&       device_;
    RenderTarget& target_;
    State         state_;
    Memory::V     vertices_;  // per-renderer, per-frame vertex scratch

    // Command storage — simple vector + arena, reset each frame
    std::vector<DrawCommand> commands_;
    Arena                    arena_;

    // DrawOp → Pipeline* lookup, built from registerPipeline()
    Pipeline* pipelineForOp_[static_cast<size_t>(DrawOp::COUNT)] = {};

    uint64_t frameCounter_ = 0;
};

} // jvk
```

**Design note:** The Renderer replaces the old Application class. It's a core concept with no
JUCE dependency. The JUCE-specific orchestration (creating a `jvk::Graphics` context, calling
`paintEntireComponent`, etc.) lives in `graphics/AudioProcessorEditor.h` which holds a Renderer
and drives it. A ShaderImage or custom app holds its own Renderer and pushes commands directly.

---

## RenderTarget (core/RenderTarget.h)

```cpp
namespace jvk {

// Abstract: something the Renderer draws to. Manages the Vulkan surface,
// swapchain or offscreen image, framebuffers, sync primitives, and MSAA.
//
// The Renderer holds a RenderTarget& and calls beginFrame/endFrame each frame.
// Pipeline modules never touch the RenderTarget directly — they go through
// the Renderer and State.
//
// RenderTarget is responsible for:
//   - Creating and owning the VkRenderPass (scene pass)
//   - Creating framebuffers (one per swapchain image, or one for offscreen)
//   - MSAA color + depth/stencil images (allocated from L1)
//   - Sync primitives (fences, semaphores — per frame-in-flight)
//   - Frame slot management (which frame-in-flight we're on)
//   - Resize handling (recreate swapchain/framebuffers/MSAA images)

class RenderTarget {
public:
    virtual ~RenderTarget() = default;

    // Per-frame state returned by beginFrame(). The Renderer uses this to
    // record Vulkan commands. Not exposed to Pipeline execute() functions —
    // they interact through Renderer/State only.
    struct Frame {
        VkCommandBuffer cmd;           // allocated from Device::commandPool
        VkFramebuffer   framebuffer;   // current frame's framebuffer
        VkExtent2D      extent;        // current dimensions in pixels
        uint32_t        imageIndex;    // swapchain image index (0 for offscreen)
        int             frameSlot;     // 0 or 1 — which frame-in-flight slot
        VkImage         resolveImage;  // MSAA resolve target (for effects that
                                       // need current frame as texture input)
    };

    // Acquire the next frame. Waits on this slot's fence, signals image-available
    // semaphore. Returns Frame with a fresh command buffer.
    // Returns Frame with cmd=VK_NULL_HANDLE if the target is unavailable (minimized).
    virtual Frame beginFrame() = 0;

    // Submit the command buffer, present. Signals render-finished semaphore,
    // queues the fence for this frame slot.
    virtual void endFrame(const Frame& frame) = 0;

    // Recreate on window resize. Rebuilds swapchain, framebuffers, MSAA images.
    virtual void resize(uint32_t w, uint32_t h) = 0;

    // Properties needed by Pipeline::build() and Renderer setup.
    virtual uint32_t              width()       const = 0;
    virtual uint32_t              height()      const = 0;
    virtual VkFormat              format()      const = 0;
    virtual VkSampleCountFlagBits msaaSamples() const = 0;
    virtual VkRenderPass          renderPass()  const = 0;  // scene render pass

protected:
    Device& device;  // all targets hold a Device ref for L1, command pool, etc.
    RenderTarget(Device& device) : device(device) {}
};


// Renders to a window via swapchain.
class SwapchainTarget : public RenderTarget {
public:
    // Creates VkSurfaceKHR from platform window, picks format, creates swapchain,
    // allocates MSAA color image + depth/stencil image from L1,
    // creates framebuffers and per-frame sync primitives.
    SwapchainTarget(Device& device, VkSurfaceKHR surface);
    ~SwapchainTarget();

    Frame    beginFrame() override;
    void     endFrame(const Frame& frame) override;
    void     resize(uint32_t w, uint32_t h) override;
    uint32_t width()       const override;
    uint32_t height()      const override;
    VkFormat format()      const override;
    VkSampleCountFlagBits msaaSamples() const override;
    VkRenderPass renderPass() const override;

private:
    VkSurfaceKHR   surface_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkRenderPass   renderPass_ = VK_NULL_HANDLE;

    // Per-swapchain-image
    std::vector<VkImage>       swapImages_;
    std::vector<VkImageView>   swapImageViews_;
    std::vector<VkFramebuffer> framebuffers_;

    // MSAA + depth (allocated from L1)
    Image msaaColorImage_;    // multisampled render target
    Image depthStencilImage_; // depth + stencil for path clipping

    // Per-frame-in-flight sync (2 slots)
    static constexpr int MAX_FRAMES = 2;
    VkSemaphore imageAvailable_[MAX_FRAMES];
    VkSemaphore renderFinished_[MAX_FRAMES];
    VkFence     inFlightFence_[MAX_FRAMES];
    VkCommandBuffer commandBuffers_[MAX_FRAMES];
    int currentFrame_ = 0;
};


// Renders to a GPU image. Result can be sampled as a texture by the main renderer.
class OffscreenTarget : public RenderTarget {
public:
    OffscreenTarget(Device& device, uint32_t w, uint32_t h, VkFormat format);
    ~OffscreenTarget();

    Frame    beginFrame() override;
    void     endFrame(const Frame& frame) override;
    void     resize(uint32_t w, uint32_t h) override;
    uint32_t width()       const override;
    uint32_t height()      const override;
    VkFormat format()      const override;
    VkSampleCountFlagBits msaaSamples() const override;
    VkRenderPass renderPass() const override;

    // The result image, readable as a texture after endFrame().
    // Can be bound to a descriptor set and sampled by the main renderer.
    Image& getImage();

private:
    VkRenderPass  renderPass_ = VK_NULL_HANDLE;
    Image         renderImage_;    // the output image (allocated from L1)
    VkFramebuffer framebuffer_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

} // jvk
```

**Integration with other components:**

- **Renderer** holds `RenderTarget&`. Calls `beginFrame()` at frame start (gets Frame with
  command buffer and frame slot). Calls `endFrame()` to submit and present. Uses
  `renderPass()` to lazily build pipelines. Uses `Frame::frameSlot` for Memory::V alternation
  and Device::flushRetired().

- **Memory::V** uses `Frame::frameSlot` to know which slot to write into and when to reset.

- **Pipeline::build()** takes `target.renderPass()` and `target.msaaSamples()` to compile
  against the correct render pass configuration. ResolveEffect pipelines build against the
  effect render pass (from EffectImages) instead.

- **EffectImages** (in resolve/ pipeline module) are sized to `target.width() x target.height()`
  and use `target.format()`. They're created lazily by the resolve pipeline on first effect use
  and resized when the target resizes.

- **MSAA and depth/stencil images** in SwapchainTarget are allocated from `Device::pool()` (L1).
  They live for the lifetime of the swapchain and are recreated on resize.

- **OffscreenTarget::getImage()** returns an `Image` (our Resource subclass) that can be bound
  to a descriptor set via `Memory::M` and sampled by the main renderer's color pipeline —
  this is how ShaderImage composites into the main scene.

---

## Resource

```cpp
namespace jvk {

// Base class for persistent GPU resources. Owns an L1 memory allocation.
// RAII: destructor frees the L1 allocation.
//
// A Resource does NOT own a descriptor set. Descriptor sets map 1-to-1 with
// shaders/pipelines, not with resources. A single Image may be referenced by
// multiple descriptor sets (different shaders sampling the same texture),
// and a single descriptor set may reference multiple Resources (an image at
// binding 0 + a buffer at binding 1). The descriptor set is owned by whoever
// created it — typically the Cache (for built-in draws) or the Shader (for
// user shaders).
//
// Per-frame transient data (shader uniforms, audio buffers) is NOT a Resource.
// That data is written directly into Memory::V scratch memory each frame and
// bound via dynamic offsets on a reusable descriptor set. No L1, no caching.

class Resource {
public:
    Resource() = default;
    Resource(Memory::L1& pool, Memory::L1::Allocation alloc);
    virtual ~Resource();  // frees L1 allocation

    Resource(Resource&& other) noexcept;
    Resource& operator=(Resource&& other) noexcept;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;

    Memory::L1::Allocation allocation() const;
    bool valid() const;

protected:
    Memory::L1*            pool_  = nullptr;
    Memory::L1::Allocation alloc_ = {};
};


// GPU image: texture, atlas page, gradient LUT, effect ping-pong image.
// Lives on L1 permanently. Uploaded once via L2 staging, sampled many times.
class Image : public Resource {
public:
    Image() = default;

    // Create: allocates L1 memory, creates VkImage, binds memory,
    // creates VkImageView and VkSampler.
    Image(Memory::L1& pool, VkDevice device,
          uint32_t width, uint32_t height, VkFormat format,
          VkImageUsageFlags usage,
          VkFilter filter = VK_FILTER_LINEAR,
          VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    ~Image() override;  // destroys VkImage, VkImageView, VkSampler, frees L1

    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    VkImage     image()   const;
    VkImageView view()    const;
    VkSampler   sampler() const;
    uint32_t    width()   const;
    uint32_t    height()  const;

private:
    VkDevice    device_  = VK_NULL_HANDLE;
    VkImage     image_   = VK_NULL_HANDLE;
    VkImageView view_    = VK_NULL_HANDLE;
    VkSampler   sampler_ = VK_NULL_HANDLE;
    uint32_t    w_ = 0, h_ = 0;
};


// GPU buffer: static mesh vertex data, lookup tables, any data uploaded once.
// Lives on L1 permanently. Uploaded once via L2 staging.
//
// NOT for per-frame shader uniforms — those are written directly to Memory::V
// and bound with dynamic descriptor set offsets.
class Buffer : public Resource {
public:
    Buffer() = default;

    // Create: allocates L1 memory, creates VkBuffer, binds memory.
    Buffer(Memory::L1& pool, VkDevice device,
           VkDeviceSize size, VkBufferUsageFlags usage);

    ~Buffer() override;  // destroys VkBuffer, frees L1

    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    VkBuffer     buffer() const;
    VkDeviceSize size()   const;

private:
    VkDevice     device_ = VK_NULL_HANDLE;
    VkBuffer     buffer_ = VK_NULL_HANDLE;
    VkDeviceSize size_   = 0;
};

} // jvk
```

---

## Cache

```cpp
namespace jvk {

// Generic frame-based LRU cache. Templated on key and value.
// Works for GPU resources (CachedImage, CachedBuffer), CPU data, or anything movable.
//
// Eviction policy: no reference counting. Each entry has a lastUsedFrame timestamp
// that updates on every find() hit. Once per frame, evict() scans for entries where
// currentFrame - lastUsedFrame > maxAge and destroys them.
//
// maxAge is tunable per cache instance:
//   - Gradients: 120 frames (~2s) — cheap to recreate from color stops
//   - Textures:  120 frames (~2s) — moderate, requires re-upload via L2
//   - Glyphs:    36000 frames (~10min) — expensive MSDF rasterization
//   - CPU paths: 60 frames (~1s) — cheap to re-flatten
//
// No reference counting means a deleted component's resources linger for maxAge
// frames before eviction. For a 10-100 MB audio plugin UI this is negligible.
// If a component is re-shown within the window, it gets a cache hit for free.
//
// The frame counter is global (on Device), not per-Application, so shared caches
// evict correctly even with multiple plugin instances at different frame rates.
template <typename K, typename V>
class Cache {
public:
    explicit Cache(uint64_t maxAge = 120);

    V*   find(const K& key);       // nullptr if miss, marks as used
    V&   insert(const K& key, V&& value);
    void evict(uint64_t currentFrame);
    void clear();
    bool contains(const K& key) const;
    size_t size() const;

private:
    struct Entry {
        V        value;
        uint64_t lastUsedFrame = 0;
    };
    std::unordered_map<K, Entry> entries;
    uint64_t maxAge;
};


// A cached entry pairs a Resource with its descriptor set.
// On eviction, the Image/Buffer destructor frees L1, and the CachedEntry
// destructor returns the descriptor set to M via the stored M reference.
struct CachedImage {
    Image           image;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    Memory::M*      bindings = nullptr;  // to return descriptor on destruction

    ~CachedImage() { if (bindings && descriptorSet) bindings->free(descriptorSet); }
    CachedImage() = default;
    CachedImage(CachedImage&& o) noexcept
        : image(std::move(o.image)), descriptorSet(o.descriptorSet), bindings(o.bindings)
    { o.descriptorSet = VK_NULL_HANDLE; }
    CachedImage& operator=(CachedImage&& o) noexcept {
        if (this != &o) { image = std::move(o.image); descriptorSet = o.descriptorSet;
                          bindings = o.bindings; o.descriptorSet = VK_NULL_HANDLE; }
        return *this;
    }
};

struct CachedBuffer {
    Buffer          buffer;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    Memory::M*      bindings = nullptr;

    ~CachedBuffer() { if (bindings && descriptorSet) bindings->free(descriptorSet); }
    CachedBuffer() = default;
    CachedBuffer(CachedBuffer&& o) noexcept
        : buffer(std::move(o.buffer)), descriptorSet(o.descriptorSet), bindings(o.bindings)
    { o.descriptorSet = VK_NULL_HANDLE; }
    CachedBuffer& operator=(CachedBuffer&& o) noexcept {
        if (this != &o) { buffer = std::move(o.buffer); descriptorSet = o.descriptorSet;
                          bindings = o.bindings; o.descriptorSet = VK_NULL_HANDLE; }
        return *this;
    }
};


// Bundles all caches for a Device. Shared across all plugin instances.
class ResourceCaches {
public:
    ResourceCaches(Device& device);
    ~ResourceCaches();

    // GPU-resident image caches
    VkDescriptorSet getGradient(const juce::ColourGradient& gradient);
    VkDescriptorSet getTexture(const juce::Image& image);
    GlyphAtlas&     glyphs();

    // Default descriptor set (1x1 black pixel). Fallback when no texture is bound.
    VkDescriptorSet defaultDescriptor() const;

    // Call once per frame.
    void beginFrame(uint64_t frameId);

private:
    Device& device;

    // GPU image caches (gradients, textures — each entry is an Image + descriptor set)
    Cache<uint64_t, CachedImage> gradients;
    Cache<uint64_t, CachedImage> textures;

    // GPU buffer caches (static meshes — each entry is a Buffer + descriptor set)
    // Cache<uint64_t, CachedBuffer> meshes;

    // CPU caches (no GPU involvement — flattened paths, computed intermediaries)
    // Cache<uint64_t, std::vector<UIVertex>> paths;

    std::unique_ptr<GlyphAtlas> glyphAtlas;
    CachedImage                 blackPixel;
    uint64_t currentFrame = 0;
};

} // jvk
```

---

## Command Types (core/Renderer.h)

```cpp
namespace jvk {

// Ops come in two kinds:
//   DrawOps  — sortable within a z-group by pipeline affinity
//   StateOps — push/pop pairs that create scopes; commands inside a scope
//              can only be sorted within that scope; entire scopes sort as a unit
enum class DrawOp : uint8_t {
    // Draw ops
    FillRect, FillRectList, FillRoundedRect, FillEllipse,
    DrawImage, DrawGlyphs, DrawLine,
    DrawShader,
    EffectBlend,    // blend-mode effect (darken, tint, etc.)
    EffectResolve,  // single-pass resolve effect (hue, saturation, color correction)
    EffectKernel,   // multi-pass kernel effect (blur, sharpen, bloom)

    // State ops (scoped — must have matching push/pop)
    PushClipRect,   // scissor clip to a rectangle
    PushClipPath,   // stencil clip to a path
    PopClip,        // undo the most recent push

    // Note: FillPath is NOT a draw op. It is decomposed at record time into:
    //   PushClipPath(path) → FillRect(pathBounds) → PopClip()
    // This means any fill type (solid, gradient) works inside a path fill,
    // and path clipping shares the exact same code path.
};

struct DrawCommand {
    float                zOrder;       // sort key — default: component tree depth
    DrawOp               op;           // which function to replay
    uint32_t             dataOffset;   // byte offset into arena
    uint32_t             dataSize;
    juce::Rectangle<int> clipBounds;   // captured at record time (scissor)
    uint8_t              stencilDepth; // captured at record time
    uint32_t             scopeDepth;   // nesting depth for StateOp scopes (0 = top level)
};

// Flat byte arena for command parameter data.
// Grows linearly, never shrinks, reset (not freed) each frame.
class Arena {
public:
    template <typename T> uint32_t push(const T& data);
    template <typename T> uint32_t pushSpan(std::span<const T> data);
    template <typename T> const T& read(uint32_t offset) const;
    template <typename T> std::span<const T> readSpan(uint32_t offset, uint32_t count) const;
    void reset();

private:
    std::vector<std::byte> buffer;
    uint32_t head = 0;
};

} // jvk
```

DrawOp, DrawCommand, and Arena are defined in `core/Renderer.h` alongside the Renderer class
that owns and operates on them. They have no JUCE dependency.

---

## Pipeline (core/Pipeline.h)

```cpp
namespace jvk {

// ============================================================================
// Pipelines are never user-facing.
//
// The user calls fillRect, drawText, blur, drawShader. The system picks the
// pipeline. Built-in draw commands use one of 6 fixed pipelines from the
// PipelineStore. User shaders create their own pipeline internally.
//
// A pipeline is: one vertex shader + one fragment shader + fixed state
// (blend mode, stencil config, color write, MSAA). Immutable after creation.
// You can't toggle layers or swap shaders — you switch to a different pipeline.
//
// State.setPipeline() is how draw functions request a pipeline. State tracks
// what's bound and skips redundant switches. The deferred command buffer
// sub-sorts by pipeline within z-groups to minimize switches further.
// ============================================================================

enum class PipelineID : uint32_t {
    // Scene pipelines (vertex-driven, UIVertex layout, scene render pass)
    Color,          // Alpha blend. Rects, text, images, SDF shapes, gradients.
    StencilWrite,   // No color write, stencil increment. Writes path geometry to stencil buffer.
    StencilCover,   // No color write, stencil decrement. Undoes stencil on PopClip.

    // Effect pipelines (fullscreen triangle, no vertex input)
    BlendEffect,    // Multiplicative blend against existing framebuffer. No resolve needed.
    ResolveEffect,  // Samples resolved frame texture, writes to effect render target.

    COUNT
    // Note: no ClipTest ID. Stencil-testing is handled by clip variants (see below).
};


// Base class for all pipeline modules. Subclass this to add a new
// rendering capability. Each pipeline module is a self-contained folder
// in graphics/pipelines/ with its shaders, config, and draw functions.
//
// ===== Stencil Clip Variants =====
//
// The stencil buffer is per-pixel state that persists across pipeline switches
// within a render pass. Stencil clipping works in three steps:
//
//   1. PushClipPath: StencilWrite pipeline draws path geometry → stencil ref increments
//   2. Any draws inside the clip use the CLIP VARIANT of their pipeline
//      (same shaders, same blend, but stencil test enabled: ref > 0 passes)
//   3. PopClip: StencilCover pipeline draws reversed fan → stencil ref decrements
//
// Pipelines that support stencil clipping (Color, BlendEffect) build TWO
// VkPipelines internally: a normal variant and a clip variant. The only
// difference is stencil test enable + compare op. Same shaders, same layout.
//
// State auto-selects: when stencilDepth > 0, it binds clipHandle() instead of
// handle(). Pipeline modules that don't support clipping (StencilWrite,
// StencilCover, ResolveEffect) return handle() for both.
//
// ResolveEffect doesn't need a clip variant because it runs on separate
// ping-pong framebuffers (no stencil buffer). The composite-back step that
// writes the result into the scene framebuffer uses the Color pipeline's
// clip variant, which does respect the stencil.
//
class Pipeline {
public:
    Pipeline(Device& device);
    virtual ~Pipeline();

    Pipeline(Pipeline&&) noexcept;
    Pipeline(const Pipeline&) = delete;

    // Subclasses override to define their fixed-function configuration.
    virtual PipelineConfig config() const = 0;

    // Override to return a config with stencil test enabled.
    // Default returns std::nullopt (no clip variant — handle() used for both).
    // Pipelines that need stencil clipping override this.
    virtual std::optional<PipelineConfig> clipConfig() const { return std::nullopt; }

    // Which DrawOps this pipeline handles. Renderer builds a lookup table from this.
    virtual std::span<const DrawOp> supportedOps() const = 0;

    // The draw function called during replay for each supported DrawOp.
    virtual void execute(Renderer& r, const Arena& arena, const DrawCommand& cmd) = 0;

    // Load SPIR-V shaders. Called by subclass constructors.
    void loadVertexShader(std::span<const uint32_t> spirv);
    void loadFragmentShader(std::span<const uint32_t> spirv);

    // Define the descriptor set layout. Default: 1 combined image sampler at binding 0.
    virtual void defineLayout(Memory::M& bindings);

    // Build the Vulkan pipeline(s). Called lazily on first use by the Renderer.
    // If clipConfig() returns a value, builds both normal and clip variants.
    void build(VkRenderPass renderPass, VkSampleCountFlagBits msaa);
    bool isBuilt() const;

    // Access for State binding during replay.
    // State calls clipHandle() when stencilDepth > 0, handle() otherwise.
    VkPipeline       handle() const;       // normal variant
    VkPipeline       clipHandle() const;   // stencil-test variant (same as handle if no clip config)
    VkPipelineLayout layout() const;       // shared between both variants (owned)

protected:
    Device& device;

private:
    VkPipeline       pipeline_     = VK_NULL_HANDLE;  // normal
    VkPipeline       clipPipeline_ = VK_NULL_HANDLE;  // clip variant (VK_NULL_HANDLE if none)
    VkPipelineLayout layout_       = VK_NULL_HANDLE;  // owned, shared by both
    std::vector<uint32_t> vertSpirv_;
    std::vector<uint32_t> fragSpirv_;
    bool built_ = false;
};

} // jvk
```

**How pipelines connect to State and draw commands:**

During replay, the Renderer looks up `pipelineForOp_[cmd.op]` to find the Pipeline module.
The module's `execute()` calls `state.setPipeline(this)`, `state.setResource(descriptorSet)`,
and `state.draw(cmd, vertices, count)`. State dirty-tracks the bound pipeline and skips
redundant `vkCmdBindPipeline` calls.

For user shaders, the Shader class creates its own Pipeline internally (custom layout from
SPIR-V reflection). During `drawShader` replay, State binds the shader's pipeline directly
via `state.setCustomPipeline(shader.pipeline(), shader.layout())`, then invalidates after
so the next built-in pipeline rebinds. The user never sees any of this.

```
Built-in draw:  pipelineForOp_[FillRect]->execute(r, arena, cmd)
                  → state.setPipeline(this)
                  → State checks stencilDepth: 0 → handle(), >0 → clipHandle()
User shader:    state.setCustomPipeline(...)  → direct bind with stencilRef, invalidate after
```

**Stencil clip flow (concrete example):**

```
User code:                          State internals:
─────────                           ────────────────
g.reduceClipRegion(path)         →  PushClipPath recorded
g.fillRect(bounds, color)        →  FillRect recorded
g.fillRect(otherBounds, color)   →  FillRect recorded
g.restoreState()                 →  PopClip recorded

Replay:
  PushClipPath:
    State binds StencilWrite pipeline
    Draws path triangle fan → stencil buffer has ref=1 inside path
    stencilDepth_ = 1

  FillRect:
    ColorPipeline::execute() calls state.setPipeline(this)
    State sees stencilDepth_=1 → binds clipHandle() + stencilRef=1
    Pixels inside path pass stencil test, outside are rejected
    (ColorPipeline doesn't know about stencil — State handled it)

  FillRect (second):
    State sees same pipeline + same stencilRef → no-op (dirty tracking)
    Just draws

  PopClip:
    State binds StencilCover pipeline
    Draws reversed fan → stencil buffer back to 0
    stencilDepth_ = 0
    Next setPipeline() will bind handle() (normal, no stencil test)
```

---

## State (core/Renderer.h, internal to Renderer)

```cpp
namespace jvk {

// Internal to Renderer. Pipeline execute() functions interact with State via
// the Renderer reference. State is never exposed outside core/.
//
// Draw functions set three things: pipeline, resource, vertices.
// State handles everything else (scissor, stencil, push constants, dirty tracking).
//
// ===== Stencil clip auto-selection =====
// When setPipeline() is called, State checks stencilDepth(). If > 0 and the
// pipeline has a clip variant, State binds clipHandle() instead of handle().
// Also sets stencilRef = stencilDepth so the stencil test passes for the
// correct nesting level. Pipeline execute() functions don't need to know
// whether they're inside a stencil clip — State handles it transparently.
//
class State {
public:
    State() = default;

    // The three things pipeline execute() functions set:
    void setPipeline(Pipeline* pipeline);          // auto-selects clip variant if stencilDepth > 0
    void setCustomPipeline(VkPipeline pipeline,    // user shader (direct bind, invalidates after)
                           VkPipelineLayout layout);
    void setResource(VkDescriptorSet set);
    void setResource(VkDescriptorSet set,          // with dynamic offset (V scratch buffers)
                     uint32_t dynamicOffset);
    void draw(const DrawCommand& cmd, const UIVertex* verts, uint32_t count);

    // ===== Clip stack =====
    // Managed by the stencil pipeline module's execute() during StateOp replay.
    //
    // pushClipPath():
    //   1. Flush pending vertices
    //   2. Bind StencilWrite pipeline
    //   3. Draw path as triangle fan → stencil buffer incremented inside path
    //   4. Increment stencilDepth
    //   5. Restore previous pipeline (now auto-selects clip variant because depth > 0)
    //
    // popClip():
    //   1. Flush pending vertices
    //   2. Bind StencilCover pipeline
    //   3. Draw reversed fan → stencil buffer decremented
    //   4. Decrement stencilDepth
    //   5. Restore previous pipeline (clip variant if still nested, normal if depth == 0)
    //
    // Between push and pop, ALL draws automatically use clip variants:
    //   - Color draws (fillRect, drawImage, drawGlyphs) → Color clip variant
    //   - BlendEffect (darken, tint) → BlendEffect clip variant
    //   - ResolveEffect (blur) → runs on separate framebuffer (no stencil),
    //     but the composite-back quad uses Color clip variant
    //   - User shaders → State::setCustomPipeline binds with stencil test enabled
    //     via dynamic stencil reference (user shader pipelines are built with
    //     stencil test enabled, ref set dynamically)

    void pushClipRect(const juce::Rectangle<int>& rect);
    void pushClipPath(const juce::Path& path,
                      const juce::AffineTransform& transform);
    void popClip();

    juce::Rectangle<int> clipBounds() const;
    uint8_t              stencilDepth() const;

    void invalidate();
    void begin(VkCommandBuffer cmd, Memory::V& vertices,
               float vpWidth, float vpHeight);

private:
    VkCommandBuffer  cmd_       = VK_NULL_HANDLE;
    Memory::V*       vertices_  = nullptr;
    float            vpWidth_   = 0;
    float            vpHeight_  = 0;

    // Dirty-tracked bound state
    Pipeline*        currentPipeline_ = nullptr;  // tracks which pipeline was requested
    VkPipeline       boundPipeline   = VK_NULL_HANDLE;  // what's actually bound (may be clip variant)
    VkPipelineLayout boundLayout     = VK_NULL_HANDLE;
    VkDescriptorSet  boundDescriptor = VK_NULL_HANDLE;
    uint32_t         boundStencilRef = 0;
    juce::Rectangle<int> boundScissor { -1, -1, 0, 0 };

    // Clip stack
    struct ClipEntry {
        enum Type { Rect, Path };
        Type type;
        juce::Rectangle<int> rect;          // for Rect clips: scissor intersection
        std::vector<UIVertex> fanVerts;     // for Path clips: undo geometry
        juce::Rectangle<int> fanBounds;     // for Path clips: tight scissor on undo
    };
    std::vector<ClipEntry> clipStack;
    uint8_t stencilDepth_ = 0;
};

} // jvk
```

---

## Shader (graphics/Shader.h)

```cpp
namespace jvk {

// User-loadable fragment shader with SPIR-V reflection.
//
// Two-phase interface matching the deferred rendering model:
//
// RECORD PHASE (during paint):
//   shader.set("background", myImage)   → resolves image through ResourceCaches.
//                                          Cache hit: grabs existing descriptor info.
//                                          Cache miss: creates Image on L1, stages L2 upload.
//                                          First frame after miss gets black pixel default
//                                          (upload completes next frame via flushUploads).
//
//   shader.set("gain", 1.5f)            → stores value locally in uniformData_ buffer.
//   shader.set("audioData", audioSpan)  → copies floats into uniformData_ buffer.
//   vk->drawShader(shader, region)      → pushes DrawCommand into CommandBuffer.
//
// REPLAY PHASE (during sorted command execution):
//   draw::drawShader() executes:
//     1. Writes shader's uniformData_ into V.write() → gets byte offset
//     2. Binds shader's descriptor set with dynamic offset for buffer binding
//     3. Draws fullscreen triangle with scissor/stencil from DrawCommand
//
// Images are resolved eagerly (during set()) because they involve cache lookups.
// Floats are deferred (written to V during replay) because V is per-frame scratch.
//
class Shader {
public:
    Shader() = default;

    // Load SPIR-V and reflect all bindings. Discovers names, types, binding indices.
    // Can be expensive — call once at construction, not per frame.
    void load(std::span<const uint32_t> fragSpirv);

    // ===== Set bindings by name (during paint/record phase) =====

    // Image bindings: resolved through ResourceCaches immediately.
    // The cache manages the Image's lifetime (frame-based LRU eviction).
    // The shader writes the image view + sampler into its descriptor set.
    void set(const juce::String& name, const juce::Image& image,
             ResourceCaches& caches);

    // Float/buffer bindings: stored locally, written to V during replay.
    // All scalar and array data packs into one contiguous buffer:
    //   [gain | frequency | attack | ... | audio[512] | sidechain[512]]
    // One buffer binding in the descriptor set, dynamic offset at bind time.
    void set(const juce::String& name, float value);
    void set(const juce::String& name, std::span<const float> data);

    // ===== Lifecycle =====

    // Lazy-init on first draw: creates pipeline, descriptor set layout (from
    // reflected bindings), allocates descriptor set from M, binds V's VkBuffer
    // to the buffer binding, binds black pixel default to any unset image bindings.
    void ensureCreated(Device& device, VkRenderPass renderPass,
                       VkSampleCountFlagBits msaa);
    bool isReady() const;

    // ===== Access for replay =====

    VkPipeline       pipeline()      const;
    VkPipelineLayout layout()        const;
    VkDescriptorSet  descriptorSet() const;

    // The packed uniform data to write into V during replay.
    const float* uniformData() const;
    size_t       uniformSize() const;  // in bytes

    ~Shader();  // Destroys pipeline, layout, frees descriptor set to M.

private:
    // Reflected binding metadata (populated by load() via spirv_reflect)
    struct BindingInfo {
        juce::String     name;
        uint32_t         binding;
        VkDescriptorType type;         // COMBINED_IMAGE_SAMPLER or STORAGE_BUFFER
        uint32_t         offsetInBuffer = 0;  // byte offset in uniformData_ (for buffer fields)
        uint32_t         sizeInBuffer   = 0;  // byte size (for buffer fields)
        bool             bound = false;
    };
    std::vector<BindingInfo> reflectedBindings;

    std::vector<uint32_t> spirv;
    VkPipeline            pipeline_      = VK_NULL_HANDLE;
    VkPipelineLayout      layout_        = VK_NULL_HANDLE;
    VkDescriptorSet       descriptorSet_ = VK_NULL_HANDLE;

    // Packed uniform buffer: all scalars + arrays contiguous.
    // Layout determined by SPIR-V reflection (offsets match shader struct).
    // Written to V.write() during replay, bound with dynamic offset.
    std::vector<float>    uniformData_;

    double startTime = 0;
    bool   created   = false;
};

} // jvk
```

---

## Graphics (graphics/Graphics.h)

```cpp
namespace jvk {

// jvk::Graphics implements juce::LowLevelGraphicsContext so JUCE can drive it,
// and adds GPU effects + z-order control. This is the only JUCE-dependent class.
// It holds a reference to a Renderer and pushes commands into it.
// It does NOT touch Vulkan — purely recording.
class Graphics : public juce::LowLevelGraphicsContext {
public:
    Graphics(Renderer& renderer, float displayScale);
    ~Graphics() override;

    // ===== Z-order control =====
    void setZOrder(float z);

    // ===== Recording state stack (no Vulkan) =====
    struct RecordState {
        juce::AffineTransform transform;
        juce::FillType        fill { juce::Colours::black };
        float                 opacity = 1.0f;
        juce::Font            font { juce::FontOptions {} };
        juce::Rectangle<int>  clipBounds;
        float                 zOrder = 0.0f;
        uint32_t              scopeDepth = 0;
        uint8_t               stencilDepth = 0;
    };

    // ===== LowLevelGraphicsContext overrides (push DrawCommands into Renderer) =====
    void fillRect(const juce::Rectangle<float>& r) override;
    void fillRect(const juce::Rectangle<int>& r, bool) override;
    void fillRectList(const juce::RectangleList<float>& list) override;
    void fillRoundedRectangle(const juce::Rectangle<float>& r, float cornerSize) override;
    void fillEllipse(const juce::Rectangle<float>& area) override;
    void drawImage(const juce::Image& img, const juce::AffineTransform& t) override;
    void drawGlyphs(juce::Span<const uint16_t> glyphs,
                    juce::Span<const juce::Point<float>> positions,
                    const juce::AffineTransform& t) override;
    void drawLine(const juce::Line<float>& line) override;

    // fillPath decomposes into PushClipPath + FillRect + PopClip
    void fillPath(const juce::Path& path, const juce::AffineTransform& t) override;

    // Clipping pushes StateOps (scoped)
    bool clipToRectangle(const juce::Rectangle<int>& r) override;
    void clipToPath(const juce::Path& path, const juce::AffineTransform& t) override;
    void saveState() override;
    void restoreState() override;
    // ... remaining LowLevelGraphicsContext overrides

    // ===== GPU Effects (push effect DrawOps into Renderer) =====
    void multiplyRGB(float r, float g, float b, juce::Rectangle<float> region = {});
    void darken(float amount, juce::Rectangle<float> region = {});
    void brighten(float amount, juce::Rectangle<float> region = {});
    void tint(juce::Colour c, juce::Rectangle<float> region = {});
    void warmth(float amount, juce::Rectangle<float> region = {});
    void hueShift(float amount, juce::Rectangle<float> region = {});
    void saturate(float amount, juce::Rectangle<float> region = {});
    void desaturate(float amount, juce::Rectangle<float> region = {});
    void blur(float radius);
    void blur(juce::Rectangle<float> region, float radius);
    void drawShader(Shader& shader, juce::Rectangle<float> region = {});

    Renderer& getRenderer();

private:
    Renderer&   renderer;
    float       displayScale;
    std::vector<RecordState> stateStack;
};

} // jvk
```

---

## Pipeline Modules (graphics/pipelines/)

Each pipeline module is a self-contained folder with its shaders, pipeline config,
and draw function implementations. Adding a new rendering capability = adding a new folder.

```cpp
// Example: graphics/pipelines/color/ColorPipeline.h
namespace jvk::pipelines {

class ColorPipeline : public Pipeline {
public:
    ColorPipeline(Device& device)
        : Pipeline(device)
    {
        loadVertexShader(ui2d_vert_spv);
        loadFragmentShader(ui2d_frag_spv);
    }

    PipelineConfig config() const override {
        return { .blendMode = BlendMode::AlphaBlend,
                 .topology  = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
    }

    // Clip variant: same config but with stencil test enabled.
    // State auto-selects this when stencilDepth > 0.
    std::optional<PipelineConfig> clipConfig() const override {
        auto cfg = config();
        cfg.stencilTestEnable = true;
        cfg.stencilCompareOp  = VK_COMPARE_OP_EQUAL;  // pass when ref == stencil value
        cfg.stencilFailOp     = VK_STENCIL_OP_KEEP;
        cfg.stencilPassOp     = VK_STENCIL_OP_KEEP;
        return cfg;
    }

    std::span<const DrawOp> supportedOps() const override {
        static constexpr DrawOp ops[] = {
            DrawOp::FillRect, DrawOp::FillRoundedRect, DrawOp::FillEllipse,
            DrawOp::DrawImage, DrawOp::DrawGlyphs, DrawOp::DrawLine
        };
        return ops;
    }

    void execute(Renderer& r, const Arena& arena, const DrawCommand& cmd) override;
};

// BlendPipeline also provides clipConfig() — same pattern.
// StencilWrite/Cover and ResolveEffect do NOT — they return std::nullopt.

} // jvk::pipelines
```

**Param structs** (packed into arena during recording by Graphics):

```cpp
struct FillRectParams {
    juce::Rectangle<float> rect;
    juce::FillType         fill;
    juce::AffineTransform  transform;
    float                  opacity;
    float                  scale;
};

struct DrawGlyphsParams {
    uint32_t               glyphCount;
    juce::AffineTransform  transform;
    juce::Font             font;
    juce::FillType         fill;
    float                  opacity;
    float                  scale;
    // followed in arena by: uint16_t[glyphCount] + Point<float>[glyphCount]
};

struct BlurParams { float radius; };
// ... one param struct per DrawOp
```

---

## Render Loop

```cpp
// In jvk/AudioProcessorEditor.h — the JUCE-specific entrypoint.
// This is the only file that knows about both JUCE components and the Renderer.

void AudioProcessorEditor::render() {
    renderer.reset();
    device->caches().beginFrame(frameCounter++);

    // Phase 1: Record — JUCE paints, Graphics pushes commands into Renderer
    Graphics graphics(renderer, displayScale);
    juce::Graphics g(graphics);
    paintEntireComponent(g, true);

    // Phase 2: Sort → beginFrame → flushUploads → beginRenderPass →
    //          Replay → endRenderPass → endFrame (submit + present)
    // All handled internally by execute().
    renderer.execute();
}

// Or without JUCE (e.g. ShaderImage, custom app):
void customRender() {
    renderer.reset();
    renderer.push(DrawOp::FillRect, 0.0f, clip, 0, 0, myRectParams);
    renderer.push(DrawOp::DrawShader, 1.0f, clip, 0, 0, myShaderParams);
    renderer.execute();
}
```

---

## File Layout

```
include/jvk/

  // ===== User-facing classes (jvk root) =====
  AudioProcessorEditor.h     // jvk::AudioProcessorEditor — JUCE plugin entrypoint.
                             // Creates Device, Renderer, SwapchainTarget, registers
                             // built-in pipelines, drives the render loop via paint().
                             // This is what plugin developers subclass.

  ShaderImage.h / .cpp       // jvk::ShaderImage — offscreen rendering to a GPU image.
                             // Creates its own Renderer + OffscreenTarget.
                             // Users push draw commands or use jvk::Graphics, call render(),
                             // and the result is a texture usable in the main renderer.
                             // Composable with AudioProcessorEditor — same Device, shared caches.

  Shader.h / .cpp            // jvk::Shader — user custom shaders with SPIR-V reflection.
                             // load() reflects bindings, set() by name, drawShader() to render.

  // ===== Core (no JUCE dependency) =====
  core/
    Device.h / .cpp            // Global Vulkan lifetime + memory tiers
    Memory.h                   // Memory::L1, L2, M, V, CPU
    Resource.h                 // Resource base + Image + Buffer
    Cache.h                    // Cache<K,V> template + ResourceCaches
    Pipeline.h / .cpp          // Pipeline base class (developer-friendly interface)
    PipelineConfig.h           // Blend/stencil/depth config struct
    Renderer.h / .cpp          // Command buffer + Arena + State + sort + replay dispatch
    RenderTarget.h             // Abstract render target base
    SwapchainTarget.h / .cpp   // Windowed rendering
    OffscreenTarget.h / .cpp   // Offscreen rendering

  // ===== Graphics (JUCE 2D rendering backend) =====
  graphics/
    Graphics.h / .cpp          // jvk::Graphics — LowLevelGraphicsContext + effects + z-order

    pipelines/
      color/
        ColorPipeline.h        // : Pipeline — alpha blend, 2D scene rendering
        ColorDraw.h / .cpp     // fillRect, fillRoundedRect, fillEllipse,
                               // drawImage, drawGlyphs, drawLine
        UIVertex.h             // Vertex format for 2D pipelines
        GlyphAtlas.h / .cpp    // MSDF text atlas
        ui2d.vert.glsl
        ui2d.frag.glsl

      stencil/
        StencilPipeline.h      // : Pipeline — StencilWrite + StencilCover
        StencilDraw.h / .cpp   // pushClipPath, popClip
        // reuses ui2d vertex shader

      blend/
        BlendPipeline.h        // : Pipeline — multiplicative blend
        BlendDraw.h / .cpp     // effectBlend (darken, tint, brighten, warmth)

      resolve/
        ResolvePipeline.h      // : Pipeline — reads resolved frame as texture
        ResolveDraw.h / .cpp   // effectResolve, effectKernel (blur, hue, sat)
        EffectImages.h         // Ping-pong images for multi-pass effects
        fullscreen.vert.glsl
        pp_blur.frag.glsl
        pp_huesat.frag.glsl

      // Future — add a folder, add a pipeline:
      scene/
        ScenePipeline.h
        SceneDraw.h / .cpp
        scene.vert.glsl
        scene.frag.glsl
```

---

## Key Invariants

1. **Two-phase rendering.** Recording pushes commands into the Renderer. `execute()` sorts and replays. No Vulkan during recording.

2. **Renderer is JUCE-independent.** Anyone can push commands and call `execute()`. Graphics.h is the JUCE adapter that pushes during `paint()`.

3. **Pipeline modules are self-contained.** Each pipeline folder has its shaders, config, and draw implementations. Adding a new capability = adding a folder. The Renderer routes DrawOps to pipelines via `supportedOps()` lookup.

4. **Z-order defaults to component depth.** Default JUCE apps work identically. Opt-in `setZOrder()` enables cross-component batching.

5. **Pipeline execute() sets three things: pipeline, resource, vertices.** State handles everything else (scissor, stencil clip variant auto-selection, push constants, dirty tracking). Pipeline modules don't know whether they're inside a stencil clip.

6. **StateOps are scoped.** Push/pop clip pairs create scopes in the command buffer. Commands inside a scope are only sortable within that scope.

7. **fillPath is not a draw op.** It decomposes into PushClipPath + FillRect + PopClip.

8. **Resources and descriptor sets are independent.** Resource owns L1 allocation. Descriptor sets are owned by caches or shaders. Per-frame uniforms go to V scratch with dynamic offsets.

9. **Four memory tiers.** L1 (persistent GPU), L2 (staging), M (descriptors), V (per-frame vertices). L1/L2/M on Device (shared). V on Renderer (per-instance).

10. **Three-layer resource management.** Memory allocates. Cache manages lifetime. Resource owns the handle.

11. **Caches are shared across plugin instances.** Multiple plugins sharing typefaces, images, or gradients get cache hits for free.

12. **Three effect categories, two effect pipelines.** BlendEffect (in-place), ResolveEffect (single-pass and multi-pass).

13. **Shaders are reflected, not registered.** SPIR-V reflection discovers bindings. Unset bindings get 1x1 black pixel default.

14. **Zero-allocation steady state.** Arena resets. V doubles then stabilizes. L1 sub-allocates from existing blocks. No per-frame heap allocation.

15. **Pipeline switches minimized by sorting.** Within z-groups and StateOp scopes, commands sub-sort by pipeline.
