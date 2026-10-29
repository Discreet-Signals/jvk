/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: VulkanBackend.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::vulkan
{

// =============================================================================
// vulkan::Backend — gpu::Backend on Vulkan
//
// One per process (acquire()), shared by every editor, offscreen surface and
// cache through shared_ptr. Owns the Device, every texture, program and
// pipeline behind the core's handles, the staging for texture writes, and the
// deferred destruction of all of it.
//
// Locks, always taken in this order (a later one never waits on an earlier):
//
//   Device::queueLock()  THE submit lock. Every queue operation, the record of
//                        a frame (Surface::submit records under it, so texture
//                        layout tracking is single-threaded), the framebuffer
//                        cache, collect(). Reentrant.
//   tableMutex_          handle-table create/destroy, the pipeline intern map,
//                        the retire list, samplers.
//   writeMutex_          queued texture writes and their staging.
//   renderPassMutex_     the VkRenderPass cache (pipelines are built on any
//                        thread, frames under the submit lock).
//
// Resolving a handle takes no lock: slots live in chunks that never move, and
// a slot is reused only after the GPU and every recorder are done with it.
// =============================================================================

// The stages a sampled texture is read in.
inline constexpr VkPipelineStageFlags kShaderStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
                                                    | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

// ---- Host-visible buffers ---------------------------------------------------

struct HostBuffer
{
    VkBuffer      buffer     = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    void*         mapped     = nullptr;
    VkDeviceSize  size       = 0;

    explicit operator bool() const { return buffer != VK_NULL_HANDLE; }
};

HostBuffer createHostBuffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage, bool readback = false);
void       destroyHostBuffer(VmaAllocator allocator, HostBuffer& buffer);

// ---- Handle tables ----------------------------------------------------------
//
// Handle value = generation << 24 | (slot + 1). 0 is null; a stale handle
// (its slot freed, maybe reused) fails the generation check and resolves to
// nothing rather than aliasing the new occupant.
template <typename T>
class SlotTable
{
public:
    static constexpr uint32_t kChunkBits = 10;
    static constexpr uint32_t kChunkSize = 1u << kChunkBits;
    static constexpr uint32_t kMaxChunks = (1u << 24) / kChunkSize;

    struct Entry
    {
        T                     value {};
        std::atomic<uint32_t> tag { 0 };   // generation << 1 | live
    };

    SlotTable() : chunks_(new std::atomic<Entry*>[kMaxChunks])
    {
        for (uint32_t i = 0; i < kMaxChunks; ++i) chunks_[i].store(nullptr, std::memory_order_relaxed);
    }

    // Under the owner's mutex. Returns the handle; `out` is the slot to fill
    // BEFORE the handle is handed to anyone.
    uint32_t allocate(T*& out)
    {
        uint32_t index;
        if (!free_.empty()) { index = free_.back(); free_.pop_back(); }
        else
        {
            index = next_++;
            if (index >= kMaxChunks * kChunkSize - 1) { --next_; out = nullptr; return 0; }
            const uint32_t c = index >> kChunkBits;
            if (chunks_[c].load(std::memory_order_relaxed) == nullptr)
            {
                owned_.push_back(std::make_unique<Entry[]>(kChunkSize));
                chunks_[c].store(owned_.back().get(), std::memory_order_release);
            }
        }
        Entry& e = entry(index);
        uint32_t generation = (e.tag.load(std::memory_order_relaxed) >> 1) & 0xFFu;
        if (generation == 0) generation = 1;
        e.value = T {};
        e.tag.store((generation << 1) | 1u, std::memory_order_release);
        out = &e.value;
        return (generation << 24) | (index + 1);
    }

    // Lock-free. Null for a null, stale or freed handle.
    T* resolve(uint32_t handle) const
    {
        if (handle == 0) return nullptr;
        const uint32_t index = (handle & 0xFFFFFFu) - 1;
        Entry* chunk = chunks_[index >> kChunkBits].load(std::memory_order_acquire);
        if (chunk == nullptr) return nullptr;
        Entry& e = chunk[index & (kChunkSize - 1)];
        const uint32_t tag = e.tag.load(std::memory_order_acquire);
        if ((tag & 1u) == 0 || ((tag >> 1) & 0xFFu) != (handle >> 24)) return nullptr;
        return &e.value;
    }

    // Under the owner's mutex: the slot is free, its generation moves on.
    void release(uint32_t handle)
    {
        if (resolve(handle) == nullptr) return;
        const uint32_t index = (handle & 0xFFFFFFu) - 1;
        Entry& e = entry(index);
        uint32_t generation = ((handle >> 24) + 1) & 0xFFu;
        if (generation == 0) generation = 1;
        e.tag.store(generation << 1, std::memory_order_release);
        e.value = T {};
        free_.push_back(index);
    }

    // Under the owner's mutex (teardown): every live handle.
    template <typename F>
    void forEachLive(F&& f)
    {
        for (uint32_t index = 0; index < next_; ++index)
        {
            Entry& e = entry(index);
            const uint32_t tag = e.tag.load(std::memory_order_relaxed);
            if (tag & 1u) f((((tag >> 1) & 0xFFu) << 24) | (index + 1), e.value);
        }
    }

private:
    Entry& entry(uint32_t index) const
    {
        return chunks_[index >> kChunkBits].load(std::memory_order_acquire)[index & (kChunkSize - 1)];
    }

    std::unique_ptr<std::atomic<Entry*>[]> chunks_;
    std::vector<std::unique_ptr<Entry[]>>  owned_;
    std::vector<uint32_t>                  free_;
    uint32_t                               next_ = 0;
};

// ---- What the handles name ----------------------------------------------------

// Where a texture's last use left it: what the next use's barrier waits on.
// Read and written under the submit lock only.
struct TextureState
{
    VkImageLayout        layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags stage  = 0;   // the stages that last touched it
    VkAccessFlags        writes = 0;   // the writes not yet made visible
};

struct TextureSlot
{
    VkImage            image      = VK_NULL_HANDLE;
    VmaAllocation      allocation = nullptr;          // null: not ours (a swap image)
    VkImageView        view       = VK_NULL_HANDLE;   // sampling: every layer and mip
    VkImageView        attachView = VK_NULL_HANDLE;   // rendering: layer 0, mip 0 (may equal view)
    VkFormat           format     = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect     = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageUsageFlags  usage      = 0;
    PixelFormat        pixelFormat = PixelFormat::RGBA8;
    gpu::TextureKind   kind       = gpu::TextureKind::Color;
    uint32_t           width = 0, height = 0, layers = 1, mips = 1;
    TextureState       state;
};

struct ProgramSlot
{
    VkShaderModule                     vertex   = VK_NULL_HANDLE;
    VkShaderModule                     fragment = VK_NULL_HANDLE;
    std::string                        vertexEntry, fragmentEntry;
    VkPipelineLayout                   layout   = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayout> setLayouts;      // by set number; gaps hold the shared empty layout
    std::vector<uint8_t>               ownsSetLayout;   // 1 where setLayouts[i] is the program's own
    std::vector<gpu::BindingDesc>      bindings;        // every binding, the portable ones spelled out
    uint32_t                           constantsBytes = 0;
    std::vector<uint32_t>              pipelines;       // pipeline handles built from it (under tableMutex_)
};

struct PipelineSlot
{
    gpu::PipelineDesc desc;
    VkPipeline        pipeline = VK_NULL_HANDLE;   // null: the build failed; its draws are skipped
    uint32_t          program  = 0;                // handle value
};

// ---- Render pass cache key ------------------------------------------------------

struct RenderPassKey
{
    uint8_t  colorCount = 0;
    VkFormat color[gpu::kMaxColorAttachments] {};   // VK_FORMAT_UNDEFINED: an unused slot
    uint8_t  colorLoad[gpu::kMaxColorAttachments] {}, colorStore[gpu::kMaxColorAttachments] {};
    VkFormat depthStencil = VK_FORMAT_UNDEFINED;
    uint8_t  dsLoad = 0, dsStore = 0;

    bool operator==(const RenderPassKey& o) const
    {
        if (colorCount != o.colorCount || depthStencil != o.depthStencil
            || dsLoad != o.dsLoad || dsStore != o.dsStore) return false;
        for (uint32_t i = 0; i < colorCount; ++i)
            if (color[i] != o.color[i] || colorLoad[i] != o.colorLoad[i] || colorStore[i] != o.colorStore[i])
                return false;
        return true;
    }
};

struct RenderPassKeyHash
{
    size_t operator()(const RenderPassKey& k) const noexcept
    {
        uint64_t h = 1469598103934665603ull;
        auto mix = [&h](uint64_t v) { h ^= v; h *= 1099511628211ull; };
        mix(k.colorCount);
        for (uint32_t i = 0; i < k.colorCount; ++i)
            mix((uint64_t(k.color[i]) << 16) | (uint64_t(k.colorLoad[i]) << 8) | k.colorStore[i]);
        mix((uint64_t(k.depthStencil) << 16) | (uint64_t(k.dsLoad) << 8) | k.dsStore);
        return static_cast<size_t>(h);
    }
};

// =============================================================================

class Backend final : public gpu::Backend, public std::enable_shared_from_this<Backend>
{
public:
    // The process's backend, made on first ask. A LOST backend never renders
    // again: whoever still holds it falls back, and the next acquire makes a
    // fresh one. An unusable one (no Vulkan, no GPU) is returned too: check
    // usable().
    static std::shared_ptr<Backend> acquire();

    ~Backend() override;
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    bool                usable() const noexcept            { return device_->usable(); }
    const juce::String& unavailableReason() const noexcept { return device_->unavailableReason(); }
    Device&             device() noexcept                  { return *device_; }

    // ---- gpu::Backend ---------------------------------------------------------
    const gpu::Caps& caps() const override { return caps_; }
    bool isLost() const override { return device_->isLost(); }
    void waitIdle() override;

    gpu::TextureId createTexture(const gpu::TextureDesc& desc) override;
    void           destroyTexture(gpu::TextureId texture) override;
    bool           canRenderTo(PixelFormat format) const override;
    bool writeTexture(gpu::TextureId texture, uint32_t layer, uint32_t mip,
                      gpu::IRect region, std::span<const std::byte> data) override;
    void clearTexture(gpu::TextureId texture, gpu::ClearColor colour) override;
    bool readTexture(gpu::TextureId texture, gpu::IRect region, std::span<std::byte> dst) override;

    gpu::ProgramId  createProgram(const gpu::ProgramDesc& desc) override;
    void            destroyProgram(gpu::ProgramId program) override;
    gpu::PipelineId createPipeline(const gpu::PipelineDesc& desc) override;

    std::unique_ptr<gpu::Surface> createOffscreenSurface(gpu::TextureId target) override;

    // ---- For the surfaces (VulkanSurface.cpp) --------------------------------
    TextureSlot*  texture(gpu::TextureId t)   const { return textures_.resolve(t.value); }
    ProgramSlot*  program(uint32_t handle)    const { return programs_.resolve(handle); }
    PipelineSlot* pipeline(gpu::PipelineId p) const { return pipelines_.resolve(p.value); }

    // A swap image as a texture (a colour attachment the core draws into). The
    // image stays the swapchain's; the texture owns only its view.
    gpu::TextureId adoptImage(VkImage image, VkFormat format, PixelFormat pixelFormat, uint32_t width, uint32_t height);
    // Frees a texture NOW, no deferral: the GPU must already be done with it
    // (after a device-wait-idle). Under the submit lock.
    void destroyTextureNow(gpu::TextureId texture);

    VkFormat      depthStencilFormat() const noexcept { return depthStencilFormat_; }
    VkSampler     sampler(gpu::SamplerDesc desc);
    VkRenderPass  renderPass(const RenderPassKey& key);
    // Framebuffer over `attachments` (texture handle values; 0 for an unused
    // slot is skipped), cached until one of them is destroyed. Under the
    // submit lock.
    VkFramebuffer framebuffer(VkRenderPass compatible, std::span<const uint32_t> attachments,
                              uint32_t width, uint32_t height);
    // A 1x1 texture bound where a draw names none (or a dead one).
    TextureSlot*  dummyTexture() const { return texture(dummy_); }

    // ---- Submissions (under the submit lock) ---------------------------------
    // The serial of the submission being recorded now; everything destroyed
    // from here on waits for it.
    uint64_t beginSubmission();
    // `serial` went to the GPU fenced by `fence`. A fence passed again here is
    // reused: its previous submission has been waited on.
    void     submitted(uint64_t serial, VkFence fence);
    // The surface waited `fence` and it signalled, or is about to destroy it.
    void     fenceDone(VkFence fence);
    // Frees whatever every submission it could be in has finished with.
    void     collect();
    // Records every queued texture write and clear into `cmd`, ahead of the
    // frame's passes.
    void     recordPendingWrites(VkCommandBuffer cmd, uint64_t serial);

    // Adds the barrier that takes `t` from its tracked state to (layout, stage,
    // access); none when nothing needs ordering. `discard`: the old contents
    // don't matter (the old layout becomes UNDEFINED).
    struct BarrierBatch
    {
        std::vector<VkImageMemoryBarrier> barriers;
        VkPipelineStageFlags src = 0, dst = 0;
        void clear() { barriers.clear(); src = dst = 0; }
        void record(VkCommandBuffer cmd);
    };
    static void transition(BarrierBatch& batch, TextureSlot& t, VkImageLayout layout,
                           VkPipelineStageFlags stage, VkAccessFlags access, bool discard);

private:
    Backend();
    void initCaps();
    void flushWritesLocked(VkCommandBuffer cmd, uint64_t serial);
    uint64_t completedSerial();   // under the submit lock
    void freeTexture(uint32_t handle);   // under the submit lock + tableMutex_
    void freeProgram(uint32_t handle);   // under the submit lock + tableMutex_
    void destroyFramebuffersOf(uint32_t textureHandle);
    gpu::TextureId createTextureLocked(const gpu::TextureDesc& desc);

    std::unique_ptr<Device> device_;
    gpu::Caps caps_;
    VkFormat  depthStencilFormat_ = VK_FORMAT_D32_SFLOAT_S8_UINT;

    SlotTable<TextureSlot>  textures_;
    SlotTable<ProgramSlot>  programs_;
    SlotTable<PipelineSlot> pipelines_;
    std::mutex tableMutex_;
    std::unordered_map<gpu::PipelineDesc, uint32_t, gpu::PipelineDescHash> pipelineIds_;

    // ---- Deferred destruction --------------------------------------------------
    enum class RetiredKind : uint8_t { Texture, Program };
    struct Retired { RetiredKind kind; uint32_t handle; uint64_t serial; };
    std::vector<Retired>  retired_;
    std::vector<Retired>  retiredScratch_;
    std::atomic<uint64_t> startedSerial_ { 0 };
    struct InFlight { uint64_t serial; VkFence fence; };
    std::vector<InFlight> inFlight_;              // under the submit lock

    // ---- Texture writes ----------------------------------------------------------
    struct StagingChunk { HostBuffer buffer; VkDeviceSize head = 0; uint64_t serial = 0; };
    struct PendingWrite
    {
        uint32_t          texture = 0;
        uint32_t          layer = 0, mip = 0;
        gpu::IRect        region;
        VkBuffer          buffer = VK_NULL_HANDLE;
        VkDeviceSize      offset = 0;
        bool              clear = false;
        VkClearColorValue colour {};
    };
    static constexpr VkDeviceSize kStagingChunk = 4u << 20;
    std::mutex writeMutex_;
    std::vector<PendingWrite> pendingWrites_, writesScratch_;
    std::vector<std::unique_ptr<StagingChunk>> stagingFilling_;   // the last is being filled
    std::vector<std::unique_ptr<StagingChunk>> stagingRetired_;   // recorded, waiting on their serial
    std::vector<std::unique_ptr<StagingChunk>> stagingFree_;

    // ---- Caches ---------------------------------------------------------------------
    std::mutex renderPassMutex_;
    std::unordered_map<RenderPassKey, VkRenderPass, RenderPassKeyHash> renderPasses_;
    struct FramebufferEntry
    {
        VkRenderPass  renderPass;
        uint32_t      width, height, count;
        uint32_t      attachments[gpu::kMaxColorAttachments + 1];
        VkFramebuffer framebuffer;
    };
    std::vector<FramebufferEntry> framebuffers_;   // under the submit lock
    struct SamplerEntry { gpu::SamplerDesc desc; VkSampler sampler; };
    std::vector<SamplerEntry> samplers_;           // under tableMutex_
    VkDescriptorSetLayout emptySetLayout_ = VK_NULL_HANDLE;
    gpu::TextureId dummy_;

    // One-off submissions (readTexture): their own pool and fence.
    VkCommandPool oneOffPool_  = VK_NULL_HANDLE;
    VkFence       oneOffFence_ = VK_NULL_HANDLE;
};

} // namespace jvk::vulkan
