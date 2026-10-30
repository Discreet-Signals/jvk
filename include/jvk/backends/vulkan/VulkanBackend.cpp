/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: VulkanBackend.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

namespace jvk::vulkan
{

// =============================================================================
// Host-visible buffers
// =============================================================================

HostBuffer createHostBuffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage, bool readback)
{
    HostBuffer b;
    if (allocator == nullptr || size == 0) return b;

    VkBufferCreateInfo bci {};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = size;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // Coherent, so a memcpy is all a write (or, for readback, a read) needs.
    VmaAllocationCreateInfo aci {};
    aci.usage         = VMA_MEMORY_USAGE_AUTO;
    aci.flags         = VMA_ALLOCATION_CREATE_MAPPED_BIT
                      | (readback ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                                  : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VmaAllocationInfo info {};
    if (vmaCreateBuffer(allocator, &bci, &aci, &b.buffer, &b.allocation, &info) != VK_SUCCESS)
        return {};
    b.mapped = info.pMappedData;
    b.size   = size;
    if (b.mapped == nullptr) destroyHostBuffer(allocator, b);
    return b;
}

void destroyHostBuffer(VmaAllocator allocator, HostBuffer& b)
{
    if (b.buffer != VK_NULL_HANDLE && allocator != nullptr)
        vmaDestroyBuffer(allocator, b.buffer, b.allocation);
    b = {};
}

// =============================================================================
// Mapping the core's vocabulary to Vulkan
// =============================================================================

namespace
{

VkBlendFactor toVk(gpu::BlendFactor f)
{
    switch (f)
    {
        case gpu::BlendFactor::Zero:             return VK_BLEND_FACTOR_ZERO;
        case gpu::BlendFactor::One:              return VK_BLEND_FACTOR_ONE;
        case gpu::BlendFactor::SrcColor:         return VK_BLEND_FACTOR_SRC_COLOR;
        case gpu::BlendFactor::OneMinusSrcColor: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case gpu::BlendFactor::SrcAlpha:         return VK_BLEND_FACTOR_SRC_ALPHA;
        case gpu::BlendFactor::OneMinusSrcAlpha: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case gpu::BlendFactor::DstColor:         return VK_BLEND_FACTOR_DST_COLOR;
        case gpu::BlendFactor::OneMinusDstColor: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case gpu::BlendFactor::DstAlpha:         return VK_BLEND_FACTOR_DST_ALPHA;
        case gpu::BlendFactor::OneMinusDstAlpha: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    }
    return VK_BLEND_FACTOR_ONE;
}

VkBlendOp toVk(gpu::BlendOp op)
{
    switch (op)
    {
        case gpu::BlendOp::Add:             return VK_BLEND_OP_ADD;
        case gpu::BlendOp::Subtract:        return VK_BLEND_OP_SUBTRACT;
        case gpu::BlendOp::ReverseSubtract: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case gpu::BlendOp::Min:             return VK_BLEND_OP_MIN;
        case gpu::BlendOp::Max:             return VK_BLEND_OP_MAX;
    }
    return VK_BLEND_OP_ADD;
}

VkCompareOp toVk(gpu::CompareOp op)
{
    switch (op)
    {
        case gpu::CompareOp::Never:          return VK_COMPARE_OP_NEVER;
        case gpu::CompareOp::Less:           return VK_COMPARE_OP_LESS;
        case gpu::CompareOp::Equal:          return VK_COMPARE_OP_EQUAL;
        case gpu::CompareOp::LessOrEqual:    return VK_COMPARE_OP_LESS_OR_EQUAL;
        case gpu::CompareOp::Greater:        return VK_COMPARE_OP_GREATER;
        case gpu::CompareOp::NotEqual:       return VK_COMPARE_OP_NOT_EQUAL;
        case gpu::CompareOp::GreaterOrEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case gpu::CompareOp::Always:         return VK_COMPARE_OP_ALWAYS;
    }
    return VK_COMPARE_OP_ALWAYS;
}

VkStencilOp toVk(gpu::StencilOp op)
{
    switch (op)
    {
        case gpu::StencilOp::Keep:           return VK_STENCIL_OP_KEEP;
        case gpu::StencilOp::Zero:           return VK_STENCIL_OP_ZERO;
        case gpu::StencilOp::Replace:        return VK_STENCIL_OP_REPLACE;
        case gpu::StencilOp::IncrementClamp: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case gpu::StencilOp::DecrementClamp: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case gpu::StencilOp::Invert:         return VK_STENCIL_OP_INVERT;
        case gpu::StencilOp::IncrementWrap:  return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case gpu::StencilOp::DecrementWrap:  return VK_STENCIL_OP_DECREMENT_AND_WRAP;
    }
    return VK_STENCIL_OP_KEEP;
}

VkFormat toVk(gpu::VertexFormat f)
{
    switch (f)
    {
        case gpu::VertexFormat::Float:      return VK_FORMAT_R32_SFLOAT;
        case gpu::VertexFormat::Float2:     return VK_FORMAT_R32G32_SFLOAT;
        case gpu::VertexFormat::Float3:     return VK_FORMAT_R32G32B32_SFLOAT;
        case gpu::VertexFormat::Float4:     return VK_FORMAT_R32G32B32A32_SFLOAT;
        case gpu::VertexFormat::UInt:       return VK_FORMAT_R32_UINT;
        case gpu::VertexFormat::UByte4Norm: return VK_FORMAT_R8G8B8A8_UNORM;
    }
    return VK_FORMAT_R32G32B32A32_SFLOAT;
}

VkAttachmentLoadOp toVkLoad(uint8_t op)
{
    switch (static_cast<gpu::LoadOp>(op))
    {
        case gpu::LoadOp::Load:     return VK_ATTACHMENT_LOAD_OP_LOAD;
        case gpu::LoadOp::Clear:    return VK_ATTACHMENT_LOAD_OP_CLEAR;
        case gpu::LoadOp::DontCare: return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    }
    return VK_ATTACHMENT_LOAD_OP_LOAD;
}

VkAttachmentStoreOp toVkStore(uint8_t op)
{
    return static_cast<gpu::StoreOp>(op) == gpu::StoreOp::DontCare ? VK_ATTACHMENT_STORE_OP_DONT_CARE
                                                                   : VK_ATTACHMENT_STORE_OP_STORE;
}

VkDescriptorType toVk(gpu::BindingKind k)
{
    switch (k)
    {
        case gpu::BindingKind::Texture:                return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        case gpu::BindingKind::Sampler:                return VK_DESCRIPTOR_TYPE_SAMPLER;
        case gpu::BindingKind::CombinedTextureSampler: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        case gpu::BindingKind::StorageBuffer:          return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        case gpu::BindingKind::UniformBuffer:          return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    }
    return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
}

constexpr VkAccessFlags kWriteAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                     | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                                     | VK_ACCESS_TRANSFER_WRITE_BIT
                                     | VK_ACCESS_SHADER_WRITE_BIT
                                     | VK_ACCESS_HOST_WRITE_BIT
                                     | VK_ACCESS_MEMORY_WRITE_BIT;

VkDeviceSize alignTo(VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) / a * a; }

void logOnce(std::atomic<bool>& flag, const juce::String& line)
{
    if (!flag.exchange(true)) diag::log(line);
}

} // namespace

// =============================================================================
// Singleton
// =============================================================================

static std::weak_ptr<Backend> g_backend;
static juce::CriticalSection  g_backendLock;

std::shared_ptr<Backend> Backend::acquire()
{
    // Locked around lock() / new / assign so two first acquires can't make two.
    const juce::ScopedLock lk(g_backendLock);
    auto b = g_backend.lock();
    // A LOST backend never renders again: holders fall back, and the next
    // window starts a fresh one instead of inheriting the dead device.
    if (b && !b->isLost()) return b;
    b = std::shared_ptr<Backend>(new Backend());
    g_backend = b;
    return b;
}

Backend::Backend()
    : device_(std::make_unique<Device>())
{
    if (!usable()) return;
    initCaps();

    VkDevice d = device_->device();

    VkDescriptorSetLayoutCreateInfo empty {};
    empty.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    vkCreateDescriptorSetLayout(d, &empty, nullptr, &emptySetLayout_);

    VkCommandPoolCreateInfo pci {};
    pci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = device_->graphicsFamily();
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
                         | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    vkCreateCommandPool(d, &pci, nullptr, &oneOffPool_);

    VkFenceCreateInfo fci {};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(d, &fci, nullptr, &oneOffFence_);

    // Bound wherever a draw names no texture (or a dead one): opaque black,
    // as jvk's default pixel always was.
    gpu::TextureDesc dd;
    dd.width = dd.height = 1;
    dd.label = "jvk default";
    dummy_ = createTexture(dd);
    clearTexture(dummy_, { 0.0f, 0.0f, 0.0f, 1.0f });
}

void Backend::initCaps()
{
    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(device_->physicalDevice(), &props);
    const auto& limits = props.limits;

    caps_.shaderFormat          = gpu::ShaderFormat::SPIRV;
    caps_.maxTextureSize        = limits.maxImageDimension2D;
    caps_.maxColorAttachments   = std::min<uint32_t>(limits.maxColorAttachments, gpu::kMaxColorAttachments);
    // Vulkan has no per-sample byte budget of its own: every attachment at 16 bytes fits.
    caps_.maxColorBytesPerSample = caps_.maxColorAttachments * 16u;
    caps_.maxDrawConstantsBytes = std::min<uint32_t>(limits.maxPushConstantsSize, 128u);
    caps_.constantsAlignment    = 4;
    caps_.storageAlignment      = static_cast<uint32_t>(std::max<VkDeviceSize>(
        { limits.minStorageBufferOffsetAlignment, limits.minUniformBufferOffsetAlignment, VkDeviceSize(4) }));
    caps_.independentBlend      = device_->supportsIndependentBlend();
    caps_.textureCompressionBC  = device_->supportsTextureCompressionBC();
    initComputeCaps();

    // The 8-bit stencil clip stacks need: D32S8 (everywhere but some older
    // desktop parts), else D24S8 (missing on Apple silicon), else D16S8.
    const VkFormat candidates[] = { VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT };
    for (VkFormat f : candidates)
    {
        VkFormatProperties fp {};
        vkGetPhysicalDeviceFormatProperties(device_->physicalDevice(), f, &fp);
        if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
        {
            depthStencilFormat_ = f;
            break;
        }
    }
}

Backend::~Backend()
{
    VkDevice d = device_->device();
    if (d == VK_NULL_HANDLE) return;

    const juce::ScopedLock lk(device_->queueLock());
    device_->waitIdle();
    inFlight_.clear();

    {
        std::lock_guard<std::mutex> tl(tableMutex_);
        retired_.clear();
        std::vector<uint32_t> live;
        programs_.forEachLive([&](uint32_t h, ProgramSlot&) { live.push_back(h); });
        for (uint32_t h : live) freeProgram(h);
        live.clear();
        textures_.forEachLive([&](uint32_t h, TextureSlot&) { live.push_back(h); });
        for (uint32_t h : live) freeTexture(h);
        live.clear();
        buffers_.forEachLive([&](uint32_t h, BufferSlot&) { live.push_back(h); });
        for (uint32_t h : live) freeBuffer(h);
        live.clear();
        kernels_.forEachLive([&](uint32_t h, KernelSlot&) { live.push_back(h); });
        for (uint32_t h : live) freeKernel(h);
        for (const auto& r : retiredNative_) freeNative(r);
        retiredNative_.clear();
        for (auto& s : samplers_) vkDestroySampler(d, s.sampler, nullptr);
        samplers_.clear();
    }
    for (auto& f : framebuffers_) vkDestroyFramebuffer(d, f.framebuffer, nullptr);
    framebuffers_.clear();
    {
        std::lock_guard<std::mutex> rl(renderPassMutex_);
        for (auto& [key, rp] : renderPasses_) vkDestroyRenderPass(d, rp, nullptr);
        renderPasses_.clear();
    }
    {
        std::lock_guard<std::mutex> wl(writeMutex_);
        for (auto* list : { &stagingFilling_, &stagingRetired_, &stagingFree_ })
        {
            for (auto& c : *list) destroyHostBuffer(device_->allocator(), c->buffer);
            list->clear();
        }
        pendingWrites_.clear();
    }
    if (emptySetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(d, emptySetLayout_, nullptr);
    if (oneOffFence_ != VK_NULL_HANDLE)    vkDestroyFence(d, oneOffFence_, nullptr);
    if (oneOffPool_ != VK_NULL_HANDLE)     vkDestroyCommandPool(d, oneOffPool_, nullptr);
}

void Backend::waitIdle()
{
    if (!usable()) return;
    const juce::ScopedLock lk(device_->queueLock());
    device_->waitGraphicsIdle();   // every surface's frames (streams wait for themselves)
    inFlight_.clear();
    collect();
}

// =============================================================================
// Textures
// =============================================================================

bool Backend::canRenderTo(PixelFormat format) const
{
    if (!usable()) return false;
    const auto info = formatInfo(format);
    if (info.compressed || info.integer) return false;
    VkFormatProperties fp {};
    vkGetPhysicalDeviceFormatProperties(device_->physicalDevice(), toVkFormat(format), &fp);
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT
                                    | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
                                    | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
    return (fp.optimalTilingFeatures & need) == need;
}

gpu::TextureId Backend::createTexture(const gpu::TextureDesc& desc)
{
    if (!usable() || desc.width == 0 || desc.height == 0) return {};
    VkDevice d = device_->device();

    const bool ds = desc.kind == gpu::TextureKind::DepthStencil;
    const VkFormat format = ds ? depthStencilFormat_ : toVkFormat(desc.format);
    const uint32_t layers = std::max(1u, desc.layers), mips = std::max(1u, desc.mips);

    VkImageUsageFlags usage = 0;
    if (ds)
    {
        usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }
    else
    {
        usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;   // writes, and clearTexture, on any colour texture
        if (gpu::hasUsage(desc.usage, gpu::TextureUsage::Sampled))      usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if (gpu::hasUsage(desc.usage, gpu::TextureUsage::RenderTarget)) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (gpu::hasUsage(desc.usage, gpu::TextureUsage::Readback))     usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        // Always SAMPLED: writes, clears, readbacks and offscreen frames leave a
        // colour texture SHADER_READ_ONLY, a layout that needs it.
        usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }

    VkImageCreateInfo ici {};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = format;
    ici.extent        = { desc.width, desc.height, 1 };
    ici.mipLevels     = mips;
    ici.arrayLayers   = layers;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = usage;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci {};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    if (vmaCreateImage(device_->allocator(), &ici, &aci, &image, &allocation, nullptr) != VK_SUCCESS)
        return {};

    const VkImageAspectFlags aspect = ds ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
                                         : VK_IMAGE_ASPECT_COLOR_BIT;

    auto makeView = [&](VkImageViewType type, uint32_t mipCount, uint32_t layerCount) {
        VkImageViewCreateInfo vci {};
        vci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image            = image;
        vci.viewType         = type;
        vci.format           = format;
        vci.subresourceRange = { aspect, 0, mipCount, 0, layerCount };
        VkImageView v = VK_NULL_HANDLE;
        vkCreateImageView(d, &vci, nullptr, &v);
        return v;
    };

    VkImageView view = makeView(layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, mips, layers);
    VkImageView attachView = view;
    const bool attachable = ds || (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0;
    if (attachable && (layers > 1 || mips > 1))
        attachView = makeView(VK_IMAGE_VIEW_TYPE_2D, 1, 1);

    if (view == VK_NULL_HANDLE || attachView == VK_NULL_HANDLE)
    {
        if (attachView != VK_NULL_HANDLE && attachView != view) vkDestroyImageView(d, attachView, nullptr);
        if (view != VK_NULL_HANDLE) vkDestroyImageView(d, view, nullptr);
        vmaDestroyImage(device_->allocator(), image, allocation);
        return {};
    }

    std::lock_guard<std::mutex> lk(tableMutex_);
    TextureSlot* t = nullptr;
    const uint32_t handle = textures_.allocate(t);
    if (handle == 0)
    {
        if (attachView != view) vkDestroyImageView(d, attachView, nullptr);
        vkDestroyImageView(d, view, nullptr);
        vmaDestroyImage(device_->allocator(), image, allocation);
        return {};
    }
    t->image       = image;
    t->allocation  = allocation;
    t->view        = view;
    t->attachView  = attachView;
    t->format      = format;
    t->aspect      = aspect;
    t->usage       = usage;
    t->pixelFormat = desc.format;
    t->kind        = desc.kind;
    t->width       = desc.width;
    t->height      = desc.height;
    t->layers      = layers;
    t->mips        = mips;
    return { handle };
}

gpu::TextureId Backend::adoptImage(VkImage image, VkFormat format, PixelFormat pixelFormat, uint32_t width, uint32_t height)
{
    if (!usable() || image == VK_NULL_HANDLE) return {};
    VkImageViewCreateInfo vci {};
    vci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image            = image;
    vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    vci.format           = format;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(device_->device(), &vci, nullptr, &view) != VK_SUCCESS) return {};

    std::lock_guard<std::mutex> lk(tableMutex_);
    TextureSlot* t = nullptr;
    const uint32_t handle = textures_.allocate(t);
    if (handle == 0) { vkDestroyImageView(device_->device(), view, nullptr); return {}; }
    t->image       = image;
    t->view        = view;
    t->attachView  = view;
    t->format      = format;
    t->usage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    t->pixelFormat = pixelFormat;
    t->width       = width;
    t->height      = height;
    return { handle };
}

void Backend::destroyTexture(gpu::TextureId texture)
{
    if (!texture) return;
    std::lock_guard<std::mutex> lk(tableMutex_);
    if (textures_.resolve(texture.value) == nullptr) return;
    // Every submission started before now may name it; later ones can't
    // (the core never destroys what an unsubmitted frame names).
    retired_.push_back({ RetiredKind::Texture, texture.value, startedSerial_.load(std::memory_order_acquire) });
}

void Backend::destroyTextureNow(gpu::TextureId texture)
{
    if (!texture) return;
    std::lock_guard<std::mutex> lk(tableMutex_);
    freeTexture(texture.value);
}

void Backend::freeTexture(uint32_t handle)
{
    TextureSlot* t = textures_.resolve(handle);
    if (t == nullptr) return;
    VkDevice d = device_->device();
    destroyFramebuffersOf(handle);
    if (t->attachView != VK_NULL_HANDLE && t->attachView != t->view) vkDestroyImageView(d, t->attachView, nullptr);
    if (t->view != VK_NULL_HANDLE) vkDestroyImageView(d, t->view, nullptr);
    if (t->allocation != nullptr) vmaDestroyImage(device_->allocator(), t->image, t->allocation);
    textures_.release(handle);
}

// ---- Writes -----------------------------------------------------------------

bool Backend::writeTexture(gpu::TextureId texture, uint32_t layer, uint32_t mip,
                           gpu::IRect region, std::span<const std::byte> data)
{
    TextureSlot* t = textures_.resolve(texture.value);
    if (t == nullptr || t->kind != gpu::TextureKind::Color || t->allocation == nullptr
        || region.isEmpty() || region.x < 0 || region.y < 0 || layer >= t->layers || mip >= t->mips)
        return false;
    const uint32_t mipW = std::max(1u, t->width >> mip), mipH = std::max(1u, t->height >> mip);
    if (uint32_t(region.x + region.width) > mipW || uint32_t(region.y + region.height) > mipH)
        return false;
    const uint64_t bytes = formatInfo(t->pixelFormat).bytesFor(uint32_t(region.width), uint32_t(region.height));
    if (bytes == 0 || data.size() < bytes) return false;

    std::lock_guard<std::mutex> lk(writeMutex_);
    StagingChunk* c = stagingFilling_.empty() ? nullptr : stagingFilling_.back().get();
    VkDeviceSize offset = c != nullptr ? alignTo(c->head, 16) : 0;
    if (c == nullptr || offset + bytes > c->buffer.size)
    {
        std::unique_ptr<StagingChunk> fresh;
        if (bytes <= kStagingChunk && !stagingFree_.empty())
        {
            fresh = std::move(stagingFree_.back());
            stagingFree_.pop_back();
        }
        else
        {
            fresh = std::make_unique<StagingChunk>();
            fresh->buffer = createHostBuffer(device_->allocator(), std::max<VkDeviceSize>(kStagingChunk, bytes),
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            if (!fresh->buffer) return false;
        }
        fresh->head = 0;
        c = fresh.get();
        stagingFilling_.push_back(std::move(fresh));
        offset = 0;
    }
    std::memcpy(static_cast<std::byte*>(c->buffer.mapped) + offset, data.data(), static_cast<size_t>(bytes));
    c->head = offset + bytes;

    PendingWrite w;
    w.texture = texture.value;
    w.layer   = layer;
    w.mip     = mip;
    w.region  = region;
    w.buffer  = c->buffer.buffer;
    w.offset  = offset;
    pendingWrites_.push_back(w);
    return true;
}

void Backend::clearTexture(gpu::TextureId texture, gpu::ClearColor colour)
{
    TextureSlot* t = textures_.resolve(texture.value);
    if (t == nullptr || t->kind != gpu::TextureKind::Color) return;
    PendingWrite w;
    w.texture = texture.value;
    w.clear   = true;
    w.colour.float32[0] = colour.r;
    w.colour.float32[1] = colour.g;
    w.colour.float32[2] = colour.b;
    w.colour.float32[3] = colour.a;
    std::lock_guard<std::mutex> lk(writeMutex_);
    pendingWrites_.push_back(w);
}

void Backend::recordPendingWrites(VkCommandBuffer cmd, uint64_t serial)
{
    flushWritesLocked(cmd, serial);
}

void Backend::flushWritesLocked(VkCommandBuffer cmd, uint64_t serial)
{
    {
        std::lock_guard<std::mutex> lk(writeMutex_);
        writesScratch_.clear();
        writesScratch_.swap(pendingWrites_);
        // Every chunk written so far is read by THIS submission.
        for (auto& c : stagingFilling_)
        {
            c->serial = serial;
            stagingRetired_.push_back(std::move(c));
        }
        stagingFilling_.clear();
    }

    for (const auto& w : writesScratch_)
    {
        TextureSlot* t = textures_.resolve(w.texture);   // destroyed meanwhile: dropped
        if (t == nullptr || t->kind != gpu::TextureKind::Color) continue;

        // A texture's first write (or a clear) takes every subresource out of
        // UNDEFINED, so the whole texture is SHADER_READ_ONLY afterwards; a
        // later write moves only its own subresource and keeps the rest.
        const bool whole = w.clear || t->state.layout == VK_IMAGE_LAYOUT_UNDEFINED;
        const VkImageSubresourceRange range = whole
            ? VkImageSubresourceRange { t->aspect, 0, t->mips, 0, t->layers }
            : VkImageSubresourceRange { t->aspect, w.mip, 1, w.layer, 1 };

        VkImageMemoryBarrier b {};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = t->image;
        b.subresourceRange    = range;
        b.oldLayout           = whole ? VK_IMAGE_LAYOUT_UNDEFINED : t->state.layout;
        b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcAccessMask       = t->state.writes;
        b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        // Earlier frames' reads (and writes) of it finish first: write-after-read.
        const VkPipelineStageFlags src = t->state.stage != 0 ? t->state.stage : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        vkCmdPipelineBarrier(cmd, src, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);

        if (w.clear)
        {
            vkCmdClearColorImage(cmd, t->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &w.colour, 1, &range);
        }
        else
        {
            VkBufferImageCopy copy {};
            copy.bufferOffset     = w.offset;   // rows tightly packed
            copy.imageSubresource = { t->aspect, w.mip, w.layer, 1 };
            copy.imageOffset      = { w.region.x, w.region.y, 0 };
            copy.imageExtent      = { uint32_t(w.region.width), uint32_t(w.region.height), 1 };
            vkCmdCopyBufferToImage(cmd, w.buffer, t->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        }

        b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, kShaderStages, 0, 0, nullptr, 0, nullptr, 1, &b);

        t->state = { VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, kShaderStages, 0 };
    }
    writesScratch_.clear();
}

// ---- Readback -----------------------------------------------------------------

bool Backend::readTexture(gpu::TextureId texture, gpu::IRect region, std::span<std::byte> dst)
{
    if (!usable() || isLost()) return false;
    TextureSlot* t = textures_.resolve(texture.value);
    if (t == nullptr || t->kind != gpu::TextureKind::Color || (t->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0
        || region.isEmpty() || region.x < 0 || region.y < 0
        || uint32_t(region.x + region.width) > t->width || uint32_t(region.y + region.height) > t->height)
        return false;
    const uint64_t bytes = formatInfo(t->pixelFormat).bytesFor(uint32_t(region.width), uint32_t(region.height));
    if (dst.size() < bytes) return false;

    VkDevice d = device_->device();
    HostBuffer buffer = createHostBuffer(device_->allocator(), bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    if (!buffer) return false;

    bool ok = false;
    {
        const juce::ScopedLock lk(device_->queueLock());
        collect();
        const uint64_t serial = beginSubmission();

        VkCommandBufferAllocateInfo ai {};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = oneOffPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(d, &ai, &cmd) == VK_SUCCESS)
        {
            VkCommandBufferBeginInfo bi {};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &bi);

            flushWritesLocked(cmd, serial);

            BarrierBatch batch;
            transition(batch, *t, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT, false);
            batch.record(cmd);

            VkBufferImageCopy copy {};
            copy.imageSubresource = { t->aspect, 0, 0, 1 };
            copy.imageOffset      = { region.x, region.y, 0 };
            copy.imageExtent      = { uint32_t(region.width), uint32_t(region.height), 1 };
            vkCmdCopyImageToBuffer(cmd, t->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer.buffer, 1, &copy);

            VkBufferMemoryBarrier hb {};
            hb.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            hb.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
            hb.dstAccessMask       = VK_ACCESS_HOST_READ_BIT;
            hb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            hb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            hb.buffer              = buffer.buffer;
            hb.size                = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 0, nullptr, 1, &hb, 0, nullptr);

            // Back to where every sampled texture lives between frames.
            transition(batch, *t, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, kShaderStages,
                       VK_ACCESS_SHADER_READ_BIT, false);
            batch.record(cmd);
            vkEndCommandBuffer(cmd);

            vkResetFences(d, 1, &oneOffFence_);
            VkSubmitInfo si {};
            si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers    = &cmd;
            if (vkQueueSubmit(device_->graphicsQueue(), 1, &si, oneOffFence_) == VK_SUCCESS)
            {
                const VkResult r = vkWaitForFences(d, 1, &oneOffFence_, VK_TRUE, 5'000'000'000ull);
                if (r == VK_SUCCESS) ok = true;
                else
                {
                    // The copy may still be running: never free its command
                    // buffer or the buffer it writes before the GPU is done.
                    if (r == VK_ERROR_DEVICE_LOST || r == VK_TIMEOUT) device_->markLost();
                    device_->waitGraphicsIdle();
                }
            }
            else
            {
                device_->markLost();
            }
            vkFreeCommandBuffers(d, oneOffPool_, 1, &cmd);
        }
    }

    if (ok) std::memcpy(dst.data(), buffer.mapped, static_cast<size_t>(bytes));
    destroyHostBuffer(device_->allocator(), buffer);
    return ok;
}

// =============================================================================
// Programs and pipelines
// =============================================================================

gpu::ProgramId Backend::createProgram(const gpu::ProgramDesc& desc)
{
    if (!usable() || desc.vertex.empty() || desc.fragment.empty()
        || desc.vertex.size() % 4 != 0 || desc.fragment.size() % 4 != 0)
        return {};
    VkDevice d = device_->device();

    auto makeModule = [d](std::span<const std::byte> bytes) {
        // Copied: a std::byte span promises no uint32_t alignment.
        std::vector<uint32_t> words(bytes.size() / 4);
        std::memcpy(words.data(), bytes.data(), bytes.size());
        VkShaderModuleCreateInfo ci {};
        ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = bytes.size();
        ci.pCode    = words.data();
        VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(d, &ci, nullptr, &m);
        return m;
    };

    ProgramSlot p;
    p.vertex   = makeModule(desc.vertex);
    p.fragment = makeModule(desc.fragment);
    p.vertexEntry   = desc.vertexEntry   != nullptr ? desc.vertexEntry   : "main";
    p.fragmentEntry = desc.fragmentEntry != nullptr ? desc.fragmentEntry : "main";

    auto fail = [&]() -> gpu::ProgramId {
        if (p.vertex != VK_NULL_HANDLE)   vkDestroyShaderModule(d, p.vertex, nullptr);
        if (p.fragment != VK_NULL_HANDLE) vkDestroyShaderModule(d, p.fragment, nullptr);
        for (size_t i = 0; i < p.setLayouts.size(); ++i)
            if (p.ownsSetLayout[i]) vkDestroyDescriptorSetLayout(d, p.setLayouts[i], nullptr);
        if (p.layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(d, p.layout, nullptr);
        diag::log(juce::String("vulkan: program ") + (desc.name != nullptr ? desc.name : "?") + " could not be created");
        return {};
    };
    if (p.vertex == VK_NULL_HANDLE || p.fragment == VK_NULL_HANDLE) return fail();

    // The portable convention spelled out, or the program's own bindings.
    if (desc.layout.bindings.empty())
    {
        for (uint8_t i = 0; i < desc.layout.textures && i < gpu::kPortableDrawTextures; ++i)
        {
            p.bindings.push_back({ gpu::BindingKind::Texture, i, 0, i });
            p.bindings.push_back({ gpu::BindingKind::Sampler, i, 0, uint8_t(8 + i) });
        }
        for (uint8_t j = 0; j < desc.layout.storageBuffers && j < gpu::kMaxDrawStorage; ++j)
            p.bindings.push_back({ gpu::BindingKind::StorageBuffer, j, 0, uint8_t(16 + j) });
    }
    else
    {
        p.bindings.assign(desc.layout.bindings.begin(), desc.layout.bindings.end());
    }

    uint32_t setCount = 0;
    for (const auto& b : p.bindings) setCount = std::max<uint32_t>(setCount, uint32_t(b.set) + 1);
    std::vector<VkDescriptorSetLayoutBinding> lb;
    for (uint32_t s = 0; s < setCount; ++s)
    {
        lb.clear();
        for (const auto& b : p.bindings)
        {
            if (b.set != s) continue;
            VkDescriptorSetLayoutBinding x {};
            x.binding         = b.binding;
            x.descriptorType  = toVk(b.kind);
            x.descriptorCount = 1;
            x.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            lb.push_back(x);
        }
        if (lb.empty())
        {
            p.setLayouts.push_back(emptySetLayout_);
            p.ownsSetLayout.push_back(0);
            continue;
        }
        VkDescriptorSetLayoutCreateInfo ci {};
        ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = uint32_t(lb.size());
        ci.pBindings    = lb.data();
        VkDescriptorSetLayout l = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(d, &ci, nullptr, &l) != VK_SUCCESS) return fail();
        p.setLayouts.push_back(l);
        p.ownsSetLayout.push_back(1);
    }

    VkPhysicalDeviceProperties props {};
    vkGetPhysicalDeviceProperties(device_->physicalDevice(), &props);
    p.constantsBytes = std::min<uint32_t>(uint32_t(desc.layout.constantsBytes), props.limits.maxPushConstantsSize) & ~3u;

    VkPushConstantRange range {};
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    range.size       = p.constantsBytes;

    VkPipelineLayoutCreateInfo pli {};
    pli.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount         = uint32_t(p.setLayouts.size());
    pli.pSetLayouts            = p.setLayouts.empty() ? nullptr : p.setLayouts.data();
    pli.pushConstantRangeCount = p.constantsBytes > 0 ? 1u : 0u;
    pli.pPushConstantRanges    = p.constantsBytes > 0 ? &range : nullptr;
    if (vkCreatePipelineLayout(d, &pli, nullptr, &p.layout) != VK_SUCCESS) return fail();

    std::lock_guard<std::mutex> lk(tableMutex_);
    ProgramSlot* slot = nullptr;
    const uint32_t handle = programs_.allocate(slot);
    if (handle == 0) return fail();
    *slot = std::move(p);
    return { handle };
}

void Backend::destroyProgram(gpu::ProgramId program)
{
    if (!program) return;
    std::lock_guard<std::mutex> lk(tableMutex_);
    if (programs_.resolve(program.value) == nullptr) return;
    retired_.push_back({ RetiredKind::Program, program.value, startedSerial_.load(std::memory_order_acquire) });
}

void Backend::freeProgram(uint32_t handle)
{
    ProgramSlot* p = programs_.resolve(handle);
    if (p == nullptr) return;
    VkDevice d = device_->device();
    for (uint32_t ph : p->pipelines)
    {
        if (PipelineSlot* ps = pipelines_.resolve(ph))
        {
            if (ps->pipeline != VK_NULL_HANDLE) vkDestroyPipeline(d, ps->pipeline, nullptr);
            pipelineIds_.erase(ps->desc);
            pipelines_.release(ph);
        }
    }
    if (p->layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(d, p->layout, nullptr);
    for (size_t i = 0; i < p->setLayouts.size(); ++i)
        if (p->ownsSetLayout[i]) vkDestroyDescriptorSetLayout(d, p->setLayouts[i], nullptr);
    if (p->vertex != VK_NULL_HANDLE)   vkDestroyShaderModule(d, p->vertex, nullptr);
    if (p->fragment != VK_NULL_HANDLE) vkDestroyShaderModule(d, p->fragment, nullptr);
    programs_.release(handle);
}

gpu::PipelineId Backend::createPipeline(const gpu::PipelineDesc& desc)
{
    if (!usable()) return {};
    {
        std::lock_guard<std::mutex> lk(tableMutex_);
        if (auto it = pipelineIds_.find(desc); it != pipelineIds_.end()) return { it->second };
    }
    ProgramSlot* prog = programs_.resolve(desc.program.value);
    if (prog == nullptr) return {};
    VkDevice d = device_->device();
    const auto& att = desc.attachments;

    // Built against a render pass of the same formats: any pass whose
    // attachments match is compatible with it, whatever its loads and stores.
    RenderPassKey key;
    key.colorCount = std::min<uint8_t>(att.colorCount, uint8_t(gpu::kMaxColorAttachments));
    for (uint32_t i = 0; i < key.colorCount; ++i)
        key.color[i] = (att.unusedMask & (1u << i)) != 0 ? VK_FORMAT_UNDEFINED : toVkFormat(att.colorFormats[i]);
    key.depthStencil = att.depthStencil ? depthStencilFormat_ : VK_FORMAT_UNDEFINED;
    VkRenderPass rp = renderPass(key);

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (rp != VK_NULL_HANDLE)
    {
        VkPipelineShaderStageCreateInfo stages[2] {};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = prog->vertex;
        stages[0].pName  = prog->vertexEntry.c_str();
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = prog->fragment;
        stages[1].pName  = prog->fragmentEntry.c_str();

        const auto& vl = desc.vertexLayout;
        VkVertexInputBindingDescription binding {};
        binding.binding   = 0;
        binding.stride    = vl.stride;
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        VkVertexInputAttributeDescription attrs[gpu::kMaxVertexAttributes] {};
        const uint32_t attrCount = vl.stride > 0 ? std::min<uint32_t>(vl.attributeCount, gpu::kMaxVertexAttributes) : 0;
        for (uint32_t i = 0; i < attrCount; ++i)
            attrs[i] = { vl.attributes[i].location, 0, toVk(vl.attributes[i].format), vl.attributes[i].offset };
        VkPipelineVertexInputStateCreateInfo vi {};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        if (vl.stride > 0)
        {
            vi.vertexBindingDescriptionCount   = 1;
            vi.pVertexBindingDescriptions      = &binding;
            vi.vertexAttributeDescriptionCount = attrCount;
            vi.pVertexAttributeDescriptions    = attrs;
        }

        VkPipelineInputAssemblyStateCreateInfo ia {};
        ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = desc.topology == gpu::Topology::TriangleStrip ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP
                                                                    : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp {};
        vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo raster {};
        raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode    = VK_CULL_MODE_NONE;
        raster.frontFace   = VK_FRONT_FACE_CLOCKWISE;
        raster.lineWidth   = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms {};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo dss {};
        dss.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        dss.depthCompareOp = VK_COMPARE_OP_ALWAYS;
        if (att.depthStencil && desc.stencil.enabled)
        {
            VkStencilOpState so {};
            so.failOp      = toVk(desc.stencil.failOp);
            so.passOp      = toVk(desc.stencil.passOp);
            so.depthFailOp = VK_STENCIL_OP_KEEP;
            so.compareOp   = toVk(desc.stencil.compare);
            so.compareMask = desc.stencil.readMask;
            so.writeMask   = desc.stencil.writeMask;
            dss.stencilTestEnable = VK_TRUE;
            dss.front = so;
            dss.back  = so;
        }

        VkPipelineColorBlendAttachmentState blends[gpu::kMaxColorAttachments] {};
        for (uint32_t i = 0; i < key.colorCount; ++i)
        {
            const auto& b = desc.blend[i];
            auto& v = blends[i];
            v.blendEnable         = b.enabled ? VK_TRUE : VK_FALSE;
            v.srcColorBlendFactor = toVk(b.srcColor);
            v.dstColorBlendFactor = toVk(b.dstColor);
            v.colorBlendOp        = toVk(b.colorOp);
            v.srcAlphaBlendFactor = toVk(b.srcAlpha);
            v.dstAlphaBlendFactor = toVk(b.dstAlpha);
            v.alphaBlendOp        = toVk(b.alphaOp);
            v.colorWriteMask      = (b.writeMask & gpu::WriteR ? VK_COLOR_COMPONENT_R_BIT : 0)
                                  | (b.writeMask & gpu::WriteG ? VK_COLOR_COMPONENT_G_BIT : 0)
                                  | (b.writeMask & gpu::WriteB ? VK_COLOR_COMPONENT_B_BIT : 0)
                                  | (b.writeMask & gpu::WriteA ? VK_COLOR_COMPONENT_A_BIT : 0);
        }
        VkPipelineColorBlendStateCreateInfo cb {};
        cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = key.colorCount;
        cb.pAttachments    = blends;

        const VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                       VK_DYNAMIC_STATE_STENCIL_REFERENCE };
        VkPipelineDynamicStateCreateInfo ds {};
        ds.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        ds.dynamicStateCount = 3;
        ds.pDynamicStates    = dyn;

        VkGraphicsPipelineCreateInfo pci {};
        pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pci.stageCount          = 2;
        pci.pStages             = stages;
        pci.pVertexInputState   = &vi;
        pci.pInputAssemblyState = &ia;
        pci.pViewportState      = &vp;
        pci.pRasterizationState = &raster;
        pci.pMultisampleState   = &ms;
        pci.pDepthStencilState  = &dss;
        pci.pColorBlendState    = &cb;
        pci.pDynamicState       = &ds;
        pci.layout              = prog->layout;
        pci.renderPass          = rp;
        if (vkCreateGraphicsPipelines(d, device_->pipelineCache(), 1, &pci, nullptr, &pipeline) != VK_SUCCESS)
            pipeline = VK_NULL_HANDLE;
    }
    if (pipeline == VK_NULL_HANDLE)
    {
        // Remembered as null below: never retried, its draws skipped.
        static std::atomic<bool> logged { false };
        logOnce(logged, "vulkan: a pipeline could not be built; its draws are skipped");
    }

    std::lock_guard<std::mutex> lk(tableMutex_);
    if (auto it = pipelineIds_.find(desc); it != pipelineIds_.end())
    {
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(d, pipeline, nullptr);
        return { it->second };
    }
    prog = programs_.resolve(desc.program.value);   // destroyed meanwhile?
    PipelineSlot* slot = nullptr;
    const uint32_t handle = prog != nullptr ? pipelines_.allocate(slot) : 0;
    if (handle == 0)
    {
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(d, pipeline, nullptr);
        return {};
    }
    slot->desc     = desc;
    slot->pipeline = pipeline;
    slot->program  = desc.program.value;
    prog->pipelines.push_back(handle);
    pipelineIds_.emplace(desc, handle);
    return { handle };
}

// =============================================================================
// Caches
// =============================================================================

VkSampler Backend::sampler(gpu::SamplerDesc desc)
{
    std::lock_guard<std::mutex> lk(tableMutex_);
    for (const auto& s : samplers_)
        if (s.desc == desc) return s.sampler;

    VkSamplerAddressMode address = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (desc.address == gpu::AddressMode::Repeat)         address = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    if (desc.address == gpu::AddressMode::MirroredRepeat) address = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    const bool linear = desc.filter == gpu::Filter::Linear;

    VkSamplerCreateInfo ci {};
    ci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.magFilter    = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.minFilter    = ci.magFilter;
    ci.mipmapMode   = linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = address;
    ci.addressModeV = address;
    ci.addressModeW = address;
    ci.maxLod       = VK_LOD_CLAMP_NONE;
    ci.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    VkSampler s = VK_NULL_HANDLE;
    if (vkCreateSampler(device_->device(), &ci, nullptr, &s) != VK_SUCCESS) return VK_NULL_HANDLE;
    samplers_.push_back({ desc, s });
    return s;
}

VkRenderPass Backend::renderPass(const RenderPassKey& key)
{
    std::lock_guard<std::mutex> lk(renderPassMutex_);
    if (auto it = renderPasses_.find(key); it != renderPasses_.end()) return it->second;

    // Initial and final layouts are the attachment layouts, and there are no
    // subpass dependencies: the executor's barriers before each pass order
    // it after every earlier use, and the barrier before each later use
    // orders that after this pass (VulkanSurface.cpp).
    VkAttachmentDescription atts[gpu::kMaxColorAttachments + 1] {};
    VkAttachmentReference   colorRefs[gpu::kMaxColorAttachments] {};
    VkAttachmentReference   dsRef {};
    uint32_t count = 0;
    for (uint32_t i = 0; i < key.colorCount; ++i)
    {
        if (key.color[i] == VK_FORMAT_UNDEFINED)
        {
            colorRefs[i] = { VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED };
            continue;
        }
        auto& a = atts[count];
        a.format         = key.color[i];
        a.samples        = VK_SAMPLE_COUNT_1_BIT;
        a.loadOp         = toVkLoad(key.colorLoad[i]);
        a.storeOp        = toVkStore(key.colorStore[i]);
        a.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        a.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorRefs[i] = { count, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        ++count;
    }
    const bool hasDs = key.depthStencil != VK_FORMAT_UNDEFINED;
    if (hasDs)
    {
        auto& a = atts[count];
        a.format         = key.depthStencil;
        a.samples        = VK_SAMPLE_COUNT_1_BIT;
        a.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;   // depth: never tested
        a.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.stencilLoadOp  = toVkLoad(key.dsLoad);
        a.stencilStoreOp = toVkStore(key.dsStore);
        a.initialLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        a.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        dsRef = { count, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
        ++count;
    }

    VkSubpassDescription sub {};
    sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount    = key.colorCount;
    sub.pColorAttachments       = key.colorCount > 0 ? colorRefs : nullptr;
    sub.pDepthStencilAttachment = hasDs ? &dsRef : nullptr;

    VkRenderPassCreateInfo ci {};
    ci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.attachmentCount = count;
    ci.pAttachments    = count > 0 ? atts : nullptr;
    ci.subpassCount    = 1;
    ci.pSubpasses      = &sub;

    VkRenderPass rp = VK_NULL_HANDLE;
    if (vkCreateRenderPass(device_->device(), &ci, nullptr, &rp) != VK_SUCCESS) return VK_NULL_HANDLE;
    renderPasses_.emplace(key, rp);
    return rp;
}

VkFramebuffer Backend::framebuffer(VkRenderPass compatible, std::span<const uint32_t> attachments,
                                   uint32_t width, uint32_t height)
{
    const uint32_t count = static_cast<uint32_t>(std::min<size_t>(attachments.size(), gpu::kMaxColorAttachments + 1));
    for (const auto& f : framebuffers_)
    {
        if (f.renderPass != compatible || f.width != width || f.height != height || f.count != count) continue;
        bool same = true;
        for (uint32_t i = 0; i < count && same; ++i) same = f.attachments[i] == attachments[i];
        if (same) return f.framebuffer;
    }

    VkImageView views[gpu::kMaxColorAttachments + 1] {};
    for (uint32_t i = 0; i < count; ++i)
    {
        TextureSlot* t = textures_.resolve(attachments[i]);
        if (t == nullptr) return VK_NULL_HANDLE;
        views[i] = t->attachView;
    }
    VkFramebufferCreateInfo ci {};
    ci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    ci.renderPass      = compatible;
    ci.attachmentCount = count;
    ci.pAttachments    = views;
    ci.width           = width;
    ci.height          = height;
    ci.layers          = 1;
    VkFramebuffer fb = VK_NULL_HANDLE;
    if (vkCreateFramebuffer(device_->device(), &ci, nullptr, &fb) != VK_SUCCESS) return VK_NULL_HANDLE;

    FramebufferEntry e {};
    e.renderPass  = compatible;
    e.width       = width;
    e.height      = height;
    e.count       = count;
    for (uint32_t i = 0; i < count; ++i) e.attachments[i] = attachments[i];
    e.framebuffer = fb;
    framebuffers_.push_back(e);
    return fb;
}

void Backend::destroyFramebuffersOf(uint32_t textureHandle)
{
    VkDevice d = device_->device();
    for (size_t i = 0; i < framebuffers_.size();)
    {
        const auto& f = framebuffers_[i];
        bool uses = false;
        for (uint32_t k = 0; k < f.count && !uses; ++k) uses = f.attachments[k] == textureHandle;
        if (uses)
        {
            vkDestroyFramebuffer(d, f.framebuffer, nullptr);
            framebuffers_[i] = framebuffers_.back();
            framebuffers_.pop_back();
        }
        else
        {
            ++i;
        }
    }
}

// =============================================================================
// Submissions and deferred destruction
// =============================================================================

uint64_t Backend::beginSubmission()
{
    return startedSerial_.fetch_add(1, std::memory_order_acq_rel) + 1;
}

void Backend::submitted(uint64_t serial, VkFence fence)
{
    // A fence passed again was waited on before its reset: whatever it fenced is done.
    fenceDone(fence);
    inFlight_.push_back({ serial, fence });
}

void Backend::fenceDone(VkFence fence)
{
    for (size_t i = 0; i < inFlight_.size();)
    {
        if (inFlight_[i].fence == fence) { inFlight_[i] = inFlight_.back(); inFlight_.pop_back(); }
        else ++i;
    }
}

Backend::Completed Backend::completedSerials()
{
    VkDevice d = device_->device();
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < inFlight_.size();)
    {
        if (vkGetFenceStatus(d, inFlight_[i].fence) == VK_SUCCESS)
        {
            inFlight_[i] = inFlight_.back();
            inFlight_.pop_back();
            continue;
        }
        oldest = std::min(oldest, inFlight_[i].serial);
        ++i;
    }
    // A compute stream's submission is listed from the moment its serial is
    // taken until its fence has signalled or the stream ends it, under
    // computeMutex_; the started serial is read there too, so none begun is
    // missed. A fence is polled only once submitted (vkQueueSubmit takes it
    // externally synchronized) and the stream ends it before resetting it.
    uint64_t started = 0, oldestCompute = std::numeric_limits<uint64_t>::max();
    {
        std::lock_guard<std::mutex> lk(computeMutex_);
        for (size_t i = 0; i < computeActive_.size();)
        {
            const auto& c = computeActive_[i];
            if (c.submitted && vkGetFenceStatus(d, c.fence) == VK_SUCCESS)
            {
                computeActive_[i] = computeActive_.back();
                computeActive_.pop_back();
                continue;
            }
            oldestCompute = std::min(oldestCompute, c.serial);
            ++i;
        }
        started = startedSerial_.load(std::memory_order_acquire);
    }
    // Nothing in flight: every serial handed out has finished (collect() runs
    // before a new frame is begun, so none is being recorded).
    Completed c;
    c.frames = oldest == std::numeric_limits<uint64_t>::max() ? started : oldest - 1;
    c.all    = oldestCompute == std::numeric_limits<uint64_t>::max() ? c.frames : std::min(c.frames, oldestCompute - 1);
    return c;
}

void Backend::collect()
{
    if (!usable()) return;
    const auto completed = completedSerials();
    const uint64_t done = completed.frames;   // what frames alone use
    {
        std::lock_guard<std::mutex> lk(tableMutex_);
        for (size_t i = 0; i < retiredNative_.size();)
        {
            if (retiredNative_[i].serial > completed.all) { ++i; continue; }
            freeNative(retiredNative_[i]);
            retiredNative_[i] = retiredNative_.back();
            retiredNative_.pop_back();
        }
        if (!retired_.empty())
        {
            retiredScratch_.clear();
            for (const auto& r : retired_)
            {
                if (r.serial > done) { retiredScratch_.push_back(r); continue; }
                if (r.kind == RetiredKind::Texture) freeTexture(r.handle);
                else                                freeProgram(r.handle);
            }
            retired_.swap(retiredScratch_);
        }
    }
    std::lock_guard<std::mutex> wl(writeMutex_);
    for (size_t i = 0; i < stagingRetired_.size();)
    {
        auto& c = stagingRetired_[i];
        if (c->serial > done) { ++i; continue; }
        if (c->buffer.size == kStagingChunk && stagingFree_.size() < 4)
        {
            c->head = 0;
            stagingFree_.push_back(std::move(c));
        }
        else
        {
            destroyHostBuffer(device_->allocator(), c->buffer);
        }
        stagingRetired_[i] = std::move(stagingRetired_.back());
        stagingRetired_.pop_back();
    }
}

// =============================================================================
// Barriers
// =============================================================================

void Backend::BarrierBatch::record(VkCommandBuffer cmd)
{
    if (!barriers.empty())
        vkCmdPipelineBarrier(cmd, src != 0 ? src : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             dst != 0 ? dst : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr,
                             static_cast<uint32_t>(barriers.size()), barriers.data());
    clear();
}

void Backend::transition(BarrierBatch& batch, TextureSlot& t, VkImageLayout layout,
                         VkPipelineStageFlags stage, VkAccessFlags access, bool discard)
{
    const bool layoutChange = t.state.layout != layout || (discard && t.state.layout != VK_IMAGE_LAYOUT_UNDEFINED);
    const bool pendingWrites = t.state.writes != 0;
    // Write-after-read in the same layout still needs the reads finished.
    const bool writeAfterRead = (access & kWriteAccess) != 0 && t.state.stage != 0;
    if (!layoutChange && !pendingWrites && !writeAfterRead)
    {
        t.state.stage |= stage;
        return;
    }

    VkImageMemoryBarrier b {};
    b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask       = t.state.writes;
    b.dstAccessMask       = access;
    b.oldLayout           = discard ? VK_IMAGE_LAYOUT_UNDEFINED : t.state.layout;
    b.newLayout           = layout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image               = t.image;
    b.subresourceRange    = { t.aspect, 0, t.mips, 0, t.layers };
    batch.barriers.push_back(b);
    batch.src |= t.state.stage != 0 ? t.state.stage : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    batch.dst |= stage;

    t.state.layout = layout;
    t.state.stage  = stage;
    t.state.writes = access & kWriteAccess;
}

// =============================================================================
// Surfaces
// =============================================================================

std::unique_ptr<gpu::Surface> Backend::createOffscreenSurface(gpu::TextureId target)
{
    TextureSlot* t = textures_.resolve(target.value);
    if (!usable() || t == nullptr || t->kind != gpu::TextureKind::Color
        || (t->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0)
        return nullptr;
    auto s = std::make_unique<OffscreenSurface>(shared_from_this(), target);
    if (!s->valid()) return nullptr;
    return s;
}

} // namespace jvk::vulkan
