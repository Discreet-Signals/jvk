/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Context.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

namespace jvk
{

// =============================================================================
// Context
// =============================================================================

std::shared_ptr<Context> Context::forBackend(const std::shared_ptr<gpu::Backend>& backend)
{
    // One per backend, alive while anything holds it. Message thread (editor
    // and offscreen construction), but locked: cheap, and an offscreen
    // renderer made elsewhere stays safe.
    static std::mutex lock;
    static std::vector<std::pair<gpu::Backend*, std::weak_ptr<Context>>> contexts;
    if (backend == nullptr) return nullptr;

    const std::lock_guard<std::mutex> lk(lock);
    std::erase_if(contexts, [](const auto& c) { return c.second.expired(); });
    for (auto& [b, c] : contexts)
        if (b == backend.get())
            if (auto ctx = c.lock()) return ctx;
    auto ctx = std::make_shared<Context>(backend);
    contexts.emplace_back(backend.get(), ctx);
    return ctx;
}

Context::Context(std::shared_ptr<gpu::Backend> backend)
    : backend_(std::move(backend))
{
    using namespace shaders::bundle;
    programs_.ui2d      = createProgram(ui2d);
    programs_.noise     = createProgram(noise);
    programs_.path      = createProgram(path);
    programs_.clip      = createProgram(clip);
    programs_.clipMask  = createProgram(clipMask);
    programs_.blur      = createProgram(blur);
    programs_.copy      = createProgram(copy);
    programs_.hsv       = createProgram(hsv);
    programs_.shapeBlur = createProgram(shapeBlur);
    programs_.pathBlur  = createProgram(pathBlur);

    // The default pixel: opaque black, as every unbound slot has always read.
    const uint32_t black = 0xFF000000u;
    gpu::TextureDesc d;
    d.width = d.height = 1;
    d.label = "jvk default texture";
    defaultTexture_ = backend_->createTexture(d);
    d.layers = 2;
    d.label = "jvk default array texture";
    defaultArrayTexture_ = backend_->createTexture(d);
    const auto pixel = std::as_bytes(std::span<const uint32_t>(&black, 1));
    if (defaultTexture_)
        backend_->writeTexture(defaultTexture_, 0, 0, { 0, 0, 1, 1 }, pixel);
    if (defaultArrayTexture_)
        for (uint32_t layer = 0; layer < 2; ++layer)
            backend_->writeTexture(defaultArrayTexture_, layer, 0, { 0, 0, 1, 1 }, pixel);

    images_ = std::make_unique<ImageCache>(*this);
}

Context::~Context()
{
    // Nothing renders on this backend any more (every Renderer holds the
    // Context): the cache's textures and the programs can go at once.
    images_.reset();
    {
        const std::lock_guard<std::mutex> lk(lock_);
        for (auto& r : retired_)
        {
            if (r.program) backend_->destroyProgram(gpu::ProgramId { r.handle });
            else           backend_->destroyTexture(gpu::TextureId { r.handle });
        }
        retired_.clear();
    }
    for (auto id : { programs_.ui2d, programs_.noise, programs_.path, programs_.clip, programs_.clipMask,
                     programs_.blur, programs_.copy, programs_.hsv, programs_.shapeBlur, programs_.pathBlur })
        if (id) backend_->destroyProgram(id);
    if (defaultTexture_)      backend_->destroyTexture(defaultTexture_);
    if (defaultArrayTexture_) backend_->destroyTexture(defaultArrayTexture_);
}

gpu::ProgramId Context::createProgram(const shaders::Program& p)
{
    gpu::ProgramDesc desc;
    desc.name = p.name;
    if (!shaders::setStages(desc, caps().shaderFormat, p.vertex, p.fragment)) return {};
    desc.layout.textures       = p.textures;
    desc.layout.storageBuffers = p.storageBuffers;
    desc.layout.constantsBytes = p.constantsBytes;
    desc.layout.colorOutputs   = p.colorOutputs;
    return backend_->createProgram(desc);
}

gpu::PipelineId Context::pipeline(const gpu::PipelineDesc& desc)
{
    if (!desc.program) return {};
    const std::lock_guard<std::mutex> lk(pipelineLock_);
    if (auto it = pipelines_.find(desc); it != pipelines_.end())
        return it->second;
    const auto id = backend_->createPipeline(desc);
    pipelines_.emplace(desc, id);   // a failure (null) is remembered: never retried per draw
    return id;
}

// ---- Lifetimes ---------------------------------------------------------------

void Context::addRenderer(std::atomic<uint64_t>& pending)
{
    const std::lock_guard<std::mutex> lk(lock_);
    renderers_.push_back(&pending);
}

void Context::removeRenderer(std::atomic<uint64_t>& pending)
{
    {
        const std::lock_guard<std::mutex> lk(lock_);
        std::erase(renderers_, &pending);
    }
    collect();
}

uint64_t Context::beginRecord(std::atomic<uint64_t>& pending)
{
    uint64_t s;
    {
        const std::lock_guard<std::mutex> lk(lock_);
        s = serial_.fetch_add(1, std::memory_order_acq_rel) + 1;
        pending.store(s, std::memory_order_release);
    }
    images_->sweep(s);
    collect();
    return s;
}

uint64_t Context::safeSerial() const
{
    // Every frame up to here is submitted (or will never be): the oldest
    // unsubmitted frame's serial, less one; else everything recorded so far.
    uint64_t safe = serial_.load(std::memory_order_acquire);
    for (auto* p : renderers_)
        if (const uint64_t s = p->load(std::memory_order_acquire); s != 0)
            safe = std::min(safe, s - 1);
    return safe;
}

void Context::retire(gpu::TextureId texture, uint64_t lastUsed)
{
    if (!texture) return;
    const std::lock_guard<std::mutex> lk(lock_);
    retired_.push_back({ texture.value, lastUsed, false });
}

void Context::retire(gpu::ProgramId program, uint64_t lastUsed)
{
    if (!program) return;
    {
        const std::lock_guard<std::mutex> lk(lock_);
        retired_.push_back({ program.value, lastUsed, true });
    }
    // Its pipelines go with it: forget them, so a later program reusing the
    // handle's slot never finds a stale one.
    const std::lock_guard<std::mutex> lk(pipelineLock_);
    std::erase_if(pipelines_, [program](const auto& e) { return e.first.program == program; });
}

void Context::collect()
{
    {
        const std::lock_guard<std::mutex> lk(lock_);
        if (retired_.empty()) return;
        const uint64_t safe = safeSerial();
        for (size_t i = 0; i < retired_.size();)
        {
            if (retired_[i].lastUsed <= safe)
            {
                collectScratch_.push_back(retired_[i]);
                retired_[i] = retired_.back();
                retired_.pop_back();
            }
            else ++i;
        }
    }
    // Outside the lock: the backend defers the native objects until the GPU
    // is done, which may take its own locks.
    for (auto& r : collectScratch_)
    {
        if (r.program) backend_->destroyProgram(gpu::ProgramId { r.handle });
        else           backend_->destroyTexture(gpu::TextureId { r.handle });
    }
    collectScratch_.clear();
}

// =============================================================================
// ImageCache
// =============================================================================

ImageCache::~ImageCache()
{
    for (auto& [key, e] : entries_)
        context_.retire(e.texture, e.lastUsed);
    entries_.clear();
}

gpu::TextureId ImageCache::get(const ImageSource& src, uint64_t serial)
{
    if (auto it = entries_.find(src.key); it != entries_.end())
    {
        it->second.lastUsed = serial;
        return it->second.texture;
    }
    if (!src.isValid()) return {};

    gpu::TextureDesc d;
    d.format = src.format;
    d.width  = src.width;
    d.height = src.height;
    const auto texture = context_.backend().createTexture(d);
    if (!texture) return {};

    scratch_.resize(static_cast<size_t>(formatInfo(src.format).bytesFor(src.width, src.height)));
    src.fill(src, reinterpret_cast<uint8_t*>(scratch_.data()));
    if (!context_.backend().writeTexture(texture, 0, 0,
            { 0, 0, static_cast<int32_t>(src.width), static_cast<int32_t>(src.height) }, scratch_))
    {
        context_.backend().destroyTexture(texture);   // never named by a frame
        return {};
    }

    Entry e;
    e.texture = texture;
    e.lastUsed = serial;
    // Hold what it came from (ImageSource): never a jvk image, which keeps its
    // own texture and would only hold itself here.
    if (src.image != nullptr && src.image->getPixelData()->createType()->getTypeID() != imageTypeID)
        e.heldImage = *src.image;
    e.heldTexels = src.owner;
    entries_.emplace(src.key, std::move(e));
    return texture;
}

void ImageCache::sweep(uint64_t serial)
{
    if (serial - lastSweep_ < 16) return;
    lastSweep_ = serial;
    for (auto it = entries_.begin(); it != entries_.end();)
    {
        auto& e = it->second;
        if (serial - e.lastUsed > kMaxAge || e.orphaned())
        {
            context_.retire(e.texture, e.lastUsed);
            it = entries_.erase(it);
        }
        else ++it;
    }
}

} // namespace jvk
