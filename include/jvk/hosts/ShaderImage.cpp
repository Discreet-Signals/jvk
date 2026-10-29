/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: ShaderImage.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

namespace jvk
{

namespace
{
    // shader_region.vert's constants block: jvk's 7 floats (and one spare).
    struct RegionConstants
    {
        float resolutionX, resolutionY;
        float time;
        float viewportW, viewportH;
        float regionX, regionY;
        float reserved;
    };
}

ShaderImage::ShaderImage(const char* fragmentSpv, int fragmentSpvSize,
                         int width, int height, int storageBufSize)
    : w(std::max(1, width)), h(std::max(1, height)),
      storageSize(storageBufSize)
{
    if (storageBufSize > 0)
    {
        const int cap = storageBufSize * 3;
        fifo       = std::make_unique<juce::AbstractFifo>(cap);
        fifoBuffer.assign(static_cast<size_t>(cap), 0.0f);
        storageData.assign(static_cast<size_t>(storageBufSize), 0.0f);
    }

    displayImage = juce::Image(juce::Image::ARGB, w, h, true);
    if (fragmentSpv == nullptr || fragmentSpvSize < 4) return;
    fragmentSpirv_.resize(static_cast<size_t>(fragmentSpvSize) / 4);
    std::memcpy(fragmentSpirv_.data(), fragmentSpv, fragmentSpirv_.size() * 4);

    backend_ = defaultBackend();
    if (backend_ == nullptr || backend_->caps().shaderFormat != gpu::ShaderFormat::SPIRV) return;
    context_ = Context::forBackend(backend_);
    const auto& caps = backend_->caps();
    builder_.setAlignments(caps.constantsAlignment, caps.storageAlignment);

    reflectShader();
    if (textureSlots_ > gpu::kMaxDrawTextures || storageSlots_ > gpu::kMaxDrawStorage)
    {
        diag::log("jvk::ShaderImage: " + juce::String(textureSlots_) + " textures and " + juce::String(storageSlots_)
                  + " blocks, a draw binds at most " + juce::String(gpu::kMaxDrawTextures) + " and "
                  + juce::String(gpu::kMaxDrawStorage) + "; it draws nothing");
        jassertfalse;
        return;
    }
    gpu::ProgramDesc desc;
    desc.name = "jvk::ShaderImage";
    desc.vertex = std::as_bytes(shaders::bundle::shaderRegionVert.spirv);
    desc.fragment = std::as_bytes(std::span<const uint32_t>(fragmentSpirv_));
    desc.layout.textures = textureSlots_;
    desc.layout.storageBuffers = storageSlots_;
    desc.layout.constantsBytes = sizeof(RegionConstants);
    desc.layout.colorOutputs = 1;
    desc.layout.bindings = nativeBindings_;
    program_ = backend_->createProgram(desc);
    if (!program_) return;

    gpu::PipelineDesc pd;
    pd.program = program_;
    pd.attachments.colorCount = 1;
    pd.attachments.colorFormats[0] = PixelFormat::BGRA8;
    pd.blend[0] = gpu::BlendState::replace();
    pipeline_ = context_->pipeline(pd);

    ready = pipeline_ && createTarget();
}

ShaderImage::~ShaderImage()
{
    stopTimer();
    surface_.reset();   // waits for its frame on the GPU
    if (backend_ == nullptr) return;
    for (auto& b : bindings_)
        if (b.texture) backend_->destroyTexture(b.texture);
    destroyTarget();
    // Its pipeline goes with the program (Context: forgotten from the intern table).
    if (program_ && context_ != nullptr) context_->retire(program_, context_->serial());
}

void ShaderImage::reflectShader()
{
    SpvReflectShaderModule module;
    if (spvReflectCreateShaderModule(fragmentSpirv_.size() * 4, fragmentSpirv_.data(), &module) != SPV_REFLECT_RESULT_SUCCESS)
        return;
    uint32_t count = 0;
    spvReflectEnumerateDescriptorBindings(&module, &count, nullptr);
    std::vector<SpvReflectDescriptorBinding*> refl(count);
    spvReflectEnumerateDescriptorBindings(&module, &count, refl.data());
    for (auto* rb : refl)
    {
        Binding b;
        b.desc.set = static_cast<uint8_t>(rb->set);
        b.desc.binding = static_cast<uint8_t>(rb->binding);
        if (rb->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
        {
            b.desc.kind = gpu::BindingKind::CombinedTextureSampler;
            b.desc.slot = textureSlots_++;
        }
        else if (rb->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER
              || rb->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
        {
            b.desc.kind = rb->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER
                        ? gpu::BindingKind::StorageBuffer : gpu::BindingKind::UniformBuffer;
            b.desc.slot = storageSlots_++;
            b.storage = true;
        }
        else continue;
        nativeBindings_.push_back(b.desc);
        bindings_.push_back(b);
    }
    spvReflectDestroyShaderModule(&module);
}

bool ShaderImage::createTarget()
{
    gpu::TextureDesc d;
    d.format = PixelFormat::BGRA8;   // juce ARGB's byte order: the readback is a straight copy
    d.usage  = gpu::TextureUsage::RenderTarget | gpu::TextureUsage::Sampled | gpu::TextureUsage::Readback;
    d.width  = static_cast<uint32_t>(w);
    d.height = static_cast<uint32_t>(h);
    d.label  = "jvk::ShaderImage";
    target_ = backend_->createTexture(d);
    if (!target_) return false;
    surface_ = backend_->createOffscreenSurface(target_);
    pendingRead = false;
    return surface_ != nullptr;
}

void ShaderImage::destroyTarget()
{
    surface_.reset();
    if (target_) backend_->destroyTexture(target_);
    target_ = {};
}

void ShaderImage::setSize(int width, int height)
{
    width = std::max(1, width);
    height = std::max(1, height);
    if (width == w && height == h) return;
    w = width;
    h = height;
    displayImage = juce::Image(juce::Image::ARGB, w, h, true);
    if (backend_ == nullptr || !program_) return;
    destroyTarget();
    ready = pipeline_ && createTarget();
}

void ShaderImage::pushData(const float* data, int count)
{
    if (storageSize <= 0 || count <= 0 || !fifo) return;
    int start1, size1, start2, size2;
    fifo->prepareToWrite(count, start1, size1, start2, size2);
    if (size1 > 0) std::memcpy(fifoBuffer.data() + start1, data, (size_t)size1 * sizeof(float));
    if (size2 > 0) std::memcpy(fifoBuffer.data() + start2, data + size1, (size_t)size2 * sizeof(float));
    fifo->finishedWrite(size1 + size2);
}

void ShaderImage::loadTexture(int binding, const juce::Image& image)
{
    if (backend_ == nullptr || !image.isValid()) return;
    for (auto& b : bindings_)
    {
        if (b.storage || b.desc.binding != binding) continue;
        if (b.texture) backend_->destroyTexture(b.texture);   // deferred past any frame that reads it
        gpu::TextureDesc d;
        d.format = PixelFormat::RGBA8;
        d.width  = static_cast<uint32_t>(image.getWidth());
        d.height = static_cast<uint32_t>(image.getHeight());
        b.texture = backend_->createTexture(d);
        if (!b.texture) return;
        std::vector<uint8_t> rgba(static_cast<size_t>(d.width) * d.height * 4);
        ImageSource::writeRGBA(image, rgba.data());
        backend_->writeTexture(b.texture, 0, 0, { 0, 0, (int32_t) d.width, (int32_t) d.height },
                               std::as_bytes(std::span<const uint8_t>(rgba)));
        return;
    }
}

void ShaderImage::setFrameRate(int fps)
{
    if (fps > 0) startTimerHz(fps);
    else         stopTimer();
}

void ShaderImage::render()
{
    if (!ready || surface_ == nullptr) return;

    gpu::FrameInfo info;
    if (!surface_->beginFrame(info)) return;   // waits for the previous frame

    // The previous frame's pixels: finished by now (beginFrame waited for it).
    if (pendingRead)
    {
        readback_.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
        if (backend_->readTexture(target_, { 0, 0, w, h }, readback_))
        {
            juce::Image::BitmapData bd(displayImage, juce::Image::BitmapData::writeOnly);
            for (int y = 0; y < h; ++y)
                std::memcpy(bd.getLinePointer(y), readback_.data() + static_cast<size_t>(y) * w * 4,
                            static_cast<size_t>(w) * 4);
        }
        pendingRead = false;
    }

    // Drain the FIFO into the storage data.
    if (fifo && storageSize > 0)
    {
        const int toRead = std::min(fifo->getNumReady(), storageSize);
        if (toRead > 0)
        {
            int start1, size1, start2, size2;
            fifo->prepareToRead(toRead, start1, size1, start2, size2);
            if (size1 > 0) std::memcpy(storageData.data(),         fifoBuffer.data() + start1, (size_t)size1 * sizeof(float));
            if (size2 > 0) std::memcpy(storageData.data() + size1, fifoBuffer.data() + start2, (size_t)size2 * sizeof(float));
            fifo->finishedRead(toRead);
        }
    }

    builder_.reset(++frame_);
    gpu::Draw d;
    d.pipeline = pipeline_;
    d.scissor = { 0, 0, w, h };
    for (const auto& b : bindings_)
    {
        if (b.storage)
            d.storage[b.desc.slot] = storageData.empty()
                ? builder_.pushStorage(nullptr, 0)
                : builder_.pushStorage(storageData.data(), static_cast<uint32_t>(storageData.size() * sizeof(float)));
        else
            d.textures[b.desc.slot] = { b.texture ? b.texture : context_->defaultTexture(),
                                        { gpu::Filter::Linear, gpu::AddressMode::Repeat } };
    }
    d.textureCount = textureSlots_;
    d.storageCount = storageSlots_;
    const RegionConstants k { (float) w, (float) h, context_->time(), (float) w, (float) h, 0.0f, 0.0f, 0.0f };
    d.constants = builder_.pushConstants(k);
    d.vertexCount = 3;

    auto& pass = builder_.beginPass("jvk::ShaderImage");
    pass.colorCount = 1;
    pass.color[0] = { info.output, gpu::LoadOp::Clear, gpu::StoreOp::Store, {} };
    pass.renderArea = { 0, 0, w, h };
    builder_.draw(d);
    builder_.endPass();
    surface_->submit(builder_.packet());
    pendingRead = true;
}

} // namespace jvk
