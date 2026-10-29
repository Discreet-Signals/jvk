/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Shader.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

namespace jvk {

Shader::~Shader()
{
    // Whatever a recorded frame names goes once those frames are submitted;
    // the backend then waits for the GPU before freeing it.
    if (context_ == nullptr) return;
    context_->retire(program_, lastUsed_);
    for (auto& b : bindings_)
        if (b.owned) context_->retire(b.owned, lastUsed_);
}

void Shader::reflectShader()
{
    // Idempotent: a second load() replaces the previous reflection.
    bindings_.clear();
    uniformData_.clear();
    nativeBindings_.clear();
    drawImageNames_.clear();
    textureSlots_ = storageSlots_ = 0;
    sceneSlot_ = -1;
    std::fill(std::begin(drawImageSlot_), std::end(drawImageSlot_), int8_t(-1));
    readsScene_ = false;
    outputLocations_.clear();
    outputNames_.clear();
    resolvedFor_ = nullptr;
    if (context_ != nullptr && program_)
        context_->retire(program_, lastUsed_);
    program_ = {};
    failed_ = false;

    SpvReflectShaderModule module;
    if (spvReflectCreateShaderModule(spirv_.size() * 4, spirv_.data(), &module) != SPV_REFLECT_RESULT_SUCCESS)
        return;

    uint32_t count = 0;
    spvReflectEnumerateDescriptorBindings(&module, &count, nullptr);
    std::vector<SpvReflectDescriptorBinding*> reflBindings(count);
    spvReflectEnumerateDescriptorBindings(&module, &count, reflBindings.data());

    // Use CharPointer_UTF8 rather than the raw-char-ptr String ctor: SPIRV-
    // Reflect can hand back pointers into bytecode where non-ASCII bytes (or
    // empty names) trip juce::String's ASCII jassert in Debug builds.
    auto str = [](const char* s) { return s != nullptr ? juce::String(juce::CharPointer_UTF8(s)) : juce::String(); };

    uint32_t bufferOffset = 0;
    std::vector<SpvReflectDescriptorBinding*> drawImages(kShaderDrawImages, nullptr);
    SpvReflectDescriptorBinding* scene = nullptr;
    for (auto* rb : reflBindings) {
        const bool combined = rb->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        if (rb->set == 1) {   // set 1, binding 0: the scene (readsScene); jvk binds it
            if (rb->binding == 0 && combined) scene = rb;
            continue;
        }
        if (rb->set >= 2) {   // a set of its own, binding 0: a per-draw image (setDrawImage)
            const auto slot = rb->set - 2;
            jassert (rb->binding == 0 && slot < kShaderDrawImages && combined);
            if (rb->binding == 0 && slot < kShaderDrawImages && combined)
                drawImages[slot] = rb;
            continue;
        }

        BindingInfo info;
        info.name = str(rb->name);
        if (rb->type_description != nullptr)
            info.blockName = str(rb->type_description->type_name);
        info.set = rb->set;
        info.binding = rb->binding;
        if (combined) {
            info.kind = gpu::BindingKind::CombinedTextureSampler;
            info.arrayed = rb->image.arrayed != 0;
        } else if (rb->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER
                || rb->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
            info.kind = rb->descriptor_type == SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER
                      ? gpu::BindingKind::StorageBuffer : gpu::BindingKind::UniformBuffer;
            // SPIRV-Reflect zeros block.size for every storage buffer (it assumes
            // a runtime-sized array). Recover the actual size by walking members
            // — the last member's offset + padded_size is the concrete block
            // footprint for fixed-size declarations like `float data[15]`.
            uint32_t blockSize = rb->block.size;
            if (blockSize == 0 && rb->block.member_count > 0) {
                const auto& last = rb->block.members[rb->block.member_count - 1];
                blockSize = last.offset + (last.padded_size != 0 ? last.padded_size : last.size);
            }
            bufferOffset = (bufferOffset + kBlockAlign - 1) / kBlockAlign * kBlockAlign;
            info.offset = bufferOffset;
            info.size   = std::max(blockSize, 4u);
            bufferOffset += info.size;
        } else {
            jassertfalse;   // a binding kind jvk can't feed (separate samplers, storage images): ignored
            continue;
        }
        bindings_.push_back(std::move(info));
    }
    uniformData_.assign(bufferOffset, std::byte {});

    // Slots: set 0's images in declaration order, then the scene, then the
    // per-draw images; set 0's blocks in order. The program's native layout
    // names where each slot lives. Every binding takes one: a shader that
    // needs more than a draw binds is refused whole (ensureProgram), never
    // built without some of them.
    for (auto& b : bindings_)
    {
        b.slot = b.isBuffer() ? storageSlots_++ : textureSlots_++;
        nativeBindings_.push_back({ b.kind, b.slot, static_cast<uint8_t>(b.set), static_cast<uint8_t>(b.binding) });
    }
    if (scene != nullptr) {
        readsScene_ = true;
        sceneSlot_ = static_cast<int8_t>(textureSlots_++);
        nativeBindings_.push_back({ gpu::BindingKind::CombinedTextureSampler, static_cast<uint8_t>(sceneSlot_), 1, 0 });
    }
    for (uint32_t i = 0; i < kShaderDrawImages; ++i)
    {
        if (drawImages[i] == nullptr) continue;
        if (drawImageNames_.size() <= i) drawImageNames_.resize(i + 1);
        drawImageNames_[i] = str(drawImages[i]->name);
        drawImageSlot_[i] = static_cast<int8_t>(textureSlots_++);
        nativeBindings_.push_back({ gpu::BindingKind::CombinedTextureSampler,
                                    static_cast<uint8_t>(drawImageSlot_[i]), static_cast<uint8_t>(2 + i), 0 });
    }

    // Fragment outputs: which colour locations this shader writes (location 0
    // = main colour, i + 1 = render target i, or the scaled target an output
    // is named after). An output array covers consecutive locations.
    uint32_t outCount = 0;
    spvReflectEnumerateOutputVariables(&module, &outCount, nullptr);
    std::vector<SpvReflectInterfaceVariable*> outs(outCount);
    spvReflectEnumerateOutputVariables(&module, &outCount, outs.data());
    for (auto* v : outs) {
        if (v == nullptr || (v->decoration_flags & SPV_REFLECT_DECORATION_BUILT_IN) != 0) continue;
        const uint32_t n = v->array.dims_count > 0 ? std::max(1u, v->array.dims[0]) : 1u;
        for (uint32_t k = 0; k < n; ++k) {
            outputLocations_.push_back(v->location + k);
            outputNames_.push_back(n == 1 ? str(v->name) : juce::String());
        }
    }

    spvReflectDestroyShaderModule(&module);
}

bool Shader::ensureProgram(Renderer& r)
{
    if (program_ && context_.get() == &r.context()) return true;
    if (failed_ || spirv_.empty()) return false;
    if (context_ != nullptr && context_.get() != &r.context()) {
        jassertfalse;   // one Shader draws on one backend: the first it was drawn with
        return false;
    }
    failed_ = true;   // cleared below once the program exists: a failure is never retried

    const auto& caps = r.context().caps();
    if (caps.shaderFormat != gpu::ShaderFormat::SPIRV) {
        diag::log("jvk::Shader: this backend doesn't take SPIR-V; the shader draws nothing");
        return false;
    }
    if (textureSlots_ > gpu::kMaxDrawTextures || storageSlots_ > gpu::kMaxDrawStorage) {
        diag::log("jvk::Shader: " + juce::String(textureSlots_) + " textures and " + juce::String(storageSlots_)
                  + " blocks, a draw binds at most " + juce::String(gpu::kMaxDrawTextures) + " and "
                  + juce::String(gpu::kMaxDrawStorage) + "; the shader draws nothing");
        jassertfalse;
        return false;
    }

    uint32_t colorOutputs = 1;
    for (auto l : outputLocations_) colorOutputs = std::max(colorOutputs, l + 1);

    gpu::ProgramDesc desc;
    desc.name = "jvk::Shader";
    desc.vertex = std::as_bytes(shaders::bundle::shaderRegionVert.spirv);
    desc.fragment = std::as_bytes(std::span<const uint32_t>(spirv_));
    desc.layout.textures = textureSlots_;
    desc.layout.storageBuffers = storageSlots_;
    desc.layout.constantsBytes = 32 + kShaderDrawConstants * 4;
    desc.layout.colorOutputs = static_cast<uint8_t>(colorOutputs);
    desc.layout.bindings = nativeBindings_;
    program_ = r.backend().createProgram(desc);
    if (!program_) return false;
    context_ = r.context().shared_from_this();
    failed_ = false;
    return true;
}

void Shader::resolveTargets(const Renderer& r)
{
    if (resolvedFor_ == &r) return;
    const auto& targets = r.targets();
    readMask_ = 0;
    for (size_t i = 0; i < targets.size(); ++i) {
        const auto name = targets[i].name.toString();
        for (auto& b : bindings_)
            if (b.kind == gpu::BindingKind::CombinedTextureSampler && !b.arrayed && b.name == name)
                readMask_ |= static_cast<TargetMask>(1u << i);
    }
    slots_ = 0;
    writesDropped_ = false;
    for (size_t k = 0; k < outputLocations_.size(); ++k) {
        const uint32_t l = outputLocations_[k];
        if (l < 1 || l > 7) continue;
        for (const auto& dropped : r.droppedTargets())
            if (outputNames_[k].isNotEmpty() && dropped.toString() == outputNames_[k])
                writesDropped_ = true;
        int t = -1;
        for (size_t i = 0; i < targets.size() && outputNames_[k].isNotEmpty(); ++i)
            if (targets[i].isScaled() && targets[i].name.toString() == outputNames_[k]) t = static_cast<int>(i);
        if (t < 0)
            t = l - 1 < targets.size() ? static_cast<int>(l - 1) : -1;
        if (t >= 0) slots_ |= static_cast<TargetSlots>(t + 1) << (8 * l);
    }
    resolvedFor_ = &r;
}

TargetMask  Shader::targetsRead (const Renderer& r) { resolveTargets(r); return readMask_; }
TargetSlots Shader::targetSlots (const Renderer& r) { resolveTargets(r); return slots_; }
bool        Shader::writesDroppedTarget (const Renderer& r) { resolveTargets(r); return writesDropped_; }

void Shader::update(const juce::String& name, const juce::Image& image, Renderer& r)
{
    if (!image.isValid()) return;
    auto* b = imageBinding(name);
    if (b == nullptr || !ensureProgram(r)) return;
    auto& backend = r.backend();
    const auto w = static_cast<uint32_t>(image.getWidth());
    const auto h = static_cast<uint32_t>(image.getHeight());

    // (Re)make the owned texture on first use or resize. BGRA8 matches juce
    // ARGB's little-endian byte order (B,G,R,A), so the rows go in as they are.
    if (!b->owned || b->ownedW != w || b->ownedH != h)
    {
        if (b->owned) context_->retire(b->owned, lastUsed_);
        gpu::TextureDesc d;
        d.format = PixelFormat::BGRA8;
        d.width = w;
        d.height = h;
        d.label = "jvk::Shader dynamic image";
        b->owned = backend.createTexture(d);
        b->ownedW = w;
        b->ownedH = h;
        b->image = {};
        b->texture = {};
        if (!b->owned) return;
    }

    juce::Image::BitmapData bd(image, juce::Image::BitmapData::readOnly);
    if (image.getFormat() == juce::Image::ARGB && bd.lineStride == static_cast<int>(w) * 4)
    {
        backend.writeTexture(b->owned, 0, 0, { 0, 0, (int32_t) w, (int32_t) h },
                             { reinterpret_cast<const std::byte*>(bd.data), static_cast<size_t>(w) * h * 4 });
        return;
    }
    // A padded or non-ARGB image: repack its rows (converted to ARGB first if need be).
    const juce::Image argb = image.getFormat() == juce::Image::ARGB ? image : image.convertedToFormat(juce::Image::ARGB);
    juce::Image::BitmapData src(argb, juce::Image::BitmapData::readOnly);
    std::vector<std::byte> rows(static_cast<size_t>(w) * h * 4);
    for (uint32_t y = 0; y < h; ++y)
        std::memcpy(rows.data() + static_cast<size_t>(y) * w * 4, src.getLinePointer(static_cast<int>(y)), static_cast<size_t>(w) * 4);
    backend.writeTexture(b->owned, 0, 0, { 0, 0, (int32_t) w, (int32_t) h }, rows);
}

bool Shader::capture(Renderer& r, ShaderCapture& c)
{
    if (!ensureProgram(r)) return false;
    lastUsed_ = r.serial();

    c.program = program_;
    c.outputs = 0;
    for (auto l : outputLocations_)
        if (l < 8) c.outputs |= static_cast<uint8_t>(1u << l);
    if (outputLocations_.empty()) c.outputs = 1;
    c.readsScene = readsScene_;
    c.sceneSlot = sceneSlot_;
    c.textureCount = textureSlots_;
    std::fill(std::begin(c.slotTarget), std::end(c.slotTarget), int8_t(-1));

    const auto& ctx = r.context();
    const auto& targets = r.targets();
    for (auto& b : bindings_)
    {
        if (b.isBuffer()) {
            c.storage[b.slot] = r.pushShaderData({ uniformData_.data() + b.offset, b.size });
            continue;
        }
        // A sampler named after a target reads that target (the lowering binds
        // it), whatever else was set on it; else what was set, else black.
        gpu::TextureBinding t { b.arrayed ? ctx.defaultArrayTexture() : ctx.defaultTexture(), {} };
        int target = -1;
        if (!b.arrayed)
            for (size_t i = 0; i < targets.size() && target < 0; ++i)
                if (targets[i].name.toString() == b.name) target = static_cast<int>(i);
        if (target >= 0)            c.slotTarget[b.slot] = static_cast<int8_t>(target);
        else if (b.owned)           t = { b.owned, {} };
        else if (b.texture)         t = { b.texture, b.sampler };
        else if (b.image.isValid()) { if (auto tex = r.texture(b.image)) t = { tex, {} }; }
        c.textures[b.slot] = t;
    }
    c.storageCount = storageSlots_;
    if (sceneSlot_ >= 0)
        c.textures[sceneSlot_] = { ctx.defaultTexture(), {} };   // the lowering binds the scene here
    for (uint32_t i = 0; i < kShaderDrawImages; ++i)
    {
        if (drawImageSlot_[i] < 0) continue;
        const auto source = takeDrawImage(i);
        const auto tex = source.isValid() ? r.texture(source) : gpu::TextureId {};
        c.textures[drawImageSlot_[i]] = { tex ? tex : ctx.defaultTexture(), {} };
    }
    return true;
}

// =============================================================================
// Graphics::drawShader's record-time step
// =============================================================================

void Graphics::prepareShaderDraw(Shader& shader, DrawShaderParams& params)
{
    auto& r = renderer_;
    params.targetsRead    = shader.targetsRead(r);
    params.targetSlots    = shader.targetSlots(r);
    params.targetsWritten = slotTargets(params.targetSlots);
    params.writesMain     = shader.writesMain();
    const TargetMask scaled = r.scaledTargets(), full = static_cast<TargetMask>(~scaled);
    params.inScene        = !shader.readsScene() && (params.targetsRead & full) == 0
                         && (params.targetsWritten & scaled) == 0;
    // A pass's attachments are one size: a shader writing the main colour or a target at
    // scale 1 together with a scaled target, or scaled targets of two sizes, draws nothing.
    // So does one writing an output named after a target the frame dropped.
    const bool fullSize = params.writesMain || (params.targetsWritten & full) != 0;
    bool valid = !(fullSize && (params.targetsWritten & scaled) != 0) && !shader.writesDroppedTarget(r);
    const Target* passSize = nullptr;
    for (size_t i = 0; i < r.targets().size(); ++i)
        if ((params.targetsWritten & scaled & (1u << i)) != 0) {
            valid = valid && (passSize == nullptr || r.targets()[i].sameSize(*passSize));
            passSize = &r.targets()[i];
        }
    jassert (valid);
    const auto constants = shader.drawConstants();
    params.constantCount = static_cast<uint32_t>(constants.size());
    std::copy(constants.begin(), constants.end(), params.constants);
    ShaderCapture capture;
    params.shaderIndex = valid && shader.capture(r, capture) ? r.captureShader(capture) : ~0u;
}

} // namespace jvk
