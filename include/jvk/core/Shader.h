/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Shader.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk {

// =============================================================================
// Shader — a user fragment shader, drawn with jvk::Graphics::drawShader
//
// The GLSL contract (Vulkan conventions), compiled to SPIR-V, the form the
// Vulkan backend draws and reflection reads; Metal draws the MSL that jvk_msl
// translates from that SPIR-V at build time (CMake: jvk_msl). A shader without
// the backend's form draws nothing:
//
//   - The vertex stage is jvk's region quad (shaders/shader_region.vert):
//     `layout(location = 0) in vec2 fragUV`, 0..1 over the draw's region.
//   - Push constants: jvk's 8 floats, then the draw's own (setDrawConstants):
//
//       layout(push_constant) uniform PC {
//           float resolutionX, resolutionY, time, viewportW, viewportH,
//                 regionX, regionY, reserved;           // jvk's, bytes 0..31
//           vec4  frame;                                // the draw's: bytes 32..
//           ...                                         //   up to 24 floats
//       } pc;
//
//   - Set 0: its own bindings — sampler2D / sampler2DArray (set), uniform and
//     storage blocks (set(name, value)). A sampler2D named after a render
//     target reads that target.
//   - Set 1, binding 0: `uniform sampler2D scene;` — the main colour as painted
//     so far (the shader then runs as a pass of its own and REPLACES its
//     region).
//   - Sets 2..5, binding 0: per-draw images (setDrawImage).
//   - Outputs: location 0 the main colour, location i + 1 render target i (or,
//     for a scaled target, an output named after it).
//
// Everything a draw needs is CAPTURED when it is recorded (capture): the
// program, the textures, the uniform values, the per-draw constants. Nothing
// reads the Shader after that, so it may be destroyed at any time; its
// program and textures go through the Context's retire list, which waits for
// the frames that name them.
// =============================================================================

class Shader {
public:
    Shader() = default;
    ~Shader();

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    void load(std::span<const uint32_t> fragSpirv, std::string_view fragMsl = {})
    {
        spirv_.assign(fragSpirv.begin(), fragSpirv.end());
        msl_.assign(fragMsl);
        reflectShader();
    }

    // ===== Images ============================================================
    //
    // An image a sampler2D binding samples. It resolves through the shared
    // image cache each frame it is drawn — one upload per content key, shared
    // by every draw and window — and the binding holds the image, so its key
    // stays valid. (The Renderer is accepted for source compatibility.)
    void set(const juce::String& name, const juce::Image& image)
    {
        if (auto* b = imageBinding(name)) { b->image = image; b->texture = {}; }
    }
    void set(const juce::String& name, const juce::Image& image, Renderer&) { set(name, image); }

    // A texture the caller made on the backend (Renderer::backend()) and fills
    // itself (Backend::writeTexture), for example a texture array or an atlas
    // page: nothing is copied. The caller keeps it alive while this Shader can
    // draw with it, and destroys it through the Context (Context::retire), which
    // waits for the frames that name it.
    void set(const juce::String& name, gpu::TextureId texture, gpu::SamplerDesc sampler = {})
    {
        if (auto* b = imageBinding(name)) { b->image = {}; b->texture = texture; b->sampler = sampler; }
    }

    // Dynamic images: for sources whose CONTENT changes every frame (video
    // feeds, CPU-composited animations). set() keys the shared cache on the
    // pixel-buffer address and never re-uploads on a hit, so a mutated buffer
    // keeps showing its first upload. update() gives the binding its own
    // texture and writes the pixels into it on every call (Backend::
    // writeTexture: ordered after the frames already submitted, so they
    // finish sampling the old contents first). Pixels are copied at call time
    // (BGRA, premultiplied — juce ARGB's memory layout). Message thread,
    // during record.
    void update(const juce::String& name, const juce::Image& image, Renderer& r);

    // ===== Values ============================================================

    // Uniform and storage blocks: stored here, captured by every draw.
    void set(const juce::String& name, float value)
    {
        for (auto& b : bindings_) {
            if (b.isBuffer() && (b.name == name || b.blockName == name)
                && b.offset + sizeof(float) <= uniformData_.size()) {
                std::memcpy(uniformData_.data() + b.offset, &value, sizeof(float));
                return;
            }
        }
    }

    void set(const juce::String& name, std::span<const float> data)
    {
        for (auto& b : bindings_) {
            if (b.isBuffer() && (b.name == name || b.blockName == name)) {
                if (b.offset + data.size_bytes() <= uniformData_.size())
                    std::memcpy(uniformData_.data() + b.offset, data.data(), data.size_bytes());
                return;
            }
        }
    }

    // ===== Per-draw constants ================================================
    //
    // Up to 24 floats that belong to ONE draw: Graphics::drawShader copies
    // what was set last into the draw it records, so one Shader can be drawn
    // many times in a frame with different values (an atlas drawn at several
    // places, each with its own cell). Blocks set with set() are per shader
    // instead: every draw sees the value at the time it was recorded. The
    // shader reads the constants from its push-constant block, after jvk's own
    // eight floats (see the contract above).
    static constexpr uint32_t kDrawConstantsOffset = 32;

    void setDrawConstants (std::span<const float> values)
    {
        jassert (values.size() <= static_cast<size_t>(kShaderDrawConstants));
        drawConstantCount_ = static_cast<uint32_t>(std::min(values.size(), static_cast<size_t>(kShaderDrawConstants)));
        std::copy_n(values.begin(), drawConstantCount_, drawConstants_.begin());
    }
    std::span<const float> drawConstants() const { return { drawConstants_.data(), drawConstantCount_ }; }

    // ===== Per-draw images ===================================================
    //
    // A sampler declared in a descriptor set of its own, set 2 or above, at
    // binding 0 (`layout(set = 2, binding = 0) uniform sampler2D albedo;`), is a
    // PER-DRAW image: setDrawImage names what the NEXT draw binds there (the draw
    // takes it; a slot left unnamed samples the black pixel). It resolves
    // through the image cache when the draw is recorded — one upload per
    // content key, shared by every draw and window — so one Shader draws any
    // number of images a frame. Up to kShaderDrawImages.
    void setDrawImage (const juce::String& name, const ImageSource& source)
    {
        for (size_t i = 0; i < drawImageNames_.size(); ++i)
            if (drawImageNames_[i] == name)
                drawImages_[i] = source;
    }

    uint32_t drawImageCount() const { return static_cast<uint32_t>(drawImageNames_.size()); }
    ImageSource takeDrawImage (uint32_t slot) { return std::exchange (drawImages_[slot], {}); }

    // ===== Reading and writing anything ======================================
    //
    // A shader reads a target by declaring a sampler2D with the target's name
    // (`layout(binding = n) uniform sampler2D normal;`), THE SCENE (the main
    // colour as painted so far) by declaring
    // `layout(set = 1, binding = 0) uniform sampler2D scene;`, and writes the
    // main colour at location 0 and targets at theirs (or, for a scaled
    // target, at an output named after it). Any reads with any writes of one
    // size; jvk works out the passes (Renderer::lower, the render graph):
    //   - INSIDE THE SCENE PASS when everything it writes is attached there
    //     (the main colour, targets at scale 1) and nothing it reads is
    //     (scaled targets it may read): the cheapest, no pass boundary;
    //   - else as ITS OWN PASS at that point in paint order, attached to
    //     exactly what it writes, seeing everything drawn before it. Writing
    //     the main colour, the clip still applies; a scene reader REPLACES the
    //     main colour inside its region with what it outputs (it has the scene
    //     to blend with itself: a lighting pass, a grade, a distortion), any
    //     other blends Over. A target it both reads and writes is versioned:
    //     it draws into the target's scratch while sampling the target, and
    //     the result is copied back over its region.
    bool readsScene() const { return readsScene_; }

    // The targets of `r` this shader samples, by sampler name (bit i = target
    // i), and the ones it writes at each output location (TargetSlots): the
    // scaled target its output there is named after, else target location - 1.
    TargetMask  targetsRead (const Renderer& r);
    TargetSlots targetSlots (const Renderer& r);

    // Whether it writes the main colour: location 0, or no declared output.
    bool writesMain() const { return writesLocation(0) || outputLocations_.empty(); }

    // Whether an output is named after a target the frame declared but DROPPED
    // (Renderer::droppedTargets): such a draw is invalid and draws nothing.
    // Falling back to the location would land it in another target.
    bool writesDroppedTarget (const Renderer& r);

    // Message thread, while recording: what the draw needs (see the class
    // comment). False when it can't draw (no program on this backend).
    bool capture(Renderer& r, ShaderCapture& c);

private:
    struct BindingInfo {
        juce::String     name;
        juce::String     blockName;   // a uniform block's declared name; set() takes either
        uint32_t         set = 0, binding = 0;
        gpu::BindingKind kind = gpu::BindingKind::CombinedTextureSampler;
        uint32_t         offset = 0, size = 0;   // a block's bytes in uniformData_
        uint8_t          slot = 0;               // its texture or storage slot
        bool             arrayed = false;        // an image binding declared as an array (sampler2DArray)

        // What an image binding samples: an image through the cache, a texture
        // the caller owns, or one this Shader owns (update).
        juce::Image      image;
        gpu::TextureId   texture;
        gpu::SamplerDesc sampler;
        gpu::TextureId   owned;
        uint32_t         ownedW = 0, ownedH = 0;

        bool isBuffer() const
        {
            return kind == gpu::BindingKind::StorageBuffer || kind == gpu::BindingKind::UniformBuffer;
        }
    };

    BindingInfo* imageBinding(const juce::String& name)
    {
        for (auto& b : bindings_)
            if (b.kind == gpu::BindingKind::CombinedTextureSampler && b.set == 0 && b.name == name)
                return &b;
        return nullptr;
    }

    void reflectShader();
    bool ensureProgram(Renderer& r);
    void resolveTargets(const Renderer& r);

    bool writesLocation (uint32_t location) const
    {
        return std::find(outputLocations_.begin(), outputLocations_.end(), location) != outputLocations_.end();
    }

    // Every block starts on a multiple of this: the largest
    // min{Uniform,Storage}BufferOffsetAlignment any API asks for.
    static constexpr uint32_t kBlockAlign = 256;

    std::vector<uint32_t>     spirv_;
    std::string               msl_;
    std::vector<BindingInfo>  bindings_;
    std::vector<std::byte>    uniformData_;
    std::vector<gpu::BindingDesc> nativeBindings_;  // the program's layout, by slot
    uint8_t                   textureSlots_ = 0, storageSlots_ = 0;
    int8_t                    sceneSlot_ = -1;
    int8_t                    drawImageSlot_[kShaderDrawImages] { -1, -1, -1, -1 };
    std::vector<uint32_t>     outputLocations_;     // reflected fragment output locations
    std::vector<juce::String> outputNames_;         // and their names ("" for an array's)
    std::vector<juce::String> drawImageNames_;      // per-draw image slots (set - 2), by sampler name
    std::array<ImageSource, kShaderDrawImages> drawImages_ {};   // what the next draw binds there
    std::array<float, kShaderDrawConstants> drawConstants_ {};   // setDrawConstants
    uint32_t                  drawConstantCount_ = 0;
    bool                      readsScene_ = false;   // set 1, binding 0: the scene

    // resolveTargets: for the targets of `resolvedFor_`.
    const Renderer*           resolvedFor_ = nullptr;
    TargetMask                readMask_ = 0;
    TargetSlots               slots_ = 0;
    bool                      writesDropped_ = false;

    std::shared_ptr<Context>  context_;     // the backend the program lives on
    gpu::ProgramId            program_;
    bool                      failed_ = false;   // ensureProgram tried and failed: never retried
    uint64_t                  lastUsed_ = 0;     // the record serial of the last frame that drew it
};

} // namespace jvk
