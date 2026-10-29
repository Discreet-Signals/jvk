/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: ShaderImage.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

// =============================================================================
// ShaderImage — a fragment shader, run into a juce::Image
//
// An offscreen host (TRANSLATION_LAYER.md, "Hosts"): every frame is one pass
// of one draw of the shader into a texture on the process's shared backend
// (defaultBackend()), read back into getImage(). The read lands one frame
// late: a frame's pixels are read when the next one begins, by which time the
// GPU has finished them.
//
// Bindings (combined image samplers, storage and uniform buffers, any set) are
// discovered from the fragment SPIR-V by reflection. Constants follow
// shader_region.vert: (resolutionX, resolutionY, time, viewportW, viewportH,
// regionX, regionY); a fragment shader only needs to declare the first three.
// Needs a backend that takes SPIR-V.
// =============================================================================

class ShaderImage : private juce::Timer
{
public:
    ShaderImage(const char* fragmentSpv, int fragmentSpvSize,
                int width, int height, int storageBufferSize = 0);
    ~ShaderImage() override;

    ShaderImage(const ShaderImage&) = delete;
    ShaderImage& operator=(const ShaderImage&) = delete;

    const juce::Image& getImage() const { return displayImage; }

    void setSize(int width, int height);
    juce::Point<int> getSize() const { return { w, h }; }

    // Audio-thread safe. Pushes up to `count` floats into the ring buffer;
    // render() consumes them into the storage buffer each frame.
    void pushData(const float* data, int count);

    // Loads a juce::Image into the combined image sampler at `binding` (in
    // whichever set declares it). Message thread.
    void loadTexture(int binding, const juce::Image& image);

    void setFrameRate(int fps);
    void render();
    bool isReady() const { return ready; }

private:
    void timerCallback() override { render(); }
    void reflectShader();
    bool createTarget();
    void destroyTarget();

    struct Binding
    {
        gpu::BindingDesc desc;
        bool             storage = false;
        gpu::TextureId   texture;   // a loaded image (loadTexture)
    };

    std::shared_ptr<gpu::Backend> backend_;
    std::shared_ptr<Context>      context_;
    std::unique_ptr<gpu::Surface> surface_;

    int  w = 0, h = 0;
    bool ready = false;
    bool pendingRead = false;   // a frame was submitted whose pixels aren't read yet

    std::vector<uint32_t>         fragmentSpirv_;
    std::vector<Binding>          bindings_;
    std::vector<gpu::BindingDesc> nativeBindings_;
    uint8_t                       textureSlots_ = 0, storageSlots_ = 0;
    gpu::ProgramId                program_;
    gpu::PipelineId               pipeline_;
    gpu::TextureId                target_;

    FrameBuilder               builder_;
    uint64_t                   frame_ = 0;
    std::vector<std::byte>     readback_;

    // Audio-thread → render-thread float FIFO
    int                                 storageSize = 0;
    std::unique_ptr<juce::AbstractFifo> fifo;
    std::vector<float>                  fifoBuffer;
    std::vector<float>                  storageData;

    juce::Image displayImage;
};

} // namespace jvk
