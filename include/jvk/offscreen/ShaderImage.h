#pragma once

namespace jvk {

class ShaderImage {
public:
    ShaderImage(std::shared_ptr<Device> device, uint32_t width, uint32_t height,
                VkFormat format = VK_FORMAT_R8G8B8A8_UNORM)
        : device_(std::move(device)),
          target_(*device_, width, height, format),
          renderer_(*device_, target_)
    {
        // Register color pipeline for offscreen drawing
        auto spv = [](const char* d, int s) { return std::span<const uint32_t>{reinterpret_cast<const uint32_t*>(d), static_cast<size_t>(s/4)}; };
        colorPipeline_ = std::make_unique<pipelines::ColorPipeline>(
            *device_,
            spv(shaders::ui2d::vert_spv, shaders::ui2d::vert_spvSize),
            spv(shaders::ui2d::frag_spv, shaders::ui2d::frag_spvSize));
        renderer_.registerPipeline(*colorPipeline_);
    }

    void resize(uint32_t width, uint32_t height)
    {
        target_.resize(width, height);
    }

    // Record commands and execute
    void render(std::function<void(Graphics&)> paintFn, float scale = 1.0f)
    {
        renderer_.reset();
        Graphics graphics(renderer_, scale);
        paintFn(graphics);
        renderer_.execute();
    }

    // Push raw commands
    template <typename Params>
    void push(DrawOp op, float z, const juce::Rectangle<int>& clip, const Params& params)
    {
        renderer_.push(op, z, clip, 0, 0, params);
    }

    void execute()
    {
        renderer_.execute();
    }

    void reset()
    {
        renderer_.reset();
    }

    // Result image — can be sampled by the main renderer
    Image& getImage() { return target_.getImage(); }

    Renderer&  renderer() { return renderer_; }
    Device&    device()   { return *device_; }

private:
    std::shared_ptr<Device> device_;
    OffscreenTarget         target_;
    Renderer                renderer_;
    std::unique_ptr<pipelines::ColorPipeline> colorPipeline_;
};

} // namespace jvk
