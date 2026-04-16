#pragma once

namespace jvk::pipelines {

class ResolvePipeline : public Pipeline {
public:
    ResolvePipeline(Device& device, std::span<const uint32_t> vertSpv, std::span<const uint32_t> fragSpv)
        : Pipeline(device)
    {
        loadVertexShader(vertSpv);
        loadFragmentShader(fragSpv);
    }

    PipelineConfig config() const override
    {
        PipelineConfig cfg;
        cfg.blendMode = BlendMode::Opaque;
        cfg.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        cfg.cullMode = VK_CULL_MODE_NONE;
        return cfg;
    }

    // No clip variant — resolve runs on separate framebuffer without stencil

    std::span<const DrawOp> supportedOps() const override
    {
        static constexpr DrawOp ops[] = { DrawOp::EffectResolve, DrawOp::EffectKernel };
        return ops;
    }

    void execute(Renderer& r, const Arena& arena, const DrawCommand& cmd) override
    {
        // TODO: Full implementation requires ending the render pass,
        // processing on ping-pong framebuffers, and restarting.
        // This is a z-order barrier. Stub for now.
    }
};

} // namespace jvk::pipelines
