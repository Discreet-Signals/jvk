#pragma once

namespace jvk::pipelines {

class StencilWritePipeline : public Pipeline {
public:
    StencilWritePipeline(Device& device, std::span<const uint32_t> vertSpv, std::span<const uint32_t> fragSpv)
        : Pipeline(device)
    {
        loadVertexShader(vertSpv);
        loadFragmentShader(fragSpv);
    }

    PipelineConfig config() const override
    {
        PipelineConfig cfg;
        cfg.blendMode = BlendMode::Opaque;
        cfg.colorWriteEnable = false;
        cfg.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        cfg.cullMode = VK_CULL_MODE_NONE;
        cfg.stencilTestEnable = true;
        cfg.stencilCompareOp = VK_COMPARE_OP_ALWAYS;
        cfg.stencilPassOp = VK_STENCIL_OP_INCREMENT_AND_WRAP;
        cfg.stencilFailOp = VK_STENCIL_OP_KEEP;
        cfg.stencilWriteMask = 0xFF;
        cfg.pushConstantRanges = {{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) * 2 }};
        return cfg;
    }

    std::span<const DrawOp> supportedOps() const override
    {
        static constexpr DrawOp ops[] = { DrawOp::PushClipPath };
        return ops;
    }

    void execute(Renderer& r, const Arena& arena, const DrawCommand& cmd) override
    {
        auto& state = r.state();
        auto& p = arena.read<PushClipPathParams>(cmd.dataOffset);
        uint32_t after = cmd.dataOffset + static_cast<uint32_t>(sizeof(PushClipPathParams));
        auto verts = arena.readSpan<UIVertex>(after, p.vertexCount);
        state.draw(cmd, verts.data(), p.vertexCount);
        state.pushClipPath(juce::Path(), juce::AffineTransform());
    }
};


class StencilCoverPipeline : public Pipeline {
public:
    StencilCoverPipeline(Device& device, std::span<const uint32_t> vertSpv, std::span<const uint32_t> fragSpv)
        : Pipeline(device)
    {
        loadVertexShader(vertSpv);
        loadFragmentShader(fragSpv);
    }

    PipelineConfig config() const override
    {
        PipelineConfig cfg;
        cfg.blendMode = BlendMode::Opaque;
        cfg.colorWriteEnable = false;
        cfg.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        cfg.cullMode = VK_CULL_MODE_NONE;
        cfg.stencilTestEnable = true;
        cfg.stencilCompareOp = VK_COMPARE_OP_ALWAYS;
        cfg.stencilPassOp = VK_STENCIL_OP_DECREMENT_AND_WRAP;
        cfg.stencilFailOp = VK_STENCIL_OP_KEEP;
        cfg.stencilWriteMask = 0xFF;
        cfg.pushConstantRanges = {{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) * 2 }};
        return cfg;
    }

    std::span<const DrawOp> supportedOps() const override
    {
        static constexpr DrawOp ops[] = { DrawOp::PopClip };
        return ops;
    }

    void execute(Renderer& r, const Arena& arena, const DrawCommand& cmd) override
    {
        r.state().popClip();
    }
};

} // namespace jvk::pipelines
