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
        cfg.stencilPassOp = VK_STENCIL_OP_INVERT;
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
        if (p.vertexCount > 0) {
            // Set write mask to this level's bit (depth is N before push, write bit N)
            state.setStencilWriteMask(1u << state.stencilDepth());
            uint32_t after = cmd.dataOffset + static_cast<uint32_t>(sizeof(PushClipPathParams));
            auto verts = arena.readSpan<UIVertex>(after, p.vertexCount);
            // Stencil-only writes don't read textures, but the pipeline layout
            // still has two sets — bind the default to both.
            auto def = r.caches().defaultDescriptor();
            state.setResources(def, def);
            state.draw(cmd, verts.data(), p.vertexCount);
        }
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
        cfg.stencilPassOp = VK_STENCIL_OP_INVERT;
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
        auto& state = r.state();
        auto& p = arena.read<PopClipParams>(cmd.dataOffset);
        if (p.vertexCount > 0) {
            // Set write mask to the level being popped (depth is N, clean bit N-1)
            uint8_t depth = state.stencilDepth();
            if (depth > 0)
                state.setStencilWriteMask(1u << (depth - 1));
            uint32_t after = cmd.dataOffset + static_cast<uint32_t>(sizeof(PopClipParams));
            auto verts = arena.readSpan<UIVertex>(after, p.vertexCount);
            auto def = r.caches().defaultDescriptor();
            state.setResources(def, def);
            state.draw(cmd, verts.data(), p.vertexCount);
        }
        state.popClip();
    }
};

} // namespace jvk::pipelines
