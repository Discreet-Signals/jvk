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
        // viewportSize (2f) + affine transform packed as 3 × vec2 (6f) = 32 bytes.
        cfg.pushConstantRanges = {{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) * 8 }};
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
            state.setStencilWriteMask(1u << state.stencilDepth());
            auto def = r.caches().defaultDescriptor();
            state.setResources(def, def);

            // Affine transform → stencil vert shader (lives in push-const
            // range at byte offset 8, right after the viewport).
            state.pushConstants(sizeof(float) * 2, sizeof(float) * 6, p.transform);

            if (p.cachedMesh) {
                state.drawCached(cmd, r.caches().pathMeshPool().buffer(),
                                 p.cachedMesh->firstVertex, p.vertexCount);
            } else {
                auto verts = arena.readSpan<UIVertex>(p.vertexArenaOffset, p.vertexCount);
                state.draw(cmd, verts.data(), p.vertexCount);
            }
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
        // viewportSize (2f) + affine transform packed as 3 × vec2 (6f) = 32 bytes.
        cfg.pushConstantRanges = {{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) * 8 }};
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
            uint8_t depth = state.stencilDepth();
            if (depth > 0)
                state.setStencilWriteMask(1u << (depth - 1));
            auto def = r.caches().defaultDescriptor();
            state.setResources(def, def);
            state.pushConstants(sizeof(float) * 2, sizeof(float) * 6, p.transform);

            if (p.cachedMesh) {
                state.drawCached(cmd, r.caches().pathMeshPool().buffer(),
                                 p.cachedMesh->firstVertex, p.vertexCount);
            } else {
                auto verts = arena.readSpan<UIVertex>(p.vertexArenaOffset, p.vertexCount);
                state.draw(cmd, verts.data(), p.vertexCount);
            }
        }
        state.popClip();
    }
};

} // namespace jvk::pipelines
