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

    // Even-odd INVERT writer for clipToPath (user-level path clips).
    // Each clip level owns one bit in the HIGH NIBBLE of the stencil (bits
    // 4..7), set by an INVERT pass — a pixel's clip bit is 1 iff an odd
    // number of the path's fan triangles cover it, which matches the even-
    // odd fill rule. This supports up to 4 levels of nested path clips,
    // with the LOW NIBBLE reserved for fillPath's winding counter so the
    // two mechanisms never collide.
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
        cfg.stencilWriteMask = 0xFF; // overridden dynamically to 1 << (4+level)
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
            // Clip level N owns bit (4+N) — write ONLY that bit so the
            // low nibble (fillPath's winding counter scratch space) and
            // the other clip levels' bits are untouched.
            state.setStencilWriteMask(1u << (4 + state.stencilDepth()));
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

    // Clip-pop — re-INVERT the same bit the matching push set, restoring
    // it to zero (XOR with itself cancels).
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
            // Undo the paired push. State.stencilDepth() is the pre-pop
            // value here (the matching push was at level depth-1), so the
            // bit to re-INVERT is 4 + (depth - 1).
            uint8_t depth = state.stencilDepth();
            if (depth > 0)
                state.setStencilWriteMask(1u << (4 + depth - 1));
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
