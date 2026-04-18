#pragma once

namespace jvk::pipelines {

class ColorPipeline : public Pipeline {
public:
    ColorPipeline(Device& device, std::span<const uint32_t> vertSpv, std::span<const uint32_t> fragSpv)
        : Pipeline(device)
    {
        loadVertexShader(vertSpv);
        loadFragmentShader(fragSpv);
        atlas_.init(device);
    }

    PipelineConfig config() const override
    {
        PipelineConfig cfg;
        cfg.blendMode = BlendMode::AlphaBlend;
        cfg.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        cfg.cullMode = VK_CULL_MODE_NONE;
        cfg.frontFace = VK_FRONT_FACE_CLOCKWISE;
        cfg.pushConstantRanges = {{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) * 2 }};
        return cfg;
    }

    std::optional<PipelineConfig> clipConfig() const override
    {
        // Even-odd clip test against the high-nibble clip mask — passes
        // where all active clip-level bits are set (stencil & mask == mask).
        // compareMask + reference are pushed dynamically per draw by State.
        auto cfg = config();
        cfg.stencilTestEnable    = true;
        cfg.stencilCompareOp     = VK_COMPARE_OP_EQUAL;
        cfg.stencilFailOp        = VK_STENCIL_OP_KEEP;
        cfg.stencilPassOp        = VK_STENCIL_OP_KEEP;
        cfg.stencilDepthFailOp   = VK_STENCIL_OP_KEEP;
        return cfg;
    }

    std::span<const DrawOp> supportedOps() const override
    {
        static constexpr DrawOp ops[] = {
            DrawOp::FillRect, DrawOp::FillRectList, DrawOp::FillRoundedRect,
            DrawOp::FillEllipse, DrawOp::StrokeRoundedRect, DrawOp::StrokeEllipse,
            DrawOp::DrawImage, DrawOp::DrawGlyphs, DrawOp::DrawLine
        };
        return ops;
    }

    GlyphAtlas& atlas() { return atlas_; }

    void prepare() override { atlas_.stageDirtyPages(); }
    void execute(Renderer& r, const Arena& arena, const DrawCommand& cmd) override;

private:
    GlyphAtlas atlas_;
};

} // namespace jvk::pipelines
