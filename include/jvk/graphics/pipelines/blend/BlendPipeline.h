#pragma once

namespace jvk::pipelines {

class BlendPipeline : public Pipeline {
public:
    BlendPipeline(Device& device, std::span<const uint32_t> vertSpv, std::span<const uint32_t> fragSpv)
        : Pipeline(device)
    {
        loadVertexShader(vertSpv);
        loadFragmentShader(fragSpv);
    }

    PipelineConfig config() const override
    {
        PipelineConfig cfg;
        cfg.blendMode = BlendMode::Multiply;
        cfg.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        cfg.cullMode = VK_CULL_MODE_NONE;
        cfg.pushConstantRanges = {{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) * 2 }};
        return cfg;
    }

    std::optional<PipelineConfig> clipConfig() const override
    {
        auto cfg = config();
        cfg.stencilTestEnable = true;
        cfg.stencilCompareOp = VK_COMPARE_OP_EQUAL;
        cfg.stencilFailOp = VK_STENCIL_OP_KEEP;
        cfg.stencilPassOp = VK_STENCIL_OP_KEEP;
        return cfg;
    }

    std::span<const DrawOp> supportedOps() const override
    {
        static constexpr DrawOp ops[] = { DrawOp::EffectBlend };
        return ops;
    }

    void execute(Renderer& r, const Arena& arena, const DrawCommand& cmd) override
    {
        auto& state = r.state();
        auto def = r.caches().defaultDescriptor();
        state.setResources(def, def);
        auto& p = arena.read<EffectBlendParams>(cmd.dataOffset);
        float x = p.region.getX() * p.scale;
        float y = p.region.getY() * p.scale;
        float w = p.region.getWidth() * p.scale;
        float h = p.region.getHeight() * p.scale;
        glm::vec4 color(p.r, p.g, p.b, 1.0f);
        // Flat fill with neutral shape / gradient fields. BlendPipeline
        // shares ColorPipeline's ui2d vertex layout but runs with
        // BlendMode::Multiply (dst *= src) — darken/brighten/tint land here.
        // Inlined here so the ColorDraw.h helpers can retire.
        glm::vec4 shape(0.0f);
        glm::vec4 grad(0.0f);
        UIVertex verts[6] = {
            { {x,     y    }, color, {0, 0}, shape, grad },
            { {x + w, y    }, color, {1, 0}, shape, grad },
            { {x + w, y + h}, color, {1, 1}, shape, grad },
            { {x,     y    }, color, {0, 0}, shape, grad },
            { {x + w, y + h}, color, {1, 1}, shape, grad },
            { {x,     y + h}, color, {0, 1}, shape, grad },
        };
        state.draw(cmd, verts, 6);
    }
};

} // namespace jvk::pipelines
