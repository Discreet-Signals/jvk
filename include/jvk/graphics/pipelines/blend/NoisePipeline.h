#pragma once

namespace jvk::pipelines {

// Procedural white-noise overlay backing jvk::Graphics::drawNoise. Reuses the
// ui2d vertex shader (a flat quad over the region) paired with noise.frag,
// which hashes gl_FragCoord per device pixel. AlphaBlend with alpha = `amount`
// yields mix(dst, whiteNoise, amount): 0 = untouched, 1 = full white-noise
// overwrite, 0.1 = a fine grain overlay. The per-draw amount + time offset ride
// in the quad's vertex colour (.a = amount, .r = time offset), so every draw is
// self-contained — no shared uniform buffer, safe to call any number of times
// per frame. The clip variant mirrors ColorPipeline so the overlay respects
// path clips (e.g. the rounded bottom-bar chassis).
class NoisePipeline : public Pipeline {
public:
    NoisePipeline(Device& device, std::span<const uint32_t> vertSpv, std::span<const uint32_t> fragSpv)
        : Pipeline(device)
    {
        loadVertexShader(vertSpv);
        loadFragmentShader(fragSpv);
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
        auto cfg = config();
        cfg.stencilTestEnable = true;
        cfg.stencilCompareOp = VK_COMPARE_OP_EQUAL;
        cfg.stencilFailOp = VK_STENCIL_OP_KEEP;
        cfg.stencilPassOp = VK_STENCIL_OP_KEEP;
        return cfg;
    }

    std::span<const DrawOp> supportedOps() const override
    {
        static constexpr DrawOp ops[] = { DrawOp::EffectNoise };
        return ops;
    }

    void execute(Renderer& r, const Arena& arena, const DrawCommand& cmd) override
    {
        auto& state = r.state();
        auto def = r.caches().defaultDescriptor();
        state.setResources(def, def);
        auto& p = arena.read<NoiseParams>(cmd.dataOffset);
        float x = p.region.getX()     * p.scale;
        float y = p.region.getY()     * p.scale;
        float w = p.region.getWidth()  * p.scale;
        float h = p.region.getHeight() * p.scale;
        // colour.r carries the per-frame time offset (0 = static), colour.a the amount.
        glm::vec4 color(p.timeOffset, 0.0f, 0.0f, p.amount);
        emitQuad(state, cmd, x, y, w, h, color);
    }
};

} // namespace jvk::pipelines
