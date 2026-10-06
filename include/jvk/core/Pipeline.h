#pragma once

namespace jvk {

// The VkPipelines of one built-in pipeline, one per (draw target, clip, the
// targets its paint clears) — see DrawTargets: the main-colour defaults built
// up front, the others on first use. Shared by Pipeline and PathPipeline;
// owns the handles.
class PipelineVariants
{
public:
    // The pipeline for a draw into `attachment` (0 = main colour, i + 1 =
    // target i of `targetCount`), the stencil-clip variant when `clip`,
    // clearing the targets in `clears` (paintClears); built by `build()` the
    // first time it is asked for. Null for a target the pass doesn't have. A
    // failed build is remembered (as null) so it is never retried per draw.
    template <typename Build>
    VkPipeline get(uint8_t attachment, bool clip, uint8_t clears, size_t targetCount, Build&& build)
    {
        for (auto& v : variants_)
            if (v.attachment == attachment && v.clip == clip && v.clears == clears)
                return v.handle;
        if (attachment > targetCount) return VK_NULL_HANDLE;
        VkPipeline h = build();
        add(attachment, clip, clears, h);
        return h;
    }

    // A variant built up front (the defaults).
    void add(uint8_t attachment, bool clip, uint8_t clears, VkPipeline handle)
    {
        variants_.push_back({ attachment, clip, clears, handle });
    }

    void destroy(VkDevice d)
    {
        for (auto& v : variants_)
            if (v.handle != VK_NULL_HANDLE) vkDestroyPipeline(d, v.handle, nullptr);
        variants_.clear();
    }

private:
    struct Variant { uint8_t attachment; bool clip; uint8_t clears; VkPipeline handle; };
    std::vector<Variant> variants_;
};

class Pipeline {
public:
    Pipeline(Device& device) : device_(device) {}
    virtual ~Pipeline();

    Pipeline(Pipeline&&) noexcept;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    virtual PipelineConfig config() const = 0;
    virtual std::optional<PipelineConfig> clipConfig() const { return std::nullopt; }
    virtual std::span<const DrawOp> supportedOps() const = 0;
    virtual void execute(Renderer& r, const Arena& arena, const DrawCommand& cmd) = 0;

    // Called once per frame before the render pass. Pipelines stage pending
    // uploads (atlas pages, deferred textures) here — push to the Renderer's
    // upload queue, NOT Device's, so two editors' workers never share the
    // queue. Default is no-op.
    virtual void prepare(Renderer& r) { (void)r; }

    void loadVertexShader(std::span<const uint32_t> spirv);
    void loadFragmentShader(std::span<const uint32_t> spirv);


    // Builds the layout and the default VkPipelines — normal and (optional)
    // clip variant, drawing into the main colour — sharing one set of shader
    // modules. Single-sample. The scene render passes (clear + load variants)
    // are pipeline-compatible, so one build covers both. `targets` are the
    // render targets the pass carries besides the main colour; how a draw
    // lands on them is DrawTargets (PipelineConfig.h).
    void build(VkRenderPass renderPass, const std::vector<Target>& targets = {});
    bool isBuilt() const { return built_; }

    // The VkPipeline for a draw into colour attachment `attachment` (0 = main
    // colour, i + 1 = target i) while the targets in `live` are still read
    // later this frame (Renderer::liveTargets). Variants other than the
    // defaults are built on first use, on the render worker. The built-in
    // fragment shaders write the same colour to every output location; a
    // variant enables writes on its one attachment only (plus, for paint into
    // the main colour, the live targets it clears).
    VkPipeline       handle    (uint8_t attachment = 0, uint8_t live = 0xFF) { return variant(attachment, false, live); }
    VkPipeline       clipHandle(uint8_t attachment = 0, uint8_t live = 0xFF) { return variant(attachment, true, live); }
    VkPipelineLayout layout()     const { return layout_; }

protected:
    Device& device_;

    // A pipeline whose fragment shader is PAINT (ui2d.frag: it writes every
    // output location and takes PaintSpecialization): its draws into the main
    // colour clear the targets declared Target::clearedByMain. Set before build().
    bool paintClearsTargets_ = false;

private:
    PipelineVariants variants_;
    VkPipelineLayout layout_       = VK_NULL_HANDLE;
    VkRenderPass     renderPass_   = VK_NULL_HANDLE;
    std::vector<Target> targets_;        // the pass's render targets
    std::vector<uint32_t> vertSpirv_;
    std::vector<uint32_t> fragSpirv_;
    bool built_ = false;
    bool hasClip_ = false;

    VkPipeline variant(uint8_t attachment, bool clip, uint8_t live);
    VkPipeline buildVariant(const PipelineConfig& cfg, VkRenderPass renderPass,
                            VkPipelineLayout layout, uint8_t attachment, uint8_t clears);
};

} // namespace jvk
