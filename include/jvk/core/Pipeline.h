#pragma once

namespace jvk {

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

    virtual void defineLayout(Memory::M& bindings);

    // Builds up to two VkPipelines — normal and (optional) clip variant —
    // sharing one layout and one set of shader modules. Both are single-
    // sample. The scene render passes (clear + load variants) are
    // pipeline-compatible, so one build covers both.
    void build(VkRenderPass renderPass);
    bool isBuilt() const { return built_; }

    VkPipeline       handle()     const { return pipeline_; }
    VkPipeline       clipHandle() const { return clipPipeline_ ? clipPipeline_ : pipeline_; }
    VkPipelineLayout layout()     const { return layout_; }

protected:
    Device& device_;

private:
    VkPipeline       pipeline_     = VK_NULL_HANDLE;
    VkPipeline       clipPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_       = VK_NULL_HANDLE;
    std::vector<uint32_t> vertSpirv_;
    std::vector<uint32_t> fragSpirv_;
    bool built_ = false;

    VkPipeline buildVariant(const PipelineConfig& cfg, VkRenderPass renderPass,
                            VkPipelineLayout layout);
};

} // namespace jvk
