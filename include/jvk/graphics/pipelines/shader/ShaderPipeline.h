#pragma once

namespace jvk {

// =============================================================================
// ShaderPipeline — dispatcher for DrawOp::DrawShader.
//
// User shaders own their own VkPipeline + layout (lazy-built inside
// jvk::Shader the first time they render, against the scene render pass
// captured at init time here). This module's job is the per-command wiring:
//
//   1. Lazy-init the user Shader (ensureCreated) against the scene render pass.
//   2. Bind the shader's pipeline via State::setCustomPipeline so the state
//      tracker knows the normal pipeline cache is stale.
//   3. Set scissor/viewport and push constants matching shader_region.vert's
//      layout (resolution, time, viewport, region origin — all physical px).
//   4. Bind the shader's descriptor set (only if it has any reflected bindings).
//   5. Issue one fullscreen triangle draw.
//
// Registered with the Renderer via setShaderPipeline(); execute() hands off
// DrawShader commands to dispatch() while the scene render pass is active.
// =============================================================================

class ShaderPipeline {
public:
    ShaderPipeline() = default;

    // `targets`: the render targets of the scene pass (user shaders write
    // the ones whose output locations they declare). `passRenderPass`: the
    // single-colour pass dispatchPass() draws in (the effect render pass),
    // where shaders that READ targets run.
    void init(Device& device, VkRenderPass sceneRenderPass,
              VkSampleCountFlagBits msaa = VK_SAMPLE_COUNT_1_BIT,
              std::vector<Target> targets = {},
              VkRenderPass passRenderPass = VK_NULL_HANDLE)
    {
        device_     = &device;
        renderPass_ = sceneRenderPass;
        msaa_       = msaa;
        targets_   = std::move(targets);
        passRenderPass_ = passRenderPass;
    }

    // Draws `shader` over `regionPixels` into the colour attachment of an
    // effect render pass the caller has begun, outside the scene pass: how a
    // shader that reads targets runs (Shader::targetsRead). The pass carries
    // the shared stencil, so path clips still apply. Such a shader is built
    // against that pass on first use and is only ever drawn this way.
    // `sceneSet`: for a scene reader (Shader::readsScene), the descriptor set of
    // the scene half it reads, bound as its set 1.
    void dispatchPass(VkCommandBuffer cmd, Shader& shader,
                      juce::Rectangle<float> regionPixels,
                      float viewportW, float viewportH,
                      const juce::Rectangle<int>& clipBoundsPixels,
                      uint8_t stencilDepth,
                      float frameTime, int frameSlot,
                      std::span<const float> drawConstants = {},
                      VkDescriptorSet sceneSet = VK_NULL_HANDLE)
    {
        if (!device_ || passRenderPass_ == VK_NULL_HANDLE) return;
        shader.ensureCreated(*device_, passRenderPass_, VK_SAMPLE_COUNT_1_BIT);
        if (!shader.isReady()) return;
        jassert (shader.isBuiltFor ({}));   // built for this (target-less) pass: see Shader::isBuiltFor

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          stencilDepth > 0 ? shader.clipPipeline() : shader.pipeline());
        if (stencilDepth > 0)
            vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencilDepth);

        // Scissor: the clip bounds within the region, edge-clamped as in dispatch().
        const auto clip = clipBoundsPixels.getIntersection(regionPixels.getSmallestIntegerContainer());
        const int x0 = std::max(0, clip.getX());
        const int y0 = std::max(0, clip.getY());
        const int x1 = std::max(x0, clip.getRight());
        const int y1 = std::max(y0, clip.getBottom());
        const VkRect2D scissor { { x0, y0 }, { static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0) } };
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        VkViewport vp {};
        vp.width    = viewportW;
        vp.height   = viewportH;
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);

        const float push[7] = {
            regionPixels.getWidth(), regionPixels.getHeight(), frameTime,
            viewportW, viewportH, regionPixels.getX(), regionPixels.getY(),
        };
        vkCmdPushConstants(cmd, shader.layout(),
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), push);
        pushDrawConstants(cmd, shader, drawConstants);

        VkDescriptorSet own = shader.descriptorSet(frameSlot);
        if (own != VK_NULL_HANDLE) {
            if (void* dst = shader.uniformMapped(frameSlot))
                std::memcpy(dst, shader.uniformData(), shader.uniformSize());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                shader.layout(), 0, 1, &own, 0, nullptr);
        }
        if (shader.readsScene()) {
            if (sceneSet == VK_NULL_HANDLE) return;   // nothing to read: no draw
            // Set 1 = the scene; set 0, if the shader has none of its own, the same set
            // (its layout is the scene's: Shader::ensureCreated).
            VkDescriptorSet sets[2] = { own != VK_NULL_HANDLE ? own : sceneSet, sceneSet };
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                shader.layout(), 0, 2, sets, 0, nullptr);
        }
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    bool ready() const { return device_ != nullptr; }

    // Dispatch one DrawShader command inside the active scene render pass.
    // `regionPixels` is the physical-pixel region the shader fills. The vertex
    // shader places ONE triangle at regionX/Y that covers twice the region
    // (fragUV runs to 2), so the SCISSOR is what limits it to the region: the
    // clip bounds within the region. A shader never runs outside its region,
    // where fragUV leaves 0..1.
    //
    // `stencilDepth` mirrors the nesting depth of active path clips. When
    // non-zero we bind the stencil-tested pipeline variant and configure the
    // dynamic reference + compare mask to match all currently-set bits —
    // identical to ColorPipeline's clip path, so DrawShader composes correctly
    // inside clipToPath / clipToRectangle scopes.
    // `frameTime` is the per-frame snapshot of jvk::Device::time() taken once
    // by Renderer::execute() and passed unchanged to every dispatch this frame.
    // Lands in the built-in `time` push-constant slot every shader sees.
    // `frameSlot` is the frame-in-flight slot being recorded: it picks the
    // shader's descriptor set and buffer slice for this frame.
    void dispatch(State& state, VkCommandBuffer cmd,
                  Shader& shader,
                  juce::Rectangle<float> regionPixels,
                  float viewportW, float viewportH,
                  const juce::Rectangle<int>& clipBoundsPixels,
                  uint8_t stencilDepth,
                  float frameTime,
                  int frameSlot,
                  std::span<const float> drawConstants = {})
    {
        if (!device_) return;
        shader.ensureCreated(*device_, renderPass_, msaa_, targets_);
        if (!shader.isReady()) return;
        jassert (shader.isBuiltFor (targets_));   // built for this frame's targets: see Shader::isBuiltFor

        const bool useClipVariant = stencilDepth > 0;

        // Custom pipeline: invalidates the normal pipeline cache in State so
        // the next setPipeline() re-binds and re-pushes viewport constants.
        state.setCustomPipeline(useClipVariant ? shader.clipPipeline()
                                               : shader.pipeline(),
                                shader.layout());

        if (useClipVariant) {
            // Stencil reference = current clip depth. compareOp=EQUAL in
            // the pipeline's clip variant passes only where stencil buffer
            // == depth, i.e. inside every active clip.
            vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencilDepth);
        }

        // Scissor — clip bounds within the region (the triangle covers twice
        // the region: this is what keeps it there), clamped to the framebuffer,
        // EDGE-clamped the same way State::draw does it: clamp each edge, then
        // derive the extent from the clamped edges. The old form clamped offset
        // but kept the un-clipped width, so a clip starting at x = -40 produced
        // offset 0 + full width — leaking 40 px past the clip's right edge.
        {
            const auto clip = clipBoundsPixels.getIntersection(regionPixels.getSmallestIntegerContainer());
            const int x0 = std::max(0, clip.getX());
            const int y0 = std::max(0, clip.getY());
            const int x1 = std::max(x0, clip.getRight());
            const int y1 = std::max(y0, clip.getBottom());
            VkRect2D sc {};
            sc.offset = { x0, y0 };
            sc.extent = { static_cast<uint32_t>(x1 - x0),
                          static_cast<uint32_t>(y1 - y0) };
            vkCmdSetScissor(cmd, 0, 1, &sc);
        }

        // Viewport covers the whole framebuffer — the shader_region vertex
        // shader places the triangle at `regionX/Y` in pixel space.
        VkViewport vp {};
        vp.width    = viewportW;
        vp.height   = viewportH;
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);

        // Built-in shader inputs in the shared 7-float push block (matches
        // shader_region.vert): user shaders read `resolution` and `time` for
        // free without declaring any uniform. `time` is the per-frame snapshot
        // captured once by Renderer::execute() so all draws this frame agree.
        const float push[7] = {
            regionPixels.getWidth(),  regionPixels.getHeight(),   // resolution (frag + vert)
            frameTime,                                            // time       (frag + vert)
            viewportW,               viewportH,                   // viewport   (vert only)
            regionPixels.getX(),     regionPixels.getY(),         // region XY  (vert only)
        };
        vkCmdPushConstants(cmd, shader.layout(),
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(push), push);
        pushDrawConstants(cmd, shader, drawConstants);

        // Bind the shader's descriptor set only if it actually has bindings.
        // Fragment-only shaders (push-constant-driven) leave descriptorSet()
        // VK_NULL_HANDLE and use a zero-set pipeline layout.
        VkDescriptorSet set = shader.descriptorSet(frameSlot);
        if (set != VK_NULL_HANDLE) {
            // Refresh this frame slot's slice of the GPU-visible uniform/
            // storage buffer from the shader's CPU shadow before binding —
            // this is what makes set(name, value) actually reach the GPU each
            // draw. The memory is HOST_COHERENT so no flush is needed before
            // the descriptor read, and the slice belongs to a slot whose
            // previous frame the worker has already waited out.
            if (void* dst = shader.uniformMapped(frameSlot)) {
                std::memcpy(dst, shader.uniformData(), shader.uniformSize());
            }

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                shader.layout(), 0, 1, &set, 0, nullptr);
        }

        vkCmdDraw(cmd, 3, 1, 0, 0);

        // We wrote scissor + stencil ref behind State's back — invalidate so
        // the next scene draw re-establishes its own state instead of
        // inheriting ours. ClipPipeline and PathPipeline both do this; this
        // dispatcher was the one of the three that forgot.
        state.invalidate();
    }

private:
    // The draw's own constants (Shader::setDrawConstants), after jvk's floats.
    static void pushDrawConstants(VkCommandBuffer cmd, Shader& shader, std::span<const float> values)
    {
        if (values.empty()) return;
        vkCmdPushConstants(cmd, shader.layout(),
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            Shader::kDrawConstantsOffset, static_cast<uint32_t>(values.size_bytes()), values.data());
    }

    Device*               device_     = nullptr;
    VkRenderPass          renderPass_ = VK_NULL_HANDLE;
    VkRenderPass          passRenderPass_ = VK_NULL_HANDLE;
    VkSampleCountFlagBits msaa_       = VK_SAMPLE_COUNT_1_BIT;
    std::vector<Target>  targets_;
};

} // namespace jvk
