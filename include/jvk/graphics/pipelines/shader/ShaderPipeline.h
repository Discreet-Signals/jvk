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

    void init(Device& device, VkRenderPass sceneRenderPass,
              VkSampleCountFlagBits msaa = VK_SAMPLE_COUNT_1_BIT)
    {
        device_     = &device;
        renderPass_ = sceneRenderPass;
        msaa_       = msaa;
    }

    bool ready() const { return device_ != nullptr; }

    // Dispatch one DrawShader command inside the active scene render pass.
    // `regionPixels` is the physical-pixel region the shader fills (the
    // fullscreen triangle is clipped to it via the vertex shader's regionX/Y
    // + resolution push constants, and further by the scissor).
    //
    // `stencilDepth` mirrors the nesting depth of active path clips. When
    // non-zero we bind the stencil-tested pipeline variant and configure the
    // dynamic reference + compare mask to match all currently-set bits —
    // identical to ColorPipeline's clip path, so DrawShader composes correctly
    // inside clipToPath / clipToRectangle scopes.
    // `frameTime` is the per-frame snapshot of jvk::Device::time() taken once
    // by Renderer::execute() and passed unchanged to every dispatch this frame.
    // Lands in the built-in `time` push-constant slot every shader sees.
    void dispatch(State& state, VkCommandBuffer cmd,
                  Shader& shader,
                  juce::Rectangle<float> regionPixels,
                  float viewportW, float viewportH,
                  const juce::Rectangle<int>& clipBoundsPixels,
                  uint8_t stencilDepth,
                  float frameTime)
    {
        if (!device_) return;
        shader.ensureCreated(*device_, renderPass_, msaa_);
        if (!shader.isReady()) return;

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

        // Scissor — clip bounds clamped to the framebuffer. Matches the
        // convention used by State::draw so subsequent scene draws don't
        // inherit an unexpected scissor.
        VkRect2D sc {};
        sc.offset = { std::max(0, clipBoundsPixels.getX()),
                      std::max(0, clipBoundsPixels.getY()) };
        sc.extent = { static_cast<uint32_t>(std::max(0, clipBoundsPixels.getWidth())),
                      static_cast<uint32_t>(std::max(0, clipBoundsPixels.getHeight())) };
        vkCmdSetScissor(cmd, 0, 1, &sc);

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

        // Bind the shader's descriptor set only if it actually has bindings.
        // Fragment-only shaders (push-constant-driven) leave descriptorSet()
        // VK_NULL_HANDLE and use a zero-set pipeline layout.
        VkDescriptorSet set = shader.descriptorSet();
        if (set != VK_NULL_HANDLE) {
            // Refresh the GPU-visible uniform/storage buffer from the shader's
            // CPU shadow before binding — this is what makes set(name, value)
            // actually reach the GPU each draw. memcpy is cheap (a handful of
            // bytes typically) and the memory is HOST_COHERENT so no flush is
            // needed before the descriptor read.
            if (void* dst = shader.uniformMapped()) {
                std::memcpy(dst, shader.uniformData(), shader.uniformSize());
            }

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                shader.layout(), 0, 1, &set, 0, nullptr);
        }

        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

private:
    Device*               device_     = nullptr;
    VkRenderPass          renderPass_ = VK_NULL_HANDLE;
    VkSampleCountFlagBits msaa_       = VK_SAMPLE_COUNT_1_BIT;
};

} // namespace jvk
