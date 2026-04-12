/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 ██████╗  ██╗ ███████╗  ██████╗ ██████╗  ███████╗ ███████╗ ████████╗
 ██╔══██╗ ██║ ██╔════╝ ██╔════╝ ██╔══██╗ ██╔════╝ ██╔════╝ ╚══██╔══╝
 ██║  ██║ ██║ ███████╗ ██║      ██████╔╝ █████╗   █████╗      ██║
 ██║  ██║ ██║ ╚════██║ ██║      ██╔══██╗ ██╔══╝   ██╔══╝      ██║
 ██████╔╝ ██║ ███████║ ╚██████╗ ██║  ██║ ███████╗ ███████╗    ██║
 ╚═════╝  ╚═╝ ╚══════╝  ╚═════╝ ╚═╝  ╚═╝ ╚══════╝ ╚══════╝    ╚═╝

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: VulkanGraphicsContext.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

#include "Types.h"

namespace jvk
{

class VulkanGraphicsContext : public juce::LowLevelGraphicsContext
{
public:
    VulkanGraphicsContext(VkCommandBuffer cmd, VkPipeline colorPipeline,
                          VkPipelineLayout pipelineLayout,
                          float vpWidth, float vpHeight,
                          VkPhysicalDevice physDevice, VkDevice device,
                          float scale = 2.0f,
                          core::Buffer* externalBuffer = nullptr,
                          void** externalMappedPtr = nullptr,
                          Pipeline* stencilPipeline = nullptr,
                          Pipeline* stencilCoverPipeline = nullptr,
                          VkDescriptorSet defaultDescSet = VK_NULL_HANDLE,
                          bool srgb = false,
                          Pipeline* multiplyPipeline = nullptr,
                          Pipeline* hsvPipeline = nullptr,
                          Pipeline* blurPipeline = nullptr,
                          RenderPassInfo rpInfo = {},
                          core::Image* blurTempImage = nullptr,
                          VkDescriptorSet blurDescSet = VK_NULL_HANDLE,
                          TextureCache* textureCache = nullptr,
                          GlyphAtlas* glyphAtlas = nullptr,
                          GradientCache* gradientCache = nullptr,
                          core::DeletionQueue* deletionQ = nullptr,
                          core::VertexRingBuffer* ringBuf = nullptr);

    ~VulkanGraphicsContext() override;

    // Path rendering backend.
    enum class PathBackend { Stencil };
    PathBackend pathBackend = PathBackend::Stencil;

    // ==== State ====
    struct SavedState
    {
        juce::Point<int> origin;     // physical pixels
        juce::AffineTransform transform;
        juce::FillType fillType { juce::Colours::black };
        float opacity = 1.0f;
        juce::Font font { juce::FontOptions {} };
        juce::Rectangle<int> clipBounds; // physical pixels

        // Stencil-based path clipping
        uint8_t stencilClipDepth = 0;    // current nesting depth (0 = no clip)
        std::vector<UIVertex> clipFanVerts; // fan verts to unstencil on restore
    };

    SavedState& state() { return stateStack.back(); }
    const SavedState& state() const { return stateStack.back(); }

    // ==== Query ====
    bool isVectorDevice() const override { return false; }
    uint64_t getFrameId() const override { return frameId++; }
    float getPhysicalPixelScaleFactor() const override { return scale; }

    std::unique_ptr<juce::ImageType> getPreferredImageTypeForTemporaryImages() const
    {
        return std::make_unique<juce::NativeImageType>();
    }

    // ==== LowLevelGraphicsContext overrides (defined after free function includes) ====
    void setOrigin(juce::Point<int> o) override;
    void addTransform(const juce::AffineTransform& t) override;

    bool clipToRectangle(const juce::Rectangle<int>& r) override;
    bool clipToRectangleList(const juce::RectangleList<int>& r) override;
    void excludeClipRectangle(const juce::Rectangle<int>& r) override;
    void clipToPath(const juce::Path& path, const juce::AffineTransform& t) override;
    void clipToImageAlpha(const juce::Image& img, const juce::AffineTransform& t) override;
    bool clipRegionIntersects(const juce::Rectangle<int>& r) override;
    juce::Rectangle<int> getClipBounds() const override;
    bool isClipEmpty() const override;

    void saveState() override;
    void restoreState() override;
    void beginTransparencyLayer(float opacity) override;
    void endTransparencyLayer() override;

    void setFill(const juce::FillType& f) override { state().fillType = f; }
    void setOpacity(float o) override { state().opacity = o; }
    void setInterpolationQuality(juce::Graphics::ResamplingQuality) override {}

    void setFont(const juce::Font& f) override { state().font = f; }
    const juce::Font& getFont() override { return state().font; }

    void fillRoundedRectangle(const juce::Rectangle<float>& r, float cornerSize) override;
    void fillEllipse(const juce::Rectangle<float>& area) override;

    void fillRect(const juce::Rectangle<int>& r, bool) override;
    void fillRect(const juce::Rectangle<float>& r) override;
    void fillRectList(const juce::RectangleList<float>& list) override;

    void fillPath(const juce::Path& path, const juce::AffineTransform& t) override;

    void drawImage(const juce::Image& image, const juce::AffineTransform& t) override;

    void drawLine(const juce::Line<float>& line) override;
    void drawLineWithThickness(const juce::Line<float>& line, float lineThickness) override;

    void drawGlyphs(juce::Span<const uint16_t> glyphs,
                     juce::Span<const juce::Point<float>> positions,
                     const juce::AffineTransform& t) override;

    // ==== Public members (accessible by graphics:: free functions) ====
    VkCommandBuffer cmd;
    VkPipeline colorPipeline;
    VkPipelineLayout pipelineLayout;
    float vpWidth, vpHeight; // physical pixels
    float scale;             // logical->physical multiplier (2.0)
    bool srgbMode;           // true = sRGB pipeline, false = UNORM
    VkPhysicalDevice physDevice;
    VkDevice device;

    std::vector<SavedState> stateStack;
    std::vector<UIVertex> vertices;
    core::VertexRingBuffer* ringBuffer = nullptr;
    core::Buffer* extBuffer = nullptr;
    void** extMappedPtr = nullptr;
    VkDeviceSize bufferWriteOffset = 0;
    Pipeline* stencilPipeline = nullptr;
    Pipeline* stencilCoverPipeline = nullptr;
    TextureCache* textureCache = nullptr;
    GlyphAtlas* glyphAtlas = nullptr;
    GradientCache* gradientCache = nullptr;
    Pipeline* mainClipPipeline = nullptr;  // main pipeline + stencil test for path clipping
    Pipeline* multiplyPipeline = nullptr;
    Pipeline* hsvPipeline = nullptr;
    Pipeline* blurPipeline = nullptr;
    RenderPassInfo rpInfo;
    core::Image* blurTempImage = nullptr;
    VkDescriptorSet blurDescSet = VK_NULL_HANDLE;
    VkDescriptorSet defaultDescriptorSet = VK_NULL_HANDLE;

    VkPipeline boundPipeline = VK_NULL_HANDLE;
    VkDescriptorSet boundDescriptorSet = VK_NULL_HANDLE;

    std::vector<glm::vec2> scratchPoints;
    std::vector<UIVertex> scratchFanVerts;

    core::DeletionQueue* deletionQueue = nullptr;
    mutable uint64_t frameId = 0;

    friend class Graphics;
};

} // jvk

// ============================================================================
// Free function implementations (require full VulkanGraphicsContext definition)
// ============================================================================
#include "Helpers.h"
#include "Flush.h"
#include "Clipping.h"
#include "FillRect.h"
#include "FillShapes.h"
#include "FillPath.h"
#include "DrawImage.h"
#include "DrawLine.h"
#include "DrawGlyphs.h"
#include "Effects.h"

// ============================================================================
// Deferred Clipping implementations (need FillPath.h for writeStencilClip)
// ============================================================================
namespace jvk::graphics
{

inline void clipToPath(VulkanGraphicsContext& ctx, const juce::Path& path, const juce::AffineTransform& t)
{
    auto combined = getFullTransform(ctx, t);

    // Always intersect bounding box for fast scissor rejection
    auto pathBounds = path.getBoundsTransformed(combined).getSmallestIntegerContainer();
    ctx.state().clipBounds = ctx.state().clipBounds.getIntersection(pathBounds);

    // Write the path into the stencil buffer for pixel-accurate clipping.
    auto fanVerts = writeStencilClip(ctx, path, combined);
    if (!fanVerts.empty())
    {
        auto& s = ctx.state();
        s.stencilClipDepth++;
        s.clipFanVerts = std::move(fanVerts);
        vkCmdSetStencilReference(ctx.cmd, VK_STENCIL_FACE_FRONT_AND_BACK, s.stencilClipDepth);
        // Force rebind to the clip-testing pipeline now that depth > 0.
        // writeStencilClip called ensureMainPipeline while depth was still 0,
        // which bound the regular (non-testing) pipeline.
        ctx.boundPipeline = VK_NULL_HANDLE;
        ensureMainPipeline(ctx);
    }
}

inline void restoreState(VulkanGraphicsContext& ctx)
{
    flush(ctx);
    if (ctx.stateStack.size() <= 1) return;

    auto& popped = ctx.stateStack.back();

    // If the popped state had a stencil clip, decrement by re-drawing
    // the same fan with reversed winding (flips increment→decrement).
    if (popped.stencilClipDepth > 0 && !popped.clipFanVerts.empty() && ctx.stencilPipeline)
    {
        auto reversed = popped.clipFanVerts;
        for (size_t i = 0; i + 2 < reversed.size(); i += 3)
            std::swap(reversed[i + 1], reversed[i + 2]);

        VkRect2D fullScissor = { { 0, 0 }, { static_cast<uint32_t>(ctx.vpWidth),
                                              static_cast<uint32_t>(ctx.vpHeight) } };
        vkCmdSetScissor(ctx.cmd, 0, 1, &fullScissor);
        ensurePipeline(ctx, ctx.stencilPipeline->getInternal(), ctx.stencilPipeline->getLayout());
        uploadAndDraw(ctx, reversed.data(), static_cast<uint32_t>(reversed.size()));
        ensureMainPipeline(ctx);
    }

    ctx.stateStack.pop_back();

    vkCmdSetStencilReference(ctx.cmd, VK_STENCIL_FACE_FRONT_AND_BACK,
                              ctx.state().stencilClipDepth);
    // Force rebind — may need to switch between clip and regular pipeline
    ctx.boundPipeline = VK_NULL_HANDLE;
    ensureMainPipeline(ctx);
}

} // jvk::graphics

// ============================================================================
// VulkanGraphicsContext method definitions (delegate to free functions above)
// ============================================================================
namespace jvk
{

inline VulkanGraphicsContext::VulkanGraphicsContext(
    VkCommandBuffer cmd, VkPipeline colorPipeline,
    VkPipelineLayout pipelineLayout,
    float vpWidth, float vpHeight,
    VkPhysicalDevice physDevice, VkDevice device,
    float scale,
    core::Buffer* externalBuffer, void** externalMappedPtr,
    Pipeline* stencilPipeline, Pipeline* stencilCoverPipeline,
    VkDescriptorSet defaultDescSet, bool srgb,
    Pipeline* multiplyPipeline, Pipeline* hsvPipeline,
    Pipeline* blurPipeline, RenderPassInfo rpInfo,
    core::Image* blurTempImage, VkDescriptorSet blurDescSet,
    TextureCache* textureCache, GlyphAtlas* glyphAtlas,
    GradientCache* gradientCache, core::DeletionQueue* deletionQ,
    core::VertexRingBuffer* ringBuf)
    : cmd(cmd), colorPipeline(colorPipeline), pipelineLayout(pipelineLayout),
      vpWidth(vpWidth), vpHeight(vpHeight), scale(scale), srgbMode(srgb),
      physDevice(physDevice), device(device),
      ringBuffer(ringBuf),
      extBuffer(externalBuffer), extMappedPtr(externalMappedPtr),
      stencilPipeline(stencilPipeline), stencilCoverPipeline(stencilCoverPipeline),
      multiplyPipeline(multiplyPipeline), hsvPipeline(hsvPipeline),
      blurPipeline(blurPipeline), rpInfo(rpInfo), blurTempImage(blurTempImage), blurDescSet(blurDescSet),
      defaultDescriptorSet(defaultDescSet), textureCache(textureCache),
      glyphAtlas(glyphAtlas), gradientCache(gradientCache),
      deletionQueue(deletionQ)
{
    stateStack.push_back({});
    stateStack.back().clipBounds = juce::Rectangle<int>(0, 0,
        static_cast<int>(vpWidth), static_cast<int>(vpHeight));
    graphics::ensureMainPipeline(*this);
}

inline VulkanGraphicsContext::~VulkanGraphicsContext() { graphics::flush(*this); }

// Transform
inline void VulkanGraphicsContext::setOrigin(juce::Point<int> o) { graphics::setOrigin(*this, o); }
inline void VulkanGraphicsContext::addTransform(const juce::AffineTransform& t) { graphics::addTransform(*this, t); }

// Clipping
inline bool VulkanGraphicsContext::clipToRectangle(const juce::Rectangle<int>& r) { return graphics::clipToRectangle(*this, r); }
inline bool VulkanGraphicsContext::clipToRectangleList(const juce::RectangleList<int>& r) { return graphics::clipToRectangleList(*this, r); }
inline void VulkanGraphicsContext::excludeClipRectangle(const juce::Rectangle<int>& r) { graphics::excludeClipRectangle(*this, r); }
inline void VulkanGraphicsContext::clipToPath(const juce::Path& path, const juce::AffineTransform& t) { graphics::clipToPath(*this, path, t); }
inline void VulkanGraphicsContext::clipToImageAlpha(const juce::Image& img, const juce::AffineTransform& t) { graphics::clipToImageAlpha(*this, img, t); }
inline bool VulkanGraphicsContext::clipRegionIntersects(const juce::Rectangle<int>& r) { return graphics::clipRegionIntersects(*this, r); }
inline juce::Rectangle<int> VulkanGraphicsContext::getClipBounds() const { return graphics::getClipBounds(*this); }
inline bool VulkanGraphicsContext::isClipEmpty() const { return graphics::isClipEmpty(*this); }

// State stack
inline void VulkanGraphicsContext::saveState() { graphics::saveState(*this); }
inline void VulkanGraphicsContext::restoreState() { graphics::restoreState(*this); }
inline void VulkanGraphicsContext::beginTransparencyLayer(float opacity) { graphics::beginTransparencyLayer(*this, opacity); }
inline void VulkanGraphicsContext::endTransparencyLayer() { graphics::endTransparencyLayer(*this); }

// SDF shape overrides
inline void VulkanGraphicsContext::fillRoundedRectangle(const juce::Rectangle<float>& r, float cornerSize)
{
    if (graphics::hasNonTrivialTransform(*this))
    {
        juce::LowLevelGraphicsContext::fillRoundedRectangle(r, cornerSize);
        return;
    }
    graphics::fillRoundedRectangle(*this, r, cornerSize);
}

inline void VulkanGraphicsContext::fillEllipse(const juce::Rectangle<float>& area)
{
    if (graphics::hasNonTrivialTransform(*this))
    {
        juce::LowLevelGraphicsContext::fillEllipse(area);
        return;
    }
    graphics::fillEllipse(*this, area);
}

// Drawing
inline void VulkanGraphicsContext::fillRect(const juce::Rectangle<int>& r, bool) { graphics::fillRect(*this, r.toFloat()); }
inline void VulkanGraphicsContext::fillRect(const juce::Rectangle<float>& r) { graphics::fillRect(*this, r); }
inline void VulkanGraphicsContext::fillRectList(const juce::RectangleList<float>& list) { graphics::fillRectList(*this, list); }
inline void VulkanGraphicsContext::fillPath(const juce::Path& path, const juce::AffineTransform& t) { graphics::fillPath(*this, path, t); }
inline void VulkanGraphicsContext::drawImage(const juce::Image& image, const juce::AffineTransform& t) { graphics::drawImage(*this, image, t); }
inline void VulkanGraphicsContext::drawLine(const juce::Line<float>& line) { graphics::drawLine(*this, line); }
inline void VulkanGraphicsContext::drawLineWithThickness(const juce::Line<float>& line, float lineThickness) { graphics::drawLineWithThickness(*this, line, lineThickness); }

inline void VulkanGraphicsContext::drawGlyphs(juce::Span<const uint16_t> glyphs,
                                               juce::Span<const juce::Point<float>> positions,
                                               const juce::AffineTransform& t)
{
    graphics::drawGlyphs(*this, glyphs, positions, t);
}

} // jvk
