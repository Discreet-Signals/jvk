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
 File: Graphics.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

class Graphics
{
public:
    // Factory: returns a jvk::Graphics if the juce::Graphics is backed
    // by our Vulkan renderer. Returns nullptr if using JUCE software renderer.
    static std::unique_ptr<Graphics> create(juce::Graphics& g)
    {
        auto* ctx = dynamic_cast<VulkanGraphicsContext*>(&g.getInternalContext());
        if (!ctx) return nullptr;
        return std::unique_ptr<Graphics>(new Graphics(*ctx, g));
    }

    // =========================================================================
    // juce::Graphics delegation — same API, same behavior
    // =========================================================================

    void fillAll()                                    { g.fillAll(); }
    void fillAll(juce::Colour c)                      { g.fillAll(c); }

    void setColour(juce::Colour c)                    { g.setColour(c); }
    void setOpacity(float o)                          { g.setOpacity(o); }
    void setGradientFill(const juce::ColourGradient& grad) { g.setGradientFill(grad); }

    void fillRect(juce::Rectangle<int> r)             { g.fillRect(r); }
    void fillRect(juce::Rectangle<float> r)           { g.fillRect(r); }
    void fillRect(int x, int y, int w, int h)         { g.fillRect(x, y, w, h); }
    void fillRect(float x, float y, float w, float h) { g.fillRect(x, y, w, h); }

    void fillRoundedRectangle(juce::Rectangle<float> r, float cr) { g.fillRoundedRectangle(r, cr); }
    void fillRoundedRectangle(float x, float y, float w, float h, float cr) { g.fillRoundedRectangle(x, y, w, h, cr); }
    void drawRoundedRectangle(juce::Rectangle<float> r, float cr, float lw) { g.drawRoundedRectangle(r, cr, lw); }

    void fillEllipse(juce::Rectangle<float> r)        { g.fillEllipse(r); }
    void fillEllipse(float x, float y, float w, float h) { g.fillEllipse(x, y, w, h); }
    void drawEllipse(juce::Rectangle<float> r, float lw) { g.drawEllipse(r, lw); }
    void drawEllipse(float x, float y, float w, float h, float lw) { g.drawEllipse(x, y, w, h, lw); }

    void drawLine(float x1, float y1, float x2, float y2)           { g.drawLine(x1, y1, x2, y2); }
    void drawLine(float x1, float y1, float x2, float y2, float lw) { g.drawLine(x1, y1, x2, y2, lw); }
    void drawLine(juce::Line<float> line)             { g.drawLine(line); }
    void drawLine(juce::Line<float> line, float lw)   { g.drawLine(line, lw); }

    void fillPath(const juce::Path& p)                                  { g.fillPath(p); }
    void fillPath(const juce::Path& p, const juce::AffineTransform& t)  { g.fillPath(p, t); }
    void strokePath(const juce::Path& p, const juce::PathStrokeType& s) { g.strokePath(p, s); }
    void strokePath(const juce::Path& p, const juce::PathStrokeType& s,
                     const juce::AffineTransform& t)                     { g.strokePath(p, s, t); }

    void drawText(const juce::String& text, int x, int y, int w, int h,
                   juce::Justification j, bool ellipsis = true) { g.drawText(text, x, y, w, h, j, ellipsis); }
    void drawText(const juce::String& text, juce::Rectangle<int> area,
                   juce::Justification j, bool ellipsis = true) { g.drawText(text, area, j, ellipsis); }
    void drawText(const juce::String& text, juce::Rectangle<float> area,
                   juce::Justification j, bool ellipsis = true) { g.drawText(text, area, j, ellipsis); }
    void drawFittedText(const juce::String& text, int x, int y, int w, int h,
                         juce::Justification j, int maxLines) { g.drawFittedText(text, x, y, w, h, j, maxLines); }

    void setFont(const juce::Font& f)                 { g.setFont(f); }
    void setFont(float height)                        { g.setFont(juce::FontOptions(height)); }

    void drawImage(const juce::Image& img, int x, int y, int w, int h,
                    int sx, int sy, int sw, int sh, bool fillAlpha = false)
    { g.drawImage(img, x, y, w, h, sx, sy, sw, sh, fillAlpha); }
    void drawImageAt(const juce::Image& img, int x, int y, bool fillAlpha = false)
    { g.drawImageAt(img, x, y, fillAlpha); }
    void drawImage(const juce::Image& img, juce::Rectangle<float> dest,
                    juce::RectanglePlacement placement = juce::RectanglePlacement::stretchToFit)
    { g.drawImage(img, dest, placement); }

    void saveState()                                  { g.saveState(); }
    void restoreState()                               { g.restoreState(); }
    void setOrigin(int x, int y)                      { g.setOrigin(x, y); }
    void setOrigin(juce::Point<int> p)                { g.setOrigin(p); }
    void addTransform(const juce::AffineTransform& t) { g.addTransform(t); }

    bool clipRegionIntersects(juce::Rectangle<int> r) { return g.clipRegionIntersects(r); }
    void reduceClipRegion(int x, int y, int w, int h) { g.reduceClipRegion(x, y, w, h); }
    void reduceClipRegion(juce::Rectangle<int> r)     { g.reduceClipRegion(r); }
    void reduceClipRegion(const juce::Path& p, const juce::AffineTransform& t = {})
    { g.reduceClipRegion(p, t); }

    juce::Rectangle<int> getClipBounds() const        { return g.getClipBounds(); }
    bool isClipEmpty() const                          { return g.isClipEmpty(); }

    // =========================================================================
    // GPU Post-Processing Effects
    // =========================================================================

    // =========================================================================
    // GPU Post-Processing — Multiply Blend
    // =========================================================================

    // The single primitive: multiply every existing pixel by (r, g, b).
    // One shader, one pipeline, one blend mode. Everything else is a wrapper.
    void multiplyRGB(float r, float green, float b,
                      juce::Rectangle<float> region = {})
    {
        graphics::drawEffectQuad(ctx, ctx.multiplyPipeline, { r, green, b, 1.0f }, region);
    }

    // Convenience wrappers
    void darken(float amount, juce::Rectangle<float> region = {})
    { float a = juce::jlimit(0.0f, 1.0f, amount); multiplyRGB(1.0f - a, 1.0f - a, 1.0f - a, region); }

    void brighten(float amount, juce::Rectangle<float> region = {})
    { float a = std::max(0.0f, amount); multiplyRGB(1.0f + a, 1.0f + a, 1.0f + a, region); }

    void tint(juce::Colour c, juce::Rectangle<float> region = {})
    { multiplyRGB(c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue(), region); }

    void warmth(float amount, juce::Rectangle<float> region = {})
    { multiplyRGB(1.0f + amount * 0.15f, 1.0f, 1.0f - amount * 0.15f, region); }

    // =========================================================================
    // GPU Post-Processing — HSV (reads framebuffer via input attachment)
    // =========================================================================

    // Multiply existing pixels' HSV values. (1,1,1) = no change.
    // H wraps (so 0.5 shifts hue by 180°), S and V are clamped 0-1.
    void multiplyHSV(float h, float s, float v,
                      juce::Rectangle<float> region = {})
    {
        // HSV multiply goes in vertex color, HSV add goes in shapeInfo
        glm::vec4 mulColor(h, s, v, 1.0f);
        glm::vec4 addValues(0, 0, 0, 0);
        graphics::drawEffectQuad(ctx, ctx.hsvPipeline, mulColor, region, addValues);
    }

    // Add to existing pixels' HSV values. (0,0,0) = no change.
    void addHSV(float h, float s, float v,
                 juce::Rectangle<float> region = {})
    {
        glm::vec4 mulColor(1.0f, 1.0f, 1.0f, 1.0f); // multiply by 1 (no change)
        glm::vec4 addValues(h, s, v, 0);
        graphics::drawEffectQuad(ctx, ctx.hsvPipeline, mulColor, region, addValues);
    }

    // Convenience HSV wrappers
    void saturate(float amount, juce::Rectangle<float> region = {})
    { multiplyHSV(1.0f, amount, 1.0f, region); }

    void desaturate(float amount, juce::Rectangle<float> region = {})
    { multiplyHSV(1.0f, 1.0f - juce::jlimit(0.0f, 1.0f, amount), 1.0f, region); }

    void hueShift(float amount, juce::Rectangle<float> region = {})
    { addHSV(amount, 0, 0, region); }

    // =========================================================================
    // GPU Post-Processing — Future
    // =========================================================================

    // Gaussian blur — requires render pass restart (case 3: two-buffer)
    void blur(juce::Rectangle<float> region, float radius)
    {
        graphics::executeBlur(ctx, region, radius);
    }

    void blur(float radius)
    {
        graphics::executeBlur(ctx, {}, radius);
    }

    void drawShader(Shader& shader, juce::Rectangle<float> region = {})
    {
        graphics::drawShader(ctx, shader, region);
    }

    // =========================================================================
    // Direct GPU Access
    // =========================================================================

    VkCommandBuffer getCommandBuffer() const  { return ctx.cmd; }
    VkDevice getDevice() const                { return ctx.device; }
    VkPhysicalDevice getPhysicalDevice() const { return ctx.physDevice; }
    VkPipelineLayout getPipelineLayout() const { return ctx.pipelineLayout; }
    float getViewportWidth() const            { return ctx.vpWidth; }
    float getViewportHeight() const           { return ctx.vpHeight; }
    float getScale() const                    { return ctx.scale; }

    // Flush pending juce::Graphics draws to GPU before custom Vulkan operations
    void flush()                              { graphics::flush(ctx); }

    // Submit raw vertices directly
    void submitVertices(const UIVertex* verts, int count)
    {
        for (int i = 0; i < count; i++)
            ctx.vertices.push_back(verts[i]);
    }

    // Access the underlying contexts
    VulkanGraphicsContext& getVulkanContext()  { return ctx; }
    juce::Graphics& getJuceGraphics()         { return g; }

private:
    Graphics(VulkanGraphicsContext& ctx, juce::Graphics& g) : ctx(ctx), g(g) {}

    VulkanGraphicsContext& ctx;
    juce::Graphics& g;
};

} // jvk
