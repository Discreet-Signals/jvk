/*
  ==============================================================================

    VulkanGraphicsContext.h
    Vulkan-backed juce::LowLevelGraphicsContext.

    All incoming JUCE coordinates are in logical points. This context scales
    them by `scale` (2.0) to physical pixels matching the 2x swapchain.
    EdgeTable scanline decomposition handles all shapes and text.

  ==============================================================================
*/

#pragma once

namespace jvk
{

// Render pass info needed for blur (end/restart render pass)
struct RenderPassInfo
{
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkExtent2D extent = {};
    VkImage msaaImage = VK_NULL_HANDLE;  // source for copy
    VkImageView msaaImageView = VK_NULL_HANDLE;
};

struct UIVertex
{
    glm::vec2 position;    // screen pixel coords
    glm::vec4 color;       // linearized RGBA
    glm::vec2 uv;          // 0-1 within quad (for SDF evaluation)
    glm::vec4 shapeInfo;   // x=type(0=flat,1=roundedRect,2=ellipse), y=halfW, z=halfH, w=param
};

class VulkanGraphicsContext : public juce::LowLevelGraphicsContext
{
public:
    // vpWidth/vpHeight are in PHYSICAL PIXELS (matching the swapchain)
    // scale is logical→physical multiplier (e.g. 2.0 for 2x supersampling)
    VulkanGraphicsContext(VkCommandBuffer cmd, VkPipeline colorPipeline,
                          VkPipelineLayout pipelineLayout,
                          float vpWidth, float vpHeight,
                          VkPhysicalDevice physDevice, VkDevice device,
                          float scale = 2.0f,
                          core::Buffer* externalBuffer = nullptr,
                          void** externalMappedPtr = nullptr,
                          Pipeline* stencilPipeline = nullptr,
                          VkDescriptorSet defaultDescSet = VK_NULL_HANDLE,
                          bool srgb = false,
                          Pipeline* multiplyPipeline = nullptr,
                          Pipeline* hsvPipeline = nullptr,
                          Pipeline* blurPipeline = nullptr,
                          RenderPassInfo rpInfo = {},
                          core::Image* blurTempImage = nullptr,
                          VkDescriptorSet blurDescSet = VK_NULL_HANDLE)
        : cmd(cmd), colorPipeline(colorPipeline), pipelineLayout(pipelineLayout),
          vpWidth(vpWidth), vpHeight(vpHeight), scale(scale), srgbMode(srgb),
          physDevice(physDevice), device(device),
          extBuffer(externalBuffer), extMappedPtr(externalMappedPtr),
          stencilPipeline(stencilPipeline), multiplyPipeline(multiplyPipeline), hsvPipeline(hsvPipeline),
          blurPipeline(blurPipeline), rpInfo(rpInfo), blurTempImage(blurTempImage), blurDescSet(blurDescSet),
          defaultDescriptorSet(defaultDescSet)
    {
        stateStack.push_back({});
        // Clip bounds in physical pixels
        stateStack.back().clipBounds = juce::Rectangle<int>(0, 0,
            static_cast<int>(vpWidth), static_cast<int>(vpHeight));
    }

    ~VulkanGraphicsContext() override { flush(); }

    // ==== Query ====
    bool isVectorDevice() const override { return false; }
    uint64_t getFrameId() const override { return frameId++; }
    float getPhysicalPixelScaleFactor() const override { return scale; }

    std::unique_ptr<juce::ImageType> getPreferredImageTypeForTemporaryImages() const override
    {
        return std::make_unique<juce::NativeImageType>();
    }

    // ==== Transform ====
    // JUCE passes logical point offsets. Scale to physical pixels.
    void setOrigin(juce::Point<int> o) override
    {
        state().origin += juce::Point<int>(
            static_cast<int>(o.x * scale),
            static_cast<int>(o.y * scale));
    }

    void addTransform(const juce::AffineTransform& t) override
    {
        state().transform = state().transform.followedBy(t);
    }

    // ==== Clipping ====
    // JUCE passes logical point rects. Scale to physical pixels.
    bool clipToRectangle(const juce::Rectangle<int>& r) override
    {
        auto& s = state();
        auto scaled = scaleRect(r);
        s.clipBounds = s.clipBounds.getIntersection(scaled.translated(s.origin.x, s.origin.y));
        return !s.clipBounds.isEmpty();
    }

    bool clipToRectangleList(const juce::RectangleList<int>& r) override
    {
        return clipToRectangle(r.getBounds());
    }

    void excludeClipRectangle(const juce::Rectangle<int>& r) override
    {
        auto& s = state();
        auto excluded = scaleRect(r).translated(s.origin.x, s.origin.y);
        if (s.clipBounds.intersects(excluded))
        {
            if (excluded.getX() <= s.clipBounds.getX() && excluded.getRight() >= s.clipBounds.getRight())
            {
                if (excluded.getY() <= s.clipBounds.getY())
                    s.clipBounds.setTop(excluded.getBottom());
                else if (excluded.getBottom() >= s.clipBounds.getBottom())
                    s.clipBounds.setBottom(excluded.getY());
            }
        }
    }

    void clipToPath(const juce::Path& path, const juce::AffineTransform& t) override
    {
        auto& s = state();
        auto combined = t.scaled(scale).translated(
            static_cast<float>(s.origin.x), static_cast<float>(s.origin.y));
        auto pathBounds = path.getBoundsTransformed(combined).getSmallestIntegerContainer();
        s.clipBounds = s.clipBounds.getIntersection(pathBounds);
    }

    void clipToImageAlpha(const juce::Image&, const juce::AffineTransform&) override {}

    bool clipRegionIntersects(const juce::Rectangle<int>& r) override
    {
        auto scaled = scaleRect(r);
        return state().clipBounds.intersects(scaled.translated(state().origin.x, state().origin.y));
    }

    // Return logical point bounds (JUCE expects this)
    juce::Rectangle<int> getClipBounds() const override
    {
        auto physBounds = state().clipBounds.translated(-state().origin.x, -state().origin.y);
        float inv = 1.0f / scale;
        return juce::Rectangle<int>(
            static_cast<int>(physBounds.getX() * inv),
            static_cast<int>(physBounds.getY() * inv),
            static_cast<int>(std::ceil(physBounds.getWidth() * inv)),
            static_cast<int>(std::ceil(physBounds.getHeight() * inv)));
    }

    bool isClipEmpty() const override { return state().clipBounds.isEmpty(); }

    // ==== State stack ====
    void saveState() override { stateStack.push_back(stateStack.back()); }
    void restoreState() override { if (stateStack.size() > 1) stateStack.pop_back(); }

    // ==== Transparency layers ====
    void beginTransparencyLayer(float opacity) override
    {
        saveState();
        state().opacity *= opacity;
    }
    void endTransparencyLayer() override { restoreState(); }

    // ==== Fill style ====
    void setFill(const juce::FillType& f) override { state().fillType = f; }
    void setOpacity(float o) override { state().opacity = o; }
    void setInterpolationQuality(juce::Graphics::ResamplingQuality) override {}

    // ==== Font ====
    void setFont(const juce::Font& f) override { state().font = f; }
    const juce::Font& getFont() override { return state().font; }

    // ==== SDF Shape Overrides (bypass fillPath/EdgeTable) ====
    void fillRoundedRectangle(const juce::Rectangle<float>& r, float cornerSize) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto phys = juce::Rectangle<float>(r.getX() * scale, r.getY() * scale,
                                            r.getWidth() * scale, r.getHeight() * scale);
        auto tr = phys.translated(static_cast<float>(s.origin.x), static_cast<float>(s.origin.y));
        float hw = tr.getWidth() * 0.5f, hh = tr.getHeight() * 0.5f;
        addSDFQuad(tr.getX(), tr.getY(), tr.getWidth(), tr.getHeight(),
                    getColorForFill(), 1.0f, hw, hh, cornerSize * scale);
    }

    void fillEllipse(const juce::Rectangle<float>& area) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto phys = juce::Rectangle<float>(area.getX() * scale, area.getY() * scale,
                                            area.getWidth() * scale, area.getHeight() * scale);
        auto tr = phys.translated(static_cast<float>(s.origin.x), static_cast<float>(s.origin.y));
        float hw = tr.getWidth() * 0.5f, hh = tr.getHeight() * 0.5f;
        addSDFQuad(tr.getX(), tr.getY(), tr.getWidth(), tr.getHeight(),
                    getColorForFill(), 2.0f, hw, hh, 0.0f);
    }

    // ==== Drawing: fillRect ====
    // JUCE passes logical point rects. Scale to physical pixels.
    void fillRect(const juce::Rectangle<int>& r, bool) override { fillRect(r.toFloat()); }

    void fillRect(const juce::Rectangle<float>& r) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto phys = juce::Rectangle<float>(r.getX() * scale, r.getY() * scale,
                                            r.getWidth() * scale, r.getHeight() * scale);
        auto translated = phys.translated(static_cast<float>(s.origin.x),
                                           static_cast<float>(s.origin.y));

        if (s.fillType.isGradient())
        {
            // GPU gradient: compute colors at 4 corners, hardware interpolates
            float x = translated.getX(), y = translated.getY();
            float w = translated.getWidth(), h = translated.getHeight();
            auto c00 = getColorAt(x,     y);
            auto c10 = getColorAt(x + w, y);
            auto c11 = getColorAt(x + w, y + h);
            auto c01 = getColorAt(x,     y + h);
            glm::vec4 flat(0);
            addVertex(x,     y,     c00, 0, 0, flat);
            addVertex(x + w, y,     c10, 0, 0, flat);
            addVertex(x + w, y + h, c11, 0, 0, flat);
            addVertex(x,     y,     c00, 0, 0, flat);
            addVertex(x + w, y + h, c11, 0, 0, flat);
            addVertex(x,     y + h, c01, 0, 0, flat);
        }
        else
        {
            addQuad(translated.getX(), translated.getY(),
                    translated.getWidth(), translated.getHeight(), getColorForFill());
        }
    }

    void fillRectList(const juce::RectangleList<float>& list) override
    {
        for (auto& r : list) fillRect(r);
    }

    // ==== Drawing: fillPath ====
    void fillPath(const juce::Path& path, const juce::AffineTransform& t) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto combined = t.scaled(scale).translated(
            static_cast<float>(s.origin.x), static_cast<float>(s.origin.y));

        // TODO: GPU stencil path rendering (requires D32_SFLOAT_S8_UINT format)
        // if (stencilPipeline != nullptr) { fillPathStencil(path, combined); return; }

        // EdgeTable path rendering
        juce::EdgeTable et(s.clipBounds, path, combined);

        if (s.fillType.isGradient())
        {
            GradientEdgeTableCallback callback(*this);
            et.iterate(callback);
        }
        else
        {
            SolidEdgeTableCallback callback(*this, getColorForFill());
            et.iterate(callback);
        }
    }

    void fillPathStencil(const juce::Path& path, const juce::AffineTransform& combined)
    {
        // Step 1: Flatten path to line segments
        juce::PathFlatteningIterator iter(path, combined);
        std::vector<glm::vec2> points;
        points.reserve(64);

        glm::vec2 firstPoint(0);
        bool hasFirst = false;

        while (iter.next())
        {
            if (!hasFirst) { firstPoint = { iter.x1, iter.y1 }; hasFirst = true; }
            points.push_back({ iter.x2, iter.y2 });
        }

        if (points.size() < 2 || !hasFirst) return;

        // Step 2: Flush pending main vertices
        flush();

        // Step 3: Bind stencil-write pipeline, draw triangle fan
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, stencilPipeline->getInternal());
        float pushData[2] = { vpWidth, vpHeight };
        vkCmdPushConstants(cmd, stencilPipeline->getLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);

        // Emit triangle fan: fan center → consecutive edge points
        glm::vec4 dummy(0); // color irrelevant for stencil
        glm::vec4 noShape(0);
        std::vector<UIVertex> fanVerts;
        fanVerts.reserve(points.size() * 3);

        for (size_t i = 0; i + 1 < points.size(); i++)
        {
            fanVerts.push_back({ firstPoint, dummy, {}, noShape });
            fanVerts.push_back({ points[i],  dummy, {}, noShape });
            fanVerts.push_back({ points[i+1], dummy, {}, noShape });
        }

        if (!fanVerts.empty())
        {
            // Upload and draw fan
            VkDeviceSize needed = sizeof(UIVertex) * fanVerts.size();
            core::Buffer* buf = extBuffer;
            void** mapped = extMappedPtr;
            core::Buffer tempBuf;
            void* tempMapped = nullptr;
            if (!buf) { buf = &tempBuf; mapped = &tempMapped; }

            if (!buf->isValid() || buf->getSize() < needed)
            {
                buf->destroy();
                VkDeviceSize sz = std::max(needed, static_cast<VkDeviceSize>(sizeof(UIVertex) * 65536));
                buf->create({ physDevice, device, sz, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT });
                *mapped = buf->map();
            }
            memcpy(*mapped, fanVerts.data(), static_cast<size_t>(needed));

            VkBuffer buffers[] = { buf->getBuffer() };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);
            vkCmdDraw(cmd, static_cast<uint32_t>(fanVerts.size()), 1, 0, 0);
        }

        // Step 4: Bind main pipeline with stencil test (NOT_EQUAL to 0)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, colorPipeline);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);

        // Bind descriptor set for main pipeline
        if (defaultDescriptorSet != VK_NULL_HANDLE)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                     0, 1, &defaultDescriptorSet, 0, nullptr);

        // Step 5: Draw cover quad over path bounding box
        auto bounds = path.getBoundsTransformed(combined);
        auto fillColor = getColorForFill();

        // For gradients, compute per-vertex colors at the 4 corners
        auto& s = state();
        glm::vec4 c00 = s.fillType.isGradient() ? getColorAt(bounds.getX(), bounds.getY()) : fillColor;
        glm::vec4 c10 = s.fillType.isGradient() ? getColorAt(bounds.getRight(), bounds.getY()) : fillColor;
        glm::vec4 c11 = s.fillType.isGradient() ? getColorAt(bounds.getRight(), bounds.getBottom()) : fillColor;
        glm::vec4 c01 = s.fillType.isGradient() ? getColorAt(bounds.getX(), bounds.getBottom()) : fillColor;

        float bx = bounds.getX(), by = bounds.getY();
        float bw = bounds.getWidth(), bh = bounds.getHeight();
        glm::vec4 flat(0);

        // Emit cover quad vertices directly (these go through stencil test)
        vertices.clear();
        addVertex(bx,      by,      c00, 0, 0, flat);
        addVertex(bx + bw, by,      c10, 0, 0, flat);
        addVertex(bx + bw, by + bh, c11, 0, 0, flat);
        addVertex(bx,      by,      c00, 0, 0, flat);
        addVertex(bx + bw, by + bh, c11, 0, 0, flat);
        addVertex(bx,      by + bh, c01, 0, 0, flat);

        // Flush the cover quad (will be stencil-tested by hardware)
        // Note: stencil test is part of the pipeline state, need dynamic stencil
        // For now, use vkCmdSetStencilReference + vkCmdSetStencilCompareMask
        // Actually — stencil state is baked into the pipeline. We need the main
        // pipeline to NOT have stencil test. The stencil fill works because the
        // stencil buffer has non-zero values where the path is.
        // TODO: This requires a third pipeline variant (main+stencilTest) or dynamic stencil state.
        // For now, flush as regular draw (stencil test not active on main pipeline).
        flush();

        // Step 6: Clear stencil for next path
        VkClearAttachment clearAtt = {};
        clearAtt.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
        clearAtt.clearValue.depthStencil = { 1.0f, 0 };
        VkClearRect clearRect = {};
        clearRect.rect = { { static_cast<int32_t>(bx), static_cast<int32_t>(by) },
                           { static_cast<uint32_t>(bw + 1), static_cast<uint32_t>(bh + 1) } };
        clearRect.baseArrayLayer = 0;
        clearRect.layerCount = 1;
        vkCmdClearAttachments(cmd, 1, &clearAtt, 1, &clearRect);
    }

    // ==== Drawing: drawImage ====
    void drawImage(const juce::Image& image, const juce::AffineTransform& t) override
    {
        if (isClipEmpty() || !image.isValid()) return;
        auto& s = state();
        auto combined = t.scaled(scale).translated(
            static_cast<float>(s.origin.x), static_cast<float>(s.origin.y));

        int w = image.getWidth(), h = image.getHeight();
        juce::Image::BitmapData bmp(image, juce::Image::BitmapData::readOnly);

        for (int iy = 0; iy < h; iy++)
        {
            for (int ix = 0; ix < w; ix++)
            {
                auto pixel = bmp.getPixelColour(ix, iy);
                if (pixel.getAlpha() == 0) continue;

                auto p1 = juce::Point<float>(static_cast<float>(ix), static_cast<float>(iy)).transformedBy(combined);
                auto p2 = juce::Point<float>(static_cast<float>(ix + 1), static_cast<float>(iy + 1)).transformedBy(combined);

                float a = pixel.getFloatAlpha() * s.opacity;
                addQuad(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y,
                        { pixel.getFloatRed(), pixel.getFloatGreen(), pixel.getFloatBlue(), a });
            }
        }
    }

    // ==== Drawing: drawLine ====
    void drawLine(const juce::Line<float>& line) override
    {
        if (isClipEmpty()) return;
        auto& s = state();
        auto c = getColorForFill();
        float ox = static_cast<float>(s.origin.x), oy = static_cast<float>(s.origin.y);

        float x1 = line.getStartX() * scale + ox, y1 = line.getStartY() * scale + oy;
        float x2 = line.getEndX() * scale + ox, y2 = line.getEndY() * scale + oy;
        float dx = x2 - x1, dy = y2 - y1;
        float len = std::sqrt(dx * dx + dy * dy);
        if (len < 0.001f) return;
        float nx = -dy / len * 0.5f, ny = dx / len * 0.5f;

        addVertex(x1+nx, y1+ny, c); addVertex(x2+nx, y2+ny, c); addVertex(x2-nx, y2-ny, c);
        addVertex(x1+nx, y1+ny, c); addVertex(x2-nx, y2-ny, c); addVertex(x1-nx, y1-ny, c);
    }

    // ==== Drawing: drawGlyphs ====
    void drawGlyphs(juce::Span<const uint16_t> glyphs,
                     juce::Span<const juce::Point<float>> positions,
                     const juce::AffineTransform& t) override
    {
        if (isClipEmpty() || glyphs.empty()) return;
        auto& s = state();

        for (size_t idx = 0; idx < glyphs.size(); ++idx)
        {
            // Scale the glyph transform to physical pixels
            auto glyphTransform = juce::AffineTransform::translation(positions[idx])
                                    .followedBy(t)
                                    .scaled(scale)
                                    .translated(static_cast<float>(s.origin.x),
                                                static_cast<float>(s.origin.y));

            auto fontHeight = juce::detail::FontRendering::getEffectiveHeight(s.font);
            auto fontTransform = juce::AffineTransform::scale(
                fontHeight * s.font.getHorizontalScale(), fontHeight)
                .followedBy(glyphTransform);

            auto layers = s.font.getTypefacePtr()->getLayersForGlyph(
                s.font.getMetricsKind(), glyphs[idx], fontTransform);

            auto baseColor = getColorForFill();

            for (auto& layer : layers)
            {
                if (auto* cl = std::get_if<juce::ColourLayer>(&layer.layer))
                {
                    glm::vec4 color = baseColor;
                    if (cl->colour.has_value())
                    {
                        auto jc = *cl->colour;
                        color = { jc.getFloatRed(), jc.getFloatGreen(),
                                  jc.getFloatBlue(), jc.getFloatAlpha() * s.opacity };
                    }

                    SolidEdgeTableCallback callback(*this, color);
                    cl->clip.iterate(callback);
                }
                else if (auto* il = std::get_if<juce::ImageLayer>(&layer.layer))
                {
                    drawImage(il->image, il->transform);
                }
            }
        }
    }

    // ==== Flush ====
    void flush()
    {
        if (vertices.empty()) return;

        // Scissor in physical pixels (clip bounds are already physical)
        auto& s = state();
        VkRect2D scissor;
        scissor.offset = { s.clipBounds.getX(), s.clipBounds.getY() };
        scissor.extent = {
            static_cast<uint32_t>(std::max(0, s.clipBounds.getWidth())),
            static_cast<uint32_t>(std::max(0, s.clipBounds.getHeight()))
        };
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, colorPipeline);

        // Bind default descriptor set (1x1 white texture for non-textured draws)
        if (defaultDescriptorSet != VK_NULL_HANDLE)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                     0, 1, &defaultDescriptorSet, 0, nullptr);

        // Push physical pixel viewport size
        float pushData[2] = { vpWidth, vpHeight };
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);

        VkDeviceSize needed = sizeof(UIVertex) * vertices.size();

        // Use external persistent buffer if provided, else create a temporary
        core::Buffer* buf = extBuffer;
        void** mapped = extMappedPtr;

        core::Buffer tempBuffer;
        void* tempMapped = nullptr;
        if (!buf)
        {
            buf = &tempBuffer;
            mapped = &tempMapped;
        }

        // Grow the buffer if needed (rare — only first few frames)
        if (!buf->isValid() || buf->getSize() < needed)
        {
            buf->destroy();
            VkDeviceSize allocSize = std::max(needed, static_cast<VkDeviceSize>(sizeof(UIVertex) * 65536));
            buf->create({ physDevice, device, allocSize,
                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT });
            *mapped = buf->map();
        }

        // Fast memcpy — no allocation, no GPU memory calls
        memcpy(*mapped, vertices.data(), static_cast<size_t>(needed));

        VkBuffer buffers[] = { buf->getBuffer() };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);
        vkCmdDraw(cmd, static_cast<uint32_t>(vertices.size()), 1, 0, 0);

        vertices.clear();
    }


private:
    // ==== Helpers ====
    struct SavedState
    {
        juce::Point<int> origin;     // physical pixels
        juce::AffineTransform transform;
        juce::FillType fillType { juce::Colours::black };
        float opacity = 1.0f;
        juce::Font font;
        juce::Rectangle<int> clipBounds; // physical pixels
    };

    SavedState& state() { return stateStack.back(); }
    const SavedState& state() const { return stateStack.back(); }

    // Scale a logical-point int rect to physical pixels
    juce::Rectangle<int> scaleRect(const juce::Rectangle<int>& r) const
    {
        return juce::Rectangle<int>(
            static_cast<int>(r.getX() * scale), static_cast<int>(r.getY() * scale),
            static_cast<int>(r.getWidth() * scale), static_cast<int>(r.getHeight() * scale));
    }

    void addVertex(float x, float y, const glm::vec4& color,
                    float u = 0, float v = 0, const glm::vec4& shape = glm::vec4(0))
    {
        vertices.push_back({ { x, y }, convertColor(color), { u, v }, shape });
    }

    // Flat colored quad (shapeType = 0, no SDF)
    void addQuad(float x, float y, float w, float h, const glm::vec4& color)
    {
        glm::vec4 flat(0); // type 0 = flat
        addVertex(x,     y,     color, 0, 0, flat);
        addVertex(x + w, y,     color, 0, 0, flat);
        addVertex(x + w, y + h, color, 0, 0, flat);
        addVertex(x,     y,     color, 0, 0, flat);
        addVertex(x + w, y + h, color, 0, 0, flat);
        addVertex(x,     y + h, color, 0, 0, flat);
    }

    // SDF quad — GPU evaluates the shape per-pixel
    void addSDFQuad(float x, float y, float w, float h,
                     const glm::vec4& color, float shapeType,
                     float halfW, float halfH, float param)
    {
        // Expand quad slightly for AA fringe (1px padding)
        float pad = 1.0f;
        float ex = x - pad, ey = y - pad;
        float ew = w + pad * 2, eh = h + pad * 2;

        // UV maps the padded quad. SDF uses ORIGINAL half-extents —
        // the padding gives room for the AA fringe outside the shape boundary.
        float uPad = pad / ew, vPad = pad / eh;
        float u0 = -uPad, v0 = -vPad;
        float u1 = 1.0f + uPad, v1 = 1.0f + vPad;

        glm::vec4 shape(shapeType, halfW, halfH, param);

        addVertex(ex,      ey,      color, u0, v0, shape);
        addVertex(ex + ew, ey,      color, u1, v0, shape);
        addVertex(ex + ew, ey + eh, color, u1, v1, shape);
        addVertex(ex,      ey,      color, u0, v0, shape);
        addVertex(ex + ew, ey + eh, color, u1, v1, shape);
        addVertex(ex,      ey + eh, color, u0, v1, shape);
    }

    glm::vec4 getColorForFill() const
    {
        auto& s = state();
        if (s.fillType.isColour())
        {
            auto c = s.fillType.colour;
            return { c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue(),
                     c.getFloatAlpha() * s.opacity };
        }
        if (s.fillType.isGradient())
            return getColorAt(0, 0);
        return { 0, 0, 0, s.opacity };
    }

    glm::vec4 getColorAt(float x, float y) const
    {
        auto& s = state();
        if (s.fillType.isColour())
        {
            auto c = s.fillType.colour;
            return { c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue(),
                     c.getFloatAlpha() * s.opacity };
        }
        if (s.fillType.isGradient())
        {
            auto& g = *s.fillType.gradient;
            float tx = x, ty = y;
            if (!s.fillType.transform.isIdentity())
            {
                auto inv = s.fillType.transform.inverted();
                inv.transformPoint(tx, ty);
            }
            float t;
            if (g.isRadial)
            {
                float dx = tx - g.point1.x, dy = ty - g.point1.y;
                float radius = g.point1.getDistanceFrom(g.point2);
                t = (radius > 0) ? std::sqrt(dx * dx + dy * dy) / radius : 0.0f;
            }
            else
            {
                float dx = g.point2.x - g.point1.x, dy = g.point2.y - g.point1.y;
                float len2 = dx * dx + dy * dy;
                t = (len2 > 0) ? ((tx - g.point1.x) * dx + (ty - g.point1.y) * dy) / len2 : 0.0f;
            }
            t = juce::jlimit(0.0f, 1.0f, t);
            auto c = g.getColourAtPosition(static_cast<double>(t));
            return { c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue(),
                     c.getFloatAlpha() * s.opacity };
        }
        return { 0, 0, 0, s.opacity };
    }

    // sRGB→linear conversion for physically correct (sRGB) pipeline
    static float srgbToLinear(float c)
    {
        return (c <= 0.04045f) ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
    }

    // Apply color space conversion based on pipeline mode
    glm::vec4 convertColor(const glm::vec4& c) const
    {
        if (srgbMode)
            return { srgbToLinear(c.r), srgbToLinear(c.g), srgbToLinear(c.b), c.a };
        return c; // UNORM: pass sRGB values through (matches JUCE)
    }

    // Gamma-correct alpha for UNORM mode (EdgeTable alphas need boost for linear blend)
    float correctAlpha(int level) const
    {
        float a = static_cast<float>(level) / 255.0f;
        if (!srgbMode)
            return std::pow(a, 1.0f / 2.2f); // compensate for sRGB-designed alpha in linear blend
        return a; // sRGB pipeline: alpha is correct as-is
    }

    // ==== EdgeTable Callbacks (emit vertices in physical pixels) ====
    struct SolidEdgeTableCallback
    {
        VulkanGraphicsContext& ctx;
        glm::vec4 color;
        int currentY = 0;

        SolidEdgeTableCallback(VulkanGraphicsContext& c, glm::vec4 col) : ctx(c), color(col) {}

        void setEdgeTableYPos(int y) { currentY = y; }

        void handleEdgeTablePixelFull(int x) const
        { ctx.addQuad(static_cast<float>(x), static_cast<float>(currentY), 1.0f, 1.0f, color); }

        void handleEdgeTablePixel(int x, int level) const
        {
            auto c = color;
            c.a *= ctx.correctAlpha(level);
            ctx.addQuad(static_cast<float>(x), static_cast<float>(currentY), 1.0f, 1.0f, c);
        }

        void handleEdgeTableLineFull(int x, int width) const
        { ctx.addQuad(static_cast<float>(x), static_cast<float>(currentY), static_cast<float>(width), 1.0f, color); }

        void handleEdgeTableLine(int x, int width, int level) const
        {
            auto c = color;
            c.a *= ctx.correctAlpha(level);
            ctx.addQuad(static_cast<float>(x), static_cast<float>(currentY), static_cast<float>(width), 1.0f, c);
        }
    };

    struct GradientEdgeTableCallback
    {
        VulkanGraphicsContext& ctx;
        int currentY = 0;

        GradientEdgeTableCallback(VulkanGraphicsContext& c) : ctx(c) {}

        void setEdgeTableYPos(int y) { currentY = y; }

        void handleEdgeTablePixelFull(int x) const
        {
            auto c = ctx.getColorAt(static_cast<float>(x) + 0.5f, static_cast<float>(currentY) + 0.5f);
            ctx.addQuad(static_cast<float>(x), static_cast<float>(currentY), 1.0f, 1.0f, c);
        }

        void handleEdgeTablePixel(int x, int level) const
        {
            auto c = ctx.getColorAt(static_cast<float>(x) + 0.5f, static_cast<float>(currentY) + 0.5f);
            c.a *= ctx.correctAlpha(level);
            ctx.addQuad(static_cast<float>(x), static_cast<float>(currentY), 1.0f, 1.0f, c);
        }

        void handleEdgeTableLineFull(int x, int width) const
        { for (int i = 0; i < width; ++i) handleEdgeTablePixelFull(x + i); }

        void handleEdgeTableLine(int x, int width, int level) const
        { for (int i = 0; i < width; ++i) handleEdgeTablePixel(x + i, level); }
    };

    friend class Graphics;

    // ==== Members ====
    VkCommandBuffer cmd;
    VkPipeline colorPipeline;
    VkPipelineLayout pipelineLayout;
    float vpWidth, vpHeight; // physical pixels
    float scale;             // logical→physical multiplier (2.0)
    bool srgbMode;           // true = sRGB pipeline (linearize colors), false = UNORM (JUCE-identical)
    VkPhysicalDevice physDevice;
    VkDevice device;

    std::vector<SavedState> stateStack;
    std::vector<UIVertex> vertices;
    core::Buffer* extBuffer = nullptr;   // external persistent buffer (owned by caller)
    void** extMappedPtr = nullptr;
    Pipeline* stencilPipeline = nullptr;
    Pipeline* multiplyPipeline = nullptr;
    Pipeline* hsvPipeline = nullptr;
    Pipeline* blurPipeline = nullptr;
    RenderPassInfo rpInfo;
    core::Image* blurTempImage = nullptr;
    VkDescriptorSet blurDescSet = VK_NULL_HANDLE;

    // Blur implementation — ends render pass, copies framebuffer, restarts, draws blur quads
    void executeBlur(juce::Rectangle<float> region, float radius)
    {
        if (!blurPipeline || !blurTempImage || rpInfo.renderPass == VK_NULL_HANDLE) return;

        flush(); // render all pending geometry first

        // 1. End render pass
        vkCmdEndRenderPass(cmd);

        // 2. Transition MSAA image for copy source
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = rpInfo.msaaImage;
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        // 3. Transition temp image for copy dest
        VkImageMemoryBarrier barrier2 = barrier;
        barrier2.image = blurTempImage->getImage();
        barrier2.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier2.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier2.srcAccessMask = 0;
        barrier2.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier2);

        // 4. Copy framebuffer to temp
        VkImageCopy copyRegion = {};
        copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copyRegion.extent = { rpInfo.extent.width, rpInfo.extent.height, 1 };
        vkCmdCopyImage(cmd, rpInfo.msaaImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        blurTempImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &copyRegion);

        // 5. Transition images back
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        barrier2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier2);

        // 6. Restart render pass (no clear — LOAD existing content)
        VkRenderPassBeginInfo rpBegin = {};
        rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBegin.renderPass = rpInfo.renderPass;
        rpBegin.framebuffer = rpInfo.framebuffer;
        rpBegin.renderArea = { {0, 0}, rpInfo.extent };
        // Use LOAD_OP_LOAD to preserve existing framebuffer content
        // Note: this requires the render pass loadOp to be LOAD, not CLEAR.
        // Since we're mid-frame, the content is already there.
        std::array<VkClearValue, 3> clearValues = {};
        rpBegin.clearValueCount = static_cast<uint32_t>(clearValues.size());
        rpBegin.pClearValues = clearValues.data();
        vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

        // 7. Set viewport/scissor
        VkViewport vp = {};
        vp.width = static_cast<float>(rpInfo.extent.width);
        vp.height = static_cast<float>(rpInfo.extent.height);
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc = { {0, 0}, rpInfo.extent };
        vkCmdSetScissor(cmd, 0, 1, &sc);

        // 8. Draw horizontal blur pass
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, blurPipeline->getInternal());
        float pushData[2] = { vpWidth, vpHeight };
        vkCmdPushConstants(cmd, blurPipeline->getLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);

        if (blurDescSet != VK_NULL_HANDLE)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, blurPipeline->getLayout(),
                                     0, 1, &blurDescSet, 0, nullptr);

        float x = region.isEmpty() ? 0 : region.getX();
        float y = region.isEmpty() ? 0 : region.getY();
        float w = region.isEmpty() ? vpWidth : region.getWidth();
        float h = region.isEmpty() ? vpHeight : region.getHeight();

        // Horizontal pass: direction=(1,0), radius in z
        glm::vec4 hDir(1.0f, 0.0f, radius, 1.0f);
        glm::vec4 noShape(0);
        addVertex(x, y, hDir, 0, 0, noShape);
        addVertex(x+w, y, hDir, 0, 0, noShape);
        addVertex(x+w, y+h, hDir, 0, 0, noShape);
        addVertex(x, y, hDir, 0, 0, noShape);
        addVertex(x+w, y+h, hDir, 0, 0, noShape);
        addVertex(x, y+h, hDir, 0, 0, noShape);
        flush();

        // For a proper two-pass blur, we'd need to copy again for the vertical pass.
        // For now, single-pass gives a reasonable blur effect.
        // TODO: two-pass separable gaussian (horizontal → copy → vertical)

        // 9. Rebind main pipeline
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, colorPipeline);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);
        if (defaultDescriptorSet != VK_NULL_HANDLE)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                     0, 1, &defaultDescriptorSet, 0, nullptr);
    }
    VkDescriptorSet defaultDescriptorSet = VK_NULL_HANDLE;

    // Draw a fullscreen effect quad using a specific pipeline
    void drawEffectQuad(Pipeline* effectPipeline, const glm::vec4& color,
                         juce::Rectangle<float> region = {},
                         const glm::vec4& shapeData = glm::vec4(0))
    {
        if (!effectPipeline) return;
        flush(); // render pending geometry first

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, effectPipeline->getInternal());
        float pushData[2] = { vpWidth, vpHeight };
        vkCmdPushConstants(cmd, effectPipeline->getLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);

        if (defaultDescriptorSet != VK_NULL_HANDLE)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, effectPipeline->getLayout(),
                                     0, 1, &defaultDescriptorSet, 0, nullptr);

        // Determine region (full viewport if empty)
        float x = region.isEmpty() ? 0 : region.getX();
        float y = region.isEmpty() ? 0 : region.getY();
        float w = region.isEmpty() ? vpWidth : region.getWidth();
        float h = region.isEmpty() ? vpHeight : region.getHeight();

        addVertex(x,     y,     color, 0, 0, shapeData);
        addVertex(x + w, y,     color, 0, 0, shapeData);
        addVertex(x + w, y + h, color, 0, 0, shapeData);
        addVertex(x,     y,     color, 0, 0, shapeData);
        addVertex(x + w, y + h, color, 0, 0, shapeData);
        addVertex(x,     y + h, color, 0, 0, shapeData);

        flush(); // draw the effect quad

        // Re-bind the main pipeline for subsequent draws
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, colorPipeline);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);
        if (defaultDescriptorSet != VK_NULL_HANDLE)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                     0, 1, &defaultDescriptorSet, 0, nullptr);
    }
    mutable uint64_t frameId = 0;
};

} // jvk
