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

// GPU gradient LUT cache — creates 256x1 textures from ColourGradient color stops
struct GradientCache
{
    static constexpr int LUT_WIDTH = 256;

    struct Entry
    {
        core::Image texture;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        uint64_t lastUsedFrame = 0;
    };

    std::unordered_map<uint64_t, Entry> entries;
    core::DescriptorHelper* descriptorHelper = nullptr;
    VkPhysicalDevice physDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint64_t currentFrame = 0;

    static uint64_t hashGradient(const juce::ColourGradient& g)
    {
        uint64_t h = g.isRadial ? 1ULL : 0ULL;
        h ^= static_cast<uint64_t>(g.getNumColours()) * 0x9e3779b97f4a7c15ULL;
        for (int i = 0; i < g.getNumColours(); i++)
        {
            auto c = g.getColour(i).getARGB();
            h ^= (static_cast<uint64_t>(c) + 0x517cc1b727220a95ULL) + (h << 6) + (h >> 2);
            double pos = g.getColourPosition(i);
            uint64_t posBytes;
            memcpy(&posBytes, &pos, sizeof(posBytes));
            h ^= posBytes * 0x6c62272e07bb0142ULL;
        }
        return h;
    }

    VkDescriptorSet getOrCreate(const juce::ColourGradient& gradient)
    {
        uint64_t key = hashGradient(gradient);
        auto it = entries.find(key);
        if (it != entries.end())
        {
            it->second.lastUsedFrame = currentFrame;
            return it->second.descriptorSet;
        }

        // Rasterize gradient to a 256x1 RGBA texture
        std::vector<uint32_t> pixels(LUT_WIDTH);
        for (int i = 0; i < LUT_WIDTH; i++)
        {
            double t = static_cast<double>(i) / static_cast<double>(LUT_WIDTH - 1);
            auto c = gradient.getColourAtPosition(t);
            pixels[static_cast<size_t>(i)] =
                static_cast<uint32_t>(c.getRed())
              | (static_cast<uint32_t>(c.getGreen()) << 8)
              | (static_cast<uint32_t>(c.getBlue()) << 16)
              | (static_cast<uint32_t>(c.getAlpha()) << 24);
        }

        Entry entry;
        entry.texture.create({ physDevice, device,
            static_cast<uint32_t>(LUT_WIDTH), 1,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT });
        entry.texture.upload(physDevice, commandPool, graphicsQueue,
            pixels.data(), static_cast<uint32_t>(LUT_WIDTH), 1, VK_FORMAT_R8G8B8A8_UNORM);
        entry.texture.createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

        entry.descriptorSet = descriptorHelper->allocateSet();
        core::DescriptorHelper::writeImage(device, entry.descriptorSet, 0,
            entry.texture.getView(), entry.texture.getSampler());

        entry.lastUsedFrame = currentFrame;
        auto [insertIt, _] = entries.emplace(key, std::move(entry));
        return insertIt->second.descriptorSet;
    }

    void evict(uint64_t maxAge = 120)
    {
        for (auto it = entries.begin(); it != entries.end();)
        {
            if (currentFrame - it->second.lastUsedFrame > maxAge)
            {
                it->second.texture.destroy();
                it = entries.erase(it);
            }
            else
                ++it;
        }
    }

    void clear()
    {
        for (auto& [k, e] : entries)
            e.texture.destroy();
        entries.clear();
    }
};

// GPU texture cache for drawImage() — persists across frames, owned by PaintBridge
struct TextureCache
{
    struct Entry
    {
        core::Image texture;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        uint64_t lastUsedFrame = 0;
    };

    std::unordered_map<uint64_t, Entry> entries;
    core::DescriptorHelper* descriptorHelper = nullptr;
    VkPhysicalDevice physDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint64_t currentFrame = 0;

    void evict(uint64_t maxAge = 120)
    {
        for (auto it = entries.begin(); it != entries.end();)
        {
            if (currentFrame - it->second.lastUsedFrame > maxAge)
            {
                it->second.texture.destroy();
                it = entries.erase(it);
            }
            else
                ++it;
        }
    }

    void clear()
    {
        for (auto& [k, e] : entries)
            e.texture.destroy();
        entries.clear();
    }
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
                          GradientCache* gradientCache = nullptr)
        : cmd(cmd), colorPipeline(colorPipeline), pipelineLayout(pipelineLayout),
          vpWidth(vpWidth), vpHeight(vpHeight), scale(scale), srgbMode(srgb),
          physDevice(physDevice), device(device),
          extBuffer(externalBuffer), extMappedPtr(externalMappedPtr),
          stencilPipeline(stencilPipeline), stencilCoverPipeline(stencilCoverPipeline),
          multiplyPipeline(multiplyPipeline), hsvPipeline(hsvPipeline),
          blurPipeline(blurPipeline), rpInfo(rpInfo), blurTempImage(blurTempImage), blurDescSet(blurDescSet),
          defaultDescriptorSet(defaultDescSet), textureCache(textureCache),
          glyphAtlas(glyphAtlas), gradientCache(gradientCache)
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

    std::unique_ptr<juce::ImageType> getPreferredImageTypeForTemporaryImages() const
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

    // Clips to bounding box of the rectangle list.
    // This is the standard GPU approach (matches JUCE's OpenGL renderer).
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

    // Image-based alpha clipping is not supported in the GPU pipeline.
    // Falls back to bounding-box clip (matches JUCE's OpenGL renderer behavior).
    void clipToImageAlpha(const juce::Image& img, const juce::AffineTransform& t) override
    {
        if (img.isValid())
        {
            auto bounds = img.getBounds().toFloat().transformedBy(t).getSmallestIntegerContainer();
            clipToRectangle(bounds);
        }
    }

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
            emitGradientQuad(translated.getX(), translated.getY(),
                             translated.getWidth(), translated.getHeight());
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

        // GPU stencil path rendering (triangle fan + cover quad)
        if (stencilPipeline != nullptr && stencilCoverPipeline != nullptr)
        {
            fillPathStencil(path, combined);
            return;
        }

        // Fallback: EdgeTable path rendering (CPU per-pixel)
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

        // Step 4: Bind stencil cover pipeline (stencil test NOT_EQUAL 0, zeros on pass)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, stencilCoverPipeline->getInternal());
        vkCmdPushConstants(cmd, stencilCoverPipeline->getLayout(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);

        if (defaultDescriptorSet != VK_NULL_HANDLE)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, stencilCoverPipeline->getLayout(),
                                     0, 1, &defaultDescriptorSet, 0, nullptr);

        // Step 5: Draw cover quad over path bounding box
        auto bounds = path.getBoundsTransformed(combined);
        float bx = bounds.getX(), by = bounds.getY();
        float bw = bounds.getWidth(), bh = bounds.getHeight();

        auto& s = state();
        vertices.clear();

        if (s.fillType.isGradient() && s.fillType.gradient && gradientCache)
        {
            // GPU gradient: bind LUT and emit cover quad with gradient-space UVs
            auto& g = *s.fillType.gradient;
            VkDescriptorSet gradDescSet = gradientCache->getOrCreate(g);
            if (gradDescSet != VK_NULL_HANDLE)
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    stencilCoverPipeline->getLayout(), 0, 1, &gradDescSet, 0, nullptr);

            glm::vec4 color(1.0f, 1.0f, 1.0f, s.opacity);
            float shapeType = g.isRadial ? 6.0f : 5.0f;
            glm::vec4 shape(shapeType, 0, 0, 0);

            if (g.isRadial)
            {
                float cx = g.point1.x * scale + static_cast<float>(s.origin.x);
                float cy = g.point1.y * scale + static_cast<float>(s.origin.y);
                float radius = g.point1.getDistanceFrom(g.point2) * scale;
                float invR = (radius > 0) ? 1.0f / radius : 0.0f;

                auto uv = [&](float px, float py) { return std::pair{ (px-cx)*invR, (py-cy)*invR }; };
                auto [u00,v00] = uv(bx, by);       auto [u10,v10] = uv(bx+bw, by);
                auto [u11,v11] = uv(bx+bw, by+bh); auto [u01,v01] = uv(bx, by+bh);

                addVertex(bx,      by,      color, u00, v00, shape);
                addVertex(bx+bw,   by,      color, u10, v10, shape);
                addVertex(bx+bw,   by+bh,   color, u11, v11, shape);
                addVertex(bx,      by,      color, u00, v00, shape);
                addVertex(bx+bw,   by+bh,   color, u11, v11, shape);
                addVertex(bx,      by+bh,   color, u01, v01, shape);
            }
            else
            {
                float gx1 = g.point1.x * scale + static_cast<float>(s.origin.x);
                float gy1 = g.point1.y * scale + static_cast<float>(s.origin.y);
                float gx2 = g.point2.x * scale + static_cast<float>(s.origin.x);
                float gy2 = g.point2.y * scale + static_cast<float>(s.origin.y);
                float dx = gx2-gx1, dy = gy2-gy1;
                float len2 = dx*dx + dy*dy;
                float invLen2 = (len2 > 0) ? 1.0f / len2 : 0.0f;
                auto tAt = [&](float px, float py) { return ((px-gx1)*dx + (py-gy1)*dy) * invLen2; };

                float t00 = tAt(bx,by), t10 = tAt(bx+bw,by), t11 = tAt(bx+bw,by+bh), t01 = tAt(bx,by+bh);
                addVertex(bx,    by,    color, t00, 0, shape);
                addVertex(bx+bw, by,    color, t10, 0, shape);
                addVertex(bx+bw, by+bh, color, t11, 0, shape);
                addVertex(bx,    by,    color, t00, 0, shape);
                addVertex(bx+bw, by+bh, color, t11, 0, shape);
                addVertex(bx,    by+bh, color, t01, 0, shape);
            }
        }
        else
        {
            // Solid color or fallback gradient (4-corner CPU-sampled interpolation)
            auto fillColor = getColorForFill();
            glm::vec4 c00 = s.fillType.isGradient() ? getColorAt(bx, by) : fillColor;
            glm::vec4 c10 = s.fillType.isGradient() ? getColorAt(bx+bw, by) : fillColor;
            glm::vec4 c11 = s.fillType.isGradient() ? getColorAt(bx+bw, by+bh) : fillColor;
            glm::vec4 c01 = s.fillType.isGradient() ? getColorAt(bx, by+bh) : fillColor;
            glm::vec4 flat(0);
            addVertex(bx,      by,      c00, 0, 0, flat);
            addVertex(bx + bw, by,      c10, 0, 0, flat);
            addVertex(bx + bw, by + bh, c11, 0, 0, flat);
            addVertex(bx,      by,      c00, 0, 0, flat);
            addVertex(bx + bw, by + bh, c11, 0, 0, flat);
            addVertex(bx,      by + bh, c01, 0, 0, flat);
        }
        flushWithPipeline(stencilCoverPipeline);

        // Step 6: Restore main pipeline for subsequent draws
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, colorPipeline);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);
        if (defaultDescriptorSet != VK_NULL_HANDLE)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                     0, 1, &defaultDescriptorSet, 0, nullptr);
    }

    // ==== Drawing: drawImage ====
    void drawImage(const juce::Image& image, const juce::AffineTransform& t) override
    {
        if (isClipEmpty() || !image.isValid()) return;
        auto& s = state();
        auto combined = t.scaled(scale).translated(
            static_cast<float>(s.origin.x), static_cast<float>(s.origin.y));

        int w = image.getWidth(), h = image.getHeight();

        // GPU textured quad path: upload image to GPU once, draw as single quad
        if (textureCache && textureCache->descriptorHelper)
        {
            // Content-based hash: sample pixels at corners + center for fast fingerprint
            juce::Image::BitmapData bmp(image, juce::Image::BitmapData::readOnly);
            uint64_t key = static_cast<uint64_t>(w) | (static_cast<uint64_t>(h) << 16);
            auto hashPixel = [&](int x, int y) {
                auto c = bmp.getPixelColour(x, y);
                return static_cast<uint64_t>(c.getARGB());
            };
            key ^= hashPixel(0, 0) * 0x9e3779b97f4a7c15ULL;
            key ^= hashPixel(w - 1, 0) * 0x517cc1b727220a95ULL;
            key ^= hashPixel(0, h - 1) * 0x6c62272e07bb0142ULL;
            key ^= hashPixel(w - 1, h - 1) * 0x62b821756295c58dULL;
            key ^= hashPixel(w / 2, h / 2) * 0x9e3779b97f4a7c15ULL;
            // Include data pointer for disambiguation of images with identical sampled pixels
            key ^= reinterpret_cast<uint64_t>(bmp.data);

            auto it = textureCache->entries.find(key);
            if (it == textureCache->entries.end())
            {
                // Upload image to GPU texture
                TextureCache::Entry entry;
                entry.texture.create({ textureCache->physDevice, textureCache->device,
                    static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                    VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT });

                // Convert JUCE image to RGBA8 for upload
                std::vector<uint32_t> pixels(static_cast<size_t>(w * h));
                for (int iy = 0; iy < h; iy++)
                    for (int ix = 0; ix < w; ix++)
                    {
                        auto c = bmp.getPixelColour(ix, iy);
                        pixels[static_cast<size_t>(iy * w + ix)] =
                            (static_cast<uint32_t>(c.getRed()))
                          | (static_cast<uint32_t>(c.getGreen()) << 8)
                          | (static_cast<uint32_t>(c.getBlue()) << 16)
                          | (static_cast<uint32_t>(c.getAlpha()) << 24);
                    }

                entry.texture.upload(textureCache->physDevice, textureCache->commandPool,
                    textureCache->graphicsQueue, pixels.data(),
                    static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                    VK_FORMAT_R8G8B8A8_UNORM);
                entry.texture.createSampler(VK_FILTER_LINEAR);

                entry.descriptorSet = textureCache->descriptorHelper->allocateSet();
                core::DescriptorHelper::writeImage(textureCache->device, entry.descriptorSet, 0,
                    entry.texture.getView(), entry.texture.getSampler());

                entry.lastUsedFrame = textureCache->currentFrame;
                auto [insertIt, _] = textureCache->entries.emplace(key, std::move(entry));
                it = insertIt;
            }
            else
            {
                it->second.lastUsedFrame = textureCache->currentFrame;
            }

            // Flush pending vertices, bind image texture, draw textured quad, restore default
            flush();

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                     0, 1, &it->second.descriptorSet, 0, nullptr);

            // Compute transformed corners
            auto p00 = juce::Point<float>(0, 0).transformedBy(combined);
            auto p10 = juce::Point<float>(static_cast<float>(w), 0).transformedBy(combined);
            auto p11 = juce::Point<float>(static_cast<float>(w), static_cast<float>(h)).transformedBy(combined);
            auto p01 = juce::Point<float>(0, static_cast<float>(h)).transformedBy(combined);

            glm::vec4 white(1.0f, 1.0f, 1.0f, s.opacity);
            glm::vec4 flat(0);
            addVertex(p00.x, p00.y, white, 0, 0, flat);
            addVertex(p10.x, p10.y, white, 1, 0, flat);
            addVertex(p11.x, p11.y, white, 1, 1, flat);
            addVertex(p00.x, p00.y, white, 0, 0, flat);
            addVertex(p11.x, p11.y, white, 1, 1, flat);
            addVertex(p01.x, p01.y, white, 0, 1, flat);
            flush();

            // Restore default texture
            if (defaultDescriptorSet != VK_NULL_HANDLE)
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                         0, 1, &defaultDescriptorSet, 0, nullptr);
            return;
        }

        // Fallback: per-pixel software path (no texture cache available)
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

        // GPU SDF atlas path: each glyph = 1 textured quad, resolution-independent
        if (glyphAtlas)
        {
            auto* typeface = s.font.getTypefacePtr().get();
            auto fontHeight = juce::detail::FontRendering::getEffectiveHeight(s.font);
            auto baseColor = getColorForFill();

            // Scale factor: how big to render relative to the SDF generation size
            float sdfScale = (fontHeight * scale) / static_cast<float>(GlyphAtlas::SDF_GLYPH_SIZE);
            float hScale = s.font.getHorizontalScale();

            VkDescriptorSet currentAtlasDescSet = VK_NULL_HANDLE;

            for (size_t idx = 0; idx < glyphs.size(); ++idx)
            {
                GlyphAtlas::GlyphKey key { typeface, glyphs[idx] };
                auto* entry = glyphAtlas->getGlyph(key, s.font);
                if (!entry) continue;

                auto descSet = glyphAtlas->getDescriptorSet(entry->atlasIndex);
                if (descSet == VK_NULL_HANDLE) continue;

                // Switch atlas texture if needed
                if (descSet != currentAtlasDescSet)
                {
                    flush();
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                             0, 1, &descSet, 0, nullptr);
                    currentAtlasDescSet = descSet;
                }

                // Position: glyph origin in physical pixels, scaled from SDF space
                auto glyphPos = positions[idx].transformedBy(t);
                float ox = static_cast<float>(s.origin.x);
                float oy = static_cast<float>(s.origin.y);
                // Offset by the path bounds origin (where the glyph actually starts)
                float px = glyphPos.x * scale + ox + (entry->offsetX - GlyphAtlas::SDF_PADDING) * sdfScale * hScale;
                float py = glyphPos.y * scale + oy + (entry->offsetY - GlyphAtlas::SDF_PADDING) * sdfScale;
                float gw = static_cast<float>(entry->w) * sdfScale * hScale;
                float gh = static_cast<float>(entry->h) * sdfScale;

                // shapeInfo.x = 4.0 tells the fragment shader to use SDF text evaluation
                glm::vec4 sdfType(4.0f, 0, 0, 0);
                addVertex(px,      py,      baseColor, entry->u0, entry->v0, sdfType);
                addVertex(px + gw, py,      baseColor, entry->u1, entry->v0, sdfType);
                addVertex(px + gw, py + gh, baseColor, entry->u1, entry->v1, sdfType);
                addVertex(px,      py,      baseColor, entry->u0, entry->v0, sdfType);
                addVertex(px + gw, py + gh, baseColor, entry->u1, entry->v1, sdfType);
                addVertex(px,      py + gh, baseColor, entry->u0, entry->v1, sdfType);
            }

            // Flush remaining glyphs and restore default texture
            flush();
            if (defaultDescriptorSet != VK_NULL_HANDLE)
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                         0, 1, &defaultDescriptorSet, 0, nullptr);
            return;
        }

        // Fallback: EdgeTable per-pixel path
        for (size_t idx = 0; idx < glyphs.size(); ++idx)
        {
            auto glyphTransform = juce::AffineTransform::translation(positions[idx])
                                    .followedBy(t)
                                    .scaled(scale)
                                    .translated(static_cast<float>(s.origin.x),
                                                static_cast<float>(s.origin.y));

            auto fontHeight = s.font.getHeight();
            auto fontTransform = juce::AffineTransform::scale(
                fontHeight * s.font.getHorizontalScale(), fontHeight)
                .followedBy(glyphTransform);

            auto layers = s.font.getTypefacePtr()->getLayersForGlyph(
                s.font.getMetricsKind(), glyphs[idx], fontTransform, fontHeight);

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

        // Grow the buffer if needed
        VkDeviceSize totalNeeded = bufferWriteOffset + needed;
        if (!buf->isValid() || buf->getSize() < totalNeeded)
        {
            buf->destroy();
            VkDeviceSize allocSize = std::max(totalNeeded, static_cast<VkDeviceSize>(sizeof(UIVertex) * 65536));
            buf->create({ physDevice, device, allocSize,
                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT });
            *mapped = buf->map();
            bufferWriteOffset = 0; // buffer was reallocated, reset offset
        }

        // Write at current offset
        memcpy(static_cast<char*>(*mapped) + bufferWriteOffset, vertices.data(), static_cast<size_t>(needed));

        VkBuffer buffers[] = { buf->getBuffer() };
        VkDeviceSize offsets[] = { bufferWriteOffset };
        vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);
        vkCmdDraw(cmd, static_cast<uint32_t>(vertices.size()), 1, 0, 0);

        bufferWriteOffset += needed;
        vertices.clear();
    }

    // Flush vertices using a specific pipeline (for stencil cover draws)
    void flushWithPipeline(Pipeline* pip)
    {
        if (vertices.empty() || !pip) return;

        auto& s = state();
        VkRect2D scissor;
        scissor.offset = { s.clipBounds.getX(), s.clipBounds.getY() };
        scissor.extent = {
            static_cast<uint32_t>(std::max(0, s.clipBounds.getWidth())),
            static_cast<uint32_t>(std::max(0, s.clipBounds.getHeight()))
        };
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        VkDeviceSize needed = sizeof(UIVertex) * vertices.size();

        core::Buffer* buf = extBuffer;
        void** mapped = extMappedPtr;
        core::Buffer tempBuffer;
        void* tempMapped = nullptr;
        if (!buf) { buf = &tempBuffer; mapped = &tempMapped; }

        VkDeviceSize totalNeeded = bufferWriteOffset + needed;
        if (!buf->isValid() || buf->getSize() < totalNeeded)
        {
            buf->destroy();
            VkDeviceSize allocSize = std::max(totalNeeded, static_cast<VkDeviceSize>(sizeof(UIVertex) * 65536));
            buf->create({ physDevice, device, allocSize,
                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT });
            *mapped = buf->map();
            bufferWriteOffset = 0;
        }

        memcpy(static_cast<char*>(*mapped) + bufferWriteOffset, vertices.data(), static_cast<size_t>(needed));

        VkBuffer buffers[] = { buf->getBuffer() };
        VkDeviceSize offsets[] = { bufferWriteOffset };
        vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);
        vkCmdDraw(cmd, static_cast<uint32_t>(vertices.size()), 1, 0, 0);

        bufferWriteOffset += needed;
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

    // Emit a gradient-filled quad using the GPU gradient LUT pipeline
    // Falls back to 4-corner CPU-sampled interpolation if no gradient cache
    void emitGradientQuad(float x, float y, float w, float h)
    {
        auto& s = state();
        if (!s.fillType.isGradient() || !s.fillType.gradient) return;

        auto& g = *s.fillType.gradient;

        // GPU path: bind 1D gradient LUT texture, emit quad with gradient-space UVs
        if (gradientCache)
        {
            VkDescriptorSet gradDescSet = gradientCache->getOrCreate(g);
            if (gradDescSet != VK_NULL_HANDLE)
            {
                flush();
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                         0, 1, &gradDescSet, 0, nullptr);

                // Compute UV coordinates in gradient space for each corner
                glm::vec4 color(1.0f, 1.0f, 1.0f, s.opacity);
                float shapeType = g.isRadial ? 6.0f : 5.0f;
                glm::vec4 shape(shapeType, 0, 0, 0);

                if (g.isRadial)
                {
                    // Radial: UV = normalized offset from center, length(UV)=1 at radius edge
                    float cx = g.point1.x * scale + static_cast<float>(s.origin.x);
                    float cy = g.point1.y * scale + static_cast<float>(s.origin.y);
                    float radius = g.point1.getDistanceFrom(g.point2) * scale;
                    float invR = (radius > 0) ? 1.0f / radius : 0.0f;

                    auto uv = [&](float px, float py) -> std::pair<float, float> {
                        return { (px - cx) * invR, (py - cy) * invR };
                    };

                    auto [u00, v00] = uv(x, y);
                    auto [u10, v10] = uv(x + w, y);
                    auto [u11, v11] = uv(x + w, y + h);
                    auto [u01, v01] = uv(x, y + h);

                    addVertex(x,     y,     color, u00, v00, shape);
                    addVertex(x + w, y,     color, u10, v10, shape);
                    addVertex(x + w, y + h, color, u11, v11, shape);
                    addVertex(x,     y,     color, u00, v00, shape);
                    addVertex(x + w, y + h, color, u11, v11, shape);
                    addVertex(x,     y + h, color, u01, v01, shape);
                }
                else
                {
                    // Linear: UV.x = t along gradient axis
                    float gx1 = g.point1.x * scale + static_cast<float>(s.origin.x);
                    float gy1 = g.point1.y * scale + static_cast<float>(s.origin.y);
                    float gx2 = g.point2.x * scale + static_cast<float>(s.origin.x);
                    float gy2 = g.point2.y * scale + static_cast<float>(s.origin.y);
                    float dx = gx2 - gx1, dy = gy2 - gy1;
                    float len2 = dx * dx + dy * dy;
                    float invLen2 = (len2 > 0) ? 1.0f / len2 : 0.0f;

                    auto tAt = [&](float px, float py) -> float {
                        return ((px - gx1) * dx + (py - gy1) * dy) * invLen2;
                    };

                    float t00 = tAt(x, y),     t10 = tAt(x + w, y);
                    float t11 = tAt(x + w, y + h), t01 = tAt(x, y + h);

                    addVertex(x,     y,     color, t00, 0, shape);
                    addVertex(x + w, y,     color, t10, 0, shape);
                    addVertex(x + w, y + h, color, t11, 0, shape);
                    addVertex(x,     y,     color, t00, 0, shape);
                    addVertex(x + w, y + h, color, t11, 0, shape);
                    addVertex(x,     y + h, color, t01, 0, shape);
                }

                flush();

                // Restore default texture
                if (defaultDescriptorSet != VK_NULL_HANDLE)
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout,
                                             0, 1, &defaultDescriptorSet, 0, nullptr);
                return;
            }
        }

        // Fallback: CPU-sampled 4-corner interpolation
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
        {
            // GPU gradient: emit single quad with corner colors, hardware interpolates
            float fx = static_cast<float>(x), fy = static_cast<float>(currentY);
            float fw = static_cast<float>(width);
            auto c0 = ctx.getColorAt(fx + 0.5f, fy + 0.5f);
            auto c1 = ctx.getColorAt(fx + fw - 0.5f, fy + 0.5f);
            glm::vec4 flat(0);
            ctx.addVertex(fx,      fy,       c0, 0, 0, flat);
            ctx.addVertex(fx + fw, fy,       c1, 0, 0, flat);
            ctx.addVertex(fx + fw, fy + 1.f, c1, 0, 0, flat);
            ctx.addVertex(fx,      fy,       c0, 0, 0, flat);
            ctx.addVertex(fx + fw, fy + 1.f, c1, 0, 0, flat);
            ctx.addVertex(fx,      fy + 1.f, c0, 0, 0, flat);
        }

        void handleEdgeTableLine(int x, int width, int level) const
        {
            float fx = static_cast<float>(x), fy = static_cast<float>(currentY);
            float fw = static_cast<float>(width);
            auto c0 = ctx.getColorAt(fx + 0.5f, fy + 0.5f);
            auto c1 = ctx.getColorAt(fx + fw - 0.5f, fy + 0.5f);
            c0.a *= ctx.correctAlpha(level);
            c1.a *= ctx.correctAlpha(level);
            glm::vec4 flat(0);
            ctx.addVertex(fx,      fy,       c0, 0, 0, flat);
            ctx.addVertex(fx + fw, fy,       c1, 0, 0, flat);
            ctx.addVertex(fx + fw, fy + 1.f, c1, 0, 0, flat);
            ctx.addVertex(fx,      fy,       c0, 0, 0, flat);
            ctx.addVertex(fx + fw, fy + 1.f, c1, 0, 0, flat);
            ctx.addVertex(fx,      fy + 1.f, c0, 0, 0, flat);
        }
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
    VkDeviceSize bufferWriteOffset = 0; // sub-allocation offset within persistent buffer
    Pipeline* stencilPipeline = nullptr;
    Pipeline* stencilCoverPipeline = nullptr;
    TextureCache* textureCache = nullptr;
    GlyphAtlas* glyphAtlas = nullptr;
    GradientCache* gradientCache = nullptr;
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
