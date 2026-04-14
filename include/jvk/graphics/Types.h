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
 File: Types.h
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
        uint64_t uploadFrame = 0; // frame when upload was staged
    };

    std::unordered_map<uint64_t, Entry> entries;
    core::DescriptorHelper* descriptorHelper = nullptr;
    VkPhysicalDevice physDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint64_t currentFrame = 0;
    core::StagingBelt* stagingBelt = nullptr;
    std::vector<core::PendingUpload>* pendingUploads = nullptr;
    VkDescriptorSet defaultDescriptorSet = VK_NULL_HANDLE;

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
            // If upload is still pending (staged last frame), use fallback
            if (it->second.uploadFrame + 1 > currentFrame)
                return defaultDescriptorSet;
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
        entry.texture.createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

        // Stage upload — no GPU stall. Upload commands recorded next prepareFrame().
        if (stagingBelt && pendingUploads)
            stagingBelt->stageImageUpload(entry.texture, *pendingUploads,
                pixels.data(), static_cast<uint32_t>(LUT_WIDTH), 1);
        else
            entry.texture.upload(physDevice, commandPool, graphicsQueue,
                pixels.data(), static_cast<uint32_t>(LUT_WIDTH), 1, VK_FORMAT_R8G8B8A8_UNORM);

        entry.descriptorSet = descriptorHelper->allocateSet();
        if (entry.descriptorSet == VK_NULL_HANDLE)
        {
            entry.texture.destroy();
            return defaultDescriptorSet;
        }
        core::DescriptorHelper::writeImage(device, entry.descriptorSet, 0,
            entry.texture.getView(), entry.texture.getSampler());

        entry.lastUsedFrame = currentFrame;
        entry.uploadFrame = currentFrame;
        auto [insertIt, _] = entries.emplace(key, std::move(entry));

        // Return fallback for this frame — image not uploaded yet
        if (stagingBelt && pendingUploads)
            return defaultDescriptorSet;
        return insertIt->second.descriptorSet;
    }

    void evict(uint64_t maxAge = 120)
    {
        for (auto it = entries.begin(); it != entries.end();)
        {
            if (currentFrame - it->second.lastUsedFrame > maxAge)
            {
                if (descriptorHelper)
                    descriptorHelper->freeSet(it->second.descriptorSet);
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
        {
            if (descriptorHelper)
                descriptorHelper->freeSet(e.descriptorSet);
            e.texture.destroy();
        }
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
        uint64_t uploadFrame = 0;
    };

    std::unordered_map<uint64_t, Entry> entries;
    core::DescriptorHelper* descriptorHelper = nullptr;
    VkPhysicalDevice physDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint64_t currentFrame = 0;
    core::StagingBelt* stagingBelt = nullptr;
    std::vector<core::PendingUpload>* pendingUploads = nullptr;
    VkDescriptorSet defaultDescriptorSet = VK_NULL_HANDLE;

    void evict(uint64_t maxAge = 120)
    {
        for (auto it = entries.begin(); it != entries.end();)
        {
            if (currentFrame - it->second.lastUsedFrame > maxAge)
            {
                if (descriptorHelper)
                    descriptorHelper->freeSet(it->second.descriptorSet);
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
        {
            if (descriptorHelper)
                descriptorHelper->freeSet(e.descriptorSet);
            e.texture.destroy();
        }
        entries.clear();
    }
};

} // jvk
