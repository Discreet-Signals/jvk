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
 File: DrawImage.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::graphics
{

// ==== Drawing: drawImage ====
inline void drawImage(VulkanGraphicsContext& ctx, const juce::Image& image,
                       const juce::AffineTransform& t)
{
    if (isClipEmpty(ctx) || !image.isValid()) return;
    auto& s = ctx.state();
    auto combined = getFullTransform(ctx, t);

    int w = image.getWidth(), h = image.getHeight();

    // GPU textured quad path: upload image to GPU once, draw as single quad
    if (ctx.textureCache && ctx.textureCache->descriptorHelper)
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

        auto it = ctx.textureCache->entries.find(key);
        bool newEntry = false;
        if (it == ctx.textureCache->entries.end())
        {
            TextureCache::Entry entry;
            entry.texture.create({ ctx.textureCache->physDevice, ctx.textureCache->device,
                static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT });
            entry.texture.createSampler(VK_FILTER_LINEAR);

            // Convert JUCE image to RGBA8
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

            // Stage upload — no GPU stall
            if (ctx.textureCache->stagingBelt && ctx.textureCache->pendingUploads)
                ctx.textureCache->stagingBelt->stageImageUpload(entry.texture, *ctx.textureCache->pendingUploads,
                    pixels.data(), static_cast<uint32_t>(w), static_cast<uint32_t>(h));
            else
                entry.texture.upload(ctx.textureCache->physDevice, ctx.textureCache->commandPool,
                    ctx.textureCache->graphicsQueue, pixels.data(),
                    static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                    VK_FORMAT_R8G8B8A8_UNORM);

            entry.descriptorSet = ctx.textureCache->descriptorHelper->allocateSet();
            core::DescriptorHelper::writeImage(ctx.textureCache->device, entry.descriptorSet, 0,
                entry.texture.getView(), entry.texture.getSampler());

            entry.lastUsedFrame = ctx.textureCache->currentFrame;
            entry.uploadFrame = ctx.textureCache->currentFrame;
            auto [insertIt, _] = ctx.textureCache->entries.emplace(key, std::move(entry));
            it = insertIt;
            newEntry = true;
        }
        else
        {
            it->second.lastUsedFrame = ctx.textureCache->currentFrame;
        }

        // Use fallback descriptor if upload hasn't been processed yet
        bool pending = newEntry && ctx.textureCache->stagingBelt && ctx.textureCache->pendingUploads;
        VkDescriptorSet texDescSet = pending ? ctx.defaultDescriptorSet : it->second.descriptorSet;

        if (ctx.boundDescriptorSet != texDescSet)
        {
            flush(ctx);
            ensureDescriptorSet(ctx, ctx.pipelineLayout, texDescSet);
        }

        // Compute transformed corners
        auto p00 = juce::Point<float>(0, 0).transformedBy(combined);
        auto p10 = juce::Point<float>(static_cast<float>(w), 0).transformedBy(combined);
        auto p11 = juce::Point<float>(static_cast<float>(w), static_cast<float>(h)).transformedBy(combined);
        auto p01 = juce::Point<float>(0, static_cast<float>(h)).transformedBy(combined);

        glm::vec4 white(1.0f, 1.0f, 1.0f, s.opacity);
        glm::vec4 texType(3.0f, 0, 0, 0); // type 3 = textured quad
        addVertex(ctx, p00.x, p00.y, white, 0, 0, texType);
        addVertex(ctx, p10.x, p10.y, white, 1, 0, texType);
        addVertex(ctx, p11.x, p11.y, white, 1, 1, texType);
        addVertex(ctx, p00.x, p00.y, white, 0, 0, texType);
        addVertex(ctx, p11.x, p11.y, white, 1, 1, texType);
        addVertex(ctx, p01.x, p01.y, white, 0, 1, texType);
        flush(ctx);
        ensureDescriptorSet(ctx, ctx.pipelineLayout, ctx.defaultDescriptorSet);
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
            addQuad(ctx, p1.x, p1.y, p2.x - p1.x, p2.y - p1.y,
                    { pixel.getFloatRed(), pixel.getFloatGreen(), pixel.getFloatBlue(), a });
        }
    }
}

} // jvk::graphics
