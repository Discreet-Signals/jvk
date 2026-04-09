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
 File: Texture.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

class Texture : public ecs::ComponentBase<Texture>
{
public:
    Texture() = default;

    void loadImage(VkPhysicalDevice physDevice, VkDevice device,
                    VkCommandPool cmdPool, VkQueue queue,
                    const juce::Image& image)
    {
        auto bitmapData = juce::Image::BitmapData(image, juce::Image::BitmapData::readOnly);
        int w = image.getWidth();
        int h = image.getHeight();

        // Convert to RGBA8
        std::vector<uint8_t> pixels(w * h * 4);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                auto c = image.getPixelAt(x, y);
                size_t idx = (y * w + x) * 4;
                pixels[idx + 0] = c.getRed();
                pixels[idx + 1] = c.getGreen();
                pixels[idx + 2] = c.getBlue();
                pixels[idx + 3] = c.getAlpha();
            }
        }

        gpuImage.create({
            physDevice, device,
            static_cast<uint32_t>(w), static_cast<uint32_t>(h),
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
        });

        gpuImage.upload(physDevice, cmdPool, queue, pixels.data(),
                        static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                        VK_FORMAT_R8G8B8A8_UNORM);
        gpuImage.createSampler();
    }

    void setDynamic(bool isDynamic) { dynamic = isDynamic; }
    void markUpdated() { updated.store(true); }

    VkImageView getView() const { return gpuImage.getView(); }
    VkSampler getSampler() const { return gpuImage.getSampler(); }
    bool isValid() const { return gpuImage.isValid(); }

    void prepare() override
    {
        // Dynamic re-upload could be implemented here
    }

private:
    core::Image gpuImage;
    bool dynamic = false;
    std::atomic<bool> updated { false };
};

} // jvk
