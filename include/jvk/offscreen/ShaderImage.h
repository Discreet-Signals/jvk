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
 File: ShaderImage.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

class ShaderImage : private juce::Timer
{
public:
    // storageBufferSize: number of floats for the storage buffer (0 = no storage buffer).
    // The shader's descriptor bindings (samplers, storage buffers) are auto-discovered
    // from the SPIR-V bytecode via reflection.
    ShaderImage(const char* fragmentSpv, int fragmentSpvSize,
                int width, int height, int storageBufferSize = 0);
    ~ShaderImage();

    const juce::Image& getImage() const { return displayImage; }

    void setSize(int width, int height);
    juce::Point<int> getSize() const { return { w, h }; }

    // Push float data to the storage buffer (consumed each frame via FIFO)
    void pushData(const float* data, int count);

    // Load a texture into a specific binding slot (set/binding discovered from SPIR-V)
    void loadTexture(int binding, const juce::Image& image);

    void setFrameRate(int fps);
    void render();

    bool isReady() const { return ready; }

private:
    void timerCallback() override { render(); }

    void reflectShader();
    void createRenderTarget();
    void destroyRenderTarget();
    void createPipeline();
    void destroyPipeline();
    void createDescriptors();
    void destroyDescriptors();

    // Shared context
    struct Context;
    static Context& getContext();

    // Dimensions
    int w = 0, h = 0;
    bool ready = false;

    // Shader source (kept for rebuild on resize)
    std::vector<char> fragShaderCode;

    // Render target
    VkImage colorImage = VK_NULL_HANDLE;
    VkDeviceMemory colorImageMemory = VK_NULL_HANDLE;
    VkImageView colorImageView = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;

    // Staging buffers (double-buffered, persistently mapped)
    VkBuffer stagingBuffer[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceMemory stagingMemory[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    void* mappedStagingPtr[2] = { nullptr, nullptr };

    // Pipeline
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;

    // --- Reflected descriptor bindings ---
    struct TextureSlot
    {
        int binding = -1;
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        bool loaded = false;
    };

    struct DescriptorSetInfo
    {
        uint32_t set = 0;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

        // Storage buffer (if this set has one)
        bool hasStorageBuffer = false;
        int storageBinding = -1;
        VkBuffer storageBuffer = VK_NULL_HANDLE;
        VkDeviceMemory storageBufferMemory = VK_NULL_HANDLE;
        void* mappedStoragePtr = nullptr;

        // Textures in this set
        std::vector<TextureSlot> textures;
    };

    std::vector<DescriptorSetInfo> descriptorSets;
    int storageSize = 0;

    // Command buffer + sync (double-buffered)
    VkCommandBuffer commandBuffer[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkFence fence[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    int frameIndex = 0;

    // Push constants
    struct PushConstants {
        float resolutionX, resolutionY;
        float time;
    };
    double startTime = 0.0;

    // Data streaming
    std::unique_ptr<FIFO<float>> fifo;
    std::vector<float> storageData;

    // Output image
    juce::Image displayImage;
};

} // namespace jvk
