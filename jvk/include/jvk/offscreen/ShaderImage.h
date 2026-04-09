/*
  ==============================================================================

    ShaderImage.h
    Offscreen Vulkan shader rendering to juce::Image.

    No window or surface required. Multiple instances share a single
    Vulkan device via a reference-counted singleton context.

    Usage:
        jvk::ShaderImage shader(fragSpv, fragSpvSize, 512, 256);
        shader.setFrameRate(60);

        // In paint():
        g.drawImageAt(shader.getImage(), 0, 0);

    Fragment shader interface:
        layout(location = 0) in vec2 fragUV;           // [0,1] normalized coords
        layout(set = 0, binding = 0) readonly buffer StorageBuffer { float data[]; };
        layout(push_constant) uniform PC { vec2 resolution; float time; };
        layout(location = 0) out vec4 outColor;

  ==============================================================================
*/

#pragma once

namespace jvk
{

class ShaderImage : private juce::Timer
{
public:
    ShaderImage(const char* fragmentSpv, int fragmentSpvSize,
                int width, int height, int storageBufferSize = 0);
    ~ShaderImage();

    const juce::Image& getImage() const { return displayImage; }

    void setSize(int width, int height);
    juce::Point<int> getSize() const { return { w, h }; }

    void pushData(const float* data, int count);

    void setFrameRate(int fps);
    void render();

    bool isReady() const { return ready; }

private:
    void timerCallback() override { render(); }

    void createRenderTarget();
    void destroyRenderTarget();
    void createPipeline();
    void destroyPipeline();
    void createStorageBuffer();
    void destroyStorageBuffer();

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

    // Storage buffer for shader data
    VkBuffer storageBuffer = VK_NULL_HANDLE;
    VkDeviceMemory storageBufferMemory = VK_NULL_HANDLE;
    void* mappedStoragePtr = nullptr;
    int storageSize = 0;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

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
