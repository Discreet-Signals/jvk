/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: Shader.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

// Inline Vulkan shader for drawing directly into the active render pass.
// Compatible with ShaderImage fragment shaders (same push constant layout,
// same descriptor binding conventions). No CPU readback — draws natively.
//
// Usage:
//   auto vk = jvk::Graphics::create(g);
//   if (vk) {
//       if (!shader)
//           shader = std::make_unique<jvk::Shader>(spvData, spvSize, 4);
//       shader->pushData(data, 4);
//       vk->drawShader(*shader, getLocalBounds().toFloat());
//   }
//
class Shader
{
public:
    // fragmentSpv: pre-compiled SPIR-V bytecode for the fragment shader.
    // storageBufferFloats: number of floats for the storage buffer (0 = none).
    // Descriptor bindings are auto-discovered from SPIR-V via reflection.
    Shader(const char* fragmentSpv, int fragmentSpvSize, int storageBufferFloats = 0);
    ~Shader();

    // Push float data to the storage buffer (copied to GPU before each draw)
    void pushData(const float* data, int count);

    // Load a texture into a specific binding slot (set/binding from SPIR-V)
    void loadTexture(int binding, const juce::Image& image);

    bool isReady() const { return created; }

    // --- Used by graphics::drawShader() ---
    // Lazy creation — called on first drawShader() with the active render context
    void ensureCreated(VkPhysicalDevice physDevice, VkDevice device,
                       VkRenderPass renderPass, VkSampleCountFlagBits msaa);
    void destroy();

    // SPIR-V reflection to discover descriptor bindings
    void reflectShader();

    // Resource creation (called from ensureCreated)
    void createDescriptors(VkPhysicalDevice physDevice, VkDevice device);
    void createPipeline(VkDevice device, VkRenderPass renderPass, VkSampleCountFlagBits msaa);
    void uploadPendingTextures();

    // --- Shader source ---
    std::vector<char> fragSpv;

    // --- Vulkan resources (created lazily) ---
    VkPhysicalDevice cachedPhysDevice = VK_NULL_HANDLE;
    VkDevice cachedDevice = VK_NULL_HANDLE;
    VkRenderPass cachedRenderPass = VK_NULL_HANDLE;
    VkSampleCountFlagBits cachedMsaa = VK_SAMPLE_COUNT_1_BIT;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

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

        bool hasStorageBuffer = false;
        int storageBinding = -1;
        VkBuffer storageBuffer = VK_NULL_HANDLE;
        VkDeviceMemory storageBufferMemory = VK_NULL_HANDLE;
        void* mappedStoragePtr = nullptr;

        std::vector<TextureSlot> textures;
    };

    std::vector<DescriptorSetInfo> descriptorSets;
    int storageSize = 0;

    // --- Deferred state ---
    struct PendingTexture { int binding; juce::Image image; };
    std::vector<PendingTexture> pendingTextures;
    std::vector<float> storageData;

    // --- Push constants ---
    double startTime = 0.0;

    bool created = false;
};

} // namespace jvk
