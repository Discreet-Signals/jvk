/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

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

// Runs a fragment shader into a juce::Image. All instances share one VkInstance
// / VkDevice / VkQueue via Device::acquire() so many can run concurrently
// without driver glitches. Everything else — command pool, render pass,
// pipeline, descriptors, framebuffer, staging buffers — is local to each
// instance. (The command pool is per-instance because Vulkan pools are
// externally synchronized and a ShaderImage's render timer runs on the
// message thread while an editor's jvk-render-worker is on its own thread —
// sharing one pool lets those threads race on MoltenVK's internal pool state.)
//
// Descriptor bindings (combined image samplers, storage buffers) are auto-
// discovered from the fragment SPIR-V via spirv-reflect. Push constants follow
// the shared shader_region.vert layout: (resolutionX, resolutionY, time,
// viewportW, viewportH, regionX, regionY). Fragment shaders only need to
// declare the first three.
class ShaderImage : private juce::Timer
{
public:
    ShaderImage(const char* fragmentSpv, int fragmentSpvSize,
                int width, int height, int storageBufferSize = 0);
    ~ShaderImage() override;

    ShaderImage(const ShaderImage&) = delete;
    ShaderImage& operator=(const ShaderImage&) = delete;

    const juce::Image& getImage() const { return displayImage; }

    void setSize(int width, int height);
    juce::Point<int> getSize() const { return { w, h }; }

    // Audio-thread safe. Pushes up to `count` floats into the ring buffer;
    // render() consumes them into the storage descriptor each frame.
    void pushData(const float* data, int count);

    // Loads a juce::Image into the combined-image-sampler at `binding`.
    // CPU-side stash only: the GPU destroy/upload/descriptor-write work is
    // deferred into the next render() so the caller's thread never stalls
    // on vkDeviceWaitIdle. Calling twice for the same binding before the
    // next render() coalesces to just the most recent image.
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
    void applyPendingTextureLoads();

    std::shared_ptr<Device> device_;

    int  w = 0, h = 0;
    bool ready = false;
    bool skipNextStagingCopy = false;

    std::vector<char> fragShaderCode;

    // Render target (color image + view + framebuffer + render pass)
    VkImage                colorImage      = VK_NULL_HANDLE;
    Memory::L1::Allocation colorImageAlloc {};
    VkImageView            colorImageView  = VK_NULL_HANDLE;
    VkFramebuffer          framebuffer     = VK_NULL_HANDLE;
    VkRenderPass           renderPass      = VK_NULL_HANDLE;

    // Readback staging (double-buffered, host-visible, persistently mapped —
    // raw alloc since L2 is a bump allocator meant for CPU→GPU only)
    VkBuffer       stagingBuffer[2]   = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceMemory stagingMemory[2]   = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    void*          mappedStagingPtr[2] = { nullptr, nullptr };

    // Pipeline
    VkPipelineLayout pipelineLayout   = VK_NULL_HANDLE;
    VkPipeline       graphicsPipeline = VK_NULL_HANDLE;

    struct TextureSlot {
        int                    binding = -1;
        VkImage                image   = VK_NULL_HANDLE;
        Memory::L1::Allocation alloc {};
        VkImageView            view    = VK_NULL_HANDLE;
        VkSampler              sampler = VK_NULL_HANDLE;
        bool                   loaded  = false;
    };

    struct DescriptorSetInfo {
        uint32_t              set            = 0;
        VkDescriptorSetLayout layout         = VK_NULL_HANDLE;
        VkDescriptorPool      pool           = VK_NULL_HANDLE;
        VkDescriptorSet       descriptorSet  = VK_NULL_HANDLE;

        bool            hasStorageBuffer    = false;
        int             storageBinding      = -1;
        VkBuffer        storageBuffer       = VK_NULL_HANDLE;
        VkDeviceMemory  storageBufferMemory = VK_NULL_HANDLE;
        void*           mappedStoragePtr    = nullptr;

        std::vector<TextureSlot> textures;
    };

    std::vector<DescriptorSetInfo> descriptorSets;
    int storageSize = 0;

    // Deferred loadTexture requests. loadTexture() only stores into this
    // vector; render() drains it at the top under a two-fence wait. Keyed
    // by binding so repeated calls for the same binding before the next
    // render() collapse to just the final image.
    struct PendingTextureLoad { int binding; juce::Image image; };
    std::vector<PendingTextureLoad> pendingTextureLoads_;

    // Per-instance command pool + per-frame command buffers + fences
    // (double-buffered). See the class-level comment for why the pool is
    // per-instance rather than shared via Device.
    VkCommandPool   commandPool_     = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkFence         fence[2]         = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    int             frameIndex = 0;

    // Matches shader_region.vert push constant block
    struct PushConstants {
        float resolutionX, resolutionY;
        float time;
        float viewportW, viewportH;
        float regionX, regionY;
    };

    // Audio-thread → render-thread float FIFO
    std::unique_ptr<juce::AbstractFifo> fifo;
    std::vector<float>                  fifoBuffer;
    std::vector<float>                  storageData;

    juce::Image displayImage;
};

} // namespace jvk
