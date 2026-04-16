#pragma once

namespace jvk {

class Resource {
public:
    Resource() = default;

    Resource(Memory::L1& pool, Memory::L1::Allocation alloc)
        : pool_(&pool), alloc_(alloc) {}

    virtual ~Resource() { release(); }

    Resource(Resource&& o) noexcept : pool_(o.pool_), alloc_(o.alloc_)
    { o.pool_ = nullptr; o.alloc_ = {}; }

    Resource& operator=(Resource&& o) noexcept
    {
        if (this != &o) {
            release();
            pool_ = o.pool_; alloc_ = o.alloc_;
            o.pool_ = nullptr; o.alloc_ = {};
        }
        return *this;
    }

    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;

    Memory::L1::Allocation allocation() const { return alloc_; }
    bool valid() const { return pool_ != nullptr && alloc_.memory != VK_NULL_HANDLE; }

protected:
    Memory::L1*            pool_  = nullptr;
    Memory::L1::Allocation alloc_ {};

    void release()
    {
        if (pool_) pool_->free(alloc_);
        pool_ = nullptr;
        alloc_ = {};
    }
};


class Image : public Resource {
public:
    Image() = default;

    Image(Memory::L1& pool, VkDevice device,
          uint32_t width, uint32_t height, VkFormat format,
          VkImageUsageFlags usage,
          VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
          VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
          VkFilter filter = VK_FILTER_LINEAR,
          VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
        : device_(device), w_(width), h_(height)
    {
        VkImageCreateInfo ci {};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = format;
        ci.extent = { width, height, 1 };
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = samples;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = usage;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vkCreateImage(device, &ci, nullptr, &image_) != VK_SUCCESS) return;

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, image_, &req);
        alloc_ = pool.alloc(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (alloc_.memory == VK_NULL_HANDLE) {
            vkDestroyImage(device, image_, nullptr);
            image_ = VK_NULL_HANDLE;
            return;
        }
        pool_ = &pool;
        vkBindImageMemory(device, image_, alloc_.memory, alloc_.offset);

        VkImageViewCreateInfo vi {};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = image_;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format;
        vi.subresourceRange = { aspect, 0, 1, 0, 1 };
        vkCreateImageView(device, &vi, nullptr, &view_);

        if (samples == VK_SAMPLE_COUNT_1_BIT && (usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
            VkSamplerCreateInfo si {};
            si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            si.magFilter = filter;
            si.minFilter = filter;
            si.addressModeU = addressMode;
            si.addressModeV = addressMode;
            si.addressModeW = addressMode;
            si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
            si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            vkCreateSampler(device, &si, nullptr, &sampler_);
        }
    }

    ~Image() override
    {
        if (sampler_ != VK_NULL_HANDLE) vkDestroySampler(device_, sampler_, nullptr);
        if (view_ != VK_NULL_HANDLE) vkDestroyImageView(device_, view_, nullptr);
        if (image_ != VK_NULL_HANDLE) vkDestroyImage(device_, image_, nullptr);
        // ~Resource() frees L1 allocation
    }

    Image(Image&& o) noexcept
        : Resource(std::move(o)), device_(o.device_),
          image_(o.image_), view_(o.view_), sampler_(o.sampler_),
          w_(o.w_), h_(o.h_)
    {
        o.device_ = VK_NULL_HANDLE;
        o.image_ = VK_NULL_HANDLE;
        o.view_ = VK_NULL_HANDLE;
        o.sampler_ = VK_NULL_HANDLE;
        o.w_ = 0; o.h_ = 0;
    }

    Image& operator=(Image&& o) noexcept
    {
        if (this != &o) {
            if (sampler_ != VK_NULL_HANDLE) vkDestroySampler(device_, sampler_, nullptr);
            if (view_ != VK_NULL_HANDLE) vkDestroyImageView(device_, view_, nullptr);
            if (image_ != VK_NULL_HANDLE) vkDestroyImage(device_, image_, nullptr);
            Resource::operator=(std::move(o));
            device_ = o.device_; image_ = o.image_; view_ = o.view_;
            sampler_ = o.sampler_; w_ = o.w_; h_ = o.h_;
            o.device_ = VK_NULL_HANDLE; o.image_ = VK_NULL_HANDLE;
            o.view_ = VK_NULL_HANDLE; o.sampler_ = VK_NULL_HANDLE;
            o.w_ = 0; o.h_ = 0;
        }
        return *this;
    }

    VkImage     image()   const { return image_; }
    VkImageView view()    const { return view_; }
    VkSampler   sampler() const { return sampler_; }
    uint32_t    width()   const { return w_; }
    uint32_t    height()  const { return h_; }

private:
    VkDevice    device_  = VK_NULL_HANDLE;
    VkImage     image_   = VK_NULL_HANDLE;
    VkImageView view_    = VK_NULL_HANDLE;
    VkSampler   sampler_ = VK_NULL_HANDLE;
    uint32_t    w_ = 0, h_ = 0;
};


class Buffer : public Resource {
public:
    Buffer() = default;

    Buffer(Memory::L1& pool, VkDevice device,
           VkDeviceSize size, VkBufferUsageFlags usage)
        : device_(device), size_(size)
    {
        VkBufferCreateInfo ci {};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = size;
        ci.usage = usage;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &ci, nullptr, &buffer_) != VK_SUCCESS) return;

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, buffer_, &req);
        alloc_ = pool.alloc(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (alloc_.memory == VK_NULL_HANDLE) {
            vkDestroyBuffer(device, buffer_, nullptr);
            buffer_ = VK_NULL_HANDLE;
            return;
        }
        pool_ = &pool;
        vkBindBufferMemory(device, buffer_, alloc_.memory, alloc_.offset);
    }

    ~Buffer() override
    {
        if (buffer_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, buffer_, nullptr);
    }

    Buffer(Buffer&& o) noexcept
        : Resource(std::move(o)), device_(o.device_), buffer_(o.buffer_), size_(o.size_)
    {
        o.device_ = VK_NULL_HANDLE; o.buffer_ = VK_NULL_HANDLE; o.size_ = 0;
    }

    Buffer& operator=(Buffer&& o) noexcept
    {
        if (this != &o) {
            if (buffer_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, buffer_, nullptr);
            Resource::operator=(std::move(o));
            device_ = o.device_; buffer_ = o.buffer_; size_ = o.size_;
            o.device_ = VK_NULL_HANDLE; o.buffer_ = VK_NULL_HANDLE; o.size_ = 0;
        }
        return *this;
    }

    VkBuffer     buffer() const { return buffer_; }
    VkDeviceSize size()   const { return size_; }

private:
    VkDevice     device_ = VK_NULL_HANDLE;
    VkBuffer     buffer_ = VK_NULL_HANDLE;
    VkDeviceSize size_   = 0;
};

} // namespace jvk
