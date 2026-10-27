#pragma once
// =============================================================================
// jvk::compute — general work on the GPU: kernels (GLSL compute shaders,
// compiled to SPIR-V by jvk_compute_shaders() at build time) over buffers.
// No JUCE graphics, no render passes, nothing about any one use.
//
//   auto context = compute::Context::create (jvk::Device::acquire());   // nullptr: no GPU (and why)
//   auto x = context->buffer (bytes, compute::Usage::Shared);           // mapped: write it from the CPU
//   auto scale = context->kernel (mykernels::scale, "scale");           // reflected: its bindings and push constants
//   auto stream = context->stream();
//   stream->dispatch (scale, compute::Groups::cover (n, 64, context->limits()), { x.range() }, Push { n, 2.0f });
//   stream->submit();
//   while (stream->wait (0.01) == compute::Stream::Result::pending) { /* stop if asked */ }
//
// Threads: a Context is safe to use from any thread (buffers and kernels are
// made through VMA and the device, both internally synchronized). A Stream is
// one thread's at a time; many streams, on many threads, may be in flight.
// Every Buffer, Kernel and Stream must go before its Context does.
//
// Order: a stream's commands start in order but may overlap. Put barrier()
// between a command that writes memory and any later one that reads or
// writes it: kernels, copies, fills, uploads and downloads alike.
// =============================================================================

namespace jvk::compute {

/** What the GPU can do. Size work by these; never assume them. */
struct Limits
{
    juce::String device;               // the GPU's name
    uint32_t apiVersion = 0;           // the Vulkan version in use
    uint32_t maxGroupSize[3] {};       // threads a workgroup, by dimension
    uint32_t maxGroupThreads = 0;      // threads a workgroup, in all
    uint32_t maxGroups[3] {};          // workgroups a dispatch, by dimension
    uint32_t sharedBytes = 0;          // shared memory a workgroup
    uint32_t pushBytes = 0;            // push constants a kernel
    uint32_t subgroupSize = 0;         // threads a subgroup (0: unknown, on Vulkan 1.0)
    VkDeviceSize storageAlignment = 4; // a bound range's offset is a multiple of this
    bool unifiedMemory = false;        // the GPU's memory is the CPU's (Apple silicon, integrated GPUs): Usage::Shared costs nothing
    bool ownQueue = false;             // compute has a queue of its own (false: it shares the editors' graphics queue)
    double timestampNs = 0.0;          // a timestamp tick in nanoseconds; 0: no timestamps on the compute queue
};

/** Where a buffer lives and who maps it. */
enum class Usage
{
    Device,    // the GPU's alone, the fastest for kernels: filled and read through a Stream (upload, download, copy, fill) or by kernels
    Upload,    // mapped: the CPU writes, the GPU reads (on a discrete GPU, across the bus)
    Readback,  // mapped and cached: the GPU writes, the CPU reads after a wait
    Shared     // mapped both ways: with Limits::unifiedMemory the GPU's own memory, else the CPU's (slow for big kernels on a discrete GPU)
};

/** A piece of a buffer, as kernels and copies take it. */
struct Range
{
    VkBuffer     buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize bytes = 0;
};

class Context;
class Stream;

/** GPU memory, VMA-allocated, freed when it goes. Move-only. */
class Buffer
{
public:
    Buffer() = default;
    ~Buffer();
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    bool         valid()  const noexcept { return buffer_ != VK_NULL_HANDLE; }
    VkBuffer     handle() const noexcept { return buffer_; }
    VkDeviceSize size()   const noexcept { return size_; }

    /** The CPU's view of it (nullptr for Usage::Device). */
    void* data() const noexcept { return mapped_; }
    template <typename T> T* as() const noexcept { return static_cast<T*>(mapped_); }

    /** All of it, or `bytes` (to the end, by default) from `offset`. */
    Range range(VkDeviceSize offset = 0, VkDeviceSize bytes = VK_WHOLE_SIZE) const noexcept
    {
        return { buffer_, offset, bytes == VK_WHOLE_SIZE ? size_ - offset : bytes };
    }
    /** `count` elements of T from element `first`. */
    template <typename T> Range elements(size_t first, size_t count) const noexcept
    {
        return { buffer_, (VkDeviceSize) (first * sizeof(T)), (VkDeviceSize) (count * sizeof(T)) };
    }

    /** After the CPU writes mapped memory the GPU will read, and before the CPU reads what the GPU
        wrote: no-ops on coherent memory (Apple silicon, most host memory), required elsewhere. */
    void written(VkDeviceSize offset = 0, VkDeviceSize bytes = VK_WHOLE_SIZE) const;
    void toRead(VkDeviceSize offset = 0, VkDeviceSize bytes = VK_WHOLE_SIZE) const;

private:
    friend class Context;
    std::shared_ptr<Device> device_;
    VkBuffer      buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = nullptr;
    VkDeviceSize  size_ = 0;
    void*         mapped_ = nullptr;
};

/** A specialization constant: a kernel's `layout(constant_id = id) const` set when it is made. */
struct Constant
{
    uint32_t id = 0;
    uint32_t bits = 0;   // the value's 32 bits: an int's, a uint's or a float's
    static Constant of(uint32_t id, uint32_t v) { return { id, v }; }
    static Constant of(uint32_t id, int32_t v) { return { id, (uint32_t) v }; }
    static Constant of(uint32_t id, float v) { uint32_t b; std::memcpy(&b, &v, 4); return { id, b }; }
};

/** A compute pipeline made from SPIR-V, its interface read from the SPIR-V itself: storage and
    uniform buffers in set 0 (a dispatch binds them in binding order), one push-constant block, and
    its workgroup size. Move-only. */
class Kernel
{
public:
    Kernel() = default;
    ~Kernel();
    Kernel(Kernel&& other) noexcept;
    Kernel& operator=(Kernel&& other) noexcept;
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;

    bool valid() const noexcept { return pipeline_ != VK_NULL_HANDLE; }
    const juce::String& name() const noexcept { return name_; }
    uint32_t bindings()  const noexcept { return (uint32_t) types_.size(); }   // ranges a dispatch gives: binding 0 to the highest declared
    uint32_t pushBytes() const noexcept { return pushBytes_; }
    const std::array<uint32_t, 3>& groupSize() const noexcept { return groupSize_; }

private:
    friend class Context;
    friend class Stream;
    std::shared_ptr<Device> device_;
    juce::String name_;
    VkPipeline            pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout      layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorType> types_;   // by binding; VK_DESCRIPTOR_TYPE_MAX_ENUM: declared by none (the optimizer took it out)
    uint32_t storageCount_ = 0, uniformCount_ = 0;
    uint32_t pushBytes_ = 0;
    std::array<uint32_t, 3> groupSize_ { 1, 1, 1 };
};

/** How many workgroups a dispatch runs. */
struct Groups
{
    uint32_t x = 1, y = 1, z = 1;

    /** Enough workgroups of `size` threads for `threads`, spread over y where x's limit is passed.
        A kernel finds its thread with jvkThread() (jvk/compute.glsl) and returns past the count. */
    static Groups cover(uint64_t threads, uint32_t size, const Limits& limits)
    {
        const uint64_t groups = (threads + size - 1) / size;
        const uint64_t across = std::max<uint64_t>(1, std::min<uint64_t>(groups, limits.maxGroups[0]));
        return { (uint32_t) across, (uint32_t) ((groups + across - 1) / across), 1 };
    }
};

/** Commands for the GPU, recorded, submitted, then waited for; then recorded again. Each stream
    keeps its own command buffer, descriptor pools, staging memory and timestamps, so streams on
    different threads never contend except at the queue. */
class Stream
{
public:
    enum class Result { done, pending, lost };

    ~Stream();   // waits for its work in flight
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    /** Runs `kernel` over `groups`, its bindings given in binding order (each offset a multiple of
        Limits::storageAlignment), `push` its push constants (exactly its block's size). */
    void dispatch(const Kernel& kernel, Groups groups, std::initializer_list<Range> bindings,
                  const void* push = nullptr, uint32_t pushBytes = 0);
    template <typename Push>
    void dispatch(const Kernel& kernel, Groups groups, std::initializer_list<Range> bindings, const Push& push)
    {
        static_assert(std::is_trivially_copyable_v<Push>);
        dispatch(kernel, groups, bindings, &push, (uint32_t) sizeof(Push));
    }

    /** Everything recorded before is finished, and its writes seen, before anything after starts. */
    void barrier();

    /** `range` set to `word`, every 4 bytes (offset and size multiples of 4). */
    void fill(Range range, uint32_t word = 0);
    void copy(Range from, Range to);
    /** `to.bytes` from `from`, copied now (so `from` may go at once), into `to` when the stream runs. */
    void upload(Range to, const void* from);
    /** `from.bytes` of `from` copied into `to` when the stream runs; `to` must live until wait() says done. */
    void download(Range from, void* to);

    /** A GPU timestamp: when everything recorded before it has finished. times() reads them. */
    void time(const char* label);

    /** Sends what is recorded to the GPU. Nothing recorded: nothing sent. */
    void submit();
    /** Waits up to `seconds` (negative: as long as it takes) for what was submitted. done: finished,
        downloads delivered, times() read, and the stream is ready to record again. pending: still
        running. lost: the GPU was lost (jvk::Device::isLost()); nothing more will run on it. */
    Result wait(double seconds = -1.0);
    bool busy() const noexcept { return state_ == State::submitted; }

    /** The last finished submit's timestamps: each label and its time in ms since the stream began. */
    struct Time { const char* label; double ms; };
    const std::vector<Time>& times() const noexcept { return times_; }

private:
    friend class Context;
    explicit Stream(Context& context);
    void begin();
    void finish(bool delivered);
    VkDescriptorSet allocate(const Kernel& kernel);
    Range stage(Usage usage, VkDeviceSize bytes, void*& mapped);

    enum class State { idle, recording, submitted };

    Context& context_;
    std::shared_ptr<Device> device_;
    VkDevice        vk_ = VK_NULL_HANDLE;
    VkCommandPool   pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence         fence_ = VK_NULL_HANDLE;
    VkQueryPool     queries_ = VK_NULL_HANDLE;
    uint32_t        queryCount_ = 0;
    State           state_ = State::idle;
    bool            lost_ = false;   // a submit that never ran: the next wait() says lost

    struct DescriptorPool { VkDescriptorPool pool; uint32_t sets, storage, uniform; };
    std::vector<DescriptorPool> descriptorPools_;
    size_t poolInUse_ = 0;

    struct Staging { std::vector<Buffer> blocks; size_t block = 0; VkDeviceSize head = 0; };
    Staging uploads_, readbacks_;
    struct Download { const void* from; size_t bytes; void* to; };   // staged at `from`, delivered to `to`
    std::vector<Download> downloads_;

    std::vector<const char*> labels_;
    std::vector<Time> times_;
};

/** The GPU, for compute. Made once (it is cheap to keep) and shared by everything that computes. */
class Context
{
public:
    /** nullptr when there is no usable GPU, and `whyNot` says why. */
    static std::unique_ptr<Context> create(std::shared_ptr<Device> device, juce::String* whyNot = nullptr);
    ~Context();

    const Limits& limits() const noexcept { return limits_; }
    Device& device() const noexcept { return *device_; }

    /** `bytes` of memory (invalid when it could not be had: check valid()). */
    Buffer buffer(VkDeviceSize bytes, Usage usage);

    /** A kernel from SPIR-V (an array jvk_compute_shaders() embedded), named for diagnostics, its
        specialization constants set. Invalid (and logged) when the SPIR-V is not a compute shader
        jvk can bind, or the driver refuses it. */
    Kernel kernel(std::span<const uint32_t> spirv, const juce::String& name, std::initializer_list<Constant> constants = {});

    std::unique_ptr<Stream> stream();

private:
    friend class Stream;
    explicit Context(std::shared_ptr<Device> device);

    std::shared_ptr<Device> device_;
    Limits limits_;
    uint64_t timestampMask_ = 0;
    std::atomic<int> streams_ { 0 };
};

} // namespace jvk::compute
