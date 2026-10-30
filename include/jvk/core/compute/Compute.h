/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Compute.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

// =============================================================================
// jvk::compute — general work on the GPU: kernels (GLSL compute shaders,
// built by jvk_compute_shaders() for the backends the build has) over
// buffers, on any backend (COMPUTE.md). No JUCE graphics, nothing about any
// one use.
//
//   auto context = compute::Context::create (&why);                     // nullptr: no GPU (and why)
//   auto x = context->buffer (bytes, compute::Usage::Shared);           // mapped: write it from the CPU
//   auto scale = context->kernel (mykernels::scale);                    // reflected at build time
//   auto stream = context->stream();
//   stream->dispatch (scale, compute::Groups::cover (n, 64, context->limits()), { x.range() }, Push { n, 2.0f });
//   stream->submit();
//   while (stream->wait (0.01) == compute::Stream::Result::pending) { /* stop if asked */ }
//
// The same kernels and buffers run in frames too: jvk::Graphics::dispatch,
// and Shader::set(name, range) for a draw that reads a buffer.
//
// Threads: a Context is safe to use from any thread. A Stream is one
// thread's at a time; many streams, on many threads, may be in flight.
// Buffers, Kernels and Streams each hold what they need: they may go in any
// order, the Context included, and the GPU memory of a Buffer or Kernel still
// in use by a submitted stream or frame stays until the GPU is done with it.
//
// Order: a stream's commands start in order but may overlap. Put barrier()
// between a command that writes memory and any later one that reads or
// writes it: kernels, copies, fills, uploads and downloads alike.
// =============================================================================

namespace jvk::compute {

// "To the end": a Range's or a flush's size.
inline constexpr uint64_t kWhole = ~uint64_t(0);

/** A kernel as jvk_compute_shaders() built it: its SPIR-V and MSL (each only
    where the build has a backend that takes it) and its interface, reflected
    from the SPIR-V at build time. */
struct KernelSource
{
    const char*                         name = nullptr;
    std::span<const uint32_t>           spirv;
    std::string_view                    msl;                   // entry point main0
    std::span<const gpu::KernelBinding> bindings;              // set 0, ascending
    uint16_t                            constantsBytes = 0;    // its push-constant block
    uint32_t                            groupSize[3] { 1, 1, 1 };
    uint32_t                            groupSizeId[3] { ~0u, ~0u, ~0u };   // the constant that sets each, if one does
    std::span<const gpu::SpecConstant>  constants;             // declared: id, type, default
    bool                                bufferSizes = false;   // its MSL reads bindings' sizes
};

/** What the GPU can do. Size work by these; never assume them. */
struct Limits
{
    juce::String device;               // the GPU's name
    uint32_t maxGroupSize[3] {};       // threads a workgroup, by dimension
    uint32_t maxGroupThreads = 0;      // threads a workgroup, in all
    uint32_t maxGroups[3] {};          // workgroups a dispatch, by dimension
    uint32_t sharedBytes = 0;          // shared memory a workgroup
    uint32_t pushBytes = 0;            // push constants a kernel
    uint32_t maxBindings = 0;          // buffers a dispatch binds
    uint32_t subgroupSize = 0;         // threads a subgroup (0: unknown)
    uint64_t storageAlignment = 4;     // a bound range's offset is a multiple of this
    bool unifiedMemory = false;        // the GPU's memory is the CPU's (Apple silicon, integrated GPUs): Usage::Shared costs nothing
    bool ownQueue = false;             // streams have a queue of their own (false: behind the editors' frames)
    double timestampNs = 0.0;          // > 0: time() works, at this resolution
};

/** Where a buffer lives and who maps it. */
enum class Usage
{
    Device,    // the GPU's alone, the fastest for kernels: filled and read through a Stream (upload, download, copy, fill) or by kernels
    Upload,    // mapped: the CPU writes, the GPU reads (on a discrete GPU, across the bus)
    Readback,  // mapped and cached: the GPU writes, the CPU reads after a wait
    Shared     // mapped both ways: with Limits::unifiedMemory the GPU's own memory, else the CPU's (slow for big kernels on a discrete GPU)
};

/** A piece of a buffer, as kernels, copies and draws take it. */
struct Range
{
    gpu::BufferId buffer;
    uint64_t      offset = 0;
    uint64_t      bytes = 0;

    gpu::BufferBinding binding() const noexcept { return { buffer, offset, bytes }; }
};

class Context;
class Stream;

/** GPU memory, freed when it goes. Move-only. */
class Buffer
{
public:
    Buffer() = default;
    ~Buffer() { release(); }
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    bool          valid()  const noexcept { return static_cast<bool>(id_); }
    gpu::BufferId handle() const noexcept { return id_; }
    uint64_t      size()   const noexcept { return size_; }

    /** The CPU's view of it (nullptr for Usage::Device). */
    void* data() const noexcept { return mapped_; }
    template <typename T> T* as() const noexcept { return static_cast<T*>(mapped_); }

    /** All of it, or `bytes` (to the end, by default) from `offset`. */
    Range range(uint64_t offset = 0, uint64_t bytes = kWhole) const noexcept
    {
        return { id_, offset, bytes == kWhole ? size_ - offset : bytes };
    }
    /** `count` elements of T from element `first`. */
    template <typename T> Range elements(size_t first, size_t count) const noexcept
    {
        return { id_, static_cast<uint64_t>(first * sizeof(T)), static_cast<uint64_t>(count * sizeof(T)) };
    }

    /** After the CPU writes mapped memory the GPU will read, and before the CPU reads what the GPU
        wrote: no-ops on coherent memory (Metal, most host memory), required elsewhere. */
    void written(uint64_t offset = 0, uint64_t bytes = kWhole) const;
    void toRead(uint64_t offset = 0, uint64_t bytes = kWhole) const;

private:
    friend class Context;
    void release();

    std::shared_ptr<jvk::Context> context_;
    gpu::BufferId id_;
    uint64_t      size_ = 0;
    void*         mapped_ = nullptr;
};

/** A specialization constant: a kernel's `layout(constant_id = id) const` set when it is made. */
struct Constant
{
    uint32_t id = 0;
    uint32_t bits = 0;   // the value's 32 bits: an int's, a uint's or a float's (a bool: nonzero is true)
    static Constant of(uint32_t id, uint32_t v) { return { id, v }; }
    static Constant of(uint32_t id, int32_t v) { return { id, static_cast<uint32_t>(v) }; }
    static Constant of(uint32_t id, float v) { uint32_t b; std::memcpy(&b, &v, 4); return { id, b }; }
    static Constant of(uint32_t id, bool v) { return { id, v ? 1u : 0u }; }
};

/** A compute pipeline made from a KernelSource: storage and uniform buffers in set 0 (a dispatch
    binds them in binding order), one push-constant block, its workgroup size. Move-only. */
class Kernel
{
public:
    Kernel() = default;
    ~Kernel() { release(); }
    Kernel(Kernel&& other) noexcept;
    Kernel& operator=(Kernel&& other) noexcept;
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;

    bool valid() const noexcept { return static_cast<bool>(id_); }
    gpu::KernelId handle() const noexcept { return id_; }
    const juce::String& name() const noexcept { return name_; }
    uint32_t bindings()  const noexcept { return bindings_; }   // ranges a dispatch gives: binding 0 to the highest declared
    uint32_t pushBytes() const noexcept { return pushBytes_; }
    const std::array<uint32_t, 3>& groupSize() const noexcept { return groupSize_; }

    /** The renderer-side context it lives on: a frame dispatches it only on the same backend. */
    const jvk::Context* owner() const noexcept { return context_.get(); }

private:
    friend class Context;
    void release();

    std::shared_ptr<jvk::Context> context_;
    gpu::KernelId id_;
    juce::String  name_;
    uint32_t      bindings_ = 0;
    uint32_t      pushBytes_ = 0;
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
        const uint64_t across = std::max<uint64_t>(1, std::min<uint64_t>(groups, std::max<uint32_t>(limits.maxGroups[0], 1u)));
        return { static_cast<uint32_t>(across), static_cast<uint32_t>((groups + across - 1) / across), 1 };
    }
};

/** Commands for the GPU: recorded, submitted, then waited for. Recording is CPU work alone, so a
    batch can be recorded while the last one runs; submit() waits for the last one first. Each
    stream keeps its own command memory, staging and timestamps on the backend, so streams on
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
        dispatch(kernel, groups, bindings, &push, static_cast<uint32_t>(sizeof(Push)));
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

    /** Sends what is recorded to the GPU, once the last batch is done. Nothing recorded: nothing sent. */
    void submit();
    /** Waits up to `seconds` (negative: as long as it takes) for what was submitted. done: finished,
        downloads delivered, times() read. pending: still running. lost: the GPU was lost; nothing more
        will run on it. */
    Result wait(double seconds = -1.0);
    bool busy() const noexcept { return submitted_; }

    /** The last finished batch's timestamps: each label and its time in ms since the batch began. */
    struct Time { const char* label; double ms; };
    const std::vector<Time>& times() const noexcept { return times_; }

private:
    friend class Context;
    Stream(std::shared_ptr<jvk::Context> context, std::unique_ptr<gpu::ComputeStream> stream);
    uint32_t pushUpload(const void* data, uint64_t bytes);

    std::shared_ptr<jvk::Context>       context_;
    std::unique_ptr<gpu::ComputeStream> stream_;

    std::vector<gpu::ComputeCommand> commands_;
    std::vector<gpu::Dispatch>       dispatches_;
    std::vector<std::byte>           constants_, uploads_;
    std::vector<const char*>         labels_, submittedLabels_;
    std::vector<Time>                times_;
    bool submitted_ = false;
    bool lost_ = false;   // a submit that never ran: the next wait() says lost
};

/** The GPU, for compute. Made once (it is cheap to keep) and shared by everything that computes. */
class Context
{
public:
    /** On the backend the editors draw with (jvk::defaultBackend()); nullptr when there is no usable
        GPU, and `whyNot` says why. */
    static std::unique_ptr<Context> create(juce::String* whyNot = nullptr);
    /** On `backend`. */
    static std::unique_ptr<Context> create(std::shared_ptr<gpu::Backend> backend, juce::String* whyNot = nullptr);
    ~Context() = default;

    const Limits& limits() const noexcept { return limits_; }
    gpu::Backend& backend() const noexcept { return context_->backend(); }

    /** `bytes` of memory (invalid when it could not be had: check valid()). */
    Buffer buffer(uint64_t bytes, Usage usage);

    /** A kernel, named for diagnostics (its file's name by default), its specialization constants set
        (any it doesn't declare are ignored: the optimizer may have taken them out). Invalid (and
        logged) when this backend can't run it: no form of it in the backend's shader format, a
        workgroup or push-constant block past the limits, or the driver refuses it. */
    Kernel kernel(const KernelSource& source, const juce::String& name = {}, std::initializer_list<Constant> constants = {});
    Kernel kernel(const KernelSource& source, std::initializer_list<Constant> constants) { return kernel(source, {}, constants); }

    /** nullptr when the backend can't make one. */
    std::unique_ptr<Stream> stream();

private:
    explicit Context(std::shared_ptr<jvk::Context> context);

    std::shared_ptr<jvk::Context> context_;
    Limits limits_;
};

} // namespace jvk::compute

// For the headers jvk_compute_shaders() generates (tools/jvk_kernel.cpp): a
// kernel's SPIR-V and MSL where the build has the backend that takes each,
// else empty, as the built-in shaders are (core/shaders/Bundle.h).
#if JVK_VULKAN
 #define JVK_KERNEL_SPIRV(kernel) std::span<const uint32_t>(kernel##_spirv)
#else
 #define JVK_KERNEL_SPIRV(kernel) std::span<const uint32_t>()
#endif
#if JVK_METAL
 #define JVK_KERNEL_MSL(kernel) std::string_view(kernel##_msl, sizeof(kernel##_msl) - 1)
#else
 #define JVK_KERNEL_MSL(kernel) std::string_view()
#endif
