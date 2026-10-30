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

namespace jvk::gpu
{

// =============================================================================
// Compute — buffers, kernels and the work that runs them (COMPUTE.md)
//
// Kernels run in two places, with the same Dispatch:
//
//   a ComputeStream   beside the rendering, on a queue of its own where the
//                     GPU has one. A ComputePacket of commands: dispatches,
//                     barriers, fills, copies, uploads, downloads, timestamps.
//   a frame           a Pass of kind Compute (FramePacket.h): dispatches in
//                     paint order, on the frame's queue, ordered against its
//                     draws by the backend.
//
// Buffers are the backend's memory behind BufferIds, as textures are behind
// TextureIds: destroyBuffer is always safe for one a submitted stream or
// frame still uses, and the backend frees it once the GPU is done.
//
// Binding convention (COMPUTE.md): a kernel binds buffers in set 0 by
// binding number b: Vulkan set 0 binding b; Metal [[buffer(b)]], its push
// constants at [[buffer(24)]] and, where it reads a runtime array's length,
// each binding's size in bytes at [[buffer(25)]] (a uint per binding, by b).
// =============================================================================

// Buffers a dispatch binds: every API's floor is lower (Vulkan guarantees 4
// storage buffers a stage), so a kernel is checked against
// ComputeCaps::maxBindings as well.
inline constexpr uint32_t kMaxKernelBindings = 16;

struct ComputeCaps
{
    bool     supported     = false;   // buffers, kernels and streams
    bool     frameDispatch = false;   // Compute passes in a frame packet
    bool     ownQueue      = false;   // streams run beside frames, not behind them
    bool     unifiedMemory = false;   // the GPU's memory is the CPU's: Shared costs nothing

    uint32_t maxGroupSize[3] {};      // threads a workgroup, by dimension
    uint32_t maxGroupThreads = 0;     // threads a workgroup, in all
    uint32_t maxGroups[3] {};         // workgroups a dispatch, by dimension
    uint32_t sharedBytes     = 0;     // shared (threadgroup) memory a workgroup
    uint32_t constantsBytes  = 0;     // a kernel's push constants
    uint32_t maxBindings     = 0;     // at most kMaxKernelBindings
    uint32_t subgroupSize    = 0;     // 0: unknown
    uint64_t maxBufferBytes  = 0;
    double   timestampNs     = 0.0;   // > 0: streams time their Timestamp commands, at this resolution

    const char* deviceName = "";      // the GPU's name; the backend owns the text
};

// ---- Buffers ----------------------------------------------------------------

enum class BufferMemory : uint8_t
{
    Device,     // the GPU's alone, the fastest for kernels; filled and read through streams or kernels
    Upload,     // mapped: the CPU writes, the GPU reads (on a discrete GPU, across the bus)
    Readback,   // mapped and cached: the GPU writes, the CPU reads after a wait
    Shared,     // mapped both ways: the GPU's own memory with unifiedMemory, else the CPU's
};

struct BufferDesc
{
    uint64_t     size   = 0;
    BufferMemory memory = BufferMemory::Device;
    const char*  label  = nullptr;   // debug name; may be null
};

// A piece of a buffer: what a dispatch binds, and what copies read and write.
struct BufferBinding
{
    BufferId buffer;
    uint64_t offset = 0, size = 0;

    constexpr bool operator==(const BufferBinding&) const = default;
};

// ---- Kernels ------------------------------------------------------------------

struct KernelBinding
{
    uint8_t     binding = 0;                       // set 0, this binding number
    BindingKind kind    = BindingKind::StorageBuffer;   // StorageBuffer (read and written) or UniformBuffer
    bool        writes  = false;                   // a storage buffer not declared readonly
};

enum class ScalarType : uint8_t { Bool, Int, UInt, Float };

// A specialization constant (`layout(constant_id = id) const`): its type as
// declared, and 32 bits of value (an int's, a uint's or a float's; 0 or 1 for
// a bool).
struct SpecConstant
{
    uint32_t   id   = 0;
    ScalarType type = ScalarType::Int;
    uint32_t   bits = 0;
};

struct KernelDesc
{
    const char* name = nullptr;   // debug name; may be null

    // In `format`, the backend's Caps::shaderFormat: SPIR-V words, or source
    // text (MSL). The bytes need only outlive createKernel().
    ShaderFormat               format = ShaderFormat::SPIRV;
    std::span<const std::byte> code;
    const char*                entry = "main";   // MSL: "main0"

    std::span<const KernelBinding> bindings;      // ascending binding numbers, each < kMaxKernelBindings
    uint16_t                       constantsBytes = 0;
    uint32_t                       groupSize[3] { 1, 1, 1 };   // final: after specialization
    std::span<const SpecConstant>  constants;     // the values it is made with
    bool                           bufferSizes = false;        // MSL: it reads [[buffer(25)]]
};

// ---- Dispatches ---------------------------------------------------------------

// A kernel over `groups` workgroups, its bindings 0..bindingCount-1 in
// binding order (a number the kernel doesn't declare is skipped), each offset
// a multiple of Caps::storageAlignment.
struct Dispatch
{
    KernelId      kernel;
    uint32_t      groups[3] { 1, 1, 1 };
    uint8_t       bindingCount = 0;
    BufferBinding bindings[kMaxKernelBindings] {};
    Range         constants;     // in the packet's constant data: the kernel's push constants
};

// ---- Streams --------------------------------------------------------------------

enum class ComputeOp : uint8_t
{
    Dispatch,
    Barrier,     // everything before finishes, its writes seen, before anything after starts
    Fill,        // dst set to `word`, every 4 bytes
    Copy,        // src.size bytes from src to dst
    Upload,      // `data` (in uploadData) into dst
    Download,    // src into `to`, delivered when a wait says Done
    Timestamp,   // when everything before has finished
};

struct ComputeCommand
{
    ComputeOp     op = ComputeOp::Barrier;
    uint32_t      dispatch = 0;    // Dispatch: into ComputePacket::dispatches
    BufferBinding src, dst;
    uint32_t      word = 0;        // Fill
    Range         data;            // Upload: in ComputePacket::uploadData
    void*         to = nullptr;    // Download: src.size bytes; must live until a wait says Done or Lost
};

// Valid for ComputeStream::submit alone: the backend copies what it keeps.
struct ComputePacket
{
    std::span<const ComputeCommand> commands;
    std::span<const Dispatch>       dispatches;
    std::span<const std::byte>      constantData;
    std::span<const std::byte>      uploadData;
};

// Commands run on the compute queue: one submission in flight at a time.
// Each stream has its own command memory, staging and descriptor memory, so
// streams on different threads never contend except at the queue. One
// thread at a time per stream.
class ComputeStream
{
public:
    enum class Status : uint8_t { Done, Pending, Lost };

    virtual ~ComputeStream() = default;   // waits for its submission

    // Runs `packet`. Only after the last submission was waited for (Done or
    // Lost). False when nothing was submitted (the GPU is lost).
    virtual bool submit(const ComputePacket& packet) = 0;

    // Waits up to `seconds` (negative: as long as it takes). Done: finished,
    // downloads delivered, timestamps read (Done too with nothing in flight).
    // Pending: still running. Lost: the GPU is gone; nothing more runs.
    virtual Status wait(double seconds) = 0;

    // The last finished submission's Timestamps: ms since it began, one per
    // Timestamp command in order. Empty without timestamps.
    virtual std::span<const double> timestamps() const = 0;
};

} // namespace jvk::gpu
