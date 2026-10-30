/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: MetalCompute.mm
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

namespace jvk::metal
{

namespace
{

void logComputeOnce(std::atomic<bool>& flag, const juce::String& line)
{
    if (!flag.exchange(true)) diag::log(line);
}

juce::String describeError(NSError* error)
{
    return error != nil ? juce::String::fromUTF8([[error localizedDescription] UTF8String]) : juce::String("unknown error");
}

// A specialization constant as MSL source: what SPIRV-Cross's macro for a
// constant that sizes an array is defined to.
std::string literal(const gpu::SpecConstant& c)
{
    switch (c.type)
    {
        case gpu::ScalarType::Bool: return c.bits != 0 ? "true" : "false";
        case gpu::ScalarType::Int:  return "(" + std::to_string(static_cast<int32_t>(c.bits)) + ")";
        case gpu::ScalarType::UInt: return std::to_string(c.bits) + "u";
        case gpu::ScalarType::Float:
        {
            float f = 0.0f;
            std::memcpy(&f, &c.bits, 4);
            char text[64];
            std::snprintf(text, sizeof(text), "%.9g", static_cast<double>(f));
            std::string s = text;
            if (s.find_first_of(".eEni") == std::string::npos) s += ".0";
            return "(" + s + "f)";
        }
    }
    return "0";
}

// jvk's own kernels, for what a blit can't do on macOS: fill with a word that
// isn't one byte four times; copy where an offset or the size isn't a
// multiple of 4.
const char* const kUtilityMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void jvk_fill(device uint* dst [[buffer(0)]], constant uint2& p [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    if (i < p.y) dst[i] = p.x;
}

kernel void jvk_copy(device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
                     constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    if (i < n) dst[i] = src[i];
}
)msl";

} // namespace

// =============================================================================
// Caps
// =============================================================================

void Backend::initComputeCaps()
{
    auto& c = caps_.compute;
    if (computeQueue_ == nil) return;
    deviceName_ = [[device_ name] UTF8String];

    c.supported     = true;
    c.frameDispatch = true;
    c.ownQueue      = true;
    c.unifiedMemory = unified_;
    const MTLSize most = [device_ maxThreadsPerThreadgroup];
    c.maxGroupSize[0] = static_cast<uint32_t>(most.width);
    c.maxGroupSize[1] = static_cast<uint32_t>(most.height);
    c.maxGroupSize[2] = static_cast<uint32_t>(most.depth);
    // Metal's per-threadgroup total; one pipeline may take fewer (its
    // maxTotalThreadsPerThreadgroup, checked when a kernel is made).
    c.maxGroupThreads = static_cast<uint32_t>(std::min<NSUInteger>(most.width, 1024));
    c.maxGroups[0]    = 0x7fffffffu;
    c.maxGroups[1]    = 65535u;
    c.maxGroups[2]    = 65535u;
    c.sharedBytes     = static_cast<uint32_t>([device_ maxThreadgroupMemoryLength]);
    c.constantsBytes  = 4096;   // setBytes' limit
    c.maxBindings     = gpu::kMaxKernelBindings;
    c.subgroupSize    = [device_ supportsFamily:MTLGPUFamilyApple1] ? 32u : 0u;   // AMD and Intel vary by pipeline
    c.maxBufferBytes  = static_cast<uint64_t>([device_ maxBufferLength]);
    c.timestampNs     = 1.0;      // command buffers' GPU times: seconds as doubles
    c.deviceName      = deviceName_.c_str();
}

// =============================================================================
// Buffers
// =============================================================================

gpu::BufferId Backend::createBuffer(const gpu::BufferDesc& desc)
{
    if (!usable() || !caps_.compute.supported || desc.size == 0) return {};
    MTLResourceOptions options = MTLResourceStorageModeShared;
    switch (desc.memory)
    {
        case gpu::BufferMemory::Device:   options = MTLResourceStorageModePrivate; break;
        case gpu::BufferMemory::Upload:   options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined; break;
        case gpu::BufferMemory::Readback: options = MTLResourceStorageModeShared; break;
        case gpu::BufferMemory::Shared:   options = MTLResourceStorageModeShared; break;
    }
    id<MTLBuffer> buffer = [device_ newBufferWithLength:static_cast<NSUInteger>(desc.size) options:options];
    if (buffer == nil)
    {
        static std::atomic<bool> logged { false };
        logComputeOnce(logged, "metal: a buffer could not be created (" + juce::String((juce::int64) desc.size) + " bytes)");
        return {};
    }
    if (desc.label != nullptr)
        buffer.label = [NSString stringWithUTF8String:desc.label];

    std::lock_guard<std::mutex> lk(tableMutex_);
    BufferSlot* slot = nullptr;
    const uint32_t handle = buffers_.allocate(slot);
    if (handle == 0) { [buffer release]; return {}; }
    slot->buffer = Strong<id<MTLBuffer>>::adopt(buffer);
    slot->size   = desc.size;
    slot->mapped = desc.memory == gpu::BufferMemory::Device ? nullptr : [buffer contents];
    return { handle };
}

void Backend::destroyBuffer(gpu::BufferId buffer)
{
    // Every command buffer that encoded it holds its own reference.
    std::lock_guard<std::mutex> lk(tableMutex_);
    buffers_.release(buffer.value);
}

void* Backend::bufferData(gpu::BufferId buffer)
{
    const BufferSlot* b = buffers_.resolve(buffer.value);
    return b != nullptr ? b->mapped : nullptr;
}

// =============================================================================
// Kernels
// =============================================================================

gpu::KernelId Backend::createKernel(const gpu::KernelDesc& desc)
{
    const juce::String name = desc.name != nullptr ? desc.name : "?";
    const std::string* held = nullptr;   // its library's entry in kernelLibraries_, one use of it held once found or made
    auto refused = [&](const juce::String& why) -> gpu::KernelId
    {
        if (held != nullptr)
        {
            std::lock_guard<std::mutex> lk(tableMutex_);
            releaseKernelLibrary(std::exchange(held, nullptr));
        }
        diag::log("metal: kernel " + name + ": " + why);
        return {};
    };
    if (!usable() || !caps_.compute.supported) return {};
    if (desc.format != gpu::ShaderFormat::MSL || desc.code.empty()) return refused("not MSL");

    KernelSlot k;
    for (const auto& b : desc.bindings)
    {
        if (b.binding >= gpu::kMaxKernelBindings) return refused("binds past binding " + juce::String(gpu::kMaxKernelBindings - 1));
        if (b.kind != gpu::BindingKind::StorageBuffer && b.kind != gpu::BindingKind::UniformBuffer)
            return refused("binds something other than a buffer");
        k.declared[b.binding] = true;
        k.bindingCount = std::max<uint32_t>(k.bindingCount, uint32_t(b.binding) + 1);
    }
    k.constantsBytes = desc.constantsBytes;
    k.bufferSizes    = desc.bufferSizes;
    k.groupSize      = MTLSizeMake(std::max(1u, desc.groupSize[0]), std::max(1u, desc.groupSize[1]), std::max(1u, desc.groupSize[2]));

    // A constant that sizes an array can't be a function constant: SPIRV-Cross
    // makes it a macro, which takes the kernel's value defined ahead of it.
    std::string source(reinterpret_cast<const char*>(desc.code.data()), desc.code.size());
    std::string defines;
    for (const auto& c : desc.constants)
    {
        const std::string macro = "SPIRV_CROSS_CONSTANT_ID_" + std::to_string(c.id);
        if (source.find("#ifndef " + macro + "\n") != std::string::npos)
            defines += "#define " + macro + " " + literal(c) + "\n";
    }
    if (!defines.empty()) source = defines + source;

    // The library of a live kernel of the same source, or a new one.
    Strong<id<MTLLibrary>> library;
    {
        std::lock_guard<std::mutex> lk(tableMutex_);
        if (auto it = kernelLibraries_.find(source); it != kernelLibraries_.end())
        {
            ++it->second.users;
            held = &it->first;
            library = it->second.library;
        }
    }
    if (!library)
    {
        Strong<id<MTLLibrary>> made;
        @autoreleasepool
        {
            NSString* text = [[NSString alloc] initWithBytes:source.data() length:source.size() encoding:NSUTF8StringEncoding];
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.languageVersion = MTLLanguageVersion2_1;
            NSError* err = nil;
            id<MTLLibrary> compiled = [device_ newLibraryWithSource:text options:options error:&err];
            [options release];
            [text release];
            if (compiled == nil) return refused(describeError(err));
            made = Strong<id<MTLLibrary>>::adopt(compiled);
        }
        // Another thread may have made the same source meanwhile: the first one in is shared.
        std::lock_guard<std::mutex> lk(tableMutex_);
        auto [it, inserted] = kernelLibraries_.try_emplace(source);
        if (inserted) it->second.library = std::move(made);
        ++it->second.users;
        held = &it->first;
        library = it->second.library;
    }

    @autoreleasepool
    {
        // Each scalar constant is a function constant of its declared type;
        // values are given only for those the function declares.
        MTLFunctionConstantValues* values = [[MTLFunctionConstantValues alloc] init];
        for (const auto& c : desc.constants)
        {
            if (source.find("[[function_constant(" + std::to_string(c.id) + ")]]") == std::string::npos) continue;
            switch (c.type)
            {
                case gpu::ScalarType::Bool:  { const bool v = c.bits != 0; [values setConstantValue:&v type:MTLDataTypeBool atIndex:c.id]; break; }
                case gpu::ScalarType::Int:   { const int32_t v = static_cast<int32_t>(c.bits); [values setConstantValue:&v type:MTLDataTypeInt atIndex:c.id]; break; }
                case gpu::ScalarType::UInt:  { const uint32_t v = c.bits; [values setConstantValue:&v type:MTLDataTypeUInt atIndex:c.id]; break; }
                case gpu::ScalarType::Float: { float v = 0.0f; std::memcpy(&v, &c.bits, 4); [values setConstantValue:&v type:MTLDataTypeFloat atIndex:c.id]; break; }
            }
        }
        NSError* err = nil;
        id<MTLFunction> fn = [library.get() newFunctionWithName:[NSString stringWithUTF8String:desc.entry != nullptr ? desc.entry : "main0"]
                                                 constantValues:values
                                                          error:&err];
        [values release];
        if (fn == nil) return refused(describeError(err));

        MTLComputePipelineDescriptor* pd = [[MTLComputePipelineDescriptor alloc] init];
        pd.computeFunction = fn;
        pd.label = [NSString stringWithUTF8String:name.toRawUTF8()];
        id<MTLComputePipelineState> state = [device_ newComputePipelineStateWithDescriptor:pd options:MTLPipelineOptionNone
                                                                                reflection:nil error:&err];
        [pd release];
        [fn release];
        if (state == nil) return refused(describeError(err));
        const NSUInteger threads = k.groupSize.width * k.groupSize.height * k.groupSize.depth;
        if (threads > [state maxTotalThreadsPerThreadgroup])
        {
            const auto most = juce::String((int) [state maxTotalThreadsPerThreadgroup]);
            [state release];
            return refused("its workgroup of " + juce::String((int) threads) + " threads passes what this GPU runs of it ("
                           + most + "): a smaller workgroup, or fewer registers");
        }
        k.state = Strong<id<MTLComputePipelineState>>::adopt(state);
    }

    std::lock_guard<std::mutex> lk(tableMutex_);
    KernelSlot* slot = nullptr;
    const uint32_t handle = kernels_.allocate(slot);
    if (handle == 0)
    {
        releaseKernelLibrary(held);
        return {};
    }
    k.library = held;
    *slot = std::move(k);
    return { handle };
}

void Backend::destroyKernel(gpu::KernelId kernel)
{
    // Its pipeline state and its use of its library go now: a command buffer
    // still running it holds the pipeline state, which needs no library.
    std::lock_guard<std::mutex> lk(tableMutex_);
    if (const KernelSlot* k = kernels_.resolve(kernel.value)) releaseKernelLibrary(k->library);
    kernels_.release(kernel.value);
}

void Backend::releaseKernelLibrary(const std::string* key)
{
    if (key == nullptr) return;
    const auto it = kernelLibraries_.find(*key);
    jassert(it != kernelLibraries_.end() && it->second.users > 0);
    if (it != kernelLibraries_.end() && --it->second.users == 0) kernelLibraries_.erase(it);
}

void Backend::makeUtilityStates()
{
    if (utilityTried_) return;
    utilityTried_ = true;
    @autoreleasepool
    {
        NSString* text = [NSString stringWithUTF8String:kUtilityMsl];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion2_1;
        NSError* err = nil;
        id<MTLLibrary> library = [device_ newLibraryWithSource:text options:options error:&err];
        [options release];
        if (library == nil) { diag::log("metal: jvk's fill and copy kernels: " + describeError(err)); return; }
        for (int i = 0; i < 2; ++i)
        {
            id<MTLFunction> fn = [library newFunctionWithName:i == 0 ? @"jvk_fill" : @"jvk_copy"];
            id<MTLComputePipelineState> state = fn != nil ? [device_ newComputePipelineStateWithFunction:fn error:&err] : nil;
            if (fn != nil) [fn release];
            (i == 0 ? fillState_ : copyState_) = Strong<id<MTLComputePipelineState>>::adopt(state);
        }
        [library release];
    }
}

id<MTLComputePipelineState> Backend::fillState()
{
    std::lock_guard<std::mutex> lk(tableMutex_);
    makeUtilityStates();
    return fillState_.get();
}

id<MTLComputePipelineState> Backend::copyState()
{
    std::lock_guard<std::mutex> lk(tableMutex_);
    makeUtilityStates();
    return copyState_.get();
}

// =============================================================================
// Dispatches (streams and frames)
// =============================================================================

bool Backend::encodeDispatch(id<MTLComputeCommandEncoder> enc, const gpu::Dispatch& d, std::span<const std::byte> constants) const
{
    auto skipped = [](const char* why)
    {
        static std::atomic<bool> logged { false };
        logComputeOnce(logged, juce::String("metal: a dispatch was skipped: ") + why);
        return false;
    };
    const KernelSlot* k = kernel(d.kernel);
    if (k == nullptr || !k->state) return skipped("a kernel that no longer exists");
    if (d.bindingCount < k->bindingCount) return skipped("fewer bindings than its kernel declares");

    // Everything it binds must be valid before anything is.
    const BufferSlot* buffers[gpu::kMaxKernelBindings] {};
    for (uint32_t b = 0; b < k->bindingCount; ++b)
    {
        if (!k->declared[b]) continue;
        const auto& r = d.bindings[b];
        buffers[b] = buffer(r.buffer);
        if (buffers[b] == nullptr || !buffers[b]->buffer) return skipped("a buffer that no longer exists");
        if (r.size == 0 || r.offset + r.size > buffers[b]->size) return skipped("a range outside its buffer");
    }

    [enc setComputePipelineState:k->state.get()];
    uint32_t sizes[gpu::kMaxKernelBindings] {};
    for (uint32_t b = 0; b < k->bindingCount; ++b)
    {
        if (buffers[b] == nullptr) continue;
        [enc setBuffer:buffers[b]->buffer.get() offset:static_cast<NSUInteger>(d.bindings[b].offset) atIndex:b];
        sizes[b] = static_cast<uint32_t>(std::min<uint64_t>(d.bindings[b].size, 0xffffffffu));
    }
    if (k->bufferSizes)
        [enc setBytes:sizes length:sizeof(uint32_t) * std::max(1u, k->bindingCount) atIndex:kBufferSizesIndex];
    if (k->constantsBytes > 0 && !d.constants.isEmpty()
        && uint64_t(d.constants.offset) + d.constants.size <= constants.size())
    {
        // At least the kernel's whole block (a shorter range is zero-padded).
        alignas(16) std::byte block[4096] {};
        const uint32_t length = std::min<uint32_t>((std::max(k->constantsBytes, d.constants.size) + 15) & ~15u, sizeof(block));
        std::memcpy(block, constants.data() + d.constants.offset, std::min<uint32_t>(d.constants.size, length));
        [enc setBytes:block length:length atIndex:kConstantsIndex];
    }
    [enc dispatchThreadgroups:MTLSizeMake(std::max(1u, d.groups[0]), std::max(1u, d.groups[1]), std::max(1u, d.groups[2]))
        threadsPerThreadgroup:k->groupSize];
    return true;
}

std::unique_ptr<gpu::ComputeStream> Backend::createComputeStream()
{
    if (!usable() || !caps_.compute.supported) return nullptr;
    return std::make_unique<ComputeStream>(shared_from_this());
}

// =============================================================================
// ComputeStream
// =============================================================================

ComputeStream::ComputeStream(std::shared_ptr<Backend> backend) : backend_(std::move(backend))
{
    done_ = dispatch_semaphore_create(0);
}

ComputeStream::~ComputeStream()
{
    // Bounded: a GPU that never answers is given up on. Each command buffer
    // holds what it uses (its staging included) and the semaphore it signals.
    if (!segments_.empty() && wait(5.0) == Status::Pending)
        segments_.clear();
    dispatch_release(done_);
}

bool ComputeStream::ensure(Strong<id<MTLBuffer>>& buffer, size_t size, MTLResourceOptions options)
{
    // Grown, never shrunk. The old one goes now: a command buffer still
    // reading it holds its own reference.
    if (size == 0 || (buffer && [buffer.get() length] >= size)) return true;
    const size_t grown = std::max<size_t>({ size, buffer ? [buffer.get() length] * 2 : 0, size_t(1u << 20) });
    id<MTLBuffer> b = [backend_->device() newBufferWithLength:grown options:options];
    buffer = Strong<id<MTLBuffer>>::adopt(b);
    return b != nil;
}

bool ComputeStream::submit(const gpu::ComputePacket& packet)
{
    jassert(segments_.empty());   // the last submission was waited for
    id<MTLCommandQueue> queue = backend_->computeQueue();
    if (!segments_.empty() || queue == nil || backend_->isLost())
    {
        lost_ = backend_->isLost();
        return false;
    }

    // Staging: the packet's upload bytes as they are, and a slot per download.
    size_t downloadBytes = 0;
    for (const auto& c : packet.commands)
        if (c.op == gpu::ComputeOp::Download)
            downloadBytes = ((downloadBytes + 15) & ~size_t(15)) + static_cast<size_t>(c.src.size);
    if (!ensure(upload_, packet.uploadData.size(), MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined)
        || !ensure(readback_, downloadBytes, MTLResourceStorageModeShared))
    {
        static std::atomic<bool> logged { false };
        logComputeOnce(logged, "metal: a compute stream's staging could not be had");
        return false;
    }
    if (!packet.uploadData.empty())
        std::memcpy([upload_.get() contents], packet.uploadData.data(), packet.uploadData.size());
    downloads_.clear();
    stampSegment_.clear();

    @autoreleasepool
    {
        id<MTLCommandBuffer>         cb = nil;
        id<MTLComputeCommandEncoder> ce = nil;
        id<MTLBlitCommandEncoder>    be = nil;
        auto open = [&] { if (cb == nil) cb = [queue commandBuffer]; };
        auto endEncoders = [&]
        {
            if (ce != nil) { [ce endEncoding]; ce = nil; }
            if (be != nil) { [be endEncoding]; be = nil; }
        };
        auto compute = [&]() -> id<MTLComputeCommandEncoder>
        {
            open();
            if (be != nil) { [be endEncoding]; be = nil; }
            if (ce == nil) ce = [cb computeCommandEncoder];   // serial: each dispatch sees the last one's writes
            return ce;
        };
        auto blit = [&]() -> id<MTLBlitCommandEncoder>
        {
            open();
            if (ce != nil) { [ce endEncoding]; ce = nil; }
            if (be == nil) be = [cb blitCommandEncoder];
            return be;
        };
        auto commitSegment = [&]
        {
            open();
            endEncoders();
            dispatch_semaphore_t done = done_;
            dispatch_retain(done);
            [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
                dispatch_semaphore_signal(done);
                dispatch_release(done);
            }];
            [cb commit];
            segments_.push_back(Strong<id<MTLCommandBuffer>>::retain(cb));
            ++unanswered_;
            cb = nil;
        };
        // A blit where macOS takes it (offsets and size multiples of 4), else jvk's copy kernel.
        auto copy = [&](id<MTLBuffer> from, size_t fromOffset, id<MTLBuffer> to, size_t toOffset, size_t bytes)
        {
            if (((fromOffset | toOffset | bytes) & 3u) == 0)
            {
                [blit() copyFromBuffer:from sourceOffset:fromOffset toBuffer:to destinationOffset:toOffset size:bytes];
                return;
            }
            id<MTLComputePipelineState> state = backend_->copyState();
            if (state == nil || bytes > 0xffffffffu) return;
            auto e = compute();
            const uint32_t n = static_cast<uint32_t>(bytes);
            [e setComputePipelineState:state];
            [e setBuffer:from offset:fromOffset atIndex:0];
            [e setBuffer:to offset:toOffset atIndex:1];
            [e setBytes:&n length:sizeof(n) atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };
        auto inside = [this](const gpu::BufferBinding& r) -> const BufferSlot*
        {
            const BufferSlot* b = backend_->buffer(r.buffer);
            return b != nullptr && b->buffer && r.size > 0 && r.offset + r.size <= b->size ? b : nullptr;
        };

        size_t readbackHead = 0;
        for (const auto& c : packet.commands)
        {
            switch (c.op)
            {
                case gpu::ComputeOp::Dispatch:
                    if (c.dispatch < packet.dispatches.size())
                        backend_->encodeDispatch(compute(), packet.dispatches[c.dispatch], packet.constantData);
                    break;

                case gpu::ComputeOp::Barrier:
                    break;   // a stream's encoders run in order and its buffers' hazards are tracked

                case gpu::ComputeOp::Fill:
                {
                    const auto* b = inside(c.dst);
                    if (b == nullptr || c.dst.offset % 4 != 0 || c.dst.size % 4 != 0) break;
                    const uint32_t byte = c.word & 0xffu;
                    if (byte * 0x01010101u == c.word)
                    {
                        [blit() fillBuffer:b->buffer.get() range:NSMakeRange(c.dst.offset, c.dst.size) value:static_cast<uint8_t>(byte)];
                        break;
                    }
                    id<MTLComputePipelineState> state = backend_->fillState();
                    if (state == nil || c.dst.size / 4 > 0xffffffffu) break;
                    const uint32_t p[2] = { c.word, static_cast<uint32_t>(c.dst.size / 4) };
                    auto e = compute();
                    [e setComputePipelineState:state];
                    [e setBuffer:b->buffer.get() offset:c.dst.offset atIndex:0];
                    [e setBytes:p length:sizeof(p) atIndex:1];
                    [e dispatchThreadgroups:MTLSizeMake((p[1] + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    break;
                }

                case gpu::ComputeOp::Copy:
                {
                    const auto* from = inside(c.src);
                    const auto* to   = inside({ c.dst.buffer, c.dst.offset, c.src.size });
                    if (from != nullptr && to != nullptr)
                        copy(from->buffer.get(), c.src.offset, to->buffer.get(), c.dst.offset, c.src.size);
                    break;
                }

                case gpu::ComputeOp::Upload:
                {
                    const auto* to = inside({ c.dst.buffer, c.dst.offset, c.data.size });
                    if (to != nullptr && uint64_t(c.data.offset) + c.data.size <= packet.uploadData.size())
                        copy(upload_.get(), c.data.offset, to->buffer.get(), c.dst.offset, c.data.size);
                    break;
                }

                case gpu::ComputeOp::Download:
                {
                    readbackHead = (readbackHead + 15) & ~size_t(15);
                    if (const auto* from = inside(c.src); from != nullptr && c.to != nullptr)
                    {
                        copy(from->buffer.get(), c.src.offset, readback_.get(), readbackHead, c.src.size);
                        downloads_.push_back({ readbackHead, static_cast<size_t>(c.src.size), c.to });
                    }
                    readbackHead += static_cast<size_t>(c.src.size);
                    break;
                }

                case gpu::ComputeOp::Timestamp:
                    // The end of everything before: a segment ends here.
                    stampSegment_.push_back(static_cast<uint32_t>(segments_.size()));
                    commitSegment();
                    break;
            }
        }
        commitSegment();
    }
    return true;
}

gpu::ComputeStream::Status ComputeStream::wait(double seconds)
{
    if (segments_.empty())
        return std::exchange(lost_, false) ? Status::Lost : Status::Done;
    const dispatch_time_t deadline = seconds < 0.0 ? DISPATCH_TIME_FOREVER
                                                   : dispatch_time(DISPATCH_TIME_NOW, static_cast<int64_t>(seconds * 1.0e9));
    while (unanswered_ > 0)
    {
        if (dispatch_semaphore_wait(done_, deadline) != 0)
            return Status::Pending;
        --unanswered_;
    }

    bool lost = false;
    for (const auto& s : segments_)
    {
        id<MTLCommandBuffer> cb = s.get();
        if ([cb status] != MTLCommandBufferStatusError) continue;
        static std::atomic<bool> logged { false };
        logComputeOnce(logged, "metal: a compute submission failed: " + describeError([cb error]));
        // Only a GPU that is gone is lost; a kernel the watchdog stopped costs
        // that submission (its results are undefined), not the GPU.
        if ([[cb error] code] == MTLCommandBufferErrorDeviceRemoved)
        {
            backend_->markLost();
            lost = true;
        }
    }
    finish(!lost);
    return lost ? Status::Lost : Status::Done;
}

void ComputeStream::finish(bool delivered)
{
    times_.clear();
    if (delivered)
    {
        const auto* staged = readback_ ? static_cast<const std::byte*>([readback_.get() contents]) : nullptr;
        for (const auto& d : downloads_)
            if (staged != nullptr) std::memcpy(d.to, staged + d.offset, d.bytes);
        if (!segments_.empty())
        {
            // Timestamp i: when every segment up to the one it ends had finished.
            const CFTimeInterval start = [segments_.front().get() GPUStartTime];
            CFTimeInterval end = start;
            size_t through = 0;
            for (const uint32_t segment : stampSegment_)
            {
                for (; through <= segment && through < segments_.size(); ++through)
                    end = std::max(end, [segments_[through].get() GPUEndTime]);
                times_.push_back((end - start) * 1000.0);
            }
        }
    }
    downloads_.clear();
    segments_.clear();
    stampSegment_.clear();
    unanswered_ = 0;
}

} // namespace jvk::metal
