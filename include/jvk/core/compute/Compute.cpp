/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Compute.cpp
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

namespace jvk::compute {

namespace {
constexpr size_t kUploadAlign = 16;   // an upload's bytes start on this in the batch's data
}

// =============================================================================
// Buffer
// =============================================================================

Buffer::Buffer(Buffer&& other) noexcept
    : context_(std::move(other.context_)),
      id_(std::exchange(other.id_, {})),
      size_(std::exchange(other.size_, 0)),
      mapped_(std::exchange(other.mapped_, nullptr))
{
}

Buffer& Buffer::operator=(Buffer&& other) noexcept
{
    if (this != &other)
    {
        release();
        context_ = std::move(other.context_);
        id_      = std::exchange(other.id_, {});
        size_    = std::exchange(other.size_, 0);
        mapped_  = std::exchange(other.mapped_, nullptr);
    }
    return *this;
}

void Buffer::release()
{
    // Any frame recorded so far may name it; the backend then waits for the GPU.
    if (id_ && context_ != nullptr)
        context_->retire(id_, context_->serial());
    context_.reset();
    id_ = {};
    size_ = 0;
    mapped_ = nullptr;
}

void Buffer::written(uint64_t offset, uint64_t bytes) const
{
    if (id_ && context_ != nullptr && offset < size_)
        context_->backend().flushBuffer(id_, offset, bytes == kWhole ? size_ - offset : bytes);
}

void Buffer::toRead(uint64_t offset, uint64_t bytes) const
{
    if (id_ && context_ != nullptr && offset < size_)
        context_->backend().invalidateBuffer(id_, offset, bytes == kWhole ? size_ - offset : bytes);
}

// =============================================================================
// Kernel
// =============================================================================

Kernel::Kernel(Kernel&& other) noexcept
    : context_(std::move(other.context_)),
      id_(std::exchange(other.id_, {})),
      name_(std::move(other.name_)),
      bindings_(other.bindings_),
      pushBytes_(other.pushBytes_),
      groupSize_(other.groupSize_)
{
}

Kernel& Kernel::operator=(Kernel&& other) noexcept
{
    if (this != &other)
    {
        release();
        context_   = std::move(other.context_);
        id_        = std::exchange(other.id_, {});
        name_      = std::move(other.name_);
        bindings_  = other.bindings_;
        pushBytes_ = other.pushBytes_;
        groupSize_ = other.groupSize_;
    }
    return *this;
}

void Kernel::release()
{
    if (id_ && context_ != nullptr)
        context_->retire(id_, context_->serial());
    context_.reset();
    id_ = {};
}

// =============================================================================
// Context
// =============================================================================

std::unique_ptr<Context> Context::create(juce::String* whyNot)
{
    // The build's default backend, the editors' (hosts/WindowHost.cpp).
    auto backend = defaultBackend(whyNot);
    if (backend == nullptr)
        return nullptr;
    return create(std::move(backend), whyNot);
}

std::unique_ptr<Context> Context::create(std::shared_ptr<gpu::Backend> backend, juce::String* whyNot)
{
    auto no = [whyNot](const juce::String& reason) -> std::unique_ptr<Context>
    {
        if (whyNot != nullptr) *whyNot = reason;
        return nullptr;
    };
    if (backend == nullptr)                    return no("no GPU backend");
    if (backend->isLost())                     return no("the GPU was lost");
    if (!backend->caps().compute.supported)    return no("the GPU backend has no compute");
    auto context = jvk::Context::forBackend(backend);
    if (context == nullptr)                    return no("no GPU backend");
    return std::unique_ptr<Context>(new Context(std::move(context)));
}

Context::Context(std::shared_ptr<jvk::Context> context) : context_(std::move(context))
{
    const auto& caps = context_->caps();
    const auto& c = caps.compute;
    limits_.device = juce::String::fromUTF8(c.deviceName != nullptr ? c.deviceName : "");
    for (int i = 0; i < 3; ++i)
    {
        limits_.maxGroupSize[i] = c.maxGroupSize[i];
        limits_.maxGroups[i]    = c.maxGroups[i];
    }
    limits_.maxGroupThreads  = c.maxGroupThreads;
    limits_.sharedBytes      = c.sharedBytes;
    limits_.pushBytes        = c.constantsBytes;
    limits_.maxBindings      = std::min(c.maxBindings, gpu::kMaxKernelBindings);
    limits_.subgroupSize     = c.subgroupSize;
    limits_.storageAlignment = std::max<uint64_t>(caps.storageAlignment, 4);
    limits_.unifiedMemory    = c.unifiedMemory;
    limits_.ownQueue         = c.ownQueue;
    limits_.timestampNs      = c.timestampNs;

    diag::log("compute context: " + limits_.device + ", subgroups of " + juce::String(limits_.subgroupSize)
              + ", " + juce::String(limits_.sharedBytes / 1024) + " KB shared, " + (limits_.unifiedMemory ? "unified memory" : "own memory")
              + ", " + (limits_.ownQueue ? "own queue" : "behind the frames") + (limits_.timestampNs > 0.0 ? ", timestamps" : ", no timestamps"));
}

Buffer Context::buffer(uint64_t bytes, Usage usage)
{
    gpu::BufferDesc d;
    d.size = std::max<uint64_t>(bytes, 4);
    switch (usage)
    {
        case Usage::Device:   d.memory = gpu::BufferMemory::Device;   break;
        case Usage::Upload:   d.memory = gpu::BufferMemory::Upload;   break;
        case Usage::Readback: d.memory = gpu::BufferMemory::Readback; break;
        case Usage::Shared:   d.memory = gpu::BufferMemory::Shared;   break;
    }
    Buffer b;
    const auto id = context_->backend().createBuffer(d);
    if (!id) return b;
    b.context_ = context_;
    b.id_      = id;
    b.size_    = d.size;
    b.mapped_  = context_->backend().bufferData(id);
    return b;
}

Kernel Context::kernel(const KernelSource& source, const juce::String& name, std::initializer_list<Constant> constants)
{
    const juce::String label = name.isNotEmpty() ? name : juce::String(source.name != nullptr ? source.name : "kernel");
    auto refused = [&label](const juce::String& why)
    {
        diag::log("compute kernel " + label + ": " + why);
        jassertfalse;
        return Kernel();
    };

    const auto& caps = context_->caps();
    gpu::KernelDesc d;
    d.format = caps.shaderFormat;
    if (d.format == gpu::ShaderFormat::SPIRV && !source.spirv.empty())
    {
        d.code  = std::as_bytes(source.spirv);
        d.entry = "main";
    }
    else if (d.format == gpu::ShaderFormat::MSL && !source.msl.empty())
    {
        d.code  = std::as_bytes(std::span<const char>(source.msl.data(), source.msl.size()));
        d.entry = "main0";
    }
    else
    {
        return refused("no form of it this backend runs (jvk_compute_shaders builds the forms of the build's backends)");
    }

    // The constants it declares, given or not; one given that it doesn't is
    // ignored (glslc's optimizer takes out the ones a kernel never reads).
    std::vector<gpu::SpecConstant> values;
    for (const auto& c : constants)
        for (const auto& declared : source.constants)
            if (declared.id == c.id)
            {
                values.push_back({ c.id, declared.type, declared.type == gpu::ScalarType::Bool ? (c.bits != 0 ? 1u : 0u) : c.bits });
                break;
            }

    for (int i = 0; i < 3; ++i)
    {
        d.groupSize[i] = source.groupSize[i];
        for (const auto& c : constants)
            if (source.groupSizeId[i] != ~0u && c.id == source.groupSizeId[i])
                d.groupSize[i] = c.bits;
        if (d.groupSize[i] == 0 || d.groupSize[i] > limits_.maxGroupSize[i])
            return refused("its workgroup is " + juce::String(d.groupSize[i]) + " threads in dimension " + juce::String(i)
                           + "; the GPU takes " + juce::String(limits_.maxGroupSize[i]));
    }
    const uint64_t threads = uint64_t(d.groupSize[0]) * d.groupSize[1] * d.groupSize[2];
    if (threads > limits_.maxGroupThreads)
        return refused("its workgroup is " + juce::String((juce::int64) threads) + " threads; the GPU takes "
                       + juce::String(limits_.maxGroupThreads));

    uint32_t bindings = 0;
    for (const auto& b : source.bindings)
        bindings = std::max<uint32_t>(bindings, uint32_t(b.binding) + 1);
    if (source.bindings.size() > limits_.maxBindings || bindings > gpu::kMaxKernelBindings)
        return refused("it binds " + juce::String((int) source.bindings.size()) + " buffers; the GPU takes "
                       + juce::String(limits_.maxBindings));
    if (source.constantsBytes > limits_.pushBytes)
        return refused("it has more push constants than the GPU takes (" + juce::String(limits_.pushBytes) + " bytes)");

    const auto utf8 = label.toStdString();
    d.name           = utf8.c_str();
    d.bindings       = source.bindings;
    d.constantsBytes = source.constantsBytes;
    d.constants      = values;
    d.bufferSizes    = source.bufferSizes;
    const auto id = context_->backend().createKernel(d);
    if (!id)
        return refused("the backend could not build it");

    Kernel k;
    k.context_   = context_;
    k.id_        = id;
    k.name_      = label;
    k.bindings_  = bindings;
    k.pushBytes_ = source.constantsBytes;
    k.groupSize_ = { d.groupSize[0], d.groupSize[1], d.groupSize[2] };
    return k;
}

std::unique_ptr<Stream> Context::stream()
{
    auto s = context_->backend().createComputeStream();
    if (s == nullptr)
        return nullptr;
    return std::unique_ptr<Stream>(new Stream(context_, std::move(s)));
}

// =============================================================================
// Stream
// =============================================================================

Stream::Stream(std::shared_ptr<jvk::Context> context, std::unique_ptr<gpu::ComputeStream> stream)
    : context_(std::move(context)), stream_(std::move(stream))
{
}

Stream::~Stream()
{
    if (submitted_)
        wait();
    stream_.reset();   // the backend's stream waits too, before freeing its memory
    release();         // recorded and never submitted
}

void Stream::release()
{
    if (hold_ != 0)
        context_->endHold(std::exchange(hold_, 0));
}

void Stream::dispatch(const Kernel& kernel, Groups groups, std::initializer_list<Range> bindings,
                      const void* push, uint32_t pushBytes)
{
    jassert(kernel.valid());
    jassert(bindings.size() == kernel.bindings());   // every binding up to the highest the kernel declares, in order
    jassert(pushBytes == kernel.pushBytes());
    jassert(kernel.owner() == context_.get());       // a kernel of this backend
    if (!kernel.valid() || bindings.size() > gpu::kMaxKernelBindings)
        return;

    gpu::Dispatch d;
    d.kernel = kernel.handle();
    d.groups[0] = groups.x;
    d.groups[1] = groups.y;
    d.groups[2] = groups.z;
    for (const auto& r : bindings)
        d.bindings[d.bindingCount++] = r.binding();
    if (push != nullptr && pushBytes > 0)
    {
        const size_t offset = (constants_.size() + 15) & ~size_t(15);
        constants_.resize(offset + pushBytes);
        std::memcpy(constants_.data() + offset, push, pushBytes);
        d.constants = { static_cast<uint32_t>(offset), pushBytes };
    }

    gpu::ComputeCommand c;
    c.op = gpu::ComputeOp::Dispatch;
    c.dispatch = static_cast<uint32_t>(dispatches_.size());
    hold();
    dispatches_.push_back(d);
    commands_.push_back(c);
}

void Stream::barrier()
{
    hold();
    gpu::ComputeCommand c;
    c.op = gpu::ComputeOp::Barrier;
    commands_.push_back(c);
}

void Stream::fill(Range range, uint32_t word)
{
    jassert(range.offset % 4 == 0 && range.bytes % 4 == 0);
    hold();
    gpu::ComputeCommand c;
    c.op = gpu::ComputeOp::Fill;
    c.dst = range.binding();
    c.word = word;
    commands_.push_back(c);
}

void Stream::copy(Range from, Range to)
{
    jassert(to.bytes >= from.bytes);
    hold();
    gpu::ComputeCommand c;
    c.op = gpu::ComputeOp::Copy;
    c.src = from.binding();
    c.dst = to.binding();
    commands_.push_back(c);
}

uint32_t Stream::pushUpload(const void* data, uint64_t bytes)
{
    const size_t offset = (uploads_.size() + kUploadAlign - 1) & ~(kUploadAlign - 1);
    jassert(offset + bytes <= std::numeric_limits<uint32_t>::max());   // a batch uploads under 4 GB
    uploads_.resize(offset + static_cast<size_t>(bytes));
    std::memcpy(uploads_.data() + offset, data, static_cast<size_t>(bytes));
    return static_cast<uint32_t>(offset);
}

void Stream::upload(Range to, const void* from)
{
    if (to.bytes == 0 || from == nullptr)
        return;
    hold();
    gpu::ComputeCommand c;
    c.op = gpu::ComputeOp::Upload;
    c.dst = to.binding();
    c.data = { pushUpload(from, to.bytes), static_cast<uint32_t>(to.bytes) };
    commands_.push_back(c);
}

void Stream::download(Range from, void* to)
{
    if (from.bytes == 0 || to == nullptr)
        return;
    hold();
    gpu::ComputeCommand c;
    c.op = gpu::ComputeOp::Download;
    c.src = from.binding();
    c.to = to;
    commands_.push_back(c);
}

void Stream::time(const char* label)
{
    hold();
    gpu::ComputeCommand c;
    c.op = gpu::ComputeOp::Timestamp;
    commands_.push_back(c);
    labels_.push_back(label);
}

void Stream::submit()
{
    if (commands_.empty())
        return;
    if (submitted_)   // one batch in flight: the last one finishes first
        wait();

    gpu::ComputePacket p;
    p.commands     = commands_;
    p.dispatches   = dispatches_;
    p.constantData = constants_;
    p.uploadData   = uploads_;
    if (stream_ != nullptr && stream_->submit(p))
        submitted_ = true;
    else
        lost_ = true;

    submittedLabels_.swap(labels_);
    labels_.clear();
    commands_.clear();
    dispatches_.clear();
    constants_.clear();
    uploads_.clear();
    release();   // the backend has the handles now, and defers their memory itself
}

Stream::Result Stream::wait(double seconds)
{
    if (!submitted_)
        return std::exchange(lost_, false) ? Result::lost : Result::done;
    const auto status = stream_->wait(seconds);
    if (status == gpu::ComputeStream::Status::Pending)
        return Result::pending;
    submitted_ = false;
    if (status == gpu::ComputeStream::Status::Lost)
        return Result::lost;

    times_.clear();
    const auto stamps = stream_->timestamps();
    for (size_t i = 0; i < stamps.size() && i < submittedLabels_.size(); ++i)
        times_.push_back({ submittedLabels_[i], stamps[i] });
    return Result::done;
}

} // namespace jvk::compute
