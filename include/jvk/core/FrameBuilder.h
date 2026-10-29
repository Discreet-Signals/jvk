/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: FrameBuilder.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

// =============================================================================
// FrameBuilder — the packet being lowered
//
// Owns the per-frame arenas a gpu::FramePacket points into: passes, draws and
// the three data streams. Reset (capacity kept) at the start of every frame's
// lowering, read by Surface::submit, untouched until the next frame: the
// zero-allocation steady state.
// =============================================================================

class FrameBuilder
{
public:
    void reset(uint64_t frame)
    {
        frame_ = frame;
        passes_.clear();
        draws_.clear();
        vertexData_.clear();
        constantData_.clear();
        storageData_.clear();
        lastConstants_ = {};
        open_ = false;
    }

    // ---- Streams -----------------------------------------------------------

    // At a multiple of `stride`, so a backend draws the range with
    // firstVertex = offset / stride from one binding at offset 0.
    gpu::Range pushVertices(const void* data, uint32_t bytes, uint32_t stride)
    {
        const uint32_t s = std::max(stride, 1u);
        const uint32_t offset = ((static_cast<uint32_t>(vertexData_.size()) + s - 1) / s) * s;
        return append(vertexData_, offset, data, bytes);
    }

    // Consecutive pushes of the same bytes share one range: that is what lets
    // draws with equal constants merge.
    gpu::Range pushConstants(const void* data, uint32_t bytes)
    {
        if (bytes == 0) return {};
        if (lastConstants_.size == bytes
            && std::memcmp(constantData_.data() + lastConstants_.offset, data, bytes) == 0)
            return lastConstants_;
        const uint32_t offset = alignUp(static_cast<uint32_t>(constantData_.size()), constantsAlignment_);
        lastConstants_ = append(constantData_, offset, data, bytes);
        return lastConstants_;
    }
    template <typename T>
    gpu::Range pushConstants(const T& value)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        return pushConstants(&value, static_cast<uint32_t>(sizeof(T)));
    }

    // At a multiple of the backend's storage alignment. Never empty: a binding
    // of zero bytes is invalid on every API.
    gpu::Range pushStorage(const void* data, uint32_t bytes)
    {
        static const std::byte zeros[16] {};
        if (bytes == 0) { data = zeros; bytes = sizeof(zeros); }
        const uint32_t offset = alignUp(static_cast<uint32_t>(storageData_.size()), storageAlignment_);
        return append(storageData_, offset, data, bytes);
    }

    void setAlignments(uint32_t constants, uint32_t storage)
    {
        constantsAlignment_ = std::max(constants, 4u);
        storageAlignment_   = std::max(storage, 4u);
    }

    // ---- Passes ------------------------------------------------------------

    // A new pass; draws land in it until endPass. The caller fills its
    // attachments and render area.
    gpu::Pass& beginPass(const char* label)
    {
        if (open_) endPass();
        auto& p = passes_.emplace_back();
        p.firstDraw = static_cast<uint32_t>(draws_.size());
        p.label = label;
        open_ = true;
        return p;
    }

    void endPass()
    {
        if (!open_) return;
        auto& p = passes_.back();
        p.drawCount = static_cast<uint32_t>(draws_.size()) - p.firstDraw;
        open_ = false;
    }

    bool passOpen() const { return open_; }
    bool passEmpty() const { return open_ && draws_.size() == passes_.back().firstDraw; }
    gpu::Pass& currentPass() { return passes_.back(); }

    // Takes back the open pass, which has no draws.
    void discardPass()
    {
        jassert(passEmpty());
        if (!passEmpty()) return;
        passes_.pop_back();
        open_ = false;
    }

    // Appends a draw to the open pass. A draw that continues the last one
    // (same pipeline, bindings, scissor, stencil reference and constants, its
    // vertices right after the last one's) extends it instead: consecutive
    // quads of one fill become one draw.
    void draw(const gpu::Draw& d)
    {
        jassert(open_);
        if (!open_ || d.vertexCount == 0 || d.scissor.isEmpty()) return;
        const auto& p = passes_.back();
        if (draws_.size() > p.firstDraw)
        {
            auto& last = draws_.back();
            if (continues(last, d))
            {
                last.vertices.size += d.vertices.size;
                last.vertexCount   += d.vertexCount;
                return;
            }
        }
        draws_.push_back(d);
    }

    gpu::FramePacket packet() const
    {
        jassert(!open_);
        gpu::FramePacket pk;
        pk.frame        = frame_;
        pk.passes       = passes_;
        pk.draws        = draws_;
        pk.vertexData   = vertexData_;
        pk.constantData = constantData_;
        pk.storageData  = storageData_;
        return pk;
    }

private:
    static uint32_t alignUp(uint32_t v, uint32_t a) { return ((v + a - 1) / a) * a; }

    static gpu::Range append(std::vector<std::byte>& stream, uint32_t offset, const void* data, uint32_t bytes)
    {
        stream.resize(static_cast<size_t>(offset) + bytes);
        if (bytes > 0)
            std::memcpy(stream.data() + offset, data, bytes);
        return { offset, bytes };
    }

    static bool continues(const gpu::Draw& a, const gpu::Draw& b)
    {
        if (a.pipeline != b.pipeline || a.scissor != b.scissor || a.stencilRef != b.stencilRef
            || a.constants != b.constants || a.instanceCount != 1 || b.instanceCount != 1
            || a.textureCount != b.textureCount || a.storageCount != b.storageCount
            || a.vertices.isEmpty() || b.vertices.isEmpty()
            || a.vertices.offset + a.vertices.size != b.vertices.offset)
            return false;
        for (uint32_t i = 0; i < a.textureCount; ++i)
            if (a.textures[i] != b.textures[i]) return false;
        for (uint32_t i = 0; i < a.storageCount; ++i)
            if (a.storage[i] != b.storage[i]) return false;
        return true;
    }

    uint64_t frame_ = 0;
    std::vector<gpu::Pass> passes_;
    std::vector<gpu::Draw> draws_;
    std::vector<std::byte> vertexData_, constantData_, storageData_;
    gpu::Range lastConstants_;
    uint32_t   constantsAlignment_ = 4, storageAlignment_ = 256;
    bool       open_ = false;
};

// =============================================================================
// Emitter — the state the 2D draws lower through
//
// What a scene draw needs besides its vertices: the pipeline, its two texture
// slots (the colour source and the shape source), storage, constants, and the
// clip state that becomes its scissor and stencil reference. The lowering of
// every built-in op (core/ops) talks to this, never to a packet directly.
// =============================================================================

class Emitter
{
public:
    void beginFrame(FrameBuilder& builder, gpu::Extent extent)
    {
        builder_ = &builder;
        extent_ = extent;
        // The clip state starts each frame clean (the stencil is cleared with
        // the frame's first pass): a stale stack from an unbalanced push last
        // frame (exception unwind mid-paint, a component painting outside
        // ScopedSaveState) must not survive, or a leaked stencil depth tests a
        // reference the cleared stencil never matches — a permanently black
        // window. With the reset it heals in one frame.
        clipRectStack_.clear();
        currentClipBounds_ = {};
        stencilDepth_ = 0;
        exclusions_   = 0;
        invalidate();
    }

    // Forget the bound state (a new pass, or a draw that set its own).
    void invalidate()
    {
        pipeline_ = {};
        textures_[0] = textures_[1] = {};
        textureCount_ = 2;
        storageCount_ = 0;
        constants_ = {};
    }

    void setPipeline(gpu::PipelineId p) { pipeline_ = p; }
    gpu::PipelineId pipeline() const { return pipeline_; }

    // Slot 0 = colour source (solid default, gradient atlas or a tiled image),
    // slot 1 = shape source (default, MSDF atlas page, or image texture).
    void setResources(gpu::TextureBinding color, gpu::TextureBinding shape)
    {
        textures_[0] = color;
        textures_[1] = shape;
        textureCount_ = 2;
    }
    void setColorResource(gpu::TextureBinding color) { textures_[0] = color; }
    void setShapeResource(gpu::TextureBinding shape) { textures_[1] = shape; }
    // For a program that binds fewer textures than the 2D programs' two.
    void setTextureCount(uint8_t n) { textureCount_ = n; }

    void setStorage(std::initializer_list<gpu::Range> ranges)
    {
        storageCount_ = 0;
        for (const auto& r : ranges)
            if (storageCount_ < gpu::kMaxDrawStorage) storage_[storageCount_++] = r;
    }
    void setConstants(gpu::Range c) { constants_ = c; }

    // The draw's scissor: its recorded clip within the rect clip stack,
    // clamped to the frame.
    gpu::IRect scissorFor(const juce::Rectangle<int>& recorded) const
    {
        juce::Rectangle<int> clip = recorded;
        if (!currentClipBounds_.isEmpty())
            clip = clip.getIntersection(currentClipBounds_);
        return clamp(clip, extent_);
    }

    static gpu::IRect clamp(const juce::Rectangle<int>& r, gpu::Extent e)
    {
        const int x0 = std::max(0, r.getX()), y0 = std::max(0, r.getY());
        const int x1 = std::min(static_cast<int>(e.width),  std::max(x0, r.getRight()));
        const int y1 = std::min(static_cast<int>(e.height), std::max(y0, r.getBottom()));
        return { x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0) };
    }

    // `stencilRef`: the reference the draw's stencil test uses; by default the
    // clip depth (every draw passes only inside all active clips). A clip's
    // own draws name theirs (the parent depth, an exclusion's parked value).
    void draw(const DrawCommand& cmd, const UIVertex* verts, uint32_t count, int stencilRef = -1)
    {
        if (count == 0 || builder_ == nullptr || !pipeline_) return;
        gpu::Draw d;
        d.pipeline = pipeline_;
        d.scissor = scissorFor(cmd.clipBounds);
        if (d.scissor.isEmpty()) return;
        d.stencilRef = stencilRef >= 0 ? static_cast<uint8_t>(stencilRef) : stencilDepth_;
        fill(d);
        d.vertices = builder_->pushVertices(verts, count * static_cast<uint32_t>(sizeof(UIVertex)),
                                            static_cast<uint32_t>(sizeof(UIVertex)));
        d.vertexCount = count;
        builder_->draw(d);
    }

    // A draw whose program makes its own vertices (a fullscreen triangle).
    void drawGenerated(gpu::IRect scissor, uint8_t stencilRef, uint32_t vertexCount)
    {
        if (builder_ == nullptr || !pipeline_ || scissor.isEmpty()) return;
        gpu::Draw d;
        d.pipeline = pipeline_;
        d.scissor = scissor;
        d.stencilRef = stencilRef;
        fill(d);
        d.vertexCount = vertexCount;
        builder_->draw(d);
    }

    // ---- Clip state --------------------------------------------------------
    //
    // Axis-aligned rect clips are scissors (this stack); path / rrect / image
    // clips are the stencil, whose depth is the reference every draw tests
    // (stencil == depth: inside all active clips).
    void pushClipRect(const juce::Rectangle<int>& rect)
    {
        clipRectStack_.push_back(currentClipBounds_);
        currentClipBounds_ = rect;
    }
    void popClipRect()
    {
        if (clipRectStack_.empty()) return;
        currentClipBounds_ = clipRectStack_.back();
        clipRectStack_.pop_back();
    }
    void pushStencilDepth() { ++stencilDepth_; }
    void popStencilDepth()  { if (stencilDepth_ > 0) --stencilDepth_; }
    // Excluded rects (excludeClipRectangle) park their pixels out of the
    // current depth, so a draw must test the stencil while one is active even
    // at depth 0, where no path clip is.
    void pushExclusion() { ++exclusions_; }
    void popExclusion()  { if (exclusions_ > 0) --exclusions_; }
    bool stencilTested() const { return stencilDepth_ > 0 || exclusions_ > 0; }

    juce::Rectangle<int> clipBounds() const { return currentClipBounds_; }
    uint8_t              stencilDepth() const { return stencilDepth_; }
    gpu::Extent          extent() const { return extent_; }
    FrameBuilder&        builder() { return *builder_; }

private:
    void fill(gpu::Draw& d) const
    {
        d.textureCount = textureCount_;
        for (uint32_t i = 0; i < textureCount_; ++i) d.textures[i] = textures_[i];
        d.storageCount = storageCount_;
        for (uint32_t i = 0; i < storageCount_; ++i) d.storage[i] = storage_[i];
        d.constants = constants_;
    }

    FrameBuilder* builder_ = nullptr;
    gpu::Extent   extent_;

    gpu::PipelineId     pipeline_;
    gpu::TextureBinding textures_[2] {};
    uint8_t             textureCount_ = 2;
    gpu::Range          storage_[gpu::kMaxDrawStorage] {};
    uint8_t             storageCount_ = 0;
    gpu::Range          constants_;

    std::vector<juce::Rectangle<int>> clipRectStack_;
    juce::Rectangle<int>              currentClipBounds_;
    uint8_t                           stencilDepth_ = 0;
    uint32_t                          exclusions_ = 0;
};

} // namespace jvk
