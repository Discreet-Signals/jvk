#pragma once

namespace jvk {

class Pipeline;
class RenderTarget;

// =============================================================================
// UIVertex — shared vertex format for 2D pipelines
// =============================================================================

struct UIVertex {
    glm::vec2 position;     // screen pixel coords
    glm::vec4 color;        // RGBA — used directly when gradientInfo.z==0 (solid)
    glm::vec2 uv;           // 0-1 within quad (or glyph/image UV)
    glm::vec4 shapeInfo;    // x=type(0=flat,1=rounded,2=ellipse,3=image,4=MSDF), y=halfW/pxRange, z=halfH, w=param
    glm::vec4 gradientInfo; // x,y = gradient t (linear: x; radial: length(x,y)); z = mode (0=solid,1=linear,2=radial); w=unused
};

// =============================================================================
// DrawOp — what to draw or what state change to make
// =============================================================================

enum class DrawOp : uint8_t {
    FillRect, FillRectList, FillRoundedRect, FillEllipse,
    StrokeRoundedRect, StrokeEllipse,
    DrawImage, DrawGlyphs, DrawLine,
    DrawShader,
    EffectBlend,
    EffectResolve,
    EffectKernel,
    PushClipRect, PushClipPath, PopClip,
    COUNT
};

// =============================================================================
// DrawCommand — one entry in the command vector
// =============================================================================

struct DrawCommand {
    float                zOrder;
    DrawOp               op;
    uint32_t             dataOffset;
    uint32_t             dataSize;
    juce::Rectangle<int> clipBounds;
    uint8_t              stencilDepth;
    uint32_t             scopeDepth;
};

// =============================================================================
// Arena — flat byte allocator for command parameter data
// =============================================================================

class Arena {
public:
    template <typename T>
    uint32_t push(const T& data)
    {
        static_assert(std::is_trivially_copyable_v<T>, "Arena is POD-only. Use Renderer side vectors for non-POD types.");
        uint32_t offset = head_;
        uint32_t needed = static_cast<uint32_t>(sizeof(T));
        ensureCapacity(offset + needed);
        memcpy(buffer_.data() + offset, &data, sizeof(T));
        head_ = offset + needed;
        return offset;
    }

    template <typename T>
    uint32_t pushSpan(std::span<const T> data)
    {
        uint32_t offset = head_;
        uint32_t needed = static_cast<uint32_t>(data.size_bytes());
        ensureCapacity(offset + needed);
        memcpy(buffer_.data() + offset, data.data(), data.size_bytes());
        head_ = offset + needed;
        return offset;
    }

    template <typename T>
    const T& read(uint32_t offset) const
    {
        return *reinterpret_cast<const T*>(buffer_.data() + offset);
    }

    template <typename T>
    std::span<const T> readSpan(uint32_t offset, uint32_t count) const
    {
        return { reinterpret_cast<const T*>(buffer_.data() + offset), count };
    }

    // Advance head to the next alignment boundary (e.g., align(4) for float data after uint16 data)
    void align(uint32_t alignment)
    {
        uint32_t mask = alignment - 1;
        head_ = (head_ + mask) & ~mask;
    }

    void reset() { head_ = 0; }
    uint32_t size() const { return head_; }

private:
    std::vector<std::byte> buffer_;
    uint32_t head_ = 0;

    void ensureCapacity(uint32_t needed)
    {
        if (needed > buffer_.size())
            buffer_.resize(std::max(needed, static_cast<uint32_t>(buffer_.size() * 2 + 1024)));
    }
};

// =============================================================================
// State — dirty-tracked GPU state + clip stack (internal to Renderer)
// =============================================================================

class State {
public:
    State() = default;

    void setPipeline(Pipeline* pipeline);
    void setCustomPipeline(VkPipeline pipeline, VkPipelineLayout layout);
    // set 0 = color source (solid default or gradient LUT), set 1 = shape source
    // (1x1 default, MSDF atlas page, or image texture). Each dirty-tracked.
    void setResources(VkDescriptorSet colorSet, VkDescriptorSet shapeSet);
    void setShapeResource(VkDescriptorSet shapeSet);
    void setColorResource(VkDescriptorSet colorSet);
    void draw(const DrawCommand& cmd, const UIVertex* verts, uint32_t count);

    void pushClipRect(const juce::Rectangle<int>& rect);
    void pushClipPath(const juce::Path& path, const juce::AffineTransform& transform);
    void popClip();

    juce::Rectangle<int> clipBounds() const { return currentClipBounds_; }
    uint8_t              stencilDepth() const { return stencilDepth_; }

    // Per-level stencil bit management for even-odd path clipping.
    // Each nesting level owns one stencil bit (bit N for level N).
    void setStencilWriteMask(uint32_t mask);
    void setStencilCompareMask(uint32_t mask);

    void invalidate();
    void begin(VkCommandBuffer cmd, Memory::V& vertices, float vpWidth, float vpHeight);

private:
    VkCommandBuffer  cmd_       = VK_NULL_HANDLE;
    Memory::V*       vertices_  = nullptr;
    float            vpWidth_   = 0;
    float            vpHeight_  = 0;

    Pipeline*        currentPipeline_ = nullptr;
    VkPipeline       boundPipeline_   = VK_NULL_HANDLE;
    VkPipelineLayout boundLayout_     = VK_NULL_HANDLE;
    VkDescriptorSet  boundColorSet_   = VK_NULL_HANDLE;
    VkDescriptorSet  boundShapeSet_   = VK_NULL_HANDLE;
    uint32_t         boundStencilRef_ = 0;
    juce::Rectangle<int> boundScissor_ { -1, -1, 0, 0 };

    struct ClipEntry {
        enum Type { Rect, Path };
        Type type;
        juce::Rectangle<int> rect;
        std::vector<UIVertex> fanVerts;
        juce::Rectangle<int> fanBounds;
    };
    std::vector<ClipEntry> clipStack_;
    juce::Rectangle<int>   currentClipBounds_;
    uint8_t stencilDepth_ = 0;
};

// =============================================================================
// Renderer — the core execution engine
// =============================================================================

class Renderer {
public:
    Renderer(Device& device, RenderTarget& target)
        : device_(device), target_(target),
          vertices_(device.physicalDevice(), device.device())
    {}

    ~Renderer() = default;

    template <typename Params>
    void push(DrawOp op, float zOrder, const juce::Rectangle<int>& clip,
              uint8_t stencilDepth, uint32_t scopeDepth, const Params& params)
    {
        uint32_t offset = arena_.push(params);
        commands_.push_back({
            zOrder, op, offset, static_cast<uint32_t>(sizeof(Params)),
            clip, stencilDepth, scopeDepth
        });
    }

    void registerPipeline(Pipeline& pipeline);
    void execute();

    // Capture non-POD types into side vectors. Returns index.
    uint32_t captureFont(const juce::Font& f)     { fonts_.push_back(f); return static_cast<uint32_t>(fonts_.size() - 1); }
    uint32_t captureFill(const juce::FillType& f)  { fills_.push_back(f); return static_cast<uint32_t>(fills_.size() - 1); }

    const juce::Font&     getFont(uint32_t i) const { return fonts_[i]; }
    const juce::FillType& getFill(uint32_t i) const { return fills_[i]; }

    // Forward to arena for appending POD data after params
    void arena_align(uint32_t alignment) { arena_.align(alignment); }
    template <typename T>
    void arena_pushSpan(std::span<const T> data) { arena_.pushSpan(data); }

    void reset()
    {
        commands_.clear();
        arena_.reset();
        fonts_.clear();
        fills_.clear();
    }

    Device&         device()   { return device_; }
    RenderTarget&   target()   { return target_; }
    ResourceCaches& caches()   { return device_.caches(); }
    State&          state()    { return state_; }
    Memory::V&      vertices() { return vertices_; }
    const Arena&    arena() const { return arena_; }

private:
    Device&       device_;
    RenderTarget& target_;
    State         state_;
    Memory::V     vertices_;

    std::vector<DrawCommand> commands_;
    Arena                    arena_;

    // Non-POD captures (proper RAII, cleared each frame)
    std::vector<juce::Font>     fonts_;
    std::vector<juce::FillType> fills_;

    Pipeline* pipelineForOp_[static_cast<size_t>(DrawOp::COUNT)] = {};
    uint64_t frameCounter_ = 0;
};

} // namespace jvk
