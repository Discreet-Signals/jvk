#pragma once

namespace jvk {

// =============================================================================
// UIVertex — shared vertex format for the 2D programs (ui2d, path, clip)
// =============================================================================

struct UIVertex {
    glm::vec2 position;     // screen pixel coords
    glm::vec4 color;        // RGBA — used directly when gradientInfo.z==0 (solid)
    glm::vec2 uv;           // 0-1 within quad (or glyph/image UV)
    glm::vec4 shapeInfo;    // x=type(0=flat,1=rounded,2=ellipse,3=image,4=MSDF), y=halfW/pxRange, z=halfH, w=param
    glm::vec4 gradientInfo; // x,y = gradient t (linear: x; radial: length(x,y)); z = mode (0=solid,1=linear,2=radial); w=unused
};

// Placement of the shape-blur falloff band relative to the shape edge (SDF=0).
//   Centered → falloff straddles the edge, half inside + half outside
//   Inside   → falloff sits entirely inside the shape, ending at the edge
//   Outside  → falloff sits entirely outside the shape, starting at the edge
enum class BlurEdge : uint32_t {
    Centered = 0,
    Inside   = 1,
    Outside  = 2,
};

// Blur kernel strategy. Picks the trade-off between speed and directional
// artefact tolerance:
//
//   Low    — 2 separable passes at 0° and 90°. Cheapest. Visible streaks
//            at `t` boundaries because each pass reads the previous pass's
//            variable-radius output; the axis-aligned streak direction
//            usually aligns with rectangular shape edges so it tucks into
//            the geometry, but for arbitrary shapes a cross artefact is
//            visible. Fine for most UI overlays at small radius.
//   Medium — 3 separable passes at 0°, 60°, 120°. Redistributes the
//            streak across 3 angles (6-fold). Trades the axis-aligned
//            cross for a faint hex; better on organic shapes but worse
//            on rectangles (streaks no longer align with edges).
//   High   — 1 pass, **true 2D** Gaussian — samples every physical pixel
//            in the kernel neighbourhood directly from the scene source,
//            no chained passes. Zero separable-blur artefacts. Cost is
//            O(N²) per fragment vs O(N) per pass, so reserve for hero
//            elements where Low / Medium streak visibly and the blur
//            radius is modest.
//
// TODO: this enum will grow into a real "blur type" selector. Candidates:
//   - Streaky / Anisotropic — deliberately non-isotropic directional
//     blur (Gaussian with a separable-axis imbalance) for frosted / hair-
//     line / motion-glass looks. The "tighter falloff at higher quality"
//     bug we just fixed produced a textured glass effect that's worth
//     revisiting as a proper mode.
//   - Box                    — cheap uniform-radius box filter.
//   - Kawase / DualKawase    — rotated 4-tap patterns; opt-in when we
//                              have a batched command plan that makes
//                              the pyramid setup worthwhile.
//   - BokehHex / BokehDisc   — bokeh-aesthetic non-separable kernels
//                              for DOF-style background content.
// When adding those, consider renaming the enum values to namespace-
// prefixed (GaussianLow, GaussianHigh, ..., Streaky, Box, etc.) so the
// type (Gaussian vs Kawase vs Box) is explicit at call sites.
enum class BlurMode : uint32_t {
    Low    = 0,
    Medium = 1,
    High   = 2,
};

// =============================================================================
// Parameter structs — POD only, packed into Arena via memcpy.
// Non-POD types (Font, FillType) stored in Recorder side vectors by index.
// =============================================================================

struct FillRectParams {
    juce::Rectangle<float> rect;
    uint32_t               fillIndex;
    juce::AffineTransform  transform;
    float                  opacity;
    float                  scale;
};

struct FillRoundedRectParams {
    juce::Rectangle<float> rect;
    float                  cornerSize;
    uint32_t               fillIndex;
    juce::AffineTransform  transform;
    float                  opacity;
    float                  scale;
};

struct FillEllipseParams {
    juce::Rectangle<float> area;
    uint32_t               fillIndex;
    juce::AffineTransform  transform;
    float                  opacity;
    float                  scale;
};

struct StrokeRoundedRectParams {
    juce::Rectangle<float> rect;
    float                  cornerSize;
    float                  lineWidth;
    uint32_t               fillIndex;
    juce::AffineTransform  transform;
    float                  opacity;
    float                  scale;
};

struct StrokeEllipseParams {
    juce::Rectangle<float> area;
    float                  lineWidth;
    uint32_t               fillIndex;
    juce::AffineTransform  transform;
    float                  opacity;
    float                  scale;
};

struct DrawImageParams {
    // The cached texture, resolved at record time (Graphics::drawImage via
    // Renderer::texture) so the worker never looks up the shared image cache,
    // which sibling editors' message-thread activity mutates. The cache keeps
    // it alive until every frame that names it is submitted (Context).
    gpu::TextureId         texture;
    juce::AffineTransform  transform;
    float                  opacity;
    float                  scale;
    int                    imageWidth;
    int                    imageHeight;
    // juce lowResamplingQuality: the shader snaps UVs to texel centres
    // (see ui2d.frag sampleImage) instead of carrying a second sampler.
    uint32_t               nearest = 0;
    // clipToImageAlpha fill (Graphics::fillThroughAlphaMask): the quad is
    // tinted with the brush (1 = solid: `tint` carries the colour) or the
    // brush GRADIENT (2: `fillIndex`+`gradientTransform` rebuild the ctx at
    // replay; tint = white x layer opacity), and the image contributes
    // ALPHA ONLY (shader shape type 5). Zero for plain image draws.
    glm::vec4              tint {};
    uint32_t               alphaMaskFill = 0;
    uint32_t               fillIndex = 0;
    juce::AffineTransform  gradientTransform;
};

// Tiled image fill (juce setTiledImageFill via Graphics::fillRect). Corners
// and UVs are precomputed at record time — UVs are image-normalized and
// unbounded; the shader wraps with fract() (shape type 6). Rendered by
// ColorPipeline like any other quad.
struct FillTiledImageParams {
    gpu::TextureId  texture;
    glm::vec2       pos[4];   // physical-px corners: p00 p10 p11 p01
    glm::vec2       uv[4];    // matching UVs (pre-fract)
    glm::vec4       tint;     // white x fill-colour alpha x layer opacity
    uint32_t        nearest;  // point-sample flag (see DrawImageParams)
};

// excludeClipRectangle: the rect (physical px) whose pixels get stencil-
// parked out of the current clip depth, and un-parked on restoreState.
struct ExcludeRectParams {
    juce::Rectangle<float> rect;
};

struct DrawGlyphsParams {
    uint32_t               glyphCount;
    juce::AffineTransform  transform;      // the glyphs: the run's own transform, then the paint's
    juce::AffineTransform  fillTransform;  // the fill: the paint's alone, as juce places every fill
    uint32_t               fontIndex;
    uint32_t               fillIndex;
    float                  opacity;
    float                  scale;
    // followed in arena by: uint16_t[glyphCount] + Point<float>[glyphCount]
};

struct DrawLineParams {
    juce::Line<float>      line;
    float                  lineWidth;
    uint32_t               fillIndex;
    juce::AffineTransform  transform;
    float                  opacity;
    float                  scale;
};

// Per-draw constants a jvk::Shader can take (Shader::setDrawConstants): copied
// into each draw at record time, pushed at lowering after jvk's own 8 floats.
// 24 floats + 32 bytes = 128 bytes, the push-constant size every device has.
static constexpr int kShaderDrawConstants = 24;
static constexpr int kShaderDrawImages = 4;   // per-draw images: descriptor sets 2..5 (Shader::setDrawImage)

struct DrawShaderParams {
    // What the draw needs of its jvk::Shader, captured at record so lowering
    // never touches the Shader (which its owner may destroy at any time):
    // Renderer::shaderCapture(shaderIndex). ~0u: an invalid draw, nothing drawn.
    uint32_t               shaderIndex = ~0u;
    juce::Rectangle<float> region;
    float                  scale;
    // What the shader reads and writes, captured at record: the targets it
    // samples (bit i = target i; they live up to here: Recorder::noteTargetsRead),
    // the ones it writes (at the output locations of targetSlots), the main
    // colour (location 0).
    TargetMask             targetsRead    = 0;
    TargetMask             targetsWritten = 0;
    TargetSlots            targetSlots    = 0;
    bool                   writesMain     = true;
    // Where it draws (Renderer::lower, the render graph): INSIDE THE SCENE PASS
    // when everything it writes is attached there (the main colour, targets at
    // scale 1) and nothing it reads is (it reads neither the scene nor a target at
    // scale 1; scaled targets it may), else as a pass of its own.
    bool                   inScene        = true;
    uint32_t               constantCount = 0;
    float                  constants[kShaderDrawConstants] {};
    // The draw's own images (Shader::setDrawImage), resolved through the texture
    // cache at record, bound as sets 2...
    uint32_t               imageCount = 0;
    gpu::TextureId         images[kShaderDrawImages] {};
};

// Analytical SDF path fill, TILE mode (see path_sdf.frag). Graphics::fillPath
// flattens the path to line segments in physical-pixel space, appends the
// tile-local segment lists to the frame's segment storage (via
// Recorder::pushSegments()), and records the tile quads here. `fillRule` matches
// juce::Path::isUsingNonZeroWinding() — 0 for non-zero, 1 for even-odd.
//
// Empty tiles are never drawn, interior runs merge into single flat quads,
// edge tiles carry small local segment lists. The per-tile UIVertex span
// follows this struct in the arena (4-byte aligned). This replaced the
// single bounds-covering quad whose every fragment walked the full (or
// strip-binned) segment list.
struct FillPathParams {
    uint32_t vertexCount;   // 6 × emitted quads; UIVertex span follows in arena
    uint32_t segmentStart;  // base of this path's tile-local segment data
    uint32_t fillRule;      // 0 = non-zero winding, 1 = even-odd
    uint32_t fillIndex;     // Recorder::getFill() slot → chooses the colour texture
    float    tileSize;      // tile width in physical px (16, coarsened if huge)
};

// One command + one arena span of juce::Rectangle<float> for a whole
// RectangleList fill. The rect span immediately follows this struct in the
// arena (4-byte aligned).
struct FillRectListParams {
    uint32_t              rectCount;
    uint32_t              fillIndex;
    juce::AffineTransform transform;
    float                 opacity;
    float                 scale;
};

// Clip shape for the stencil push/pop pipelines. One payload shared by both
// DrawOp::PushClipPath and DrawOp::PopClip — the shapes must match exactly
// so the INCR on push and DECR on pop cancel (same fragments touched, same
// stencil op applied symmetrically).
//
// Axis-aligned rectangle clips NEVER appear here — they fast-path through
// Graphics::clipToRectangle to a plain scissor rect. This payload covers
// the two stencil-backed cases:
//
//   shapeType == 1  Rounded rectangle
//     centerX/Y + halfW/H + cornerRadius encode the shape in physical
//     pixels; the clip fragment shader evaluates roundedRectSDF and
//     discards outside pixels before the stencil op.
//
//   shapeType == 2  Arbitrary path
//     segmentStart/segmentCount index into the shared per-frame segment
//     storage (Recorder::pushSegments). fillRule selects non-zero (0) or
//     even-odd (1) winding; the clip frag shader walks the segments and
//     computes the winding number per fragment.
//
//   shapeType == 3  Image alpha (Graphics::clipToImageAlpha)
//     mask is the mask's texture; toMask maps physical pixels to its
//     texels as clip_mask.frag reads it: the x row (mat00 mat01), the y
//     row (mat10 mat11), the offset (mat02 mat12) of the juce transform;
//     maskNearest is the resampling the clip was made with. Inside
//     where the sampled alpha is at least one half (clip_mask.frag).
//
// coverRect is the bounding box + 1px AA margin in physical pixels —
// emitted as a 6-vertex quad so the clip fragment shader runs for every
// pixel the shape could occupy.
struct ClipShapeParams {
    uint32_t               shapeType;       // 1 = rrect, 2 = path, 3 = image alpha
    float                  centerX, centerY; // rrect only (physical px)
    float                  halfW, halfH;     // rrect only (physical px)
    float                  cornerRadius;     // rrect only (physical px)
    uint32_t               segmentStart;     // path only (strip TABLE when binned)
    uint32_t               segmentCount;     // path only
    uint32_t               fillRule;         // path only
    juce::Rectangle<float> coverRect;        // quad to cover in physical px
    // Y-strip binning (path only). 0 = flat list.
    uint32_t               stripCount = 0;
    float                  stripMinY  = 0.0f;
    float                  invStripH  = 0.0f;
    // Image alpha only.
    gpu::TextureId         mask;
    float                  toMask[6] {};
    float                  maskW = 0.0f, maskH = 0.0f;
    uint32_t               maskNearest = 0;
};

struct EffectBlendParams {
    float r, g, b;
    juce::Rectangle<float> region;
    float                  scale;
};

// Procedural white-noise overlay (Graphics::drawNoise). `amount` is the alpha
// blend toward white noise (0 = none, 1 = full overwrite); `timeOffset` shifts
// the per-pixel hash per frame (0 = static / identical every frame).
struct NoiseParams {
    float                  amount;
    float                  timeOffset;
    juce::Rectangle<float> region;
    float                  scale;
};

struct BlurParams {
    float                  radius;
    juce::Rectangle<float> region;
    float                  scale;
};

// Full-screen HSV transform. `scale*` defaults to 1.0, `delta*` to 0.0 —
// identity. Applied as `hsv *= (scale...); hsv += (delta...);` in hsv.frag.
// Specialised callers like Graphics::saturate / shiftHue / brightness fill
// in only the relevant fields; everything else stays at defaults.
struct HSVParams {
    float scaleH = 1.0f, scaleS = 1.0f, scaleV = 1.0f;
    float deltaH = 0.0f, deltaS = 0.0f, deltaV = 0.0f;
    juce::Rectangle<float> region;
    float scale;
};

// Shape-aware variable-radius blur. Pre-packed in the layout the fragment
// shader's push-constant block expects, minus viewport/direction which the
// pipeline fills in per-pass. Distances are in USER-LOGICAL pixels
// (pre-transform); the shader converts kernel sample offsets to physical
// texels via `blurStep`.
struct BlurShapeParams {
    float    invXform[6];          // inverse affine: frag (physical) → shape-local (logical)
    float    shapeHalf[2];         // rect/rrect/ellipse halfsize (logical)
    float    lineB[2];             // line: endpoint B in shape-local (A is origin)
    float    maxRadius;            // user-logical pixels
    float    falloff;              // user-logical pixels
    float    blurStep;             // physical texels per user-logical pixel
                                   // (= transformScale * displayScale)
    float    cornerRadius;         // rrect only
    float    lineThickness;        // line only
    uint32_t shapeType;            // 0=rect, 1=rrect, 2=ellipse, 3=line
    uint32_t edgePlacement;        // BlurEdge as uint32
    uint32_t inverted;             // 0 or 1
    uint32_t mode;                 // BlurMode as uint32 (pass count / type lives here)
    // ROI: physical-pixel AABB of every pixel this blur can touch (shape
    // bounds + blur reach, transformed). Record-side computed; replay walks
    // the pass chain backward from it. Empty = whole clip (inverted blurs).
    juce::Rectangle<float> region;
};

// Analytical path SDF blur. Graphics::{draw,fill}BlurredPath flattens the
// path to line segments in PHYSICAL pixels (same as FillPath) and uploads
// them via Recorder::pushSegments. All distances below are also in
// PHYSICAL pixels — the C++ caller pre-multiplies by transformScale ×
// displayScale so the shader runs in one coord space (step = 1 texel).
struct BlurPathParams {
    uint32_t segmentStart;
    uint32_t segmentCount;
    uint32_t fillRule;        // 0 = non-zero, 1 = even-odd
    float    maxRadius;       // physical px
    float    falloff;         // physical px
    float    strokeHalfWidth; // physical px — 0 for fill, >0 for stroke
    uint32_t edgePlacement;   // BlurEdge as uint32
    uint32_t inverted;        // 0 or 1
    uint32_t mode;            // BlurMode as uint32 (pass count / type lives here)
    // ROI: physical-pixel AABB of the path + blur reach. Empty = whole clip.
    juce::Rectangle<float> region;
    // Y-strip binning (pad = the blur's full reach). 0 = flat list.
    uint32_t stripCount = 0;
    float    stripMinY  = 0.0f;
    float    invStripH  = 0.0f;
};

} // namespace jvk
