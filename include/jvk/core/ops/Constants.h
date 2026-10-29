/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Constants.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::ops
{

// =============================================================================
// The built-in programs' constants blocks (JvkConstants in shaders/*), byte for
// byte, each padded to the program's constantsBytes (the bundle rounds a block
// up to 16 bytes, as Metal sizes it). Scalars and vectors only, so std140 and
// std430 agree: the same bytes are push constants on Vulkan and a uniform
// buffer everywhere else.
// =============================================================================

// ui2d.vert (ui2d, noise): 16 bytes.
struct PaintConstants
{
    float   viewportW = 0, viewportH = 0;
    int32_t paintsMain = 0;
    int32_t _pad = 0;
};
static_assert(sizeof(PaintConstants) == 16);

// path_sdf.vert/.frag: 48 bytes.
struct PathConstants
{
    float    viewportW = 0, viewportH = 0;
    uint32_t segmentStart = 0;
    uint32_t segmentCount = 0;   // unused in tile mode
    uint32_t fillRule = 0;       // 0 = non-zero, 1 = even-odd
    uint32_t _r0 = 0;
    float    tileW = 0;
    float    _r2 = 0;
    int32_t  paintsMain = 0;
    int32_t  _p0 = 0, _p1 = 0, _p2 = 0;
};
static_assert(sizeof(PathConstants) == 48);

// clip.frag (clip.vert reads the viewport alone): 56 bytes of members, 64 bound.
struct ClipConstants
{
    float    viewportW = 0, viewportH = 0;
    float    centerX = 0,   centerY = 0;     // rrect center (physical px)
    float    halfW = 0,     halfH = 0;       // rrect half-extent
    float    cornerRadius = 0;
    uint32_t shapeType = 0;                  // 1 = rrect, 2 = path
    uint32_t segmentStart = 0;               // path only (strip TABLE when binned)
    uint32_t segmentCount = 0;
    uint32_t fillRule = 0;                   // 0 = non-zero, 1 = even-odd
    uint32_t stripCount = 0;                 // Y-strip binning; 0 = flat
    float    stripMinY = 0;
    float    invStripH = 0;
    float    _pad[2] {};
};
static_assert(sizeof(ClipConstants) == 64);

// clip_mask.frag: 48 bytes.
struct ClipMaskConstants
{
    float    viewportW = 0, viewportH = 0;
    float    toMask[6] {};                   // physical px → mask texels: x row, y row, offset
    float    maskW = 0, maskH = 0;
    uint32_t nearest = 0;
    uint32_t _pad = 0;
};
static_assert(sizeof(ClipMaskConstants) == 48);

// blur.frag, copy.frag: 24 bytes of members, 32 bound.
struct EffectConstants
{
    float    viewportW = 0, viewportH = 0;   // the source's extent
    float    dirX = 0, dirY = 0;
    float    scale = 1;                      // source px per destination px
    uint32_t tapCount = 0;                   // the kernel's taps in its storage; 0: one bilinear read
    float    _pad[2] {};
};
static_assert(sizeof(EffectConstants) == 32);

// hsv.frag: 32 bytes.
struct HSVConstants
{
    float viewportW = 0, viewportH = 0;
    float scaleH = 1, scaleS = 1, scaleV = 1;
    float deltaH = 0, deltaS = 0, deltaV = 0;

    // The identity transform: the region copy-back that ends a partial
    // scene-reading chain on the half it started on.
    static HSVConstants identity(float w, float h)
    {
        HSVConstants c;
        c.viewportW = w; c.viewportH = h;
        return c;
    }
};
static_assert(sizeof(HSVConstants) == 32);

// shape_blur.frag: 96 bytes.
struct ShapeBlurConstants
{
    float viewportW = 0, viewportH = 0;
    float dirX = 0, dirY = 0;
    float invCol0X = 0, invCol0Y = 0;        // affine frag.xy → shape-local
    float invCol1X = 0, invCol1Y = 0;
    float invCol2X = 0, invCol2Y = 0;
    float shapeHalfX = 0, shapeHalfY = 0;
    float lineBX = 0, lineBY = 0;            // line only
    float maxRadius = 0;                     // user-logical px
    float falloff = 0;                       // user-logical px
    float blurStep = 0;                      // physical texels per user-logical px
    float cornerRadius = 0;
    float lineThickness = 0;
    int   shapeType = 0;                     // 0=rect,1=rrect,2=ellipse,3=line
    int   edgePlacement = 0;                 // 0=centered,1=inside,2=outside
    int   inverted = 0;
    int   kernelType = 0;                    // 0 = one separable pass along dir, 1 = Poisson disc, 2 = 2D Gaussian
    uint32_t tapCount = 0;                   // kernel 0: the full radius's taps in its storage (0: none made)
};
static_assert(sizeof(ShapeBlurConstants) == 96);

// path_blur.frag: 64 bytes.
struct PathBlurConstants
{
    float    viewportW = 0, viewportH = 0;
    float    dirX = 0, dirY = 0;             // (1,0) H pass | (0,1) V pass
    float    maxRadius = 0;                  // physical px
    float    falloff = 0;                    // physical px
    float    strokeHalfWidth = 0;            // physical px — 0 for fill
    uint32_t segmentStart = 0;
    uint32_t segmentCount = 0;
    uint32_t fillRule = 0;                   // 0=non-zero, 1=even-odd
    int32_t  edgePlacement = 0;              // 0=centered, 1=inside, 2=outside
    int32_t  inverted = 0;
    int32_t  kernelType = 0;                 // 0 = 1D separable pass, 1/2 = 2D kernel
    uint32_t stripCount = 0;                 // Y-strip binning; 0 = flat list
    float    stripMinY = 0;
    float    invStripH = 0;
};
static_assert(sizeof(PathBlurConstants) == 64);

// What the paint programs read of the frame (storage): the render targets'
// clear colours, for paint into the main colour (Target::clearedByMain).
struct FrameData
{
    float targetClear[kMaxSceneTargets][4] {};
};
static_assert(sizeof(FrameData) == 112);

// The 2D programs' vertex: UIVertex, five attributes at locations 0..4.
inline gpu::VertexLayout uiVertexLayout()
{
    gpu::VertexLayout l;
    l.stride = static_cast<uint16_t>(sizeof(UIVertex));
    l.attributeCount = 5;
    l.attributes[0] = { 0, gpu::VertexFormat::Float2, static_cast<uint16_t>(offsetof(UIVertex, position)) };
    l.attributes[1] = { 1, gpu::VertexFormat::Float4, static_cast<uint16_t>(offsetof(UIVertex, color)) };
    l.attributes[2] = { 2, gpu::VertexFormat::Float2, static_cast<uint16_t>(offsetof(UIVertex, uv)) };
    l.attributes[3] = { 3, gpu::VertexFormat::Float4, static_cast<uint16_t>(offsetof(UIVertex, shapeInfo)) };
    l.attributes[4] = { 4, gpu::VertexFormat::Float4, static_cast<uint16_t>(offsetof(UIVertex, gradientInfo)) };
    return l;
}

} // namespace jvk::ops
