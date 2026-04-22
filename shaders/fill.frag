#version 450
#extension GL_GOOGLE_include_directive : require

// =============================================================================
// fill.frag — geometry-abstracted fill shader. Absorbs ui2d.frag + path_sdf.frag.
//
// One fragment shader handles every 2D fill/stroke/draw op in jvk::Graphics:
// rect, rounded rect, ellipse, capsule/line, path, MSDF glyph, image. Per-
// instance behaviour is driven by the GeometryPrimitive tag (vTag) + colour-
// source bits in vShapeFlags.
//
// Descriptor sets (shared across every geometry-abstracted fill pipeline):
//   set 0, binding 0   sampler2D  — colour LUT (gradient atlas row OR 1×1)
//   set 1, binding 0   SSBO       — GeometryPrimitive array (read in vertex)
//   set 2, binding 0   SSBO       — path segment array (tag=5 branch only)
//   set 3, binding 0   sampler2D  — shape sampler (MSDF atlas page OR image
//                                   OR 1×1 default)
//
// Output is the conventional JUCE straight-alpha blend pair: shape × color,
// multiplied component-wise. For solid shapes color = RGBA fill; for gradient
// shapes color.rgb comes from the LUT and color.a is opacity; for images the
// sampler supplies color.rgba and shape is (1,1,1,opacity).
// =============================================================================

#include "geometry.glsl"

layout(push_constant) uniform PC {
    vec2  viewportSize;
    vec2  _pad0;
    int   _pad1, _pad2, _pad3, _pad4;
} pc;

layout(set = 0, binding = 0) uniform sampler2D colorLUT;

layout(std430, set = 2, binding = 0) readonly buffer Segments {
    vec4 data[];
} segments;

layout(set = 3, binding = 0) uniform sampler2D shapeTex;

layout(location = 0)  flat in uint  vTag;
layout(location = 1)  flat in uint  vShapeFlags;
layout(location = 2)  flat in vec4  vInvXf01;
layout(location = 3)  flat in vec4  vInvXf23_half;   // m02, m12, shapeHalf.x, shapeHalf.y
layout(location = 4)  flat in vec4  vExtraA;
layout(location = 5)  flat in vec4  vExtraB;
layout(location = 6)  flat in vec4  vPayload;
layout(location = 7)  flat in uvec2 vPathRange;
layout(location = 8)       in vec2  vFragPos;
layout(location = 9)       in vec2  vQuadUV;
layout(location = 10) flat in vec4  vColor;

layout(location = 0) out vec4 outColor;

// MSDF median of three channels.
float median3(float r, float g, float b) {
    return max(min(r, g), min(max(r, g), b));
}

// ---- Shape evaluator -----------------------------------------------
// Returns an RGBA mask that the caller multiplies with evalColor(). For
// SDF + MSDF + path tags, RGB = (1,1,1) and alpha carries coverage. For
// image tag, RGBA is the sampled texel (caller's color supplies (1,1,1,
// opacity) so the final composite lands correct straight-alpha values).

vec4 evalShape() {
    bool isStroke = (vShapeFlags & SHAPE_FLAG_STROKE_BIT) != 0u;

    if (vTag == GEOM_IMAGE) {
        vec2 uv = mix(vExtraB.xy, vExtraB.zw, vQuadUV);
        return texture(shapeTex, uv);
    }

    if (vTag == GEOM_MSDF_GLYPH) {
        vec2 uv = mix(vExtraB.xy, vExtraB.zw, vQuadUV);
        vec3 msd = texture(shapeTex, uv).rgb;
        float sd = median3(msd.r, msd.g, msd.b);
        float screenPxRange = max(1.0, vPayload.z);
        float screenPxDist  = screenPxRange * (sd - 0.5);
        float alpha = clamp(screenPxDist + 0.5, 0.0, 1.0);
        // sRGB linear→gamma transfer for AA edge weight (matches ui2d.frag).
        alpha = alpha <= 0.0031308
              ? 12.92 * alpha
              : 1.055 * pow(alpha, 1.0 / 2.4) - 0.055;
        return vec4(1.0, 1.0, 1.0, alpha);
    }

    if (vTag == GEOM_PATH) {
        uint segStart = vPathRange.x;
        uint segCount = vPathRange.y;
        if (segCount == 0u) return vec4(0.0);
        float minDist = 1e20;
        int   winding = 0;
        for (uint i = 0u; i < segCount; ++i) {
            vec4 seg = segments.data[segStart + i];
            vec2 a = seg.xy;
            vec2 b = seg.zw;
            minDist  = min(minDist, sdSegment(vFragPos, a, b));
            winding += pathCrossing(vFragPos, a, b);
        }
        uint fillRule = (vShapeFlags >> SHAPE_FLAG_FILLRULE_SHIFT) & SHAPE_FLAG_FILLRULE_MASK;
        bool inside   = (fillRule == 0u) ? (winding != 0) : ((winding & 1) != 0);
        float signedDist = inside ? -minDist : minDist;
        float aa = max(fwidth(signedDist), 1e-6);
        float alpha = clamp(0.5 - signedDist / aa, 0.0, 1.0);
        return vec4(1.0, 1.0, 1.0, alpha);
    }

    // Analytical SDF shapes (tags 0..3). Evaluate in shape-local coords.
    vec2 local     = applyInverse(vInvXf01, vInvXf23_half.xy, vFragPos);
    vec2 shapeHalf = vInvXf23_half.zw;
    float cornerR  = vExtraA.x;
    float thick    = vExtraA.y;
    vec2  lineB    = vExtraA.zw;

    float d;
    if      (vTag == GEOM_RECT)    d = sdfRect(local, shapeHalf);
    else if (vTag == GEOM_RRECT)   d = sdfRoundedRect(local, shapeHalf, cornerR);
    else if (vTag == GEOM_ELLIPSE) d = sdfEllipse(local, shapeHalf);
    else                           d = sdfCapsule(local, lineB, thick);

    // Stroke / ring SDF for non-capsule shapes. Capsule's `thick` is its own
    // cross-section radius (already consumed above); never stroke-wraps.
    if (vTag != GEOM_CAPSULE && isStroke && thick > 0.0) {
        d = abs(d) - thick * 0.5;
    }

    // Tag 0 filled rect is axis-aligned — no AA bleed, preserve hard alpha.
    if (vTag == GEOM_RECT && !isStroke) {
        return d <= 0.0 ? vec4(1.0) : vec4(0.0);
    }

    float aa = max(fwidth(d), 1e-6);
    float alpha = clamp(0.5 - d / aa, 0.0, 1.0);
    return vec4(1.0, 1.0, 1.0, alpha);
}

// ---- Color evaluator -----------------------------------------------
// Solid: vColor directly. Gradient: sample colorLUT at t(fragment), multiply
// RGB by LUT row, alpha by vColor.a. Image/MSDF go through their shape
// sampler; the colour here still runs but resolves to (1,1,1,opacity) so the
// multiply in main() becomes identity on RGB and an opacity scale on alpha.
//
// Gradient coefficient slot differs by tag: MSDF glyphs keep their atlas UV
// in extraB, so gradient coefs live in extraA for that one tag; every other
// tag puts them in extraB.

vec4 evalColor() {
    uint colorSource = (vShapeFlags >> SHAPE_FLAG_COLOR_SHIFT) & SHAPE_FLAG_COLOR_MASK;
    if (colorSource == COLOR_SRC_SOLID) {
        return vColor;
    }

    vec4 gradCoef;
    if (vTag == GEOM_MSDF_GLYPH) gradCoef = vExtraA;
    else                         gradCoef = vExtraB;

    float invLen2 = vPayload.x;
    float rowNorm = vPayload.y;

    vec2 originPhys = gradCoef.xy;
    vec2 toFrag     = vFragPos - originPhys;

    float t;
    if (colorSource == COLOR_SRC_RADIAL) {
        // gradCoef.z = invRadius; gradCoef.w unused.
        t = length(toFrag) * gradCoef.z;
    } else {
        // gradCoef.zw = dir vector (physical px); invLen2 = 1 / |dir|^2.
        t = dot(toFrag, gradCoef.zw) * invLen2;
    }
    t = clamp(t, 0.0, 1.0);

    vec4 lut = texture(colorLUT, vec2(t, rowNorm));
    return vec4(lut.rgb, lut.a * vColor.a);
}

void main() {
    vec4 shape = evalShape();
    vec4 color = evalColor();
    outColor = color * shape;
}
