#version 450
#extension GL_GOOGLE_include_directive : require

// =============================================================================
// geom_blur.frag — geometry-abstracted variable-radius Gaussian blur.
//
// Absorbs shape_blur.frag + path_blur.frag. Per-instance behaviour is driven
// by the GeometryPrimitive the vertex shader passed via flat outputs.
// A batch of N primitives runs under one vkCmdDraw(6, N, 0, first); each
// instance's fragments branch on the primitive's tag to either evaluate an
// analytic SDF (tags 0..3) or walk the shared path-segment SSBO (tag 5).
//
// Descriptor sets (shared across every geometry-abstracted blur pipeline):
//   set 0, binding 0   sampler2D  — scene source (ping-pong half being read)
//   set 1, binding 0   SSBO       — GeometryPrimitive array (Primitives)
//   set 2, binding 0   SSBO       — path segment array (vec4 p0.xy,p1.xy)
//
// Shape SDFs (tags 0..3) evaluate in SHAPE-LOCAL / user-logical pixels; maps
// FRAG → LOCAL via the primitive's inverse affine. Radii/falloff/stroke on
// the primitive are in LOGICAL pixels; blurStep (= transformScale ×
// displayScale) folds logical → physical for the kernel sampler. Path tag 5
// keeps segments in physical pixels for now — that matches what
// Graphics::pushBlurPath uploads today; a later phase will move segments to
// local too. Stroke/fill distinction travels on shapeFlags.SHAPE_FLAG_STROKE
// (shape outlines) or extraB.w > 0 (path strokes).
// =============================================================================

#include "geometry.glsl"

layout(push_constant) uniform PC {
    vec2  viewportSize;
    vec2  direction;
    int   kernelType;
    int   _pad0;
    int   _pad1;
    int   _pad2;
} pc;

layout(set = 0, binding = 0) uniform sampler2D srcTexture;

layout(std430, set = 2, binding = 0) readonly buffer Segments {
    vec4 data[];
} segments;

layout(location = 0)  flat in uint  vTag;
layout(location = 1)  flat in uint  vShapeFlags;
layout(location = 2)  flat in vec4  vInvXf01;
layout(location = 3)  flat in vec4  vInvXf23_half;
layout(location = 4)  flat in vec4  vExtraA;          // cornerRadius, lineThickness, lineB.x, lineB.y
layout(location = 5)  flat in vec4  vExtraB;          // maxRadius, falloff, blurStep, strokeHalfWidth
layout(location = 6)  flat in vec4  vPayload;
layout(location = 7)  flat in uvec2 vPathRange;
layout(location = 8)       in vec2  vFragPos;
layout(location = 9)       in vec2  vQuadUV;
layout(location = 10) flat in vec4  vColor;           // unused by blur; present for layout compat

layout(location = 0) out vec4 outColor;

const int   MAX_PAIRS        = 4096;
const int   MAX_POISSON_TAPS = 1024;
const float GOLDEN_ANGLE     = 2.39996322972865332;

// ---- Shape / path SDF dispatch ---------------------------------------------
//
// Returns a signed distance in whatever coord space the primitive's SDF
// lives in (LOGICAL for tags 0..3, PHYSICAL for tag 5). The caller then
// works in that same space for the band / stroke / falloff math.

float shapeSDF(vec2 fragCoord) {
    if (vTag == GEOM_PATH) {
        uint segStart = vPathRange.x;
        uint segCount = vPathRange.y;
        if (segCount == 0u) return 1e20;

        float minDist = 1e20;
        int   winding = 0;
        for (uint i = 0u; i < segCount; ++i) {
            vec4 seg = segments.data[segStart + i];
            vec2 a   = seg.xy;
            vec2 b   = seg.zw;
            minDist  = min(minDist, sdSegment(fragCoord, a, b));
            winding += pathCrossing(fragCoord, a, b);
        }
        uint fillRule = (vShapeFlags >> SHAPE_FLAG_FILLRULE_SHIFT) & SHAPE_FLAG_FILLRULE_MASK;
        bool inside   = (fillRule == 0u) ? (winding != 0) : ((winding & 1) != 0);
        return inside ? -minDist : minDist;
    }

    // Shape tags — evaluate in shape-local coords.
    vec2 local     = applyInverse(vInvXf01, vInvXf23_half.xy, fragCoord);
    vec2 shapeHalf = vInvXf23_half.zw;
    float cornerR  = vExtraA.x;
    float thick    = vExtraA.y;
    vec2  lineB    = vExtraA.zw;

    float d;
    if      (vTag == GEOM_RECT)    d = sdfRect(local, shapeHalf);
    else if (vTag == GEOM_RRECT)   d = sdfRoundedRect(local, shapeHalf, cornerR);
    else if (vTag == GEOM_ELLIPSE) d = sdfEllipse(local, shapeHalf);
    else                           d = sdfCapsule(local, lineB, thick);

    // Stroke / ring conversion for non-capsule shapes. Capsule's lineThickness
    // is its own cross-section, already consumed by sdfCapsule.
    bool isStroke = (vShapeFlags & SHAPE_FLAG_STROKE_BIT) != 0u;
    if (vTag != GEOM_CAPSULE && isStroke && thick > 0.0) {
        d = abs(d) - thick * 0.5;
    }
    return d;
}

// ---- Per-fragment blur radius ----------------------------------------------
//
// Falloff band straddles the shape edge per edgePlacement (centered / inside /
// outside). Returns the blur radius in PHYSICAL pixels regardless of tag —
// for shape tags we convert logical → physical via blurStep here; for path
// (already physical) blurStep is 1.0 on the host side.

float computeRadius(vec2 fragCoord) {
    float d          = shapeSDF(fragCoord);
    float maxRadius  = vExtraB.x;   // logical for shapes, physical for path
    float falloff    = vExtraB.y;
    float blurStep   = vExtraB.z;
    float strokeHW   = vExtraB.w;   // path-only; 0 for fill

    // Path stroke — ring SDF after we have the signed distance to the filled
    // path. Shape outlines take the other branch above via shapeFlags.
    if (vTag == GEOM_PATH && strokeHW > 0.0) {
        d = abs(d) - strokeHW;
    }

    uint edge = (vShapeFlags >> SHAPE_FLAG_EDGE_SHIFT) & SHAPE_FLAG_EDGE_MASK;
    float bandMin, bandMax;
    if (edge == 0u)      { bandMin = -falloff * 0.5; bandMax =  falloff * 0.5; }
    else if (edge == 1u) { bandMin = -falloff;       bandMax =  0.0;            }
    else                 { bandMin =  0.0;           bandMax =  falloff;        }

    float t = 1.0 - smoothstep(bandMin, bandMax, d);
    if ((vShapeFlags & SHAPE_FLAG_INVERTED_BIT) != 0u) t = 1.0 - t;

    // Physical-pixel blur radius. blurStep == 1.0 for tag 5 (path segments
    // already in physical pixels), so `maxRadius * t * blurStep` reduces to
    // `maxRadius * t` there.
    return maxRadius * t * blurStep;
}

// Mirror-wrap UVs to avoid the CLAMP_TO_EDGE bright-edge flash on thin
// foreground elements near the viewport edge. See shape_blur.frag notes.
vec4 sampleSrc(vec2 uv) {
    vec2 m = uv - 2.0 * floor(uv * 0.5 + 0.5);
    return texture(srcTexture, abs(m));
}

void main() {
    vec2 fragCoord = vFragPos;
    vec2 fragUV    = fragCoord / pc.viewportSize;

    float radius = computeRadius(fragCoord);  // physical px
    if (radius <= 0.5) {
        outColor = texture(srcTexture, fragUV);
        return;
    }

    float sigma = radius * 0.5;
    float twoS2 = 2.0 * sigma * sigma;
    int   N     = int(ceil(radius));

    vec2  texelSize   = 1.0 / pc.viewportSize;
    vec4  color       = texture(srcTexture, fragUV);
    float totalWeight = 1.0;

    if (pc.kernelType == 0) {
        // 1D separable LST. Caller runs this shader twice (direction = (1,0),
        // then (0,1)); ping-pong between the scene halves.
        for (int p = 0; p < MAX_PAIRS; ++p) {
            int i1 = 2 * p + 1;
            if (i1 > N) break;
            int i2 = i1 + 1;
            float w1 = exp(-float(i1 * i1) / twoS2);
            float w2 = (i2 <= N) ? exp(-float(i2 * i2) / twoS2) : 0.0;
            float w  = w1 + w2;
            float o  = (float(i1) * w1 + float(i2) * w2) / w;
            vec2 off = pc.direction * o * texelSize;
            color += sampleSrc(fragUV + off) * w;
            color += sampleSrc(fragUV - off) * w;
            totalWeight += 2.0 * w;
        }
    } else if (pc.kernelType == 1) {
        // Vogel / phi-spiral blue-noise disc, N ~= 4·radius.
        int Np = clamp(4 * int(ceil(radius)), 8, MAX_POISSON_TAPS);
        float invN = 1.0 / float(Np);
        for (int i = 0; i < MAX_POISSON_TAPS; ++i) {
            if (i >= Np) break;
            float r   = sqrt((float(i) + 0.5) * invN);
            float ang = float(i) * GOLDEN_ANGLE;
            vec2  v   = vec2(cos(ang), sin(ang)) * r;
            float w   = exp(-2.0 * r * r);
            vec2  off = v * radius * texelSize;
            color += sampleSrc(fragUV + off) * w;
            totalWeight += w;
        }
    } else {
        // True 2D Gaussian via 2D linear-sampling trick. See shape_blur.frag.
        for (int py = 0; py < MAX_PAIRS; ++py) {
            int y1 = 2 * py + 1;
            if (y1 > N) break;
            int y2 = y1 + 1;
            float wy1 = exp(-float(y1 * y1) / twoS2);
            float wy2 = (y2 <= N) ? exp(-float(y2 * y2) / twoS2) : 0.0;
            float wyT = wy1 + wy2;
            float yTap = (float(y1) * wy1 + float(y2) * wy2) / wyT;

            for (int px = 0; px < MAX_PAIRS; ++px) {
                int x1 = 2 * px + 1;
                if (x1 > N) break;
                int x2 = x1 + 1;
                float wx1 = exp(-float(x1 * x1) / twoS2);
                float wx2 = (x2 <= N) ? exp(-float(x2 * x2) / twoS2) : 0.0;
                float wxT = wx1 + wx2;
                float xTap = (float(x1) * wx1 + float(x2) * wx2) / wxT;

                float w = wxT * wyT;
                vec2 off = vec2(xTap, yTap) * texelSize;
                color += sampleSrc(fragUV + vec2( off.x,  off.y)) * w;
                color += sampleSrc(fragUV + vec2(-off.x,  off.y)) * w;
                color += sampleSrc(fragUV + vec2( off.x, -off.y)) * w;
                color += sampleSrc(fragUV + vec2(-off.x, -off.y)) * w;
                totalWeight += 4.0 * w;
            }

            vec2 yOff = vec2(0.0, yTap) * texelSize;
            color += sampleSrc(fragUV + yOff) * wyT;
            color += sampleSrc(fragUV - yOff) * wyT;
            totalWeight += 2.0 * wyT;
        }
        for (int px = 0; px < MAX_PAIRS; ++px) {
            int x1 = 2 * px + 1;
            if (x1 > N) break;
            int x2 = x1 + 1;
            float wx1 = exp(-float(x1 * x1) / twoS2);
            float wx2 = (x2 <= N) ? exp(-float(x2 * x2) / twoS2) : 0.0;
            float wxT = wx1 + wx2;
            float xTap = (float(x1) * wx1 + float(x2) * wx2) / wxT;

            vec2 xOff = vec2(xTap, 0.0) * texelSize;
            color += sampleSrc(fragUV + xOff) * wxT;
            color += sampleSrc(fragUV - xOff) * wxT;
            totalWeight += 2.0 * wxT;
        }
    }

    outColor = color / totalWeight;
}
