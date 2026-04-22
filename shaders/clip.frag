#version 450
#extension GL_GOOGLE_include_directive : require

// =============================================================================
// clip.frag — geometry-abstracted clip-stencil shader. Replaces the legacy
// clip.vert + clip.frag pair.
//
// Runs once for every pixel of the clip shape's cover quad (expanded from
// bbox by geometry.vert). Evaluates the shape's SDF / path-winding in the
// shared geometry.glsl helpers, discards outside pixels, and writes no
// colour — the push + pop VkPipeline variants of ClipPipeline configure the
// stencil op (INCR_WRAP for push, DECR_WRAP for pop) so the fixed-function
// HW updates the stencil buffer for every surviving fragment.
//
// Descriptor sets:
//   set 1, binding 0   SSBO   — GeometryPrimitive array (read in vertex)
//   set 2, binding 0   SSBO   — path segment array (tag=5 branch only)
//
// Supported tags: GEOM_RRECT (1) and GEOM_PATH (5). Axis-aligned rect clips
// never reach here — those fast-path through Graphics::clipToRectangle into
// plain scissor state.
// =============================================================================

#include "geometry.glsl"

layout(push_constant) uniform PC {
    vec2  viewportSize;
    vec2  _pad0;
    int   _pad1, _pad2, _pad3, _pad4;
} pc;

layout(std430, set = 2, binding = 0) readonly buffer Segments {
    vec4 data[];
} segments;

layout(location = 0)  flat in uint  vTag;
layout(location = 1)  flat in uint  vShapeFlags;
layout(location = 2)  flat in vec4  vInvXf01;
layout(location = 3)  flat in vec4  vInvXf23_half;
layout(location = 4)  flat in vec4  vExtraA;
layout(location = 5)  flat in vec4  vExtraB;
layout(location = 6)  flat in vec4  vPayload;
layout(location = 7)  flat in uvec2 vPathRange;
layout(location = 8)       in vec2  vFragPos;
layout(location = 9)       in vec2  vQuadUV;
layout(location = 10) flat in vec4  vColor;

// No color attachment write — stencil INCR/DECR happens in fixed-function.

void main() {
    bool inside = false;

    if (vTag == GEOM_PATH) {
        uint segStart = vPathRange.x;
        uint segCount = vPathRange.y;
        if (segCount == 0u) discard;
        int winding = 0;
        for (uint i = 0u; i < segCount; ++i) {
            vec4 seg = segments.data[segStart + i];
            winding += pathCrossing(vFragPos, seg.xy, seg.zw);
        }
        uint fillRule = (vShapeFlags >> SHAPE_FLAG_FILLRULE_SHIFT) & SHAPE_FLAG_FILLRULE_MASK;
        inside = (fillRule == 0u) ? (winding != 0) : ((winding & 1) != 0);
    } else {
        // Analytical SDF clip — evaluate in shape-local coords.
        vec2 local     = applyInverse(vInvXf01, vInvXf23_half.xy, vFragPos);
        vec2 shapeHalf = vInvXf23_half.zw;
        float cornerR  = vExtraA.x;
        float d;
        if      (vTag == GEOM_RRECT)   d = sdfRoundedRect(local, shapeHalf, cornerR);
        else if (vTag == GEOM_ELLIPSE) d = sdfEllipse(local, shapeHalf);
        else                           d = sdfRect(local, shapeHalf);
        inside = d < 0.0;
    }

    if (!inside) discard;
}
