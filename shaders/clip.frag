#version 450

// Clip-stencil fragment shader.
//
// Runs for every pixel of the clip shape's cover quad. Evaluates the shape's
// SDF analytically and `discard`s outside fragments. Surviving fragments get
// their stencil value INCR'd (push) or DECR'd (pop) by fixed-function HW —
// no color is written. Hard edges only; matches juce::Graphics clip semantics.
//
// Shape types:
//   1 = rounded rectangle   (analytical SDF, center + halfSize + cornerRadius)
//   2 = arbitrary path      (winding from flattened segment SSBO)
//
// Axis-aligned rect clips never hit this shader — they're routed to scissor
// by Graphics::clipToRectangle.

layout(push_constant) uniform PC {
    vec2  viewportSize;
    vec2  center;
    vec2  halfSize;
    float cornerRadius;
    uint  shapeType;
    uint  segmentStart;
    uint  segmentCount;
    uint  fillRule;
    uint  _pad;
} pc;

// Shared segment ring — owned by PathPipeline, bound here at clip dispatch.
layout(std430, set = 0, binding = 0) readonly buffer Segments {
    vec4 data[];
} segments;

layout(location = 0) in vec2 fragPos;

float roundedRectSDF(vec2 p, vec2 b, float r) {
    vec2 d = abs(p) - b + vec2(r);
    return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - r;
}

// Half-open-on-top crossing contribution for a horizontal ray going +x.
// Returns +1 / -1 / 0 to accumulate the winding number.
int crossing(vec2 p, vec2 a, vec2 b) {
    bool aAbove = a.y > p.y;
    bool bAbove = b.y > p.y;
    if (aAbove == bAbove) return 0;
    float t = (p.y - a.y) / (b.y - a.y);
    float xCross = a.x + t * (b.x - a.x);
    if (xCross <= p.x) return 0;
    return bAbove ? 1 : -1;
}

void main() {
    bool inside = false;

    if (pc.shapeType == 1u) {
        // Rounded-rect clip — center/halfSize are in physical pixels.
        vec2 p = fragPos - pc.center;
        inside = roundedRectSDF(p, pc.halfSize, pc.cornerRadius) < 0.0;
    } else if (pc.shapeType == 2u) {
        // Path clip — walk flattened segments from the shared SSBO.
        int winding = 0;
        for (uint i = 0u; i < pc.segmentCount; ++i) {
            vec4 seg = segments.data[pc.segmentStart + i];
            winding += crossing(fragPos, seg.xy, seg.zw);
        }
        inside = (pc.fillRule == 0u) ? (winding != 0)
                                     : ((winding & 1) != 0);
    }

    if (!inside) discard;
    // No color output — stencil INCR/DECR happens in fixed-function HW.
}
