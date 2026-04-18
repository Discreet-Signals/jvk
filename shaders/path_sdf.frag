#version 450

// Analytical path-SDF fragment shader — vger / Slug style.
//
// Instead of pre-baking a distance field texture, the path is represented
// as a list of line segments (pre-flattened from beziers on the CPU) in a
// storage buffer. For each fragment we scan all segments, keeping the
// minimum unsigned distance and accumulating a winding-number crossing
// count. A pixel is "inside" the path iff:
//   non-zero winding: winding != 0
//   even-odd:         (winding & 1) != 0
//
// The signed distance is then `inside ? -minDist : minDist`, and alpha
// is a single-pixel anti-aliased smoothstep around zero. Since the path
// is evaluated *analytically* (no rasterisation), corners stay crisp at
// any zoom level and the output is resolution-independent with zero
// preprocessing cost — path can change every frame for free.

layout(push_constant) uniform PC {
    vec2  viewportSize;
    uint  segmentStart;
    uint  segmentCount;
    uint  fillRule;      // 0 = non-zero, 1 = even-odd
    float _pad;
} pc;

// Descriptor sets mirror the 2D color pipeline's layout so the existing
// ResourceCaches descriptors (`gradientDescriptor()` / `defaultDescriptor()`)
// can be bound directly at set 0 without extra plumbing. Set 1 owns this
// pipeline's per-frame line-segment SSBO — PathPipeline binds it manually.
//
//   set 0 = color LUT (1x1 default for solid, or 256x1 gradient atlas row)
//   set 1 = segments  (std430, 16 bytes per line segment as vec4)
layout(set = 0, binding = 0) uniform sampler2D colorLUT;

layout(std430, set = 1, binding = 0) readonly buffer Segments {
    vec4 data[];
} segments;

layout(location = 0) in vec2 fragPos;
layout(location = 1) in vec4 fragColor;
layout(location = 2) in vec4 fragGradientInfo;

layout(location = 0) out vec4 outColor;

// Unsigned distance from point p to the line segment a→b.
float sdSegment(vec2 p, vec2 a, vec2 b)
{
    vec2 pa = p - a;
    vec2 ba = b - a;
    float baLen2 = dot(ba, ba);
    float h = (baLen2 > 0.0) ? clamp(dot(pa, ba) / baLen2, 0.0, 1.0) : 0.0;
    return length(pa - ba * h);
}

// Crossing count contribution of segment (a→b) w.r.t. a horizontal ray
// going to +x from p. Returns +1, -1 or 0.
// Uses the standard "half-open on top" convention to handle vertices
// exactly on the ray without double-counting.
int crossing(vec2 p, vec2 a, vec2 b)
{
    bool aAbove = a.y >  p.y;
    bool bAbove = b.y >  p.y;
    if (aAbove == bAbove) return 0;

    // y-range straddles p.y — find x at the crossing.
    float t = (p.y - a.y) / (b.y - a.y);
    float xCross = a.x + t * (b.x - a.x);
    if (xCross <= p.x) return 0;

    return bAbove ? 1 : -1;
}

// Unified color source — identical contract to ui2d.frag::sampleColor().
// mode 0 = solid: vertex color carries RGB + opacity.
// mode 1 = linear: gradientInfo.x is the gradient parameter t, row at .w.
// mode 2 = radial: length(gradientInfo.xy) is t.
// Opacity always comes from fragColor.a (fillColorAttr folds it in for
// solid; fillColorAttr leaves .a = opacity for gradient).
vec4 sampleColor() {
    int mode = int(fragGradientInfo.z + 0.5);
    if (mode == 0)
        return fragColor;

    float t = (mode == 2) ? length(fragGradientInfo.xy) : fragGradientInfo.x;
    t = clamp(t, 0.0, 1.0);
    vec4 col = texture(colorLUT, vec2(t, fragGradientInfo.w));
    return vec4(col.rgb, col.a * fragColor.a);
}

void main()
{
    if (pc.segmentCount == 0u) { outColor = vec4(0.0); return; }

    float minDist = 1e20;
    int   winding = 0;

    for (uint i = 0u; i < pc.segmentCount; ++i)
    {
        vec4 seg = segments.data[pc.segmentStart + i];
        vec2 a = seg.xy;
        vec2 b = seg.zw;
        minDist = min(minDist, sdSegment(fragPos, a, b));
        winding += crossing(fragPos, a, b);
    }

    bool inside = (pc.fillRule == 0u)
        ? (winding != 0)
        : ((winding & 1) != 0);

    float signedDist = inside ? -minDist : minDist;

    // 1-pixel-wide AA kernel centred on the edge. fwidth gives the rate
    // of change of signedDist w.r.t. screen pixels, which is ~1 here
    // because fragPos *is* the physical-pixel position.
    float aa = max(fwidth(signedDist), 1e-6);
    float alpha = clamp(0.5 - signedDist / aa, 0.0, 1.0);

    vec4 col = sampleColor();
    outColor = vec4(col.rgb, col.a * alpha);
}
