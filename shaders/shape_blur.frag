#version 450

// Shape-aware variable-radius Gaussian blur. Runs as a separable 2-pass
// (horizontal, then vertical) just like blur.frag, but the per-pixel radius
// is driven by the fragment's distance to an analytic shape SDF:
//
//   inside the shape + beyond the falloff band   → radius = maxRadius
//   outside the shape + beyond the falloff band  → radius = 0 (pass-through)
//   within the falloff band                      → smoothstep between the two
//
// `edgePlacement` decides where the falloff band sits relative to SDF=0:
//   0 Centered → band = [-falloff/2, +falloff/2]
//   1 Inside   → band = [-falloff,   0]
//   2 Outside  → band = [0,         +falloff]
//
// `inverted` flips the weight (blur everything OUTSIDE the shape instead).

layout(push_constant) uniform PC {
    vec2  viewportSize;     // physical pixels of the src/dst image
    vec2  direction;        // (1,0) horizontal pass  |  (0,1) vertical
    vec2  invCol0;          // inverse affine: frag.xy (physical) → shape-local (logical)
    vec2  invCol1;
    vec2  invCol2;
    vec2  shapeHalf;        // rect / rrect / ellipse: halfsize in logical px
    vec2  lineB;            // line: endpoint B in shape-local (A is origin)
    float maxRadius;        // logical pixels
    float falloff;          // logical pixels
    float displayScale;     // logical → physical factor
    float cornerRadius;     // rrect only, logical px
    float lineThickness;    // line only, logical px
    int   shapeType;        // 0=rect, 1=rrect, 2=ellipse, 3=line
    int   edgePlacement;    // 0=centered, 1=inside, 2=outside
    int   inverted;         // 0 or 1
    float _pad0, _pad1;     // pad block to 96 bytes (std140 multiple-of-16)
} pc;

layout(set = 0, binding = 0) uniform sampler2D srcTexture;

layout(location = 0) in  vec2 fragUV;
layout(location = 0) out vec4 outColor;

// Upper bound for the driver's loop cost model. The runtime `if (i1 > N) break;`
// exits at the actual radius, so only insane radii (> 8192 px) would pay full cost.
const int MAX_PAIRS = 4096;

// ---- SDF primitives (all in shape-local coords, origin-centered) ----

float sdfRect(vec2 p, vec2 h) {
    vec2 q = abs(p) - h;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
}

float sdfRoundedRect(vec2 p, vec2 h, float r) {
    vec2 q = abs(p) - h + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

float sdfEllipse(vec2 p, vec2 radii) {
    // Iquilez-style cheap-but-good ellipse SDF.
    // Accurate enough for a blur falloff; avoids the full quartic solver.
    vec2 pr = p / max(radii, vec2(1e-6));
    float k1 = length(pr);
    float k2 = length(pr / max(radii, vec2(1e-6)));
    return (k1 - 1.0) * k1 / max(k2, 1e-6);
}

float sdfCapsule(vec2 p, vec2 b, float r) {
    // A at origin, B at `b`. Thickness r.
    float dotBB = max(dot(b, b), 1e-6);
    float h = clamp(dot(p, b) / dotBB, 0.0, 1.0);
    return length(p - b * h) - r;
}

// ---- Per-fragment blur radius ----

float computeRadius(vec2 fragCoord) {
    vec2 local = pc.invCol0 * fragCoord.x
               + pc.invCol1 * fragCoord.y
               + pc.invCol2;

    float d;
    if (pc.shapeType == 0)      d = sdfRect(local, pc.shapeHalf);
    else if (pc.shapeType == 1) d = sdfRoundedRect(local, pc.shapeHalf, pc.cornerRadius);
    else if (pc.shapeType == 2) d = sdfEllipse(local, pc.shapeHalf);
    else                        d = sdfCapsule(local, pc.lineB, pc.lineThickness);

    float bandMin, bandMax;
    if (pc.edgePlacement == 0) {
        bandMin = -pc.falloff * 0.5;
        bandMax =  pc.falloff * 0.5;
    } else if (pc.edgePlacement == 1) {
        bandMin = -pc.falloff;
        bandMax =  0.0;
    } else {
        bandMin =  0.0;
        bandMax =  pc.falloff;
    }

    // 1 deep-inside (d ≤ bandMin), 0 well-outside (d ≥ bandMax), smooth ramp.
    float t = 1.0 - smoothstep(bandMin, bandMax, d);
    if (pc.inverted != 0) t = 1.0 - t;

    // Requested blur radius (physical px) following the falloff curve.
    float requested = pc.maxRadius * t * pc.displayScale;

    // Symmetric cap — applied identically to H and V so the kernel stays a
    // proper blur, not a smear.
    //
    // Bounded shapes (rect / rrect / ellipse): isotropic SDF cap prevents
    // scene content from bleeding past the outer falloff edge into the
    // blurred region.
    //
    // Capsule: the shape is unbounded in one direction (by design — it
    // models a line segment), so an SDF-based cap would clamp the whole
    // blur to the narrow cross-section. Leave it uncapped; samples reaching
    // the texture boundary are mirrored by sampleSrc below so bright pixels
    // near the edge don't smear into a full-column flash.
    float cap;
    if (pc.shapeType == 3) {
        cap = 1e20; // effectively infinite; samples beyond the edge mirror back in
    } else {
        float sdfDist = (pc.inverted != 0) ? (d - bandMin) : (bandMax - d);
        sdfDist = max(sdfDist, 0.0);
        cap = sdfDist * pc.displayScale;
    }

    return min(requested, cap);
}

// Mirror-wrap UVs outside [0,1] back inside (triangle wave). Using this
// instead of raw texture() avoids the CLAMP_TO_EDGE artifact where a bright
// edge pixel gets weighted by every out-of-bounds sample in the Gaussian
// kernel — which looks like a full-column flash when thin bright UI (slider
// tracks, text) scrolls past the viewport edge. Reflection keeps the scene
// locally continuous so edge pixels contribute naturally to the blur.
vec4 sampleSrc(vec2 uv) {
    vec2 m = uv - 2.0 * floor(uv * 0.5 + 0.5);
    return texture(srcTexture, abs(m));
}

void main() {
    vec2 fragCoord = fragUV * pc.viewportSize;
    float radius   = computeRadius(fragCoord);

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

    // Linear-sampling trick: pairs (1,2), (3,4), ... collapse to a single
    // bilinear tap placed between the two integer texel offsets.
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

    outColor = color / totalWeight;
}
