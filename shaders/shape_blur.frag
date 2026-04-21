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
//
// `lineThickness` > 0 converts the solid SDF into a ring SDF before the band
// logic runs (rect / rrect / ellipse only — the capsule is stroke-native and
// already reads thickness as its cross-section radius). This is what drives
// the drawBlurred* (outline) API variants: the blur centre follows the ring
// boundary instead of the filled interior.

layout(push_constant) uniform PC {
    vec2  viewportSize;     // physical pixels of the src/dst image
    vec2  direction;        // (1,0) horizontal pass  |  (0,1) vertical
    vec2  invCol0;          // inverse affine: frag.xy (physical) → shape-local (logical)
    vec2  invCol1;
    vec2  invCol2;
    vec2  shapeHalf;        // rect / rrect / ellipse: halfsize in logical px
    vec2  lineB;            // line: endpoint B in shape-local (A is origin)
    float maxRadius;        // user-logical pixels (pre-transform)
    float falloff;          // user-logical pixels
    float blurStep;         // physical texels per user-logical pixel
                            // = transformScale * displayScale. Used ONLY
                            // to convert the logical radius → physical px
                            // count; the kernel still samples every
                            // physical texel (step = 1 texel) so we don't
                            // undersample and produce moiré. Cost scales
                            // with blurStep, which is the correct tradeoff
                            // for respecting the user transform.
    float cornerRadius;     // rrect only, logical px
    float lineThickness;    // line only, logical px
    int   shapeType;        // 0=rect, 1=rrect, 2=ellipse, 3=line
    int   edgePlacement;    // 0=centered, 1=inside, 2=outside
    int   inverted;         // 0 or 1
    int   kernelType;       // 0 = 1D separable (Low, caller runs this shader
                            //     twice with different `direction`s);
                            // 1 = 2D blue-noise disc, 32 fixed taps
                            //     regardless of radius (Medium);
                            // 2 = 2D grid with linear-sampling trick — exact
                            //     Gaussian at ¼ the taps of a naive N²
                            //     nested loop (High).
    float _pad0;            // pad block to 96 bytes (std140 multiple-of-16)
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

    // Stroke / ring SDF. For non-capsule shapes the C++ API's draw* (outline)
    // variants set lineThickness > 0; fill* variants leave it 0. The capsule
    // shape reads lineThickness as its own cross-section radius (above) so it
    // skips this step — its "outline" vs "fill" distinction is meaningless
    // since a capsule is already a stroke.
    if (pc.shapeType != 3 && pc.lineThickness > 0.0) {
        d = abs(d) - pc.lineThickness * 0.5;
    }

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

    // Blur radius in PHYSICAL pixels — blurStep folds in both the user's
    // transform scale and displayScale. Cost scales with it (more samples
    // at higher scale) but we sample every physical pixel, which avoids
    // the moiré you get from striding > 1 texel per sample.
    // Sampling is unclamped so the kernel can reach scene content outside
    // the shape's bounds; sampleSrc below only mirror-wraps at the
    // physical ping-pong buffer edge.
    return pc.maxRadius * t * pc.blurStep;
}

// Vogel / phi-spiral disc sampling. Given the golden angle θ = 2π(1 − 1/φ),
// the sequence (i·θ, √((i+0.5)/N)) lays N points on the unit disc with
// near-blue-noise spacing — no directional structure at any scale and no
// grid aliasing, at zero memory cost (pure arithmetic, no LUT). Not a true
// Poisson distribution (a faint spiral is detectable on still images with
// very few taps) but for Gaussian-weighted blur it's visually identical.
//
// With σ = radius/2 and |offset| = r·radius (r ∈ [0,1]), the Gaussian weight
// reduces to `exp(-2·r²)` — independent of the blur radius.
const float GOLDEN_ANGLE = 2.39996322972865332;

// Tap count scales linearly with radius so variance stays bounded as the
// disc grows: N = clamp(4·radius, 8, 1024). At radius 32 that's 128 taps,
// radius 256 → 1024 taps. The loop is bounded by MAX_POISSON_TAPS so the
// driver can cost-model it; runtime N drives the early break.
const int MAX_POISSON_TAPS = 1024;

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
    float radius   = computeRadius(fragCoord);   // physical pixels

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
        // 1D separable pass. Linear-sampling trick: pair consecutive
        // integer taps (1,2), (3,4), ... into a single bilinear sample at
        // the weighted midpoint. Caller runs this shader twice along
        // `direction`s (0°, 90°) and ping-pongs between them.
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
        // 2D blue-noise disc via Vogel / phi-spiral. Radius-scaled tap
        // count N = clamp(4·radius, 8, 1024) — roughly N samples per
        // quadrant — so variance stays bounded as the disc grows. Single
        // pass, no chained pre-blur, so no directional streak at `t`
        // boundaries. Trades Gaussian-exactness for a subtle frosted-glass
        // grain that vanishes into moving video.
        int N = clamp(4 * int(ceil(radius)), 8, MAX_POISSON_TAPS);
        float invN = 1.0 / float(N);
        for (int i = 0; i < MAX_POISSON_TAPS; ++i) {
            if (i >= N) break;
            float r  = sqrt((float(i) + 0.5) * invN);
            float ang = float(i) * GOLDEN_ANGLE;
            vec2  p  = vec2(cos(ang), sin(ang)) * r;
            float w  = exp(-2.0 * r * r);
            vec2  off = p * radius * texelSize;
            color += sampleSrc(fragUV + off) * w;
            totalWeight += w;
        }
    } else {
        // True 2D Gaussian with 2D linear-sampling trick. Each 2×2 block
        // of integer offsets collapses to a single bilinear tap at the
        // block's weighted centroid, exact for a separable Gaussian —
        // weight factorises as (w_x1+w_x2)·(w_y1+w_y2), tap position is
        // (x + w_x2/(w_x1+w_x2), y + w_y2/(w_y1+w_y2)). 4× reduction vs
        // the naïve N² nested loop, same output to float precision.
        //
        // Four-way symmetry: one quadrant iteration drives four samples
        // (±x, ±y). Axis samples (x=0 or y=0) handled in their own loops
        // since they aren't paired in both dimensions.
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

            // Axis x=0, y=±yTap.
            vec2 yOff = vec2(0.0, yTap) * texelSize;
            color += sampleSrc(fragUV + yOff) * wyT;
            color += sampleSrc(fragUV - yOff) * wyT;
            totalWeight += 2.0 * wyT;
        }
        // Axis y=0, x=±xTap.
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
