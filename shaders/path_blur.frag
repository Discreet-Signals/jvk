#version 450

// Path-SDF variable-radius Gaussian blur. Structurally a mashup:
//   - shape_blur.frag's separable H/V blur + smoothstep falloff,
//   - path_sdf.frag's analytical segment-loop distance + winding-number SDF.
//
// Set 0  → source scene texture (read/write via ping-pong like every other
//          post-process effect).
// Set 1  → the same per-frame segment SSBO PathPipeline owns (PathPipeline's
//          descriptor is bound here so we read its already-uploaded segments).
//
// All distances in this shader are in physical pixels — the segments are
// uploaded physical by Graphics::{draw,fill}BlurredPath (matching fillPath's
// convention), and the C++ side converts the caller's logical blurRadius /
// falloffRadius / strokeHalfWidth to physical before pushing.
//
// `strokeHalfWidth` > 0 converts the filled-path SDF into a ring SDF
// (`abs(d) - strokeHalfWidth`) so the blur band centres on the path's
// stroke instead of the filled interior — what drawBlurredPath uses.

layout(push_constant) uniform PC {
    vec2  viewportSize;     // physical pixels
    vec2  direction;        // (1,0) horizontal | (0,1) vertical

    float maxRadius;        // physical pixels (= user × transformScale × displayScale)
    float falloff;          // physical pixels
    float strokeHalfWidth;  // physical pixels — 0 for fill, >0 for stroke

    uint  segmentStart;
    uint  segmentCount;
    uint  fillRule;         // 0 = non-zero winding, 1 = even-odd

    int   edgePlacement;    // 0 centered, 1 inside, 2 outside
    int   inverted;

    int   kernelType;       // 0 = 1D separable; 1 = 2D Poisson disc;
                            // 2 = 2D grid with LST. See shape_blur.frag.
    float _pad0, _pad1, _pad2;  // pad to 64 bytes (multiple of 16)
} pc;

layout(set = 0, binding = 0) uniform sampler2D srcTexture;

layout(std430, set = 1, binding = 0) readonly buffer Segments {
    vec4 data[];
} segments;

layout(location = 0) in  vec2 fragUV;
layout(location = 0) out vec4 outColor;

const int MAX_PAIRS = 4096;

float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float baLen2 = dot(ba, ba);
    float h = (baLen2 > 0.0) ? clamp(dot(pa, ba) / baLen2, 0.0, 1.0) : 0.0;
    return length(pa - ba * h);
}

int crossing(vec2 p, vec2 a, vec2 b) {
    bool aAbove = a.y > p.y;
    bool bAbove = b.y > p.y;
    if (aAbove == bAbove) return 0;
    float t = (p.y - a.y) / (b.y - a.y);
    float xCross = a.x + t * (b.x - a.x);
    if (xCross <= p.x) return 0;
    return bAbove ? 1 : -1;
}

float pathSDF(vec2 fragCoord) {
    if (pc.segmentCount == 0u) return 1e20;
    float minDist = 1e20;
    int   winding = 0;
    for (uint i = 0u; i < pc.segmentCount; ++i) {
        vec4 seg = segments.data[pc.segmentStart + i];
        vec2 a = seg.xy;
        vec2 b = seg.zw;
        minDist = min(minDist, sdSegment(fragCoord, a, b));
        winding += crossing(fragCoord, a, b);
    }
    bool inside = (pc.fillRule == 0u) ? (winding != 0) : ((winding & 1) != 0);
    return inside ? -minDist : minDist;
}

float computeRadius(vec2 fragCoord) {
    // Segments and the SDF are in physical pixels; so are maxRadius,
    // falloff, and strokeHalfWidth (the C++ caller pre-multiplied them
    // by transformScale × displayScale).
    float d = pathSDF(fragCoord);

    // Stroke / ring SDF — only when the caller asked for an outline draw.
    if (pc.strokeHalfWidth > 0.0) {
        d = abs(d) - pc.strokeHalfWidth;
    }

    float bandMin, bandMax;
    if (pc.edgePlacement == 0)      { bandMin = -pc.falloff * 0.5; bandMax =  pc.falloff * 0.5; }
    else if (pc.edgePlacement == 1) { bandMin = -pc.falloff;       bandMax =  0.0; }
    else                            { bandMin =  0.0;              bandMax =  pc.falloff; }

    float t = 1.0 - smoothstep(bandMin, bandMax, d);
    if (pc.inverted != 0) t = 1.0 - t;

    // The kernel reads freely from the full ping-pong buffer — the SDF
    // only drives blur *intensity* (t), never clamps kernel reach.
    // sampleSrc mirror-wraps at the physical texture edge, not at the
    // path bounds, so a blurred path over some other scene content picks
    // up that content naturally.
    return pc.maxRadius * t;
}

// Vogel / phi-spiral disc sampling — see shape_blur.frag for design notes.
const float GOLDEN_ANGLE = 2.39996322972865332;
const int   MAX_POISSON_TAPS = 1024;

// Mirror-wrap UVs so edge pixels don't smear into a full-column flash when
// thin bright UI elements scroll past the viewport edge (same trick
// shape_blur.frag uses).
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
        // 1D separable LST. See shape_blur.frag.
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
        // Radius-scaled Vogel / phi-spiral disc. See shape_blur.frag.
        int N = clamp(4 * int(ceil(radius)), 8, MAX_POISSON_TAPS);
        float invN = 1.0 / float(N);
        for (int i = 0; i < MAX_POISSON_TAPS; ++i) {
            if (i >= N) break;
            float r   = sqrt((float(i) + 0.5) * invN);
            float ang = float(i) * GOLDEN_ANGLE;
            vec2  p   = vec2(cos(ang), sin(ang)) * r;
            float w   = exp(-2.0 * r * r);
            vec2  off = p * radius * texelSize;
            color += sampleSrc(fragUV + off) * w;
            totalWeight += w;
        }
    } else {
        // 2D grid with linear-sampling trick — ¼ the taps of a naïve
        // nested N² loop at identical output. See shape_blur.frag.
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
