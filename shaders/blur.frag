#version 450

// Separable Gaussian blur pass. Run twice — once horizontal, once vertical —
// with ping-pong framebuffers to get a 2D blur.
//
// Dynamic radius: weights are computed on-the-fly from a true Gaussian
// (sigma = radius / 2). Uses the linear-sampling trick — each paired tap
// is placed between two texels so bilinear filtering gives us two samples
// for one texture() call, halving the bandwidth for a given kernel extent.
//
// viewportSize: extent of the source image (pixels). Used to derive texel size.
// direction:    (1,0) for horizontal pass, (0,1) for vertical.
// radius:       kernel half-width in pixels. 0 = pass-through.
layout(push_constant) uniform PC {
    vec2  viewportSize;
    vec2  direction;
    float radius;
} pc;

layout(set = 0, binding = 0) uniform sampler2D srcTexture;

layout(location = 0) in  vec2 fragUV;
layout(location = 0) out vec4 outColor;

// Compile-time cap so drivers can bound the loop; supports radius up to ~128px.
const int MAX_PAIRS = 64;

void main() {
    vec2 texelSize = 1.0 / pc.viewportSize;

    if (pc.radius <= 0.5) {
        outColor = texture(srcTexture, fragUV);
        return;
    }

    float sigma = pc.radius * 0.5;
    float twoS2 = 2.0 * sigma * sigma;
    int   N     = int(ceil(pc.radius));

    // Center tap (weight = exp(0) = 1).
    vec4  color       = texture(srcTexture, fragUV);
    float totalWeight = 1.0;

    // Walk integer texel offsets in pairs (1,2), (3,4), (5,6), ...
    // Collapse each pair to a single bilinear tap between the two texels.
    for (int p = 0; p < MAX_PAIRS; ++p) {
        int i1 = 2 * p + 1;
        if (i1 > N) break;
        int i2 = i1 + 1;

        float w1 = exp(-float(i1 * i1) / twoS2);
        float w2 = (i2 <= N) ? exp(-float(i2 * i2) / twoS2) : 0.0;
        float w  = w1 + w2;
        float o  = (float(i1) * w1 + float(i2) * w2) / w;

        vec2 off = pc.direction * o * texelSize;
        color += texture(srcTexture, fragUV + off) * w;
        color += texture(srcTexture, fragUV - off) * w;
        totalWeight += 2.0 * w;
    }

    outColor = color / totalWeight;
}
