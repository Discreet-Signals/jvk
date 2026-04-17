#version 450

// Two descriptor sets, both combined image samplers:
//   set 0 = color LUT   (1x1 default for solid, or 256x1 gradient LUT)
//   set 1 = shape tex   (1x1 default, or MSDF atlas page, or image)
layout(set = 0, binding = 0) uniform sampler2D colorLUT;
layout(set = 1, binding = 0) uniform sampler2D shapeTex;

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec4 fragShapeInfo;
layout(location = 3) in vec4 fragGradientInfo;

layout(location = 0) out vec4 outColor;

// Signed distance to a rounded rectangle centered at origin
float roundedRectSDF(vec2 p, vec2 b, float r) {
    vec2 d = abs(p) - b + vec2(r);
    return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - r;
}

// Median of three values — core of MSDF rendering
float median(float r, float g, float b) {
    return max(min(r, g), min(max(r, g), b));
}

// ---- Color source --------------------------------------------------------
// mode 0: solid  — color from vertex attribute (fragColor)
// mode 1: linear — sample colorLUT row at gradientInfo.x
// mode 2: radial — sample colorLUT row at length(gradientInfo.xy)
// gradientInfo.w is the pre-normalised y of the gradient's row in the atlas
// (row-center, safe under linear filtering). Each row is one 256-sample LUT
// rasterized from juce::ColourGradient::getColourAtPosition, so N-stop
// gradients work natively. Opacity always comes from fragColor.a.
vec4 sampleColor() {
    int mode = int(fragGradientInfo.z + 0.5);
    if (mode == 0)
        return fragColor;

    float t = (mode == 2) ? length(fragGradientInfo.xy) : fragGradientInfo.x;
    t = clamp(t, 0.0, 1.0);
    vec4 col = texture(colorLUT, vec2(t, fragGradientInfo.w));
    return vec4(col.rgb, col.a * fragColor.a);
}

// ---- Shape source --------------------------------------------------------
// Returns an RGBA mask multiplied against sampled color. RGB is usually (1,1,1)
// except for images, which supply their own color; the "fill color" then acts
// as a tint.
// Types:
//   0 = flat quad            (full coverage)
//   1 = rounded rect SDF     (shapeInfo.yz = halfSize, .w = cornerRadius)
//   2 = ellipse SDF          (shapeInfo.yz = halfSize)
//   3 = sampled image        (shapeTex = image)
//   4 = MSDF text            (shapeTex = atlas page, shapeInfo.y = screenPxRange)
vec4 sampleShape() {
    int type = int(fragShapeInfo.x + 0.5);

    if (type == 0)
        return vec4(1.0);

    if (type == 3)
        return texture(shapeTex, fragUV);

    if (type == 4) {
        vec3 msd = texture(shapeTex, fragUV).rgb;
        float sd = median(msd.r, msd.g, msd.b);
        float screenPxRange = fragShapeInfo.y;

        // --- Weight shift ------------------------------------------------
        // MSDF considers `sd > 0.5` as "inside". Shifting the threshold
        // thickens (negative shift) or thins (positive shift) glyph bodies
        // uniformly. Default 0 = pure MSDF coverage, matching JUCE's
        // physically-correct edge. Exposed per-draw via jvk::Graphics so
        // callers can render variable-weight text without re-rasterizing
        // the atlas (something juce::Graphics can't do natively).
        const float WEIGHT_SHIFT = 0.0;
        float screenPxDist = screenPxRange * (sd - (0.5 - WEIGHT_SHIFT));
        float alpha = clamp(screenPxDist + 0.5, 0.0, 1.0);

        // --- sRGB encoding of alpha -------------------------------------
        // Pushes mid-alphas up so AA edges composite with the same "bolder
        // reader-friendly" weight sRGB-blended text has. This is the exact
        // sRGB linear→gamma transfer (IEC 61966-2-1), not the pow(1/2.2)
        // approximation — matters most for low-alpha pixels at stem edges.
        alpha = alpha <= 0.0031308
              ? 12.92 * alpha
              : 1.055 * pow(alpha, 1.0 / 2.4) - 0.055;

        return vec4(1.0, 1.0, 1.0, alpha);
    }

    // Stroked rounded rect (type 8), stroked ellipse (type 9):
    //   shapeInfo.yz = halfSize (of the filled shape)
    //   shapeInfo.w  = packHalf2(cornerSize, strokeWidth)  [type 8]
    //   shapeInfo.w  = packHalf2(0,          strokeWidth)  [type 9]
    // Standard stroke trick: abs(filledSDF) - strokeWidth/2 gives an annular band.
    if (type == 8 || type == 9) {
        vec2 halfSize = fragShapeInfo.yz;
        vec2 packed = unpackHalf2x16(floatBitsToUint(fragShapeInfo.w));
        float cornerSize = packed.x;
        float strokeW    = packed.y;
        vec2 p = (fragUV - 0.5) * halfSize * 2.0;

        float filled;
        if (type == 8) {
            filled = roundedRectSDF(p, halfSize, cornerSize);
        } else {
            float k = length(p / max(halfSize, vec2(0.001)));
            filled = (k - 1.0) * min(halfSize.x, halfSize.y);
        }
        float dist = abs(filled) - strokeW * 0.5;
        float aa = fwidth(dist);
        float alpha = 1.0 - smoothstep(-aa, aa, dist);
        return vec4(1.0, 1.0, 1.0, alpha);
    }

    // Capsule line (type 10) — quad is oriented along the line direction on CPU.
    // Local frame: +x along the segment, +y perpendicular. shapeInfo.yz holds
    // the unpadded (halfLen, radius); fragUV is offset to map 0..1 across the
    // padded quad so localP reaches slightly beyond (halfLen, radius) for AA.
    if (type == 10) {
        vec2 halfSize = fragShapeInfo.yz; // (halfLen, radius)
        vec2 p = (fragUV - 0.5) * halfSize * 2.0;
        vec2 q = vec2(max(abs(p.x) - halfSize.x, 0.0), p.y);
        float dist = length(q) - halfSize.y;
        float aa = fwidth(dist);
        float alpha = 1.0 - smoothstep(-aa, aa, dist);
        return vec4(1.0, 1.0, 1.0, alpha);
    }

    // Analytical SDF shapes (rounded rect, ellipse)
    vec2 halfSize = fragShapeInfo.yz;
    float param = fragShapeInfo.w;
    vec2 p = (fragUV - 0.5) * halfSize * 2.0;

    float dist;
    if (type == 1) {
        dist = roundedRectSDF(p, halfSize, param);
    } else {
        float k = length(p / max(halfSize, vec2(0.001)));
        dist = (k - 1.0) * min(halfSize.x, halfSize.y);
    }

    float aa = fwidth(dist);
    float alpha = 1.0 - smoothstep(-aa, aa, dist);
    return vec4(1.0, 1.0, 1.0, alpha);
}

void main() {
    vec4 color = sampleColor();
    vec4 shape = sampleShape();
    outColor = color * shape;
}
