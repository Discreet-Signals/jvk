#version 450

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec4 fragShapeInfo;

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

void main() {
    int type = int(fragShapeInfo.x + 0.5);

    // Type 0: Flat color (vertex-interpolated, solid fills)
    if (type == 0) {
        outColor = fragColor;
        return;
    }

    // Type 3: Textured quad (drawImage — sample texture, multiply by vertex color)
    if (type == 3) {
        vec4 texColor = texture(texSampler, fragUV);
        outColor = texColor * fragColor;
        return;
    }

    // Type 4: MSDF text — multi-channel signed distance field
    if (type == 4) {
        vec3 msd = texture(texSampler, fragUV).rgb;

        // DEBUG: if shapeInfo.z > 0.5, output raw MSDF RGB for visual inspection
        if (fragShapeInfo.z > 0.5) {
            outColor = vec4(msd, 1.0);
            return;
        }

        float sd = median(msd.r, msd.g, msd.b);
        float screenPxRange = fragShapeInfo.y;
        // JUCE's Y-flip reverses winding, so inside/outside sign is inverted.
        // Use (0.5 - sd) to compensate.
        float screenPxDist = screenPxRange * (0.5 - sd);
        float alpha = clamp(screenPxDist + 0.5, 0.0, 1.0);

        // Gamma-correct text alpha to match sRGB-blended text weight.
        // Linear blending thins AA pixels; this boosts midtones so text pops.
        alpha = pow(alpha, 0.4545);

        outColor = vec4(fragColor.rgb, fragColor.a * alpha);
        return;
    }

    // Type 5: Linear gradient — texSampler is a 1D gradient LUT
    if (type == 5) {
        float t = clamp(fragUV.x, 0.0, 1.0);
        vec4 gradColor = texture(texSampler, vec2(t, 0.5));
        outColor = vec4(gradColor.rgb, gradColor.a * fragColor.a);
        return;
    }

    // Type 6: Radial gradient — texSampler is a 1D gradient LUT
    if (type == 6) {
        float t = clamp(length(fragUV), 0.0, 1.0);
        vec4 gradColor = texture(texSampler, vec2(t, 0.5));
        outColor = vec4(gradColor.rgb, gradColor.a * fragColor.a);
        return;
    }

    // Types 1,2: Analytical SDF shapes (rounded rect, ellipse)
    vec2 halfSize = fragShapeInfo.yz;
    float param = fragShapeInfo.w;
    vec2 p = (fragUV - 0.5) * halfSize * 2.0;

    float dist = 0.0;
    if (type == 1) {
        dist = roundedRectSDF(p, halfSize, param);
    } else if (type == 2) {
        float k = length(p / max(halfSize, vec2(0.001)));
        dist = (k - 1.0) * min(halfSize.x, halfSize.y);
    }

    float aa = fwidth(dist);
    float alpha = 1.0 - smoothstep(-aa, aa, dist);
    outColor = vec4(fragColor.rgb, fragColor.a * alpha);
}
