#version 450

// Full-screen HSV transform post-process.
//
// Converts the sampled scene pixel to HSV, applies MULTIPLY-THEN-ADD per
// channel, then converts back to RGB. This one shader covers every HSV
// operation jvk will ever need:
//
//   scaleH = 1, deltaH = 0      → no hue change
//   scaleS = 0                  → grayscale (desaturate)
//   scaleS in (0,1)             → partial desaturation
//   scaleS > 1                  → boosted saturation
//   deltaH = 0.5                → 180° hue shift
//   scaleV < 1                  → darken
//   scaleV > 1, deltaV > 0      → brighten / lift
//
// H wraps modulo 1.0 (fract). S and V clamp to [0, 1]. Defaults (identity
// transform) are scales=1, deltas=0.
//
// Specialised helpers on jvk::Graphics (`saturate`, `shiftHue`, `brightness`,
// ...) set a subset of the six parameters; the shader doesn't need to know
// which helper called it.

layout(set = 0, binding = 0) uniform sampler2D inputTex;

layout(push_constant) uniform PC {
    vec2  viewportSize;
    float scaleH, scaleS, scaleV;
    float deltaH, deltaS, deltaV;
} pc;

layout(location = 0) out vec4 outColor;

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.zy, K.wz), vec4(c.yz, K.xy), vec4(step(c.z, c.y)));
    vec4 q = mix(vec4(p.xyw, c.x), vec4(c.x, p.yzx), vec4(step(p.x, c.x)));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y) * c.z;
}

void main() {
    vec4  src = texelFetch(inputTex, ivec2(gl_FragCoord.xy), 0);
    vec3  hsv = rgb2hsv(src.rgb);

    // Multiply first, then add — lets callers compose transforms cleanly
    // (e.g. saturate=0 AND shiftHue=0.25 works as expected: zero-scale
    // kills the old hue, then the delta establishes a new one).
    hsv *= vec3(pc.scaleH, pc.scaleS, pc.scaleV);
    hsv += vec3(pc.deltaH, pc.deltaS, pc.deltaV);

    hsv.x = fract(hsv.x);
    hsv.y = clamp(hsv.y, 0.0, 1.0);
    hsv.z = clamp(hsv.z, 0.0, 1.0);

    outColor = vec4(hsv2rgb(hsv), src.a);
}
