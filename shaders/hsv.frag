#version 450

// Full-screen perceptual colour transform post-process (Oklab / Oklch).
//
// The push-constant layout keeps its historical "H / S / V" labels so the C++
// callers and helpers (`saturate`, `shiftHue`, ...) stay source-compatible.
// Internally the op runs in Oklch — the polar form of Oklab — which is
// perceptually uniform:
//
//   H field  → h  (hue angle, normalized to 0..1 cycles)
//   S field  → C  (chroma; 0 = grayscale, bright sRGB colours land ~0.1..0.3)
//   V field  → L  (lightness; perceptually uniform 0..1)
//
// Why Oklab instead of classical HSV:
//   - V in HSV is max(R,G,B), so pure red, pure blue, and white all collapse
//     to the SAME grey at scaleS=0. Oklab's L reflects human luminance weighting
//     (blue → dark grey, red → mid grey, white → white) so contrast survives
//     full desaturation.
//   - Hue sweeps are visually even (no green dominating the wheel) and C∞
//     continuous — no kink at 60° boundaries.
//
// Pipeline: sRGB → linear RGB → Oklab → Oklch → ops → Oklch → Oklab → linear
// → sRGB. The sRGB↔linear step uses pow(2.2) to match the rest of the codebase,
// which stores values in UNORM attachments and treats them as sRGB-encoded.

layout(set = 0, binding = 0) uniform sampler2D inputTex;

layout(push_constant) uniform PC {
    vec2  viewportSize;
    float scaleH, scaleS, scaleV;
    float deltaH, deltaS, deltaV;
} pc;

layout(location = 0) out vec4 outColor;

const float TWO_PI = 6.28318530717958647692;

vec3 srgbToLinear(vec3 c) { return pow(max(c, 0.0), vec3(2.2));      }
vec3 linearToSrgb(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }

// Björn Ottosson — https://bottosson.github.io/posts/oklab/
vec3 linRgbToOklab(vec3 rgb) {
    float l = 0.4122214708 * rgb.r + 0.5363325363 * rgb.g + 0.0514459929 * rgb.b;
    float m = 0.2119034982 * rgb.r + 0.6806995451 * rgb.g + 0.1073969566 * rgb.b;
    float s = 0.0883024619 * rgb.r + 0.2817188376 * rgb.g + 0.6299787005 * rgb.b;

    // Sign-preserving cube root — cbrt() isn't in core GLSL, and the LMS axes
    // can go slightly negative after the transform on out-of-sRGB colours.
    float l_ = sign(l) * pow(abs(l), 1.0 / 3.0);
    float m_ = sign(m) * pow(abs(m), 1.0 / 3.0);
    float s_ = sign(s) * pow(abs(s), 1.0 / 3.0);

    return vec3(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_);
}

vec3 oklabToLinRgb(vec3 lab) {
    float l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    float m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    float s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;

    float l = l_ * l_ * l_;
    float m = m_ * m_ * m_;
    float s = s_ * s_ * s_;

    return vec3(
         4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s);
}

void main() {
    vec4 src = texelFetch(inputTex, ivec2(gl_FragCoord.xy), 0);
    vec3 lab = linRgbToOklab(srgbToLinear(src.rgb));

    // Oklab → Oklch (polar).
    float L = lab.x;
    float C = length(lab.yz);
    float h = atan(lab.z, lab.y) / TWO_PI;  // -0.5..0.5 cycles

    // MULTIPLY-then-ADD per channel, matching the old HSV semantics so the
    // existing specialised helpers compose unchanged:
    //   saturate(a)   → scaleS = a                (chroma scale)
    //   shiftHue(t)   → deltaH = t                (hue rotation in cycles)
    //   brightness(a) → scaleV = a / deltaV = ... (lightness lift)
    L = L * pc.scaleV + pc.deltaV;
    C = C * pc.scaleS + pc.deltaS;
    h = h * pc.scaleH + pc.deltaH;

    L = clamp(L, 0.0, 1.0);
    C = max(C, 0.0);
    h = fract(h);  // wrap, preserving HSV's cyclic hue behaviour

    // Oklch → Oklab.
    float theta = h * TWO_PI;
    lab = vec3(L, C * cos(theta), C * sin(theta));

    vec3 lin = max(oklabToLinRgb(lab), 0.0);
    outColor = vec4(linearToSrgb(lin), src.a);
}
