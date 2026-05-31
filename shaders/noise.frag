#version 450

// Procedural white-noise overlay backing jvk::Graphics::drawNoise. Pairs with
// the ui2d vertex shader (standard UIVertex), so it receives the usual vertex
// interpolants; only the flat vertex colour is used to carry per-draw params:
//   fragColor.r = per-frame time offset (0 = static: same pattern every frame)
//   fragColor.a = amount -> AlphaBlend yields mix(dst, whiteNoise, amount)
//                 (0 = untouched, 1 = full white-noise overwrite, 0.1 = grain)
layout(location = 0) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

// Dave Hoskins' hash12 — one pseudo-random value in [0,1] per input. Keyed on
// gl_FragCoord (device pixels), so the grain is pixel-perfect and never tiles.
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    float n = hash12(gl_FragCoord.xy + fragColor.r);
    outColor = vec4(vec3(n), fragColor.a);
}
