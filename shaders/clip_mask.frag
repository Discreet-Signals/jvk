#version 450
#include "jvk_portable.glsl"

// Image-alpha clip-stencil fragment shader (Graphics::clipToImageAlpha).
//
// The clip.frag protocol with an image for the shape: runs for every pixel of
// the cover quad, maps the pixel centre into the mask's texels through the
// transform the clip was made with, and `discard`s where the mask is outside
// or its alpha is below one half. Surviving fragments get their stencil INCR'd
// (push) or DECR'd (pop) by fixed-function HW — no color is written. Hard
// edges, like every stencil clip: alpha rounds to in or out per pixel.
//
// `nearest` is the resampling quality the clip was made with (juce reads it at
// clip time): point sampling snaps to the texel centre, as ui2d.frag does.

JVK_CONSTANTS {
    vec2  viewportSize;
    vec2  toMaskX;        // physical px → mask texels: x = dot(toMaskX, p) + toMaskOffset.x
    vec2  toMaskY;        //                            y = dot(toMaskY, p) + toMaskOffset.y
    vec2  toMaskOffset;
    vec2  maskSize;       // texels
    uint  nearest;
    uint  _pad;
} pc;

JVK_TEXTURE(0, maskTex);
JVK_SAMPLER(0, maskSampler);

layout(location = 0) in vec2 fragPos;

void main() {
    vec2 p = vec2(dot(pc.toMaskX, fragPos), dot(pc.toMaskY, fragPos)) + pc.toMaskOffset;
    if (any(lessThan(p, vec2(0.0))) || any(greaterThanEqual(p, pc.maskSize))) discard;
    vec2 uv = (pc.nearest != 0u ? floor(p) + 0.5 : p) / pc.maskSize;
    if (texture(sampler2D(maskTex, maskSampler), uv).a < 0.5) discard;
    // No color output — stencil INCR/DECR happens in fixed-function HW.
}
