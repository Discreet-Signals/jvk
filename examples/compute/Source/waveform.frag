#version 450
// Draws the peaks waveform.comp wrote this frame: the shader's storage block
// is that kernel's buffer, bound in place (jvk::Shader::set(name, range)).
layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC
{
    float resolutionX, resolutionY, time, viewportW, viewportH, regionX, regionY, reserved;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer Peaks { vec2 peaks[]; };

const int kColumns = 512;   // PluginEditor.h's

void main()
{
    const int column = clamp(int(fragUV.x * float(kColumns)), 0, kColumns - 1);
    const vec2 range = peaks[column];
    const float y = 1.0 - 2.0 * fragUV.y;   // +1 at the top
    const float edge = 1.5 / max(pc.resolutionY, 1.0);
    const float inside = smoothstep(range.x - edge, range.x, y) * (1.0 - smoothstep(range.y, range.y + edge, y));
    const vec3 back = vec3(0.06, 0.07, 0.09);
    const vec3 wave = mix(vec3(0.20, 0.55, 1.00), vec3(0.45, 0.95, 0.85), fragUV.x);
    outColor = vec4(mix(back, wave, inside), 1.0);
}
