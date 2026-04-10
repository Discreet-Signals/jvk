#version 450

layout(binding = 1) uniform sampler2D inputColor;

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 2) in vec4 fragShapeInfo;

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
    vec4 existing = texelFetch(inputColor, ivec2(gl_FragCoord.xy), 0);
    vec3 hsv = rgb2hsv(existing.rgb);
    hsv *= fragColor.xyz;
    hsv += fragShapeInfo.xyz;
    hsv.x = fract(hsv.x);
    hsv.y = clamp(hsv.y, 0.0, 1.0);
    hsv.z = clamp(hsv.z, 0.0, 1.0);
    outColor = vec4(hsv2rgb(hsv), existing.a);
}
