#version 450

layout(location = 0) in vec4 fragColor;  // Input color from vertex shader
layout(location = 0) out vec4 outColor;  // Output color

layout(binding = 0) buffer BufferType
{
    float data[];
};

float softMax(float x, float edge, float softness) {
    if (x > edge + softness) return x; // Above the soft transition range
    return mix(edge, x, smoothstep(edge, edge + softness, x));
}

void main()
{
    float value = max(0.0, data[int(fragColor.r * 1023.999999)]);
    float pixel = 1.0 - fragColor.g;
    pixel = abs(2.0 * pixel - 1.0);
    bool waveform = pixel <= value;
    float dist = value - pixel;

    vec3 background = vec3(0.1, 0.1, 0.1);
    vec3 col = vec3(0.0);
    float alpha = 0.0;
    if (waveform)
    {
        dist = 1.0 - dist;
        dist = 0.2 + 0.8 * cos((dist * dist) * 0.5 * 3.14159);
        col = smoothstep(background, vec3(1.0), vec3(0.825, 1.1, 0.675) * dist);
        alpha = dist;
    }
    
    outColor = vec4(col, alpha);
}
