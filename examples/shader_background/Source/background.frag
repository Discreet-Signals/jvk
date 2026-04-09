#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    vec2 resolution;
    float time;
};

layout(set = 0, binding = 0) readonly buffer StorageBuffer {
    float data[];
};

// data[0] = saturation envelope [0..1]

vec3 center = vec3(0., 0., -1.0);
float radius = 0.5;

struct Ray {
    vec3 origin;
    vec3 direction;
};

float hitSphere(Ray ray, vec3 c, float r) {
    vec3 oc = ray.origin - c;
    float a = dot(ray.direction, ray.direction);
    float h = dot(oc, ray.direction);
    float cc = dot(oc, oc) - r * r;
    float delta = h * h - a * cc;
    float sqrtDelta = sqrt(max(delta, 0.0));
    float coefficient = step(0.0, delta);
    return mix(-h / sqrt(a), (-h - sqrtDelta) / a, coefficient);
}

vec3 rayColor(Ray ray) {
    float t = hitSphere(ray, center, radius);
    if (t > 0.0) {
        vec3 normal = normalize(ray.origin + t * ray.direction - center);

        // Rotate normal around Y axis — full revolution in 10 seconds
        float a = 5.1;
        float cs = cos(a), sn = sin(a);
        normal = vec3(normal.x * cs - normal.z * sn,
                      normal.y,
                      normal.x * sn + normal.z * cs);

        return 0.5 * (normal + 1.0);
    }
    return vec3(0.0);
}

void main() {
    float aspect = resolution.x / resolution.y;

    vec2 uv = (fragUV - 0.5) * 2.0;
    uv.x *= aspect;
    uv.y = -uv.y;

    Ray ray = Ray(vec3(0.0), normalize(vec3(uv.x, uv.y, -1.0)));
    vec3 col = rayColor(ray);

    float sat = clamp(data[0], 0.0, 1.0);
    col.r *= (0.35 + sat);
    col = 1.4 * tanh(1.1 * col);

    outColor = vec4(col, 1.0);
}
