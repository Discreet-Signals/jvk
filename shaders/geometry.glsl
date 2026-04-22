// =============================================================================
// geometry.glsl — shared include for geometry-abstracted fragment shaders.
//
// Mirrors the C++ GeometryPrimitive struct (pipelines/Params.h) field-for-field
// under std430 layout. Fragment shaders #include this header and read their
// per-instance data from the flat outputs the shared geometry.vert produces
// (locations 0..9). None of the helpers here bind any resources themselves —
// the path-segment SSBO is bound by the consuming pipeline (conventional slot
// set=2, binding=0) and accessed via the segmentsSSBO() macro below.
//
// Coord spaces:
//   FRAG    physical pixels (fragCoord in the viewport)
//   LOCAL   shape-local / user-logical (pre-transform) pixels
//   The fragment shader calls toLocal(fragCoord) to cross the boundary via
//   the primitive's inverse affine (physical → local). SDFs evaluate entirely
//   in LOCAL; only the kernel step back out to FRAG for texture sampling.
// =============================================================================

#ifndef GEOMETRY_GLSL_INCLUDED
#define GEOMETRY_GLSL_INCLUDED

// Geometry tag values. Mirror C++ jvk::GeometryTag.
#define GEOM_RECT       0u
#define GEOM_RRECT      1u
#define GEOM_ELLIPSE    2u
#define GEOM_CAPSULE    3u
#define GEOM_MSDF_GLYPH 4u
#define GEOM_PATH       5u
#define GEOM_IMAGE      6u

// shapeFlags bit layout (flags.y).
#define SHAPE_FLAG_INVERTED_BIT   (1u << 0)
#define SHAPE_FLAG_EDGE_SHIFT     1u
#define SHAPE_FLAG_EDGE_MASK      0x3u
#define SHAPE_FLAG_FILLRULE_SHIFT 3u
#define SHAPE_FLAG_FILLRULE_MASK  0x3u
#define SHAPE_FLAG_STROKE_BIT     (1u << 5)
// Color source — tag 4 + 6 draw RGB from the shape sampler regardless, but
// tags 0..3 + 5 use this to pick solid vs gradient colour lookup.
#define SHAPE_FLAG_COLOR_SHIFT    6u
#define SHAPE_FLAG_COLOR_MASK     0x3u
#define COLOR_SRC_SOLID           0u
#define COLOR_SRC_LINEAR          1u
#define COLOR_SRC_RADIAL          2u

// -----------------------------------------------------------------------------
// SDF primitives — all in shape-local coords, origin-centred. Copied verbatim
// from the legacy shape_blur.frag; kept here so every geometry-aware fragment
// shader evaluates shapes identically.
// -----------------------------------------------------------------------------

float sdfRect(vec2 p, vec2 h) {
    vec2 q = abs(p) - h;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
}

float sdfRoundedRect(vec2 p, vec2 h, float r) {
    vec2 q = abs(p) - h + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

// Iquilez-style cheap-but-good ellipse SDF. Accurate enough for AA/falloff
// bands; avoids the full quartic solver.
float sdfEllipse(vec2 p, vec2 radii) {
    vec2 pr = p / max(radii, vec2(1e-6));
    float k1 = length(pr);
    float k2 = length(pr / max(radii, vec2(1e-6)));
    return (k1 - 1.0) * k1 / max(k2, 1e-6);
}

// A at origin, B at `b`, cross-section radius `r`.
float sdfCapsule(vec2 p, vec2 b, float r) {
    float dotBB = max(dot(b, b), 1e-6);
    float h = clamp(dot(p, b) / dotBB, 0.0, 1.0);
    return length(p - b * h) - r;
}

// Unsigned distance from p to the line segment a → b. Used by path geometry
// (tag 5) in both fill and blur shaders.
float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float baLen2 = dot(ba, ba);
    float h = (baLen2 > 0.0) ? clamp(dot(pa, ba) / baLen2, 0.0, 1.0) : 0.0;
    return length(pa - ba * h);
}

// Half-open-on-top crossing contribution for a horizontal ray going +x from p.
// Returns +1, -1 or 0. Sum across a path's segments to get its winding number.
int pathCrossing(vec2 p, vec2 a, vec2 b) {
    bool aAbove = a.y > p.y;
    bool bAbove = b.y > p.y;
    if (aAbove == bAbove) return 0;
    float t = (p.y - a.y) / (b.y - a.y);
    float xCross = a.x + t * (b.x - a.x);
    if (xCross <= p.x) return 0;
    return bAbove ? 1 : -1;
}

// -----------------------------------------------------------------------------
// Affine helpers. The shader receives the inverse affine packed as:
//   invXf01 = vec4(m00, m10, m01, m11)
//   invXf23 = vec2(m02, m12)
// Apply it to a physical fragment coord to get a shape-local coord.
// -----------------------------------------------------------------------------

vec2 applyInverse(vec4 invXf01, vec2 invXf23, vec2 fragCoord) {
    return invXf01.xy * fragCoord.x
         + invXf01.zw * fragCoord.y
         + invXf23;
}

#endif // GEOMETRY_GLSL_INCLUDED
