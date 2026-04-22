#version 450

// =============================================================================
// geometry.vert — shared vertex shader for geometry-abstracted pipelines.
//
// No vertex buffer is bound. `gl_InstanceIndex` picks a GeometryPrimitive out
// of the per-frame primitive SSBO (set=1, binding=0); `gl_VertexIndex`
// (0..5) picks one of the 6 corners of a triangle-list quad covering the
// primitive's bbox. The primitive's bbox is in PHYSICAL pixels — already
// post-transform + AA margin from the CPU — so no forward affine runs here.
// The inverse affine + geometry extras ride along as flat outputs so the
// fragment shader can evaluate SDFs in shape-local coords.
// =============================================================================

layout(push_constant) uniform PC {
    vec2  viewportSize;     // physical pixels
    vec2  direction;        // separable pass direction (ignored by 2D kernels)
    int   kernelType;       // 0 = Low (1D separable), 1 = Medium, 2 = High
    int   _pad0;
    int   _pad1;
    int   _pad2;
} pc;

// std430 mirror of jvk::GeometryPrimitive (pipelines/Params.h).
struct Primitive {
    vec4  bbox;              // xyxy — physical-pixel AABB
    uvec4 flags;             // (tag, shapeFlags, pathStart, pathCount)
    vec4  invXform01;        // m00, m10, m01, m11
    vec4  invXform23_half;   // m02, m12, shapeHalf.x, shapeHalf.y
    vec4  extraA;            // geometry-type / colorSource extras
    vec4  extraB;            // op-specific payload A
    vec4  payload;           // op-specific payload B
    vec4  color;             // solid RGBA or (1,1,1,opacity) mask
};

layout(std430, set = 1, binding = 0) readonly buffer Primitives {
    Primitive data[];
} prims;

// Flat pass-through of primitive fields. Fragment shader shares these via
// #include of geometry.glsl; locations are conventional and stable across
// every geometry-abstracted fragment shader.
layout(location = 0)  flat out uint  vTag;
layout(location = 1)  flat out uint  vShapeFlags;
layout(location = 2)  flat out vec4  vInvXf01;
layout(location = 3)  flat out vec4  vInvXf23_half;  // m02,m12,shapeHalf.xy
layout(location = 4)  flat out vec4  vExtraA;
layout(location = 5)  flat out vec4  vExtraB;
layout(location = 6)  flat out vec4  vPayload;
layout(location = 7)  flat out uvec2 vPathRange;     // (pathStart, pathCount)
layout(location = 8)       out vec2  vFragPos;       // physical-px position of this fragment
layout(location = 9)       out vec2  vQuadUV;        // 0..1 across the primitive quad
layout(location = 10) flat out vec4  vColor;         // RGBA, or (1,1,1,opacity)

// CCW triangle-list, 6 verts per quad.
const vec2 CORNERS[6] = vec2[](
    vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
);

void main() {
    Primitive p   = prims.data[gl_InstanceIndex];
    vec2 corner   = CORNERS[gl_VertexIndex];
    vec2 physPos  = mix(p.bbox.xy, p.bbox.zw, corner);

    vec2 ndc = (physPos / pc.viewportSize) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);

    vTag          = p.flags.x;
    vShapeFlags   = p.flags.y;
    vInvXf01      = p.invXform01;
    vInvXf23_half = p.invXform23_half;
    vExtraA       = p.extraA;
    vExtraB       = p.extraB;
    vPayload      = p.payload;
    vPathRange    = uvec2(p.flags.z, p.flags.w);
    vFragPos      = physPos;
    vQuadUV       = corner;
    vColor        = p.color;
}
