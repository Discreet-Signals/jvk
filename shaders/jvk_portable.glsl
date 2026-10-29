// =============================================================================
// jvk_portable.glsl — the portable shader profile (TRANSLATION_LAYER.md)
//
// Every built-in shader includes this right after #version. One GLSL source
// compiles to Vulkan SPIR-V (-DJVK_VULKAN) and to the SPIR-V the other
// backends' sources are translated from (SPIRV-Cross → MSL), so bindings are
// declared through these macros and never by hand:
//
//   JVK_TEXTURE(i, name)   texture i      set 0, binding i       (i < 8)
//   JVK_SAMPLER(i, name)   its sampler    set 0, binding 8 + i
//   JVK_STORAGE(j)         storage j      set 0, binding 16 + j  (read-only)
//   JVK_CONSTANTS          the per-draw constants block: push constants on
//                          Vulkan, a uniform block at binding 24 elsewhere
//                          (MSL buffer(24), the same bytes on both stages)
//
// Sample with texture(sampler2D(tex, smp), uv); textureSize and texelFetch
// take the texture alone (GL_EXT_samplerless_texture_functions).
//
// Constants blocks hold scalars, vectors and vec4 arrays only — no vec3, no
// nested structs, no other arrays — so std430 (push constants) and std140
// (uniform blocks) put every member at the same offset. At most 128 bytes.
// The two stages of a program declare byte-compatible views of the block:
// the same members at the same offsets, one may declare a prefix.
//
// No specialisation constants (D3D12-via-SPIRV-Cross and eacp have none): a
// value that used to be one is a constant or a storage read.
// =============================================================================

#extension GL_EXT_samplerless_texture_functions : require

#ifdef JVK_VULKAN
 #define JVK_CONSTANTS layout(push_constant) uniform JvkConstants
#else
 #define JVK_CONSTANTS layout(std140, set = 0, binding = 24) uniform JvkConstants
#endif

#define JVK_TEXTURE(i, name) layout(set = 0, binding = i) uniform texture2D name
#define JVK_SAMPLER(i, name) layout(set = 0, binding = 8 + i) uniform sampler name
#define JVK_STORAGE(j) layout(std430, set = 0, binding = 16 + j) readonly buffer
