// jvk's compute helpers: #include "jvk/compute.glsl" (jvk_compute_shaders() puts
// jvk's GLSL library on the include path).
#ifndef JVK_COMPUTE_GLSL
#define JVK_COMPUTE_GLSL

// This thread's index in a one-dimensional dispatch sized by Groups::cover(),
// which spreads its workgroups over y past the limit of one dimension. Threads
// past the count asked for run too: return when it is past.
uint jvkThread()
{
    return (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
}

// This workgroup's index, the same way.
uint jvkGroup()
{
    return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}

#endif
