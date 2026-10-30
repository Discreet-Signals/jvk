// Scans (prefix combinations) of any associative operation, in a workgroup:
// sums, but also anything that composes in order, such as affine maps (a
// linear recursion's steps: x[t] = A[t] x[t-1] + b[t], whose chunks compose,
// so a recursion runs in parallel over time). The order is kept, so the
// operation need not commute: JVK_SCAN_COMBINE(a, b) means a, then b.
//
// Before including, define:
//   JVK_SCAN_T              the element type (a struct is fine)
//   JVK_SCAN_COMBINE(a, b)  a then b
//   JVK_SCAN_IDENTITY       the element that changes nothing
//   JVK_SCAN_SIZE           the workgroup's threads: its local_size_x (y and z 1)
// and, for jvkScanSequence(), how the sequence is read and its prefixes written:
//   JVK_SCAN_LOAD(k)            element k
//   JVK_SCAN_STORE(k, prefix)   element k's exclusive prefix: everything before it combined
//
// Every thread of the workgroup must make the call (they meet at barriers).
// Shared memory: 2 * JVK_SCAN_SIZE elements.
//
// Portable by design: no subgroup operations (Vulkan 1.0's SPIR-V, Metal 2.1), and no
// waiting on other workgroups (neither Vulkan nor Metal promises that one workgroup
// makes progress while another waits for it; single-pass scans that depend on
// it can hang). A scan longer than a workgroup goes in passes: combine chunks,
// scan their totals, apply them.
#ifndef JVK_SCAN_GLSL
#define JVK_SCAN_GLSL

shared JVK_SCAN_T jvkScanShared[2 * JVK_SCAN_SIZE];

// Each thread's element in, the combination of every earlier thread's out
// (identity for the first), and the whole workgroup's in `total`.
JVK_SCAN_T jvkScanExclusive(JVK_SCAN_T value, out JVK_SCAN_T total)
{
    const uint i = gl_LocalInvocationIndex;
    uint from = 0u, to = JVK_SCAN_SIZE;
    jvkScanShared[i] = value;
    memoryBarrierShared();
    barrier();
    // Hillis and Steele: at each step, each element takes in the one `step` before it.
    for (uint step = 1u; step < JVK_SCAN_SIZE; step <<= 1u)
    {
        JVK_SCAN_T v = jvkScanShared[from + i];
        if (i >= step)
            v = JVK_SCAN_COMBINE(jvkScanShared[from + i - step], v);
        jvkScanShared[to + i] = v;
        memoryBarrierShared();
        barrier();
        const uint t = from; from = to; to = t;
    }
    total = jvkScanShared[from + JVK_SCAN_SIZE - 1u];
    const JVK_SCAN_T before = i == 0u ? JVK_SCAN_IDENTITY : jvkScanShared[from + i - 1u];
    memoryBarrierShared();
    barrier();   // the shared memory is free for the next scan
    return before;
}

#if defined(JVK_SCAN_LOAD) && defined(JVK_SCAN_STORE)
// A sequence of `count` elements, any length, scanned by one workgroup: each
// thread combines a run of it, the workgroup scans the runs, and each thread
// walks its run again writing each element's prefix. Returns the total.
JVK_SCAN_T jvkScanSequence(uint count)
{
    const uint i = gl_LocalInvocationIndex;
    const uint each = (count + JVK_SCAN_SIZE - 1u) / JVK_SCAN_SIZE;
    const uint first = min(i * each, count), last = min(first + each, count);
    JVK_SCAN_T run = JVK_SCAN_IDENTITY;
    for (uint k = first; k < last; ++k)
        run = JVK_SCAN_COMBINE(run, JVK_SCAN_LOAD(k));
    JVK_SCAN_T total;
    JVK_SCAN_T prefix = jvkScanExclusive(run, total);
    for (uint k = first; k < last; ++k)
    {
        const JVK_SCAN_T element = JVK_SCAN_LOAD(k);
        JVK_SCAN_STORE(k, prefix);
        prefix = JVK_SCAN_COMBINE(prefix, element);
    }
    return total;
}
#endif

#endif
