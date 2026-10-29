#version 450
#include "jvk_portable.glsl"

// Analytical path-SDF fragment shader — TILE mode (piet-gpu-class).
//
// The CPU (Graphics::fillPath) decomposes the path into 16px tiles:
//   - tiles the path never touches are NOT DRAWN at all;
//   - interior tiles arrive with localCount == 0 and a nonzero backdrop —
//     constant fill, zero per-fragment segment work;
//   - edge tiles carry a small local segment list (segments whose padded
//     bbox intersects the tile) plus the tile's backdrop winding.
//
// EXACT WINDING DERIVATION (why walking only local segments is not an
// approximation): define W(q) as the signed crossing count of a +x ray
// from q, half-open on top (a segment counts iff (a.y > q.y) != (b.y >
// q.y) and its x at q.y is > q.x; sign = (b.y > q.y) ? +1 : -1). Then for
// a fragment p in a tile with top edge y = T and right edge x = R:
//
//   W(p) = W((R, T))                                   … per-tile backdrop
//        + [crossings of y = T with x in (p.x, R]]     … "top" term
//        + [crossings of x = p.x with y in (T, p.y]]   … "descent" term
//
// The backdrop is the ray at the tile's top-RIGHT corner — every segment
// it counts crosses y = T strictly right of R, i.e. outside the tile, and
// is summed on the CPU. The top term's crossings happen at y = T with
// x ∈ (p.x, R] — inside the tile, so those segments are in the local
// list. The descent term's crossings happen at x = p.x ∈ tile — local
// again. Nothing non-local can contribute, so the sum is exact.
//
// EVERY boundary between the terms is decided from ONE number the CPU
// computed: xc, where the segment crosses y = T. The CPU uses it to count
// the crossing into the backdrop of every tile left of it and to hand it
// to the tile it lands in as an OWNER (the first `owners` = fragTile.w
// entries of that tile's list — only owners take part in the top term),
// and it rides with every list entry of a segment that crosses this
// tile's top (each entry is two vec4s: the segment, then (xc, 0, 0, 0)).
// Here the one comparison `xc > p.x` then answers both halves of the
// corner (p.x, T): the top term's "is the crossing right of the
// fragment?" and the descent term's "does the segment meet x = p.x below
// T?". A crossing near — or exactly on — the corner is therefore claimed
// exactly once, by construction, whatever this GPU's arithmetic does.
//
// History, because each step was a real failure: two independent
// divisions answered "no" twice at the corner and punched 1px holes down
// to the tile bottom; a shared `precise` cross product (`cornerSide`) fixed
// that on NVIDIA, but the descent gate built on it — `(bRight ? -cs : cs) >
// 0` — evaluated true for segments meeting x = p.x ABOVE the tile on an
// Intel HD 530 (Gen9, Windows), dripping fill from every glyph to its tile's
// bottom edge; and re-deriving ownership as `xc <= R` on the GPU relied on
// OpFDiv rounding like the CPU's, which Vulkan does not require (2.5 ULP).
// What is left here is comparisons of stored values plus one per-fragment
// division (the descent crossing's height) that no other term depends on.
//
// AA is derivative-free: fragPos is in physical pixels and a distance
// field is 1-Lipschitz, so the edge band is exactly 1px wide — no
// fwidth(). (fwidth across tile/strip boundaries mixes distances from
// different segment lists and produced horizontal streak artifacts.)

// Shared with path_sdf.vert (byte-identical view).
JVK_CONSTANTS {
    vec2  viewportSize;
    uint  segmentStart;
    uint  segmentCount;   // unused in tile mode
    uint  fillRule;       // 0 = non-zero, 1 = even-odd
    uint  _r0;
    float tileW;     // tile width in physical px (unread: ownership comes from the CPU)
    float _r2;
    int   paintsMain;     // 1: a draw into the main colour that clears targets (writeTargets)
    int   _p0;
    int   _p1;
    int   _p2;
} pc;

// Texture 0: the colour source (gradient LUT, a tiled fill's image, or the 1x1 default).
JVK_TEXTURE(0, colorLUT);
JVK_SAMPLER(0, colorLUTSampler);

JVK_STORAGE(0) Segments {
    vec4 data[];
} segments;

// The frame's render-target clear values (jvk::Target::clear), rgb used.
JVK_STORAGE(1) FrameData {
    vec4 targetClear[7];
} frame;

layout(location = 0) in vec2 fragPos;
layout(location = 1) in vec4 fragColor;
layout(location = 2) in vec4 fragGradientInfo;
layout(location = 3) flat in vec4 fragTile;        // x=localStart y=count z=backdrop w=owners
layout(location = 4) flat in vec2 fragTileOrigin;  // tile top-left (px)

layout(location = 0) out vec4 outColor;
// Render targets: the same colour at every other location (see ui2d.frag).
layout(location = 1) out vec4 outChannel0;
layout(location = 2) out vec4 outChannel1;
layout(location = 3) out vec4 outChannel2;
layout(location = 4) out vec4 outChannel3;
layout(location = 5) out vec4 outChannel4;
layout(location = 6) out vec4 outChannel5;
layout(location = 7) out vec4 outChannel6;

float sdSegment(vec2 p, vec2 a, vec2 b)
{
    vec2 pa = p - a;
    vec2 ba = b - a;
    float baLen2 = dot(ba, ba);
    float h = (baLen2 > 0.0) ? clamp(dot(pa, ba) / baLen2, 0.0, 1.0) : 0.0;
    return length(pa - ba * h);
}

// TODO(WGSL uniformity): texture() under the per-vertex mode branch; sample
// with explicit LOD/gradients before a WGSL build.
vec4 sampleColor() {
    int mode = int(fragGradientInfo.z + 0.5);
    if (mode == 0)
        return fragColor;
    if (mode == 3) {
        // A tiled image fill: texture 0 holds the image, gradientInfo.xy its UV
        // (unbounded: wrapped, as the tiled image quad wraps), .w the
        // point-sampling flag. Its texel is premultiplied, and this returns
        // straight alpha (main() premultiplies once): un-premultiplied here.
        vec2 uv = fract(fragGradientInfo.xy);
        if (fragGradientInfo.w > 0.5) {
            vec2 sz = vec2(textureSize(colorLUT, 0));
            uv = (floor(uv * sz) + 0.5) / sz;
        }
        vec4 px = texture(sampler2D(colorLUT, colorLUTSampler), uv);
        return px.a > 0.0 ? vec4(px.rgb / px.a, px.a * fragColor.a) : vec4(0.0);
    }
    float t = (mode == 2) ? length(fragGradientInfo.xy) : fragGradientInfo.x;
    t = clamp(t, 0.0, 1.0);
    vec4 col = texture(sampler2D(colorLUT, colorLUTSampler), vec2(t, fragGradientInfo.w));
    return vec4(col.rgb, col.a * fragColor.a);
}

// PAINT INTO THE MAIN COLOUR also sets the render targets declared
// Target::clearedByMain to their clear values, by the paint's alpha (see
// ui2d.frag): pc.paintsMain = 1 and FrameData holds the values.
void writeTargets(vec4 colour)
{
    bool paint = pc.paintsMain != 0;
    float a = colour.a;
    outChannel0 = paint ? vec4(frame.targetClear[0].rgb, a) : colour;
    outChannel1 = paint ? vec4(frame.targetClear[1].rgb, a) : colour;
    outChannel2 = paint ? vec4(frame.targetClear[2].rgb, a) : colour;
    outChannel3 = paint ? vec4(frame.targetClear[3].rgb, a) : colour;
    outChannel4 = paint ? vec4(frame.targetClear[4].rgb, a) : colour;
    outChannel5 = paint ? vec4(frame.targetClear[5].rgb, a) : colour;
    outChannel6 = paint ? vec4(frame.targetClear[6].rgb, a) : colour;
}

void main()
{
    uint  base    = pc.segmentStart + 2u * uint(fragTile.x);   // two vec4s per entry
    uint  count   = uint(fragTile.y);
    int   winding = int(fragTile.z);      // backdrop at (tileRight, tileTop)
    uint  owners  = uint(fragTile.w);     // list entries that cross the tile top here
    float tileTop = fragTileOrigin.y;

    float alpha;
    if (count == 0u)
    {
        // Interior (or CPU-kept exterior) tile: constant coverage.
        bool inside = (pc.fillRule == 0u) ? (winding != 0)
                                          : ((winding & 1) != 0);
        alpha = inside ? 1.0 : 0.0;
    }
    else
    {
        float minDist = 1e20;
        for (uint i = 0u; i < count; ++i)
        {
            vec4  seg = segments.data[base + 2u * i];
            float xc  = segments.data[base + 2u * i + 1u].x;   // where it crosses y = tileTop (if it does)
            vec2 a = seg.xy;
            vec2 b = seg.zw;

            minDist = min(minDist, sdSegment(fragPos, a, b));

            // Which ends lie past (below) the tile top — the CPU's own
            // half-open test against the same floats, so "crosses the top"
            // here is exactly the set it computed xc for.
            bool aPastT = a.y > tileTop;
            bool bPastT = b.y > tileTop;
            // The ONE corner decision (see the header): the crossing of the
            // tile top lies strictly right of this fragment.
            bool xcRight = xc > fragPos.x;

            // Top term — crossing of y = tileTop at x in (p.x, tileRight].
            // The owners are exactly the segments crossing the top inside
            // this tile, as the CPU binned them against the backdrop.
            if (i < owners && xcRight)
                winding += bPastT ? 1 : -1;

            // Descent term — crossing of x = p.x at y in (tileTop, p.y].
            // Sign from continuity of W along the downward walk: passing
            // below a left-to-right segment adds +1 (its crossing point
            // moves onto the ray), right-to-left subtracts.
            bool aRight = a.x > fragPos.x;
            bool bRight = b.x > fragPos.x;
            if (aRight != bRight)
            {
                // Is that crossing below the tile top? A segment that does
                // not cross the top line answers by its ends: wholly past it,
                // or wholly on/above it (a flat segment lying ON the top is
                // the latter — it crosses nothing in (tileTop, p.y]). One that
                // crosses it meets x = p.x below the top iff p.x lies between
                // xc and its past-the-top end, i.e. iff that end and the
                // crossing sit on opposite sides of p.x (`(bPastT == bRight)`
                // is the end's side, as the segment straddles p.x). Through
                // the corner exactly (xc == p.x) that counts the segments
                // whose past-the-top end is right of p.x — the ones whose
                // crossing moves right as y grows, which the top term then
                // leaves alone (its test is strict): claimed exactly once.
                bool belowTop = (aPastT == bPastT) ? aPastT
                                                   : ((bPastT == bRight) != xcRight);
                if (belowTop)
                {
                    float t  = (fragPos.x - a.x) / (b.x - a.x);
                    float yc = a.y + t * (b.y - a.y);
                    if (yc <= fragPos.y)
                        winding += bRight ? 1 : -1;
                }
            }
        }

        bool inside = (pc.fillRule == 0u) ? (winding != 0)
                                          : ((winding & 1) != 0);
        float signedDist = inside ? -minDist : minDist;
        // 1px AA band, derivative-free (distance is in physical pixels).
        alpha = clamp(0.5 - signedDist, 0.0, 1.0);
    }

    vec4 col = sampleColor();
    outColor = vec4(col.rgb, col.a * alpha);
    writeTargets(outColor);
}
