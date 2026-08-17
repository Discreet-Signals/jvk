#version 450

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
// The two intervals meet at the corner (p.x, T), and a crossing landing on
// it must be claimed EXACTLY once. Both terms therefore decide their shared
// boundary from ONE quantity — `cornerSide`, the corner's signed area against
// the segment — with the through-the-corner case (cornerSide == 0) settled by
// the segment's own direction. Two independent divisions instead answered
// "no" twice and punched a 1px hole from the crossing to the tile's bottom
// edge, once per tile column any shallow edge spans (see the loop).
//
// The backdrop is the ray at the tile's top-RIGHT corner — every segment
// it counts crosses y = T strictly right of R, i.e. outside the tile, and
// is summed on the CPU. The top term's crossings happen at y = T with
// x ∈ (p.x, R] — inside the tile, so those segments are in the local
// list. The descent term's crossings happen at x = p.x ∈ tile — local
// again. Nothing non-local can contribute, so the sum is exact.
//
// AA is derivative-free: fragPos is in physical pixels and a distance
// field is 1-Lipschitz, so the edge band is exactly 1px wide — no
// fwidth(). (fwidth across tile/strip boundaries mixes distances from
// different segment lists and produced horizontal streak artifacts.)

layout(push_constant) uniform PC {
    vec2  viewportSize;
    uint  segmentStart;
    uint  segmentCount;   // unused in tile mode
    uint  fillRule;       // 0 = non-zero, 1 = even-odd
    uint  _r0;
    float tileW;     // tile width in physical px (16, coarsened for huge paths)
    float _r2;
} pc;

layout(set = 0, binding = 0) uniform sampler2D colorLUT;

layout(std430, set = 1, binding = 0) readonly buffer Segments {
    vec4 data[];
} segments;

layout(location = 0) in vec2 fragPos;
layout(location = 1) in vec4 fragColor;
layout(location = 2) in vec4 fragGradientInfo;
layout(location = 3) flat in vec4 fragTile;        // x=localStart y=count z=backdrop
layout(location = 4) flat in vec2 fragTileOrigin;  // tile top-left (px)

layout(location = 0) out vec4 outColor;

float sdSegment(vec2 p, vec2 a, vec2 b)
{
    vec2 pa = p - a;
    vec2 ba = b - a;
    float baLen2 = dot(ba, ba);
    float h = (baLen2 > 0.0) ? clamp(dot(pa, ba) / baLen2, 0.0, 1.0) : 0.0;
    return length(pa - ba * h);
}

vec4 sampleColor() {
    int mode = int(fragGradientInfo.z + 0.5);
    if (mode == 0)
        return fragColor;
    float t = (mode == 2) ? length(fragGradientInfo.xy) : fragGradientInfo.x;
    t = clamp(t, 0.0, 1.0);
    vec4 col = texture(colorLUT, vec2(t, fragGradientInfo.w));
    return vec4(col.rgb, col.a * fragColor.a);
}

void main()
{
    uint  base    = pc.segmentStart + uint(fragTile.x);
    uint  count   = uint(fragTile.y);
    int   winding = int(fragTile.z);      // backdrop at (tileRight, tileTop)
    float tileTop   = fragTileOrigin.y;
    float tileRight = fragTileOrigin.x + pc.tileW;

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
            vec4 seg = segments.data[base + i];
            vec2 a = seg.xy;
            vec2 b = seg.zw;

            minDist = min(minDist, sdSegment(fragPos, a, b));

            // The top and descent terms meet at the CORNER (p.x, tileTop).
            // Decide both sides of that junction from ONE quantity — the
            // corner's signed area against the segment — so a crossing that
            // lands exactly on it is claimed exactly once. Asking each term
            // its own division instead (xc > p.x  /  yc > tileTop) lets
            // rounding answer "no" twice: the crossing is dropped, winding
            // goes to 0, and the fill shows a 1px hole running from the
            // crossing down to the tile's bottom edge. It is not a corner
            // case — every shallow edge crosses a tile top once per tile
            // column it spans.
            //
            //   cornerSide = (b-a) × (corner-a)
            //              = (b.y-a.y)·(xc - p.x)  =  (b.x-a.x)·(tileTop - yc)
            //
            // so its sign answers BOTH questions consistently, and the zero
            // (segment through the corner) is settled once, below. `precise`
            // is load-bearing twice over: it keeps xc bit-identical to the
            // CPU tiler AND keeps that zero exact (an FMA would round it off
            // the corner and the tie-break would never fire).
            precise float cornerSide = (b.x - a.x) * (tileTop    - a.y)
                                     - (b.y - a.y) * (fragPos.x  - a.x);

            // Top term — crossing of y = tileTop at x in (p.x, tileRight].
            // `precise` (no FMA contraction) keeps xc BIT-IDENTICAL to the
            // CPU tiler's statement-split evaluation: tile boundaries are
            // exact integer floats (power-of-2 tileW), so matching xc makes
            // the B-vs-local split exact — no double count / drop when a
            // crossing lands on a tile edge.
            bool aAboveT = a.y > tileTop;
            bool bAboveT = b.y > tileTop;
            if (aAboveT != bAboveT)
            {
                precise float t  = (tileTop - a.y) / (b.y - a.y);
                precise float dx = t * (b.x - a.x);
                precise float xc = a.x + dx;
                // xc > p.x, strict (sign(b.y-a.y) = bAboveT ? + : -).
                if ((bAboveT ? cornerSide : -cornerSide) > 0.0 && xc <= tileRight)
                    winding += bAboveT ? 1 : -1;
            }

            // Descent term — crossing of x = p.x at y in (tileTop, p.y].
            // Sign from continuity of W along the downward walk: passing
            // below a left-to-right segment adds +1 (its crossing point
            // moves onto the ray), right-to-left subtracts.
            bool aRight = a.x > fragPos.x;
            bool bRight = b.x > fragPos.x;
            if (aRight != bRight)
            {
                float t  = (fragPos.x - a.x) / (b.x - a.x);
                float yc = a.y + t * (b.y - a.y);
                // yc > tileTop, from the same quantity as the top term. When
                // the segment passes exactly THROUGH the corner (cornerSide
                // == 0 — routine, not exotic: integer-ish art on a 16px grid
                // makes both products exact) neither half-open test claims
                // it, so decide it the way the fragment's own ray would: the
                // fragment is below the corner, so the crossing counts iff it
                // moves RIGHT as y grows, i.e. the segment descends rightward.
                float below = bRight ? -cornerSide : cornerSide;
                bool  onCorner = cornerSide == 0.0 && ((b.x > a.x) == (b.y > a.y));
                if ((below > 0.0 || onCorner) && yc <= fragPos.y)
                    winding += bRight ? 1 : -1;
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
}
