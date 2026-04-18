/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: MSDF.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once
#include <msdfgen.h>

// =============================================================================
// jvk::msdf — standalone MSDF generator for arbitrary juce::Path input.
//
// Shared by GlyphAtlas (text, keyed on typeface+glyphId) and ShapeAtlas
// (paths, keyed on path blob + fill rule). The generator is PURE — no
// caching, no GPU state, no allocations beyond the returned Bitmap. Callers
// own whatever atlas / cache layer wraps this.
//
// The full non-Skia pipeline encapsulated here:
//
//   1. jucePathToShape          — correct closeSubPath handling (JUCE's
//                                 Path::Iterator closePath is a marker only;
//                                 we emit the explicit lineTo back to each
//                                 subpath's start).
//   2. shape.normalize()        — msdfgen convergent-edge handling.
//   3. normaliseContourWinding  — shoelace signed-area + bbox-containment
//                                 depth; enforces outer-CCW / hole-CW. Fixes
//                                 single-contour non-convex glyphs (e.g. 't')
//                                 that msdfgen's scanline-based orientContours
//                                 can leave unreversed.
//   4. edgeColoringByDistance   — picks per-edge colours so the three MSDF
//                                 channels disagree only where their median
//                                 resolves correctly. Handles tight interior
//                                 features (counters of 'e', 'a') better than
//                                 edgeColoringSimple.
//   5. generateMSDF             — with built-in error correction DISABLED so
//                                 the scanline pass below owns sign correction
//                                 (running both fights each other around
//                                 overlapping same-winding contours).
//   6. distanceSignCorrection   — scanline pass under the real FILL_NONZERO /
//                                 FILL_ODD rule. Rewrites the sign of pixels
//                                 whose per-contour distance combiner output
//                                 disagreed with the true inside/outside test.
//                                 Fixes pinhole/stripe artefacts in glyphs
//                                 with overlapping same-winding contours.
//   7. msdfErrorCorrection      — fast median-coherence pass with
//                                 DO_NOT_CHECK_DISTANCE (step 6 already did
//                                 the exact solve). Cleans edge-coloring seams.
// =============================================================================

namespace jvk::msdf
{

enum class FillRule : uint8_t { NonZero, EvenOdd };

struct Config
{
    int      maxDimension         = 32;    // max axis of the MSDF tile, in texels
    int      padding              = 4;     // padding around shape, in texels
    double   pxRange              = 4.0;   // SDF distance range in atlas pixels
    FillRule fillRule             = FillRule::NonZero;
    double   cornerAngleThreshold = 3.0;   // edgeColoringByDistance param
};

struct Result
{
    msdfgen::Bitmap<float, 3> bitmap;        // RGB MSDF — inspect .width()/.height()
    double                    texelScale = 0; // shape-unit → texel scale factor used
    msdfgen::Vector2          translate  { 0, 0 }; // translate applied pre-scale
    juce::Rectangle<float>    shapeBounds;  // path.getBounds() that scale mapped in
    bool                      valid = false; // false iff path was degenerate / rejected
};

// -----------------------------------------------------------------------------
// Path → msdfgen::Shape. JUCE's Path::Iterator emits closePath as a marker
// only (no coords); we track each subpath's start point and emit an explicit
// lineTo back on close. msdfgen requires closed contours — its scanline
// inside/outside tests are undefined for open ones.
// -----------------------------------------------------------------------------
inline msdfgen::Shape jucePathToShape(const juce::Path& path)
{
    msdfgen::Shape shape;
    msdfgen::Contour* contour = nullptr;

    juce::Path::Iterator it(path);
    msdfgen::Point2 lastPoint(0, 0);
    msdfgen::Point2 subpathStart(0, 0);
    bool haveSubpath = false;

    auto closeIfOpen = [&]() {
        if (!contour || !haveSubpath) return;
        // Avoid a zero-length closing edge — msdfgen rejects those.
        if (lastPoint.x != subpathStart.x || lastPoint.y != subpathStart.y)
            contour->addEdge(msdfgen::EdgeHolder(lastPoint, subpathStart));
        haveSubpath = false;
    };

    while (it.next())
    {
        switch (it.elementType)
        {
            case juce::Path::Iterator::startNewSubPath:
                // A new subpath without explicit closePath implicitly
                // terminates the previous one.
                closeIfOpen();
                contour = &shape.addContour();
                lastPoint = msdfgen::Point2(it.x1, it.y1);
                subpathStart = lastPoint;
                haveSubpath = true;
                break;

            case juce::Path::Iterator::lineTo:
                if (contour)
                {
                    msdfgen::Point2 p(it.x1, it.y1);
                    contour->addEdge(msdfgen::EdgeHolder(lastPoint, p));
                    lastPoint = p;
                }
                break;

            case juce::Path::Iterator::quadraticTo:
                if (contour)
                {
                    msdfgen::Point2 c(it.x1, it.y1);
                    msdfgen::Point2 p(it.x2, it.y2);
                    contour->addEdge(msdfgen::EdgeHolder(lastPoint, c, p));
                    lastPoint = p;
                }
                break;

            case juce::Path::Iterator::cubicTo:
                if (contour)
                {
                    msdfgen::Point2 c1(it.x1, it.y1);
                    msdfgen::Point2 c2(it.x2, it.y2);
                    msdfgen::Point2 p(it.x3, it.y3);
                    contour->addEdge(msdfgen::EdgeHolder(lastPoint, c1, c2, p));
                    lastPoint = p;
                }
                break;

            case juce::Path::Iterator::closePath:
                closeIfOpen();
                break;
        }
    }
    closeIfOpen();
    return shape;
}

// -----------------------------------------------------------------------------
// Winding normalisation. Replaces msdfgen's Shape::orientContours() which uses
// a scanline sampled at ~0.618 of each contour's vertical extent — for non-
// convex single-contour glyphs (e.g. 't') that ray can hit the cross-bar and
// produce intersections whose dy signs cancel, leaving the contour unreversed
// and therefore inverted in the output.
//
// Shoelace signed area is robust for any closed polygon. Bbox-containment
// depth approximates nesting cheaply (exact for the common non-overlapping
// case: letter with holes, rings, nested islands). Forced orientation:
//   even depth (outer / island nested in hole)   → CCW (winding = +1)
//   odd  depth (hole  / hole-nested-in-island)   → CW  (winding = -1)
// This matches msdfgen's expected "outer filled, hole empty" fragment output
// for Y-flipped JUCE path data.
// -----------------------------------------------------------------------------
inline void normaliseContourWinding(msdfgen::Shape& shape)
{
    if (shape.contours.empty()) return;

    std::vector<msdfgen::Shape::Bounds> bounds;
    bounds.reserve(shape.contours.size());
    for (auto& c : shape.contours)
    {
        msdfgen::Shape::Bounds b { +INFINITY, +INFINITY, -INFINITY, -INFINITY };
        c.bound(b.l, b.b, b.r, b.t);
        bounds.push_back(b);
    }

    for (size_t i = 0; i < shape.contours.size(); ++i)
    {
        int depth = 0;
        for (size_t j = 0; j < shape.contours.size(); ++j)
        {
            if (i == j) continue;
            const auto& bi = bounds[i];
            const auto& bj = bounds[j];
            if (bj.l <= bi.l && bj.b <= bi.b && bj.r >= bi.r && bj.t >= bi.t)
                ++depth;
        }
        const int expected = (depth & 1) ? -1 : +1;
        if (shape.contours[i].winding() != expected)
            shape.contours[i].reverse();
    }
}

// -----------------------------------------------------------------------------
// Main entry point — path → MSDF bitmap. Returns .valid=false for empty /
// degenerate input, or for output dims the caller should reject.
// -----------------------------------------------------------------------------
inline Result rasteriseJucePath(const juce::Path& path, const Config& cfg)
{
    Result out;
    if (path.isEmpty()) return out;

    auto pathBounds = path.getBounds();
    if (pathBounds.isEmpty()) return out;

    msdfgen::Shape shape = jucePathToShape(path);
    if (shape.contours.empty()) return out;

    shape.normalize();
    normaliseContourWinding(shape);
    msdfgen::edgeColoringByDistance(shape, cfg.cornerAngleThreshold);

    // Pick per-axis output size: scale so the larger axis fits in
    // (maxDimension - 2*padding) texels, ceil the other axis to match
    // the path's aspect ratio.
    float maxExtent = std::max(pathBounds.getWidth(), pathBounds.getHeight());
    if (maxExtent <= 0) return out;

    int innerSize = cfg.maxDimension - cfg.padding * 2;
    if (innerSize <= 0) return out;

    double texelScale = static_cast<double>(innerSize) / static_cast<double>(maxExtent);

    int outW = static_cast<int>(std::ceil(pathBounds.getWidth()  * texelScale)) + cfg.padding * 2;
    int outH = static_cast<int>(std::ceil(pathBounds.getHeight() * texelScale)) + cfg.padding * 2;
    outW = std::max(outW, 4);
    outH = std::max(outH, 4);

    msdfgen::Bitmap<float, 3> msdf(outW, outH);

    // Translate so the path's top-left lands at (padding, padding) in tile space.
    msdfgen::Vector2 translate(
        -static_cast<double>(pathBounds.getX()) + cfg.padding / texelScale,
        -static_cast<double>(pathBounds.getY()) + cfg.padding / texelScale);

    msdfgen::Projection projection(msdfgen::Vector2(texelScale), translate);
    double rangeInShapeUnits = cfg.pxRange / texelScale;

    // Step 5–7: generate, scanline-correct, then fast error-correction pass.
    msdfgen::MSDFGeneratorConfig genCfg;
    genCfg.errorCorrection.mode = msdfgen::ErrorCorrectionConfig::DISABLED;
    msdfgen::generateMSDF(msdf, shape, projection,
                          msdfgen::Range(rangeInShapeUnits), genCfg);

    msdfgen::FillRule msdfRule = (cfg.fillRule == FillRule::NonZero)
        ? msdfgen::FILL_NONZERO
        : msdfgen::FILL_ODD;
    msdfgen::distanceSignCorrection(msdf, shape, projection, msdfRule);

    msdfgen::MSDFGeneratorConfig ecCfg;
    ecCfg.errorCorrection.distanceCheckMode =
        msdfgen::ErrorCorrectionConfig::DO_NOT_CHECK_DISTANCE;
    msdfgen::msdfErrorCorrection(msdf, shape, projection,
                                 msdfgen::Range(rangeInShapeUnits), ecCfg);

    out.bitmap      = std::move(msdf);
    out.texelScale  = texelScale;
    out.translate   = translate;
    out.shapeBounds = pathBounds;
    out.valid       = true;
    return out;
}

} // namespace jvk::msdf
