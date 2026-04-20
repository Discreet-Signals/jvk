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
#include <core/ShapeDistanceFinder.h>

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
//   3. normaliseContourWinding  — canonical msdfgen / msdf-atlas-gen approach:
//                                 trust the source's relative per-contour
//                                 windings (outer vs hole, encoded in the font
//                                 itself and preserved through JUCE's internal
//                                 scale(1, -1) Y-flip), compute true distance
//                                 at a far-outside probe point, and globally
//                                 reverse every contour iff the overall sense
//                                 is inverted. No per-contour heuristic —
//                                 avoids both the 't' case that msdfgen's
//                                 Shape::orientContours mishandles AND the
//                                 bbox-nesting false-positive that filled 'g'
//                                 counters before.
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
//                                 Note this only flips ALL three channels
//                                 together (based on median), so it cannot
//                                 fix a single rogue channel — that's step 7.
//   7. msdfErrorCorrection      — error-correction with CHECK_DISTANCE_AT_EDGE
//                                 so per-channel pinholes whose median already
//                                 agreed with the scanline (step 6 a no-op for
//                                 them) but whose true shape distance disagrees
//                                 are still caught and equalized to median.
//                                 This is msdfgen's main.cpp default; msdf-
//                                 atlas-gen downgrades to DO_NOT_CHECK_DISTANCE
//                                 for batch-render speed, which we don't need
//                                 for an on-demand glyph atlas.
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
// Winding normalisation — canonical msdfgen / msdf-atlas-gen approach: trust
// the source's RELATIVE per-contour windings (outer vs hole, encoded in the
// font and preserved through JUCE's scale(1, -1) Y-flip) and only fix the
// GLOBAL sense. Pick a point guaranteed to be outside the shape's bounds,
// query true signed distance; a positive result means msdfgen sees that far-
// outside point as inside, i.e. the overall orientation is inverted — reverse
// every contour.
// -----------------------------------------------------------------------------
inline void normaliseContourWinding(msdfgen::Shape& shape)
{
    if (shape.contours.empty()) return;

    auto bounds = shape.getBounds();
    msdfgen::Point2 outerPoint(
        bounds.l - (bounds.r - bounds.l) - 1.0,
        bounds.b - (bounds.t - bounds.b) - 1.0);

    const double d = msdfgen::SimpleTrueShapeDistanceFinder::oneShotDistance(shape, outerPoint);
    if (d > 0)
        for (auto& c : shape.contours)
            c.reverse();
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

    // CHECK_DISTANCE_AT_EDGE: distanceSignCorrection above flips all three
    // channels together (based on median) and so can't fix a pixel whose
    // median already agrees with the scanline fill but whose individual
    // channels straddle the boundary — a median-coherent pinhole. Running
    // error correction with exact shape-distance checking at edges catches
    // those. Matches msdfgen main.cpp's default; msdf-atlas-gen downgrades
    // to DO_NOT_CHECK_DISTANCE purely for batch-render speed.
    msdfgen::MSDFGeneratorConfig ecCfg;
    ecCfg.errorCorrection.distanceCheckMode =
        msdfgen::ErrorCorrectionConfig::CHECK_DISTANCE_AT_EDGE;
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
