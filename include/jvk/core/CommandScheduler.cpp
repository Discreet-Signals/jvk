namespace jvk {

// =============================================================================
// Debug formatting helpers for requestDump() — stringifying DrawCommands to
// a single line. Only used when the one-shot dump flag fires, so the cost of
// the juce::String concats is irrelevant.
// =============================================================================

namespace {

// Per-op batching value — rough estimate of the dispatch cost each additional
// batched instance saves. Bigger = more valuable to cluster:
//
//   Effects / blurs (100)  — every one ends+begins a scene render pass and
//                            ping-pongs the full buffer. Clustering N of them
//                            collapses N-1 pass transitions plus N-1 full-
//                            viewport copies for clipped cases. Highest win.
//   DrawShader      (80)   — user shaders dispatch against the scene buffer,
//                            similar overhead though not as severe as blurs.
//   FillPath / path clips (60) — share the per-frame segment SSBO; batching
//                            dodges descriptor-set and SSBO-offset rebinds.
//   DrawGlyphs      (40)   — font atlas + fill descriptor rebinds.
//   DrawImage       (30)   — per-image texture descriptor bind.
//   Fills / strokes (10)   — cheapest — pipeline + colour descriptor only.
//
// The picker uses this only when ties happen (same DAG level, same match
// class against lastStateKey). It biases the *order of cluster starts*,
// never the correctness of the schedule, and never overrides a same-stateKey
// match — those are always zero-cost transitions so they win unconditionally.
// True for ops whose dispatch refs carry the common DispatchRefPrefix
// (firstInstance, primCount, arenaOffset) AND whose dispatch is a style-B
// instanced quad — FillPipeline / BlurPipeline / ClipPipeline ops. Fuse
// mode concatenates primitive spans across pairwise-independent ready-set
// members with matching stateKey for these.
//
// DrawGlyphs is intentionally EXCLUDED even though it matches the prefix
// shape. Each DrawGlyphs arena payload is the ref struct followed
// immediately by a uint32_t[] of per-glyph atlas-page indices; merging
// two glyph runs would also need to concat the atlas-page arrays, and
// the dispatcher reads them from `cmd.dataOffset + sizeof(ref)`. Solving
// that (e.g. moving pages to their own arena offset on the ref) is a
// future change; for now glyph dispatches stay singleton.
//
// Legacy effect ops (EffectBlend / EffectKernel / EffectHSV) and
// DrawShader use their own non-prefix arena structs and dispatch paths,
// so they stay single-dispatch too.
bool isFusableOp(DrawOp op) noexcept
{
    switch (op) {
        case DrawOp::FillRect:
        case DrawOp::FillRectList:
        case DrawOp::FillRoundedRect:
        case DrawOp::FillEllipse:
        case DrawOp::StrokeRoundedRect:
        case DrawOp::StrokeEllipse:
        case DrawOp::DrawImage:
        case DrawOp::DrawLine:
        case DrawOp::FillPath:
        case DrawOp::BlurShape:
        case DrawOp::BlurPath:
            return true;
        // PushClipPath / PopClipPath are stateful stencil-buffer
        // operations, NOT style-B instanced shape dispatches. Fusing N
        // pushes into one vkCmdDraw(6, N, 0, first) writes the stencil
        // counter N times in a single call, but their matching Pops
        // remain as N separate ops that unwind one at a time — so by
        // the first Pop the stencil is off by N-1, every sd>=1 op
        // inside fails its EQUAL test, and the nested clip-scope's
        // content goes blank. See the fuse-dump regression where three
        // PushClipPath inputs collapsed to one while three PopClipPaths
        // remained unmatched.
        case DrawOp::PushClipPath:
        case DrawOp::PopClipPath:
        default:
            return false;
    }
}

int opBatchPriority(DrawOp op) noexcept
{
    switch (op) {
        case DrawOp::BlurShape:
        case DrawOp::BlurPath:
        case DrawOp::EffectKernel:
        case DrawOp::EffectBlend:
        case DrawOp::EffectResolve:
        case DrawOp::EffectHSV:           return 100;
        case DrawOp::DrawShader:          return  80;
        case DrawOp::FillPath:
        case DrawOp::PushClipPath:
        case DrawOp::PopClipPath:         return  60;
        case DrawOp::DrawGlyphs:          return  40;
        case DrawOp::DrawImage:           return  30;
        case DrawOp::FillRect:
        case DrawOp::FillRectList:
        case DrawOp::FillRoundedRect:
        case DrawOp::FillEllipse:
        case DrawOp::StrokeRoundedRect:
        case DrawOp::StrokeEllipse:
        case DrawOp::DrawLine:            return  10;
        case DrawOp::PushClipRect:
        case DrawOp::PopClipRect:
        case DrawOp::COUNT:               return   0;
    }
    return 0;
}

const char* opNameFor(DrawOp op) noexcept
{
    switch (op) {
        case DrawOp::FillRect:          return "FillRect";
        case DrawOp::FillRectList:      return "FillRectList";
        case DrawOp::FillRoundedRect:   return "FillRoundedRect";
        case DrawOp::FillEllipse:       return "FillEllipse";
        case DrawOp::StrokeRoundedRect: return "StrokeRoundedRect";
        case DrawOp::StrokeEllipse:     return "StrokeEllipse";
        case DrawOp::DrawImage:         return "DrawImage";
        case DrawOp::DrawGlyphs:        return "DrawGlyphs";
        case DrawOp::DrawLine:          return "DrawLine";
        case DrawOp::FillPath:          return "FillPath";
        case DrawOp::DrawShader:        return "DrawShader";
        case DrawOp::EffectBlend:       return "EffectBlend";
        case DrawOp::EffectResolve:     return "EffectResolve";
        case DrawOp::EffectKernel:      return "EffectKernel";
        case DrawOp::EffectHSV:         return "EffectHSV";
        case DrawOp::BlurShape:         return "BlurShape";
        case DrawOp::BlurPath:          return "BlurPath";
        case DrawOp::PushClipRect:      return "PushClipRect";
        case DrawOp::PopClipRect:       return "PopClipRect";
        case DrawOp::PushClipPath:      return "PushClipPath";
        case DrawOp::PopClipPath:       return "PopClipPath";
        case DrawOp::COUNT:             return "COUNT";
    }
    return "unknown";
}

juce::String fmtRect(const juce::Rectangle<int>& r)
{
    if (r.isEmpty()) return "empty";
    return "(" + juce::String(r.getX()) + "," + juce::String(r.getY())
        + " " + juce::String(r.getWidth()) + "x" + juce::String(r.getHeight()) + ")";
}

juce::String fmtCmd(size_t idx, const DrawCommand& c)
{
    juce::String s;
    s << "  [" << juce::String(static_cast<int>(idx)).paddedLeft(' ', 3) << "] "
      << juce::String(opNameFor(c.op)).paddedRight(' ', 18)
      << "  sk=0x" << juce::String::toHexString((juce::int64) c.stateKey)
                          .paddedLeft('0', 16)
      << "  w=" << fmtRect(c.writesPx).paddedRight(' ', 22)
      << "  r=" << fmtRect(c.readsPx) .paddedRight(' ', 22)
      << "  sc=" << juce::String(static_cast<int>(c.scopeId)).paddedLeft(' ', 2)
      << "  pd=" << juce::String(static_cast<int>(c.peerDepth)).paddedLeft(' ', 3)
      << "  sd=" << juce::String(static_cast<int>(c.stencilDepth))
      << "  ord=" << juce::String(static_cast<int>(c.recordOrder))
                          .paddedLeft(' ', 4)
      << (c.isOpaque ? "  opaque" : "");
    return s;
}

// ---- Scope-tree helpers, used by both DAG edge check and peer-depth pass --

bool isAncestorOf(uint32_t maybeAncestor,
                  uint32_t descendant,
                  const std::vector<Scope>& scopes)
{
    uint32_t x = descendant;
    while (true) {
        if (x == maybeAncestor) return true;
        if (x == 0) return false;
        x = scopes[x].parentId;
    }
}

// Two scope IDs are on the same lineage iff one is an ancestor of the other
// (or they're equal). Commands across sibling subtrees are NOT same-lineage —
// their hazards should be WAW-only, never RAW.
bool sameScopeLineage(uint32_t a,
                      uint32_t b,
                      const std::vector<Scope>& scopes)
{
    return isAncestorOf(a, b, scopes) || isAncestorOf(b, a, scopes);
}

uint32_t childOnPath(uint32_t ancestor,
                     uint32_t descendant,
                     const std::vector<Scope>& scopes)
{
    // The child of `ancestor` that's on the path to `descendant`.
    uint32_t x = descendant;
    while (scopes[x].parentId != ancestor) x = scopes[x].parentId;
    return x;
}

} // unnamed namespace

// =============================================================================
// CommandScheduler::run — frame entry point.
//
// Identity mode: zero work. Output == input, byte for byte.
//
// SortOnly / SortCull / Fuse: no barrier partitioning — everything goes
// through one big DAG. The ordering invariant is the universal RAW/WAW
// rule: any two commands whose writesPx (colour or stencil) overlap get
// a DAG edge and stay in painter's order. Path clips fit that rule via
// their coverRect-sized writesPx, so PushClipPath / PopClipPath participate
// in the DAG like any other command — disjoint sibling clips get no edge
// between them and cluster freely with their matching-stateKey neighbours.
//
// Rect clips are pure scissor updates. Dispatch now reads cmd.clipBounds
// directly per-command, so PushClipRect / PopClipRect carry no semantics
// the scheduler needs and are elided from the output entirely.
//
// SortCull adds DCE in Step 4. Fuse adds batched dispatches in Step 5.
// =============================================================================

void CommandScheduler::run(std::vector<DrawCommand>& cmds,
                           const std::vector<Scope>& scopes,
                           Arena&                    arena)
{
    const Mode m = mode_.load(std::memory_order_acquire);
    // Take the dump flag once up-front so Identity-mode passthrough can also
    // dump (you can compare mode output to raw input that way).
    const bool dump = dumpNextFrame_.exchange(false, std::memory_order_acq_rel);

    if (m == Mode::Identity) {
        if (dump) {
            juce::String s;
            s << "=== CommandScheduler dump (mode=Identity, passthrough) ===\n"
              << "Input: " << static_cast<int>(cmds.size()) << " commands\n";
            for (size_t i = 0; i < cmds.size(); ++i)
                s << fmtCmd(i, cmds[i]) << "\n";
            juce::Logger::writeToLog(s);
        }
        return;
    }
    if (cmds.empty()) return;

    // Viewport bounds for the tile grid — Scope 0 is the root (repopulated
    // by Renderer::reset each frame). Falls back to a large square if the
    // scope tree isn't set up (shouldn't happen, but keeps the grid math
    // well-defined).
    juce::Rectangle<int> viewport = !scopes.empty() ? scopes[0].bounds
                                                    : juce::Rectangle<int>{};
    if (viewport.isEmpty())
        viewport = { 0, 0, 8192, 8192 };

    // Filter: elide rect clip ops. Their sole purpose was to push/pop the
    // scissor state, and dispatch now takes scissor from cmd.clipBounds per
    // draw — so leaving these in the stream would just waste a dispatch slot
    // and spuriously show up in the DAG. Path clips are kept: they write the
    // stencil buffer and genuinely participate in scheduling hazards.
    scratchInput_.clear();
    scratchInput_.reserve(cmds.size());
    for (const auto& c : cmds) {
        if (isElidedRectClip(c.op)) continue;
        scratchInput_.push_back(c);
    }

    scratchOutput_.clear();
    scratchOutput_.reserve(scratchInput_.size());

    // Peer-tree depth pass. Assigns cmd.peerDepth from scopeId + writesPx.
    // Drives both DAG edge-filtering (same-depth commands skip RAW checks)
    // and the scheduler's primary ordering.
    computePeerDepths(scratchInput_, scopes);

    // One region for the whole stream. Path clips participate in the DAG as
    // normal commands (writesPx = coverRect), so disjoint siblings cluster
    // naturally with same-stateKey interiors. Overlapping ones get the
    // standard WAW edge and stay ordered. Mode + arena plumb through so
    // Fuse can concatenate mutually-independent same-stateKey primitive
    // runs into merged spans.
    scheduleRegion(scratchInput_, 0, scratchInput_.size(),
                   viewport, scopes, scratchOutput_, m, arena);

    if (dump) {
        const int elided = static_cast<int>(cmds.size() - scratchInput_.size());
        juce::String s;
        s << "=== CommandScheduler dump (mode=" << modeName(m) << ") ===\n"
          << "Input: " << static_cast<int>(cmds.size())
          << " commands (" << elided << " rect clips elided)\n"
          << "-- Pre-sort (" << static_cast<int>(scratchInput_.size()) << ") --\n";
        for (size_t i = 0; i < scratchInput_.size(); ++i)
            s << fmtCmd(i, scratchInput_[i]) << "\n";
        s << "-- Post-sort (" << static_cast<int>(scratchOutput_.size()) << ") --\n";
        for (size_t i = 0; i < scratchOutput_.size(); ++i)
            s << fmtCmd(i, scratchOutput_[i]) << "\n";
        juce::Logger::writeToLog(s);
    }

    cmds.swap(scratchOutput_);
}

// =============================================================================
// scheduleRegion — build the RAW/WAW dataflow DAG via a tile-grid spatial
// index, then list-schedule with stateKey-locality priority.
//
// Tile-grid build:
//   For each cmd C in record order within the region:
//     Query: collect candidate prior writers by walking the tiles C.readsPx
//            covers; dedup via visited_[] using the current cmd index as a
//            per-query generation marker. Each candidate is tested for real
//            bbox overlap; matches become edges.
//     Insert: append C's index to every tile C.writesPx covers.
//
// Why readsPx (and not readsPx ∪ writesPx) is enough for the query: Graphics
// always sets readsPx ⊇ writesPx (equal for non-effects, dilated for blurs),
// so tiles covered by writesPx are a subset of those covered by readsPx.
//
// Correctness of the RAW/WAW rule: painter's-order alpha blending is a
// read-modify-write of the destination pixel, so every overlapping write
// depends on every prior overlapping write — WAW alone wouldn't suffice,
// and RAW alone wouldn't either. Testing BOTH is needed. We short-circuit
// once either hazard fires.
//
// Complexity:
//   Insert: O(tilesTouched(C.writesPx))
//   Query:  O(tilesTouched(C.readsPx) · avgOccupancy)
//   DAG build total: O(N · tilesPerCmd · avgOccupancy). For typical UI
//   workloads this is effectively O(N); it degrades to O(N²) only when
//   every command writes the viewport, in which case the DAG genuinely has
//   O(N²) real edges and no algorithm can do better.
//   List-schedule: O(N · avgReady) — small ready sets in practice.
// =============================================================================

void CommandScheduler::scheduleRegion(const std::vector<DrawCommand>& cmds,
                                      size_t                           start,
                                      size_t                           end,
                                      const juce::Rectangle<int>&      viewport,
                                      const std::vector<Scope>&        scopes,
                                      std::vector<DrawCommand>&        out,
                                      Mode                             mode,
                                      Arena&                           arena)
{
    const size_t N = end - start;
    if (N == 0) return;
    if (N == 1) { out.push_back(cmds[start]); return; }

    const size_t emitStart = out.size();

    // ---- Per-node scratch ---------------------------------------------------
    remainingDeps_.assign(N, 0u);
    if (dependents_.size() < N) dependents_.resize(N);
    for (size_t i = 0; i < N; ++i) dependents_[i].clear();
    // All-bits-set sentinel for "unvisited" — any real cmd index j < N fits
    // in the 32-bit range, so this never collides.
    visited_.assign(N, ~uint32_t{0});

    // ---- Tile grid scratch --------------------------------------------------
    // Grid dimensions cover the full viewport so any command's bbox lands in
    // valid tiles after clamping. tileWriters_ is reused across regions —
    // clear the cells we'll touch rather than reallocating.
    const int vpX = viewport.getX();
    const int vpY = viewport.getY();
    const int vpW = juce::jmax(viewport.getWidth(),  kTileSize);
    const int vpH = juce::jmax(viewport.getHeight(), kTileSize);
    tilesX_ = (vpW + kTileSize - 1) >> kTileShift;
    tilesY_ = (vpH + kTileSize - 1) >> kTileShift;
    const int tileCount = tilesX_ * tilesY_;
    if (static_cast<int>(tileWriters_.size()) < tileCount)
        tileWriters_.resize(tileCount);
    for (int t = 0; t < tileCount; ++t) tileWriters_[t].clear();

    // Helper: convert a physical-px rect to clamped tile coordinates.
    auto tileRange = [&](const juce::Rectangle<int>& r,
                         int& tx0, int& ty0, int& tx1, int& ty1)
    {
        // Half-open bottom-right → subtract 1 so a zero-size rect stays in
        // one tile rather than spilling to the next.
        const int x0 = r.getX()      - vpX;
        const int y0 = r.getY()      - vpY;
        const int x1 = r.getRight()  - vpX - 1;
        const int y1 = r.getBottom() - vpY - 1;
        tx0 = juce::jlimit(0, tilesX_ - 1, x0 >> kTileShift);
        ty0 = juce::jlimit(0, tilesY_ - 1, y0 >> kTileShift);
        tx1 = juce::jlimit(0, tilesX_ - 1, x1 >> kTileShift);
        ty1 = juce::jlimit(0, tilesY_ - 1, y1 >> kTileShift);
    };

    // ---- DAG build (incremental, record-order sweep) ------------------------
    for (size_t j = 0; j < N; ++j) {
        const auto& C = cmds[start + j];
        const uint32_t jU = static_cast<uint32_t>(j);

        // Query tiles C.readsPx covers. readsPx ⊇ writesPx so a single range
        // catches both RAW and WAW candidates.
        if (!C.readsPx.isEmpty()) {
            int tx0, ty0, tx1, ty1;
            tileRange(C.readsPx, tx0, ty0, tx1, ty1);
            for (int ty = ty0; ty <= ty1; ++ty) {
                const int rowBase = ty * tilesX_;
                for (int tx = tx0; tx <= tx1; ++tx) {
                    auto& cell = tileWriters_[rowBase + tx];
                    for (uint32_t w : cell) {
                        if (visited_[w] == jU) continue;   // already tested this query
                        visited_[w] = jU;
                        const auto& P = cmds[start + w];

                        // Both RAW and WAW edges are gated by sameScopeLineage.
                        //
                        // The contract: sibling subtrees are asserted
                        // independent by their disjoint component bounds —
                        // e.g. the 6 pedals sit in their own clip scopes
                        // and are positioned side-by-side. Two ops in
                        // disjoint sibling subtrees don't strictly order
                        // against each other; if their physical pixel
                        // bounds nick each other at the edges (tight
                        // layouts, sub-pixel overhangs, blur halos), the
                        // overlap is accepted as visually undefined. The
                        // win: cross-pedal serialization chains evaporate.
                        //
                        // Pedal-5's title fillRR no longer transitively
                        // chains back to pedal-1's blur through a
                        // pedal-1-body WAW edge, so all 6 pedal title
                        // blurs become simultaneously ready and Fuse mode
                        // collapses them into one instanced dispatch —
                        // same ping-pong cycle cost as a single blur.
                        // That's what closes the "many little blurs"
                        // perf gap to "one big blur".
                        //
                        // The same rule keeps RAW cross-peer sampling
                        // tolerated (TopBar glass buttons reading the
                        // neighbour's pre-blur scene is fine).
                        const bool sameLineage =
                            sameScopeLineage(P.scopeId, C.scopeId, scopes);

                        // WAW is UNCONDITIONAL: if two ops' writesPx
                        // actually intersect, painter's order is load-
                        // bearing — the later one must follow. This
                        // covers both genuinely overlapping colour writes
                        // (e.g. a pedal being dragged over a sibling
                        // pedal) and stencil writes (PushClipPath /
                        // PopClipPath brackets, whose shared stencil
                        // plane makes overlap destructive). The sibling-
                        // independence shortcut would merge those into
                        // one fused dispatch and visibly corrupt the
                        // result — there's no interpretation under which
                        // reordering real overlap is OK.
                        //
                        // RAW stays gated by sameLineage: cross-peer
                        // reads (blur kernels sampling a sibling's
                        // pixels — TopBar glass buttons reading a
                        // neighbour's pre-blur scene, pedal-1 blur's
                        // halo extending into pedal-2's body) are
                        // tolerated as visually undefined so that the
                        // cross-peer blur-fusion wins still hold. This
                        // is the only gate that provides a perf benefit;
                        // WAW-gating was a side effect that this fixes.
                        const bool waw = P.writesPx.intersects(C.writesPx);
                        const bool raw = sameLineage
                                      && P.writesPx.intersects(C.readsPx);

                        if (waw || raw) {
                            dependents_[w].push_back(jU);
                            ++remainingDeps_[j];
                        }
                    }
                }
            }
        }

        // Insert C into every tile its writesPx covers, so future queries
        // against these tiles find C.
        if (!C.writesPx.isEmpty()) {
            int tx0, ty0, tx1, ty1;
            tileRange(C.writesPx, tx0, ty0, tx1, ty1);
            for (int ty = ty0; ty <= ty1; ++ty) {
                const int rowBase = ty * tilesX_;
                for (int tx = tx0; tx <= tx1; ++tx)
                    tileWriters_[rowBase + tx].push_back(jU);
            }
        }
    }

    // ---- Initial ready set --------------------------------------------------
    ready_.clear();
    ready_.reserve(N);
    for (size_t i = 0; i < N; ++i)
        if (remainingDeps_[i] == 0)
            ready_.push_back(static_cast<uint32_t>(i));

    // ---- List schedule ------------------------------------------------------
    // Pick policy (in priority order):
    //   1. STATEKEY MATCH with lastEmitted — clustering is the whole point;
    //      match trumps depth when the DAG allows it. This is what lets
    //      blur2 hop past a same-scope intermediate op (e.g. a rect whose
    //      bounds don't conflict with blur2's reads/writes) to cluster
    //      next to blur1.
    //   2. LOWEST PEER DEPTH — structural ordering from the component tree.
    //      Sibling-disjoint scopes share a depth so their peer ops wind up
    //      at matching layers; draining depth-first still helps within
    //      non-matching classes.
    //   3. LOWEST RECORD ORDER — stable painter-biased tiebreak.
    uint64_t lastStateKey = 0;
    bool     haveLast     = false;

    while (!ready_.empty()) {
        size_t   bestPos   = 0;
        uint32_t bestDepth = cmds[start + ready_[0]].peerDepth;
        uint32_t bestOrder = cmds[start + ready_[0]].recordOrder;
        bool     bestMatch = haveLast
                          && cmds[start + ready_[0]].stateKey == lastStateKey;
        for (size_t k = 1; k < ready_.size(); ++k) {
            const auto idx   = ready_[k];
            const auto& C    = cmds[start + idx];
            const uint32_t D = C.peerDepth;
            const bool match = haveLast && (C.stateKey == lastStateKey);

            // Match wins over non-match, regardless of depth.
            if (match && !bestMatch) {
                bestPos = k; bestDepth = D; bestOrder = C.recordOrder; bestMatch = true;
            } else if (match == bestMatch) {
                // Same match class — tiebreak on depth, then recordOrder.
                if (D < bestDepth) {
                    bestPos = k; bestDepth = D; bestOrder = C.recordOrder;
                } else if (D == bestDepth && C.recordOrder < bestOrder) {
                    bestPos = k; bestOrder = C.recordOrder;
                }
            }
            // else (!match && bestMatch): don't downgrade from match.
        }

        const uint32_t emitIdx   = ready_[bestPos];
        const auto&    emittedRO = cmds[start + emitIdx];
        DrawCommand    emitted   = emittedRO; // mutable copy for possible ref rewrite

        // ---- Fuse mode: coalesce ready-set same-stateKey siblings ------
        //
        // Every ready command has all its DAG predecessors emitted, so any
        // two commands simultaneously in ready_ are by construction
        // mutually independent — free to reorder / merge. Collect all ready
        // cmds with matching stateKey as the winner and concatenate their
        // primitive spans into a fresh contiguous arena region. The
        // winner's DispatchRef is rewritten to point at the merged span
        // with primCount = total; the followers are dropped from ready_
        // but still propagate their dependents (since the merged dispatch
        // logically "emits" all of them at once).
        //
        // Only runs for ops that carry DispatchRefPrefix and dispatch via
        // the style-B geometry-abstracted pipelines (see isFusableOp).
        // Legacy effect ops, rect clips, and DrawShader fall through the
        // single-emit path below.
        bool didFuse = false;
        if (mode == Mode::Fuse && isFusableOp(emittedRO.op)) {
            scratchFuseReady_.clear();
            scratchFuseReady_.push_back(bestPos);
            uint32_t runPrimCount = 0;
            {
                const auto& prefix = arena.read<DispatchRefPrefix>(emittedRO.dataOffset);
                runPrimCount = prefix.primCount;
            }
            for (size_t k = 0; k < ready_.size(); ++k) {
                if (k == bestPos) continue;
                if (scratchFuseReady_.size() >= /*cap*/ 64) break;
                const auto& C = cmds[start + ready_[k]];
                if (C.stateKey != emittedRO.stateKey) continue;
                if (!isFusableOp(C.op)) continue;
                // stencilDepth MUST match — it drives
                // vkCmdSetStencilReference, one value per dispatch.
                // Primitives at different clip depths genuinely cannot
                // share a dispatch. clipBounds DIFFER is fine though —
                // the dispatch scissor will be set to the union of all
                // fused members' clipBounds below, acting as a
                // permissive outer envelope. Each primitive's quad
                // (bbox = writesPx clipped to its own scope at record
                // time) already stays within its own region at
                // rasterization, and the DAG guarantees fused
                // primitives have disjoint writesPx so there's no
                // cross-primitive overwrite to worry about.
                if (C.stencilDepth != emittedRO.stencilDepth) continue;
                const auto& pref = arena.read<DispatchRefPrefix>(C.dataOffset);
                if (runPrimCount + pref.primCount > kMaxFuseBatchSize) continue;
                runPrimCount += pref.primCount;
                scratchFuseReady_.push_back(k);
            }

            if (scratchFuseReady_.size() > 1) {
                // Concatenate every fused member's primitive span into the
                // scratch vector, then push it into the arena as a single
                // contiguous region. Also union each member's clipBounds
                // / writesPx / readsPx into the emitted cmd so the
                // dispatcher's vkCmdSetScissor = union(clipBounds)
                // envelope covers every fused primitive's quad. Each
                // primitive's quad stays within its own writesPx at
                // rasterization (set from its own scope's clipBounds at
                // record time), so the union scissor is a permissive
                // outer filter, not a truncation.
                //
                // The scheduler doesn't own an op-specific ref type — it
                // only touches the DispatchRefPrefix bytes (firstInstance
                // stays 0 for the upload pass to fill; primCount +
                // arenaOffset are what fusion rewrites).
                scratchFusePrims_.clear();
                scratchFusePrims_.reserve(runPrimCount);
                juce::Rectangle<int> unionClip   = emittedRO.clipBounds;
                juce::Rectangle<int> unionWrites = emittedRO.writesPx;
                juce::Rectangle<int> unionReads  = emittedRO.readsPx;
                for (size_t readyPos : scratchFuseReady_) {
                    const auto& C      = cmds[start + ready_[readyPos]];
                    const auto& pref   = arena.read<DispatchRefPrefix>(C.dataOffset);
                    auto        cPrims = arena.readSpan<GeometryPrimitive>(pref.arenaOffset,
                                                                            pref.primCount);
                    for (const auto& p : cPrims)
                        scratchFusePrims_.push_back(p);
                    unionClip   = unionClip.getUnion(C.clipBounds);
                    unionWrites = unionWrites.getUnion(C.writesPx);
                    unionReads  = unionReads.getUnion(C.readsPx);
                }
                // Arena reallocation can move pointers — read no spans from
                // the arena after this push until all ref reads are fresh.
                const uint32_t mergedOffset = arena.pushSpan(
                    std::span<const GeometryPrimitive>(scratchFusePrims_.data(),
                                                       scratchFusePrims_.size()));

                // Rewrite the winner's ref prefix: merged primitive run
                // lives at (mergedOffset, scratchFusePrims_.size()).
                {
                    auto& pref = arena.readMut<DispatchRefPrefix>(emitted.dataOffset);
                    pref.primCount  = static_cast<uint32_t>(scratchFusePrims_.size());
                    pref.arenaOffset = mergedOffset;
                }

                // Stamp the fused bounds onto the emitted cmd. clipBounds
                // is the scissor the dispatcher will set; writesPx /
                // readsPx are the accurate post-merge footprint for any
                // downstream inspection (dump output, later scheduler
                // passes if we add them).
                emitted.clipBounds = unionClip;
                emitted.writesPx   = unionWrites;
                emitted.readsPx    = unionReads;
                didFuse = true;
            }
        }

        out.push_back(emitted);
        lastStateKey = emitted.stateKey;
        haveLast     = true;

        if (didFuse) {
            // Remove all fused members from ready_ and propagate their
            // dependents' dep-count decrements. Descending-index swap-pop
            // keeps the bestPos slot valid across multiple removals.
            std::sort(scratchFuseReady_.begin(), scratchFuseReady_.end(),
                      std::greater<size_t>());
            for (size_t readyPos : scratchFuseReady_) {
                const uint32_t cmdIdx = ready_[readyPos];
                for (auto d : dependents_[cmdIdx])
                    if (--remainingDeps_[d] == 0)
                        ready_.push_back(d);
                ready_[readyPos] = ready_.back();
                ready_.pop_back();
            }
        } else {
            // Singleton emit — existing fast path.
            ready_[bestPos] = ready_.back();
            ready_.pop_back();
            for (auto d : dependents_[emitIdx])
                if (--remainingDeps_[d] == 0)
                    ready_.push_back(d);
        }
    }

    // SortOnly / SortCull: exactly N commands in, N commands out — DAG is
    // acyclic by construction (edges always go lower → higher j in the
    // sweep), and neither mode culls at this layer.
    //
    // Fuse: drops N-1 followers per merged run, so the output count is <=
    // N. The invariant there is "every input cmd is either emitted as
    // itself OR absorbed by an earlier fused emit". We still require
    // ready_ to have drained to empty, which is what covers this — if a
    // cmd wasn't emitted or fused, it would still sit in ready_ (or its
    // dependencies would have left it stranded, same bug class).
    if (mode == Mode::Fuse) {
        jassert(out.size() - emitStart <= N);
    } else {
        jassert(out.size() - emitStart == N);
    }
}

// =============================================================================
// computePeerDepths — maps cmd.scopeId + cmd.writesPx onto a peer-tree depth
// that reflects the component hierarchy with sibling-disjointness baked in.
//
// Rules:
//   - A scope's direct cmds get consecutive peerDepths starting at the scope's
//     entry depth + 1.
//   - When entering a child scope: enterDepth = parent.directCursor + 1,
//     bumped past any prior sibling of the parent whose subtreeBounds
//     intersects the child's clip bounds.
//   - When leaving a scope: propagate subtreeBounds + subtreeEnd to parent,
//     log the scope in parent.closedChildren for future sibling checks.
//   - Direct cmds in a scope that already has closed children skip past
//     max(prior closed child's subtreeEnd) before incrementing — this is
//     what gives paintOverChildren ops the correct higher depth.
// =============================================================================

void CommandScheduler::computePeerDepths(std::vector<DrawCommand>& cmds,
                                         const std::vector<Scope>& scopes)
{
    if (scopes.empty() || cmds.empty()) return;

    // Size + reset per-scope working state. Reuses vector capacity
    // frame-to-frame; closedChildren vectors individually cleared.
    if (scopeState_.size() < scopes.size()) scopeState_.resize(scopes.size());
    for (size_t i = 0; i < scopes.size(); ++i) {
        auto& st = scopeState_[i];
        st.directCursor      = 0;
        st.subtreeEnd        = 0;
        st.maxClosedChildEnd = 0;
        st.subtreeBounds     = {};
        st.closedChildren.clear();
    }

    // Scope-tree walk helpers, inlined as lambdas closing over state.

    auto enterChild = [&](uint32_t childId) {
        const auto parentId = scopes[childId].parentId;
        auto& parent = scopeState_[parentId];
        auto& child  = scopeState_[childId];

        // Disjoint-from-all-siblings default.
        uint32_t enterDepth = parent.directCursor + 1;

        // Bump past any prior sibling whose subtree overlaps this child's
        // clip bounds. (Overlapping siblings are layered → serialize.)
        for (uint32_t priorId : parent.closedChildren) {
            const auto& prior = scopeState_[priorId];
            if (prior.subtreeBounds.intersects(scopes[childId].bounds))
                enterDepth = juce::jmax(enterDepth, prior.subtreeEnd + 1);
        }

        // Entry itself consumes a depth slot; first direct cmd bumps from here.
        child.directCursor = enterDepth;
        child.subtreeEnd   = enterDepth;

        activeStack_.push_back(childId);
    };

    auto closeScope = [&]() {
        const uint32_t closingId = activeStack_.back();
        if (closingId == 0) return;  // never close root
        const auto parentId = scopes[closingId].parentId;
        auto& closing = scopeState_[closingId];
        auto& parent  = scopeState_[parentId];

        parent.subtreeBounds     = parent.subtreeBounds.getUnion(closing.subtreeBounds);
        parent.subtreeEnd        = juce::jmax(parent.subtreeEnd, closing.subtreeEnd);
        parent.maxClosedChildEnd = juce::jmax(parent.maxClosedChildEnd, closing.subtreeEnd);
        parent.closedChildren.push_back(closingId);

        activeStack_.pop_back();
    };

    auto syncTo = [&](uint32_t target) {
        // Ascend + descend until active top == target.
        while (activeStack_.back() != target) {
            const uint32_t top = activeStack_.back();
            if (isAncestorOf(top, target, scopes)) {
                enterChild(childOnPath(top, target, scopes));
            } else {
                closeScope();
            }
        }
    };

    // Walk cmds in record order, assigning peerDepth per cmd.
    activeStack_.clear();
    activeStack_.push_back(0);

    for (auto& cmd : cmds) {
        syncTo(cmd.scopeId);

        auto& S = scopeState_[cmd.scopeId];
        // Advance past any closed children — handles JUCE's paintOverChildren.
        const uint32_t base = juce::jmax(S.directCursor, S.maxClosedChildEnd);
        S.directCursor  = base + 1;
        cmd.peerDepth   = S.directCursor;
        S.subtreeEnd    = juce::jmax(S.subtreeEnd, S.directCursor);
        S.subtreeBounds = S.subtreeBounds.getUnion(cmd.writesPx);
    }
}

} // namespace jvk
