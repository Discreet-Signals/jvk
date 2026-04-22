#pragma once

namespace jvk {

// =============================================================================
// CommandScheduler — post-process pass over the deferred draw-command buffer.
//
// Runs on the render worker thread between Renderer::finalizeScopes() and the
// pipeline dispatch loop. Consumes the metadata Step 1 put on every command
// (writesPx, readsPx, stateKey, scopeId, isOpaque, recordOrder) plus the Scope
// tree (bounds, subtreeBounds) to build a dataflow DAG, reorder for locality,
// cull dead commands, and fuse batched dispatches.
//
// Mode is live-switchable from the UI — set by the message thread, read once
// by the worker at the top of run(). All four modes are no-op stubs in Step 2;
// Step 3 lands SortOnly, Step 4 lands SortCull, Step 5 lands Fuse.
// =============================================================================

class CommandScheduler {
public:
    enum class Mode : uint32_t {
        Identity = 0,  // passthrough — baseline for bisection / A-B testing
        SortOnly = 1,  // DAG + list-schedule, no fusion, no cull (Step 3)
        SortCull = 2,  // SortOnly + opaque-overdraw DCE         (Step 4)
        Fuse     = 3,  // SortCull + pipeline-opt-in batch fusion (Step 5)
    };

    static constexpr Mode kFirstMode = Mode::Identity;
    static constexpr Mode kLastMode  = Mode::Fuse;

    void setMode(Mode m) noexcept { mode_.store(m, std::memory_order_relaxed); }
    Mode getMode()    const noexcept { return mode_.load(std::memory_order_relaxed); }

    // Cycle to the next mode; wraps. Useful for a debug button.
    void cycleMode() noexcept
    {
        auto m = getMode();
        auto next = (m == kLastMode) ? kFirstMode
                                     : static_cast<Mode>(static_cast<uint32_t>(m) + 1);
        setMode(next);
    }

    static const char* modeName(Mode m) noexcept
    {
        switch (m) {
            case Mode::Identity: return "Identity";
            case Mode::SortOnly: return "SortOnly";
            case Mode::SortCull: return "SortCull";
            case Mode::Fuse:     return "Fuse";
        }
        return "?";
    }

    // One-shot debug capture: set from the message thread (e.g. a UI button),
    // consumed by the next run() on the worker. That frame logs its full
    // pre-sort and post-sort command streams via juce::Logger and then clears
    // the flag — no spam. Safe to call repeatedly; extra requests land on the
    // same next frame.
    void requestDump() noexcept
    {
        dumpNextFrame_.store(true, std::memory_order_release);
    }

    // Rewrites cmds in place. May shrink (cull) or reorder it; in the Fuse
    // mode it may also replace runs of commands with batched descendants
    // whose dataOffset points to arena-allocated primitive arrays.
    void run(std::vector<DrawCommand>& cmds,
             const std::vector<Scope>& scopes,
             Arena&                    arena);

private:
    // Schedule one contiguous barrier-free region [start, end) of cmds.
    // Builds the RAW/WAW DAG across the region, list-schedules it with
    // stateKey-locality priority, appends the reordered commands to `out`.
    // Does NOT modify cmds in place — the caller swaps at the end.
    // `scopes` is consulted by the RAW check (sameScopeLineage) so peer-
    // subtree commands don't falsely edge through each other's reads.
    void scheduleRegion(const std::vector<DrawCommand>& cmds,
                        size_t                           start,
                        size_t                           end,
                        const juce::Rectangle<int>&      viewport,
                        const std::vector<Scope>&        scopes,
                        std::vector<DrawCommand>&        out,
                        Mode                             mode,
                        Arena&                           arena);

    // Max primitives per fused dispatch. Caps the SSBO bandwidth of a
    // single vkCmdDraw(6, N, 0, first) — 1024 primitives × 128 B = 128 KB
    // of primitive data per fused run, fits well within the FillPipeline /
    // BlurPipeline / ClipPipeline per-slot buffer initial capacity and
    // stays under typical GPU descriptor-cache budgets.
    static constexpr uint32_t kMaxFuseBatchSize = 1024;

    // Pre-scheduling pass: walks cmds in record order, tracks scope
    // transitions via cmd.scopeId, and assigns each cmd.peerDepth based on
    // the peer-tree rules (sibling-disjointness resets depth, overlapping
    // siblings serialize, paintOverChildren advances past children).
    void computePeerDepths(std::vector<DrawCommand>&   cmds,
                           const std::vector<Scope>&   scopes);

    // Rect clips are pure scissor updates and dispatch becomes authoritative
    // per-command via cmd.clipBounds — the push/pop ops carry no semantics
    // the scheduler needs to honour, so they're ELIDED from the output.
    // Path clips DO participate: their writesPx = coverRect models the
    // stencil-buffer write, which the DAG handles as a normal RAW/WAW
    // hazard. Two Push/PopClipPaths with disjoint cover rects have no edge
    // between them and cluster freely — which is what makes the 10-letter
    // and TopBar-buttons batching cases work.
    static bool isElidedRectClip(DrawOp op) noexcept
    {
        return op == DrawOp::PushClipRect || op == DrawOp::PopClipRect;
    }

    // 64×64 px uniform spatial hash. Picked to balance tile count (~375 tiles
    // at 1600×900) against per-tile occupancy for typical UI command sizes.
    // Power of two so the index math stays as shifts.
    static constexpr int kTileShift = 6;           // 2^6 = 64
    static constexpr int kTileSize  = 1 << kTileShift;

    // Mode is read on the worker, written on the message thread. Atomic
    // ensures a well-defined value each frame; relaxed ordering is fine
    // because setMode happens between submits (no cross-thread data races
    // on other fields gated by this).
    std::atomic<Mode> mode_ { Mode::Identity };

    // One-shot debug-capture flag. Exchange-on-read inside run() makes each
    // click dump exactly one frame.
    std::atomic<bool> dumpNextFrame_ { false };

    // ---- Working buffers, reused across frames (cleared, not reallocated) --
    //
    // Indexed by position within the current region (0..N-1, NOT by global
    // cmd index). scheduleRegion() resizes them to N and fills them from
    // scratch every call.
    std::vector<uint32_t>                 remainingDeps_;
    std::vector<std::vector<uint32_t>>    dependents_;   // per-node adjacency
    std::vector<uint32_t>                 ready_;        // region-local indices
    std::vector<DrawCommand>              scratchInput_;  // rect-clips elided
    std::vector<DrawCommand>              scratchOutput_;

    // Fuse-mode scratch. scratchFuseReady_ collects indices into ready_
    // that share the winning command's stateKey for this emit cycle;
    // scratchFusePrims_ concatenates their primitive spans before we
    // re-allocate them contiguously in the arena.
    std::vector<size_t>                   scratchFuseReady_;
    std::vector<GeometryPrimitive>        scratchFusePrims_;

    // Working state for the peer-depth pre-pass. Sized to `scopes.size()`
    // per frame and cleared. `subtreeBounds` here is the scheduler's local
    // copy — Scope no longer carries it.
    struct ScopeState {
        uint32_t             directCursor      = 0;
        uint32_t             subtreeEnd        = 0;
        uint32_t             maxClosedChildEnd = 0;
        juce::Rectangle<int> subtreeBounds {};
        std::vector<uint32_t> closedChildren;
    };
    std::vector<ScopeState>               scopeState_;
    std::vector<uint32_t>                 activeStack_;  // scope-tree walk stack

    // ---- Tile-grid spatial index (per-region, reused across regions) -------
    //
    // tileWriters_[ty * tilesX_ + tx] = list of region-local cmd indices whose
    // writesPx touches that tile. Visited_ dedups candidates across a single
    // cmd's multi-tile query: visited_[w] == j iff w has already been tested
    // against j. UINT32_MAX is the "unvisited" sentinel.
    std::vector<std::vector<uint32_t>>    tileWriters_;
    std::vector<uint32_t>                 visited_;
    int                                   tilesX_ = 0;
    int                                   tilesY_ = 0;
};

} // namespace jvk
