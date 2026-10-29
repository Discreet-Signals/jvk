/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: SlotTable.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::backends
{

// ---- Handle tables ----------------------------------------------------------
//
// Handle value = generation << 24 | (slot + 1). 0 is null; a stale handle
// (its slot freed, maybe reused) fails the generation check and resolves to
// nothing rather than aliasing the new occupant.
template <typename T>
class SlotTable
{
public:
    static constexpr uint32_t kChunkBits = 10;
    static constexpr uint32_t kChunkSize = 1u << kChunkBits;
    static constexpr uint32_t kMaxChunks = (1u << 24) / kChunkSize;

    struct Entry
    {
        T                     value {};
        std::atomic<uint32_t> tag { 0 };   // generation << 1 | live
    };

    SlotTable() : chunks_(new std::atomic<Entry*>[kMaxChunks])
    {
        for (uint32_t i = 0; i < kMaxChunks; ++i) chunks_[i].store(nullptr, std::memory_order_relaxed);
    }

    // Under the owner's mutex. Returns the handle; `out` is the slot to fill
    // BEFORE the handle is handed to anyone.
    uint32_t allocate(T*& out)
    {
        uint32_t index;
        if (!free_.empty()) { index = free_.back(); free_.pop_back(); }
        else
        {
            index = next_++;
            if (index >= kMaxChunks * kChunkSize - 1) { --next_; out = nullptr; return 0; }
            const uint32_t c = index >> kChunkBits;
            if (chunks_[c].load(std::memory_order_relaxed) == nullptr)
            {
                owned_.push_back(std::make_unique<Entry[]>(kChunkSize));
                chunks_[c].store(owned_.back().get(), std::memory_order_release);
            }
        }
        Entry& e = entry(index);
        uint32_t generation = (e.tag.load(std::memory_order_relaxed) >> 1) & 0xFFu;
        if (generation == 0) generation = 1;
        e.value = T {};
        e.tag.store((generation << 1) | 1u, std::memory_order_release);
        out = &e.value;
        return (generation << 24) | (index + 1);
    }

    // Lock-free. Null for a null, stale or freed handle.
    T* resolve(uint32_t handle) const
    {
        if (handle == 0) return nullptr;
        const uint32_t index = (handle & 0xFFFFFFu) - 1;
        Entry* chunk = chunks_[index >> kChunkBits].load(std::memory_order_acquire);
        if (chunk == nullptr) return nullptr;
        Entry& e = chunk[index & (kChunkSize - 1)];
        const uint32_t tag = e.tag.load(std::memory_order_acquire);
        if ((tag & 1u) == 0 || ((tag >> 1) & 0xFFu) != (handle >> 24)) return nullptr;
        return &e.value;
    }

    // Under the owner's mutex: the slot is free, its generation moves on.
    void release(uint32_t handle)
    {
        if (resolve(handle) == nullptr) return;
        const uint32_t index = (handle & 0xFFFFFFu) - 1;
        Entry& e = entry(index);
        uint32_t generation = ((handle >> 24) + 1) & 0xFFu;
        if (generation == 0) generation = 1;
        e.tag.store(generation << 1, std::memory_order_release);
        e.value = T {};
        free_.push_back(index);
    }

    // Under the owner's mutex (teardown): every live handle.
    template <typename F>
    void forEachLive(F&& f)
    {
        for (uint32_t index = 0; index < next_; ++index)
        {
            Entry& e = entry(index);
            const uint32_t tag = e.tag.load(std::memory_order_relaxed);
            if (tag & 1u) f((((tag >> 1) & 0xFFu) << 24) | (index + 1), e.value);
        }
    }

private:
    Entry& entry(uint32_t index) const
    {
        return chunks_[index >> kChunkBits].load(std::memory_order_acquire)[index & (kChunkSize - 1)];
    }

    std::unique_ptr<std::atomic<Entry*>[]> chunks_;
    std::vector<std::unique_ptr<Entry[]>>  owned_;
    std::vector<uint32_t>                  free_;
    uint32_t                               next_ = 0;
};

} // namespace jvk::backends
