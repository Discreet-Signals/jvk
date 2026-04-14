/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.

 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: DeletionQueue.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::core
{

// Per-frame-slot deferred destruction queue.
// GPU resources that are replaced mid-frame (e.g. vertex buffer growth)
// are moved here instead of being destroyed immediately. The queue is
// flushed after the frame's fence signals, guaranteeing the GPU is done.
//
// In steady state the vectors hold capacity but are empty — flush() is a no-op.
struct DeletionQueue
{
    std::vector<Buffer> buffers;
    std::vector<Image>  images;

    void retire(Buffer&& b) { buffers.push_back(std::move(b)); }
    void retire(Image&& i)  { images.push_back(std::move(i)); }

    // Destructors of Buffer/Image call vkDestroy*. Safe because the
    // fence for this frame slot has been waited on before flush().
    void flush()
    {
        buffers.clear();
        images.clear();
    }
};

} // jvk::core
