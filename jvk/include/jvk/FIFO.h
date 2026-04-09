/*
  ==============================================================================

    FIFO.h
    Created: 8 Aug 2023 1:37:48pm
    Author:  Gavin

  ==============================================================================
*/

#pragma once

namespace jvk
{

template<typename T>
class FIFO
{
public:
    FIFO(int size, T init_value) : fifo(size), buffer(size, init_value) { }

    void resize(int size, T init_value) { fifo.setTotalSize(size); buffer.resize(size, init_value); }

    void push(T item)
    {
        juce::AbstractFifo::ScopedWrite write = fifo.write(1);
        write.forEach([&, item](int index){ buffer[index] = item; });
    }
    T pop()
    {
        juce::AbstractFifo::ScopedRead read = fifo.read(1);
        if (read.blockSize1 > 0)
            return buffer[read.startIndex1];
        if (read.blockSize2 > 0)
            return buffer[read.startIndex2];
        jassertfalse;
        return T();
    }

    void push_num(const T* items, int num)
    {
        juce::AbstractFifo::ScopedWrite write = fifo.write(num);
        for (int i = 0; i < write.blockSize1; i++)
            buffer[write.startIndex1 + i] = items[i];
        for (int i = 0; i < write.blockSize2; i++)
            buffer[write.startIndex2 + i] = items[write.blockSize1 + i];
    }

    void pop_num(T* items, int num)
    {
        juce::AbstractFifo::ScopedRead read = fifo.read(num);
        for (int i = 0; i < read.blockSize1; i++)
            items[i] = buffer[read.startIndex1 + i];
        for (int i = 0; i < read.blockSize2; i++)
            items[read.blockSize1 + i] = buffer[read.startIndex2 + i];
    }

    void pop_num(int num)
    {
        juce::AbstractFifo::ScopedRead read = fifo.read(num);
    }

    void write(const T* items, int num)
    {
        int start1, size1, start2, size2;
        fifo.prepareToWrite(num, start1, size1, start2, size2);
        for (int i = 0; i < size1; i++)
            buffer[start1 + i] = items[i];
        for (int i = 0; i < size2; i++)
            buffer[start2 + i] = items[size1 + i];
    }

    void read(T* items, int num) const
    {
        int start1, size1, start2, size2;
        fifo.prepareToRead(num, start1, size1, start2, size2);
        for (int i = 0; i < size1; i++)
            items[i] = buffer[start1 + i];
        for (int i = 0; i < size2; i++)
            items[size1 + i] = buffer[start2 + i];
    }

    T read(int index) const
    {
        int start1, size1, start2, size2;
        fifo.prepareToRead(1, start1, size1, start2, size2);
        if (index < size1)
            return buffer[start1 + index];
        else
            return buffer[start2 + (index - start1)];
    }

    T operator[] (int index) const { return read(index); }

    T* data() { return buffer.data(); }

    template<typename FN>
    void for_each(int start, int num, FN&& fn)
    {
        FN apply = std::forward<FN>(fn);
        int start1, size1, start2, size2;
        fifo.prepareToRead(num, start1, size1, start2, size2);
        for (int i = start; (i < size1 && i < num); i++)
            apply(buffer[start1 + i]);
        for (int i = 0; (i < size2 && i + start1 < num); i++)
            apply(buffer[start2 + i]);
    }

    int size() { return fifo.getNumReady(); }
    int capacity() { return fifo.getTotalSize(); }
    int space() { return fifo.getFreeSpace(); }
private:
    juce::AbstractFifo fifo;
    std::vector<T> buffer;
};

}
