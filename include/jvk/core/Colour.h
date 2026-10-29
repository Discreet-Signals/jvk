/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Colour.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

// =============================================================================
// Colour — four 32-bit floats, PREMULTIPLIED (rgb already scaled by alpha), as
// every jvk draw blends: rgb = src + dst * (1 - a).
//
// It converts to and from juce::Colour implicitly, so any juce::Colour
// (juce::Colours::red, c.withAlpha(0.5f)) goes wherever jvk asks for a Colour,
// and a Colour goes wherever juce asks for a juce::Colour, as its nearest one.
// Set with jvk::Graphics::setColour, its floats reach the target whole, and
// the target's format takes what it can hold: an 8-bit one rounds and clamps,
// a float one keeps what a juce::Colour can't, values past 1, negative ones,
// and rgb at alpha 0, an amount ADDED under the draw that covers nothing (a
// height raised by 0.05: { 0, 0, 0.05f, 0 }).
// =============================================================================

struct Colour
{
    float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;

    constexpr Colour() noexcept = default;
    constexpr Colour(float red, float green, float blue, float alpha) noexcept : r(red), g(green), b(blue), a(alpha) {}

    Colour(juce::Colour c) noexcept
        : r(c.getFloatRed() * c.getFloatAlpha()), g(c.getFloatGreen() * c.getFloatAlpha()),
          b(c.getFloatBlue() * c.getFloatAlpha()), a(c.getFloatAlpha()) {}

    // The nearest juce::Colour: straight alpha, 8 bits a channel, clamped. Rgb
    // at alpha 0 has none.
    operator juce::Colour() const noexcept
    {
        return a > 0.0f ? juce::Colour::fromFloatRGBA(r / a, g / a, b / a, a) : juce::Colours::transparentBlack;
    }

    // Faded by `k`, as an opacity fades a draw: every channel, premultiplied.
    constexpr Colour operator*(float k) const noexcept { return { r * k, g * k, b * k, a * k }; }
    constexpr bool operator==(const Colour&) const noexcept = default;
};

} // namespace jvk
