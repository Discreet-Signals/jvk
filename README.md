# jvk

A GPU renderer for JUCE plugin editors: Metal on macOS, Vulkan on Windows. Derive your editor from `jvk::AudioProcessorEditor` and your existing `paint()` code draws on the GPU. If the GPU isn't available on a machine, the editor falls back to JUCE's own renderer.

## Requirements

- CMake 3.20+ and a C++20 compiler
- JUCE 8
- The Vulkan SDK
- `bash` and `python3` on your PATH (jvk compiles its shaders during the build; on Windows, use Git Bash)

## 1. Install the Vulkan SDK

Download it from <https://vulkan.lunarg.com/sdk/home> and run the installer. Select the **GLM headers** component, because jvk uses them.

- **macOS:** install to the default `~/VulkanSDK/<version>`. jvk finds the newest SDK there when `VULKAN_SDK` isn't set.
- **Windows:** the installer sets `VULKAN_SDK` for you. Restart your terminal or IDE afterwards so it sees the variable.

The SDK is only needed to build, on every platform: its `glslc` compiles jvk's shaders, and jvk uses its GLM headers. Your users don't need to install it: on macOS, jvk draws with Metal, which every Mac has, and on Windows it uses the Vulkan driver that ships with the GPU.

## 2. Add jvk to your project

Clone it into a folder named `jvk`, so that `#include <jvk/jvk.h>` resolves:

```bash
git submodule add https://github.com/Discreet-Signals/jvk modules/jvk
```

## 3. CMake

```cmake
cmake_minimum_required(VERSION 3.20)
project(MyPlugin VERSION 1.0.0)

add_subdirectory(JUCE)          # add JUCE before jvk
add_subdirectory(modules/jvk)

juce_add_plugin(MyPlugin
    FORMATS AU VST3 Standalone
    # ...your usual plugin settings
)

target_sources(MyPlugin PRIVATE Source/PluginProcessor.cpp Source/PluginEditor.cpp)

target_link_libraries(MyPlugin PRIVATE jvk juce::juce_audio_utils)

# Copies MoltenVK into each macOS bundle when the build has the Vulkan backend
# (otherwise, and on Windows, it does nothing)
foreach(format AU VST3 Standalone)
    jvk_bundle(MyPlugin_${format})
endforeach()
```

If JUCE hasn't been added when jvk is configured, jvk downloads JUCE 8.0.12 itself. The first configure also downloads msdfgen, plus MoltenVK when a macOS build has the Vulkan backend, so it needs an internet connection.

### Backends

`JVK_BACKENDS` picks the backends a build compiles, and so what it ships. On macOS it defaults to `Metal`: nothing of Vulkan is compiled or bundled. On Windows and Linux it is `Vulkan`.

To compare the two on a Mac, configure with `-DJVK_BACKENDS="Metal;Vulkan"`. The build then carries MoltenVK as well, still draws with Metal, and draws with Vulkan when the app runs with `JVK_BACKEND=vulkan` in its environment.

## Drawing

You can use jvk at three levels. Each one builds on the one before it.

### Level 1: juce::Graphics on the GPU

Derive your editor from `jvk::AudioProcessorEditor` and change nothing else. Every `juce::Graphics` call made by the editor and its child components runs on the GPU. That includes fills, strokes, paths, gradients, text, images, clips and transforms, so JUCE's widgets and your LookAndFeels work as they are.

```cpp
#include <jvk/jvk.h>

class MyEditor : public jvk::AudioProcessorEditor
{
public:
    explicit MyEditor (MyProcessor& p) : jvk::AudioProcessorEditor (p)
    {
        addAndMakeVisible (gain);
        setSize (600, 400);
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (juce::Colours::black);
        g.setGradientFill (juce::ColourGradient (juce::Colours::orange, 0.0f, 0.0f,
                                                 juce::Colours::purple, 600.0f, 400.0f, false));
        g.fillRoundedRectangle (getLocalBounds().reduced (20).toFloat(), 12.0f);
        g.setColour (juce::Colours::white);
        g.setFont (juce::FontOptions (24.0f));
        g.drawText ("Hello, GPU", getLocalBounds(), juce::Justification::centred);
    }

    void resized() override { gain.setBounds (40, 40, 100, 120); }

private:
    juce::Slider gain { juce::Slider::RotaryVerticalDrag, juce::Slider::TextBoxBelow };
};
```

jvk repaints the whole editor every frame, at 60 Hz. `setVulkanEnabled (false)` switches to JUCE's own renderer at run time. If a machine can't run Vulkan, jvk switches to JUCE's renderer by itself.

### Level 2: GPU effects and your own shaders

`jvk::Graphics::create (g)` gives you the GPU context behind a `juce::Graphics`. It returns `nullptr` whenever JUCE's renderer is drawing, so check it, and draw a plain version in that case if you need one.

The effects act on whatever has been painted so far in their region. If you don't pass a region, they cover the current clip.

```cpp
void paint (juce::Graphics& g) override
{
    g.drawImage (background, getLocalBounds().toFloat());

    auto panel  = getLocalBounds().toFloat().reduced (60);
    auto footer = getLocalBounds().toFloat().removeFromBottom (40);

    if (auto* vk = jvk::Graphics::create (g))
    {
        vk->fillBlurredRoundedRectangle (panel, 16.0f, 20.0f, 6.0f);  // frosted glass: blur inside a shape
        vk->darken (0.3f, panel);                                     // also brighten, tint, warmth
        vk->blur (8.0f, footer);                                      // a plain blur
        vk->saturate (0.5f);                                          // half the colour; also hsv, shiftHue
        vk->drawNoise (0.04f, true);                                  // film grain
    }

    g.setColour (juce::Colours::white);
    g.drawRoundedRectangle (panel, 16.0f, 1.0f);
}
```

The blurred shapes come as both fills and strokes: `fillBlurredRectangle`, `fillBlurredEllipse`, `fillBlurredPath`, `drawBlurredLine`, `drawBlurredPath` and the rest.

**Your own shaders.** Write a fragment shader. jvk fills in the first eight push constants for you, and anything after them is yours to set per draw:

```glsl
#version 450
layout(location = 0) in vec2 fragUV;       // 0..1 across the region
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    float resolutionX, resolutionY, time;  // region size in pixels, seconds
    float viewportW, viewportH, regionX, regionY, reserved;
    float level;                           // yours: setDrawConstants
} pc;

void main()
{
    vec3 colour = 0.5 + 0.5 * cos (pc.time + fragUV.xyx * 6.0 + vec3 (0, 2, 4));
    outColor = vec4 (colour * pc.level, 1.0);
}
```

Compile it with the SDK's `glslc`, have `jvk_msl` translate that for Metal, and embed both. The translation happens at build time, so no shader translator ships in your plugin; in a build without Metal, `jvk_msl` writes an empty file.

```cmake
set(spv "${CMAKE_CURRENT_BINARY_DIR}/meter.frag.spv")
set(msl "${CMAKE_CURRENT_BINARY_DIR}/meter.frag.metal")
add_custom_command(OUTPUT "${spv}"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/Source/meter.frag" -o "${spv}"
    DEPENDS Source/meter.frag)
jvk_msl("${spv}" "${msl}")
juce_add_binary_data(MyPluginShaders SOURCES "${spv}" "${msl}")
target_link_libraries(MyPlugin PRIVATE MyPluginShaders)
```

Then draw it:

```cpp
static std::span<const uint32_t> spirv (const char* data, int size)
{
    return { reinterpret_cast<const uint32_t*> (data), static_cast<size_t> (size) / 4 };
}

void paint (juce::Graphics& g) override
{
    if (auto* vk = jvk::Graphics::create (g))
    {
        auto& meter = vk->shader (spirv (BinaryData::meter_frag_spv, BinaryData::meter_frag_spvSize),
                                  { BinaryData::meter_frag_metal, static_cast<size_t> (BinaryData::meter_frag_metalSize) });
        const float constants[] = { level };
        meter.setDrawConstants (constants);
        vk->drawShader (meter, meterArea);
    }
}
```

`vk->shader()` keeps one compiled shader per window, keyed by the address of the SPIR-V, so pass data that lives for the whole program, such as `BinaryData`. A shader without the form its backend draws (no MSL on a Mac) draws nothing, and says so in jvk's log. `time` keeps counting on its own, so an animated shader needs no `repaint()`. For a uniform block, call `shader.set ("BlockName", values)`. For an image, call `shader.set ("sampler", image, vk->getRenderer())`, or `shader.update (...)` if the image changes every frame.

### Level 3: render targets

A frame can draw more than one picture. Next to the main colour you can declare named **targets**, such as a glow layer, normals, heights or material values. Ordinary drawing can go into a target, effects can run on one, and a shader can read any of them later in the same frame.

This example draws a glow under a knob:

```cpp
class GlowEditor : public jvk::AudioProcessorEditor
{
public:
    explicit GlowEditor (MyProcessor& p) : jvk::AudioProcessorEditor (p)
    {
        jvk::Target glow;
        glow.name   = "glow";
        glow.format = jvk::Target::Format::RGBA16F;   // half floats
        setTargets ({ glow });                        // before setSize: targets are fixed while Vulkan runs
        setSize (600, 400);
    }

    void paint (juce::Graphics& g) override
    {
        auto knob = getLocalBounds().toFloat().withSizeKeepingCentre (120, 120);
        g.fillAll (juce::Colours::black);

        if (auto* vk = jvk::Graphics::create (g))
        {
            {
                juce::Graphics::ScopedSaveState state (g);   // the target is part of the saved state
                vk->setTarget ("glow");                       // ordinary draws now land in "glow"
                g.setColour (juce::Colours::orange);
                g.fillEllipse (knob.expanded (10));
                vk->blur (24.0f);                             // effects run on the selected target
            }
            vk->drawShader (vk->shader (spirv (BinaryData::glow_frag_spv, BinaryData::glow_frag_spvSize)));
        }

        g.setColour (juce::Colours::orange);
        g.fillEllipse (knob);                                 // the knob itself, in the main colour
    }
};
```

A shader reads a target by declaring a sampler with the target's name:

```glsl
#version 450
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2D glow;   // the "glow" target

void main()
{
    vec4 g = texelFetch (glow, ivec2 (gl_FragCoord.xy), 0);
    outColor = vec4 (g.rgb / max (g.a, 1e-4), g.a);    // drawn over the main colour
}
```

You don't set up any passes yourself. jvk works out from what each shader reads and writes whether it can run inside the frame's main pass or needs a pass of its own.

Other things you can do:

- **Read the scene.** Declare `layout(set = 1, binding = 0) uniform sampler2D scene;` to read everything painted so far. A shader that reads the scene replaces the main colour inside its region, which is what you want for lighting, colour grading or distortion.
- **Write targets from a shader.** Target `i` is output `layout(location = i + 1)`. Location 0 is always the main colour.
- **Blend modes.** `Target::blend` sets how a shader's writes combine with what the target holds: `Over`, `Premultiplied`, `Replace`, `Add`, `Max` or `None`.
- **Values a `juce::Colour` can't hold.** `jvk::Graphics::setColour (g, jvk::Colour { r, g, b, a })` sets the colour of the draws that follow as four floats, premultiplied. In a float target (`RGBA16F`, `RGBA32F`) they land whole: values past 1, negative ones, or rgb at alpha 0, which adds under the draw without covering it. An 8-bit target rounds and clamps them. Every `juce::Colour` converts to a `jvk::Colour` and back, so jvk functions that take one accept `juce::Colours::orange` too.
- **Smaller targets.** `Target::scale` (for example 0.25), or a fixed `Target::height` in pixels, makes a target that is cheap to blur and to compute. Ordinary drawing doesn't land on it. A shader writes it through an output named after the target.
- **`Target::clearedByMain`.** Ordinary paint into the main colour resets the target to its `clear` value underneath it, so the target always describes whatever was painted on top at each pixel.
- **Per-draw images.** A sampler in `layout(set = 2, binding = 0)` takes a different image on each draw: `shader.setDrawImage ("albedo", jvk::ImageSource::of (image))`.
- **Frame values.** `vk->values()` is a `juce::ValueTree` that is emptied every frame. Components use it to pass information, such as a light's position, to shaders painted later in the same frame.

## Examples

`examples/` has four example plugins. To build one on its own:

```bash
cmake -B build -G Ninja
cmake --build build --target VulkanCompressor_Standalone
```

The targets are `VulkanCompressor_Standalone`, `ShaderBackground_Standalone`, `JVKCompliance_Standalone` and `JVKBenchmark_Standalone`.
