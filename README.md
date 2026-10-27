# JVK
Vulkan integration for JUCE plugins.\
<img width="734" alt="Screenshot 2023-10-15 at 7 47 59 PM" src="https://github.com/ZGgof1999/JVK/assets/9949963/ebb0b266-1a13-44ed-915c-b1e1c8e46d7b">\
Background and text rendered with JUCE.
Triangles rendered with transparency, from two seperate components added to single vulkan renderer component
 
## Installation
If not using the projucer, just use JVK as any other juce module!
 
### Projucer
1. Install Vulkan (Not Bundled)
2. Ensure VULKAN\_SDK environment is set
3. run 'sh include_vulkan.sh [path/to/project] [optional VULKAN_SDK_PATH]'
 
This will process your .jucer to add the JVK module, add necessary search paths, and link necesarry external libraries. If you add a new exporter, just run the script again!
 
[Warning: If you are linking multiple external libraries, the automatic .jucer processor will not work, and it is recomended you use CMAKE to setup your project and link vulkan.]
 
## Features
- Windows and MacOS!
- Integrates seamlessly into juce::Component
- Bundled Plug 'N Play Components (ex. jvk::ShaderComponent - give it a shader, push data, awesome audio visualisations!)
- Transparent Native OS Windows to not obscure parent juce::Component (HWND not supported yet)
- [In Dev] Entity Component System for 3D rendering scenes with keyframe animations (I am porting from my juce openGL ECS rendering engine)
- Compute: GLSL compute kernels over GPU buffers, beside the rendering, on a queue of their own where the GPU has one (`jvk::compute`, see [COMPUTE.md](COMPUTE.md))

## Examples
- jvkTriangles: How to add a vulkan instance, and multpiple vulkan components as children of juce components
- jvkWaveform: How to push audio data from the processor to the GPU to render audio visualizers with shaders!

## Disclaimer
This module is IN DEV, and most likely not stable enough for production software. Use at your own risk. If you love how well it works already, and want to use it to make your plugins look incredible while running fast, please contribute! Theres lot's of error handling needed to be done in jvk::core to ensure it runs smoothly on all devices.
