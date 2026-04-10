#!/bin/bash
# Compile GLSL shaders to SPIR-V and regenerate UI2DShaders.h
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GLSLC="${GLSLC:-$(command -v glslc)}"
if [ -z "$GLSLC" ]; then
    echo "Error: glslc not found. Install the Vulkan SDK or set GLSLC env var."
    exit 1
fi

echo "Compiling shaders..."
$GLSLC -fshader-stage=vert ui2d.vert     -o ui2d.vert.spv
$GLSLC -fshader-stage=frag ui2d.frag     -o ui2d.frag.spv
$GLSLC -fshader-stage=vert stencil.vert  -o stencil.vert.spv
$GLSLC -fshader-stage=frag stencil.frag  -o stencil.frag.spv
$GLSLC -fshader-stage=frag hsv.frag      -o hsv.frag.spv
$GLSLC -fshader-stage=frag blur.frag     -o blur.frag.spv

echo "Generating UI2DShaders.h..."
python3 - << 'PYEOF'
import os

def spv_to_c_array(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    lines = []
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        lines.append('    ' + ','.join(f'0x{b:02x}' for b in chunk) + ',')
    return len(data), '\n'.join(lines)

shaders = [
    ('ui2d.vert.spv', 'vert_spv'),
    ('ui2d.frag.spv', 'frag_spv'),
    ('stencil.vert.spv', 'stencil_vert_spv'),
    ('stencil.frag.spv', 'stencil_frag_spv'),
    ('hsv.frag.spv', 'hsv_frag_spv'),
    ('blur.frag.spv', 'blur_frag_spv'),
]

out = """/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: UI2DShaders.h
 Auto-generated from GLSL sources in shaders/ — do not edit manually.
 Run: cd shaders && ./compile.sh
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::shaders::ui2d
{

"""

for spv_file, var_name in shaders:
    size, array_data = spv_to_c_array(spv_file)
    out += f'static inline const int {var_name}Size = {size};\n'
    out += f'static inline const unsigned char __{var_name}[] = {{\n{array_data}\n}};\n'
    out += f'static inline const char* {var_name} = reinterpret_cast<const char*>(__{var_name});\n\n'

out += '} // jvk::shaders::ui2d\n'

with open('../include/jvk/ui/UI2DShaders.h', 'w') as f:
    f.write(out)

for spv_file, var_name in shaders:
    print(f"  {var_name}: {os.path.getsize(spv_file)} bytes")
PYEOF

echo "Done."
