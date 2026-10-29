#!/bin/bash
# =============================================================================
# Compile the built-in shaders into the shader bundle,
# include/jvk/core/shaders/Bundle.h (its shape: core/shaders/Program.h).
#
# Every stage is compiled twice from one GLSL source (jvk_portable.glsl):
#   -DJVK_VULKAN   the Vulkan SPIR-V (constants as push constants)
#   plain          the SPIR-V the translations start from (constants as a
#                  uniform block at binding 24), translated to MSL by
#                  SPIRV-Cross with each binding number as its Metal index
# Both are validated (spirv-val), and each program's layout (textures,
# storage buffers, constants size, colour outputs) is reflected from the
# SPIR-V (spirv-cross --reflect), never written by hand. The script also
# checks that std430 (push constants) and std140 (uniform block) put every
# constants member at the same offset.
#
# Tools: GLSLC / SPIRV_CROSS / SPIRV_VAL from the environment, else the
# Vulkan SDK ($VULKAN_SDK/bin, else the newest ~/VulkanSDK/*/macOS/bin),
# else PATH. CMake runs this whenever a shader changes; by hand:
#   cd shaders && ./compile.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SDK_BINS=()
if [ -n "${VULKAN_SDK:-}" ]; then
    SDK_BINS+=("$VULKAN_SDK/bin" "$VULKAN_SDK/Bin")
fi
if [ -d "$HOME/VulkanSDK" ]; then
    NEWEST="$(ls -d "$HOME"/VulkanSDK/*/macOS/bin 2>/dev/null | sort -V | tail -n 1 || true)"
    [ -n "$NEWEST" ] && SDK_BINS+=("$NEWEST")
fi

find_tool() {   # find_tool <env value> <name>
    if [ -n "$1" ]; then echo "$1"; return; fi
    for dir in "${SDK_BINS[@]}"; do
        if [ -x "$dir/$2" ]; then echo "$dir/$2"; return; fi
        if [ -x "$dir/$2.exe" ]; then echo "$dir/$2.exe"; return; fi
    done
    command -v "$2" || true
}

GLSLC="$(find_tool "${GLSLC:-}" glslc)"
SPIRV_CROSS="$(find_tool "${SPIRV_CROSS:-}" spirv-cross)"
SPIRV_VAL="$(find_tool "${SPIRV_VAL:-}" spirv-val)"

for tool in "$GLSLC" "$SPIRV_CROSS" "$SPIRV_VAL"; do
    if [ -z "$tool" ]; then
        echo "Error: glslc, spirv-cross and spirv-val are needed (install the Vulkan SDK, or set GLSLC / SPIRV_CROSS / SPIRV_VAL)."
        exit 1
    fi
done

python3 - "$GLSLC" "$SPIRV_CROSS" "$SPIRV_VAL" "../include/jvk/core/shaders/Bundle.h" << 'PYEOF'
import json, os, subprocess, sys, tempfile

glslc, spirv_cross, spirv_val, out_path = sys.argv[1:5]

# (source, glslc stage, C++ name, translated to MSL)
STAGES = [
    ('ui2d.vert',          'vert', 'ui2d_vert',          True),
    ('ui2d.frag',          'frag', 'ui2d_frag',          True),
    ('noise.frag',         'frag', 'noise_frag',         True),
    ('path_sdf.vert',      'vert', 'path_sdf_vert',      True),
    ('path_sdf.frag',      'frag', 'path_sdf_frag',      True),
    ('clip.vert',          'vert', 'clip_vert',          True),
    ('clip.frag',          'frag', 'clip_frag',          True),
    ('clip_mask.frag',     'frag', 'clip_mask_frag',     True),
    ('blur.vert',          'vert', 'blur_vert',          True),
    ('blur.frag',          'frag', 'blur_frag',          True),
    ('copy.frag',          'frag', 'copy_frag',          True),
    ('hsv.frag',           'frag', 'hsv_frag',           True),
    ('shape_blur.frag',    'frag', 'shape_blur_frag',    True),
    ('path_blur.frag',     'frag', 'path_blur_frag',     True),
    # The vertex stage of user jvk::Shaders: Vulkan only, and its push-constant
    # block is the users' native contract, so it stays as written.
    ('shader_region.vert', 'vert', 'shader_region_vert', False),
]

# (program, vertex stage, fragment stage)
PROGRAMS = [
    ('ui2d',      'ui2d_vert',     'ui2d_frag'),
    ('noise',     'ui2d_vert',     'noise_frag'),
    ('path',      'path_sdf_vert', 'path_sdf_frag'),
    ('clip',      'clip_vert',     'clip_frag'),
    ('clipMask',  'clip_vert',     'clip_mask_frag'),
    ('blur',      'blur_vert',     'blur_frag'),
    ('copy',      'blur_vert',     'copy_frag'),
    ('hsv',       'blur_vert',     'hsv_frag'),
    ('shapeBlur', 'blur_vert',     'shape_blur_frag'),
    ('pathBlur',  'blur_vert',     'path_blur_frag'),
]

MAX_TEXTURES, SAMPLER_BASE, STORAGE_BASE, CONSTANTS_BINDING = 8, 8, 16, 24
MAX_CONSTANTS = 128

def fail(msg):
    print('Error: ' + msg)
    sys.exit(1)

def run(args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        fail(' '.join(args) + '\n' + r.stdout + r.stderr)
    return r.stdout

tmp = tempfile.mkdtemp(prefix='jvk_shaders_')

SCALAR_SIZES = {
    'float': 4, 'int': 4, 'uint': 4, 'bool': 4,
    'vec2': 8, 'ivec2': 8, 'uvec2': 8,
    'vec3': 12, 'ivec3': 12, 'uvec3': 12,
    'vec4': 16, 'ivec4': 16, 'uvec4': 16,
}

def type_size(types, t, member=None):
    if member is not None and member.get('array'):
        count = 1
        for n in member['array']:
            count *= n
        return member['array_stride'] * count
    if t in SCALAR_SIZES:
        return SCALAR_SIZES[t]
    if t in types:
        return struct_size(types, t)
    fail('unknown type in a constants block: ' + t)

def struct_size(types, t):
    end = 0
    for m in types[t]['members']:
        end = max(end, m['offset'] + type_size(types, m['type'], m))
    return end

def offsets(types, t):
    return [(m['name'], m['type'], m['offset']) for m in types[t]['members']]

stages = {}
for src, kind, name, translate in STAGES:
    info = { 'name': name, 'kind': kind, 'src': src }

    vk_spv = os.path.join(tmp, name + '.vk.spv')
    run([glslc, '-fshader-stage=' + kind, '-DJVK_VULKAN', '-I.', src, '-o', vk_spv])
    run([spirv_val, vk_spv])
    vk = json.loads(run([spirv_cross, vk_spv, '--reflect']))
    with open(vk_spv, 'rb') as f:
        info['spirv'] = f.read()
    info['reflect'] = vk

    info['constants'] = None
    pcs = vk.get('push_constants', [])
    if pcs:
        t = pcs[0]['type']
        info['constants'] = { 'size': struct_size(vk['types'], t),
                              'members': offsets(vk['types'], t) }

    info['msl'] = ''
    if translate:
        x_spv = os.path.join(tmp, name + '.x.spv')
        run([glslc, '-fshader-stage=' + kind, '-I.', src, '-o', x_spv])
        run([spirv_val, x_spv])
        x = json.loads(run([spirv_cross, x_spv, '--reflect']))
        args = [spirv_cross, x_spv, '--msl', '--msl-version', '20100', '--msl-decoration-binding']
        if kind == 'vert':
            args.append('--flip-vert-y')   # Metal's clip space is y-up; the shaders are written y-down
        info['msl'] = run(args)

        # The two builds must agree on every constants member's offset
        # (std430 push constants vs the std140 uniform block).
        ubos = [u for u in x.get('ubos', []) if u['binding'] == CONSTANTS_BINDING]
        if bool(ubos) != bool(info['constants']):
            fail(src + ': the constants block differs between the two builds')
        if ubos:
            xo = offsets(x['types'], ubos[0]['type'])
            if xo != info['constants']['members']:
                fail(src + ': std140 and std430 disagree on the constants layout:\n  '
                     + str(xo) + '\n  ' + str(info['constants']['members']))
            info['constants']['size'] = max(info['constants']['size'], ubos[0]['block_size'])
        if x.get('textures'):
            fail(src + ': combined image samplers are not portable (use JVK_TEXTURE + JVK_SAMPLER)')
        if any(u['binding'] != CONSTANTS_BINDING for u in x.get('ubos', [])):
            fail(src + ': uniform blocks other than JVK_CONSTANTS are not portable')
    stages[name] = info

def layout(stage):
    r = stage['reflect']
    textures = [i['binding'] for i in r.get('separate_images', [])]
    samplers = [s['binding'] - SAMPLER_BASE for s in r.get('separate_samplers', [])]
    storage  = [s['binding'] - STORAGE_BASE for s in r.get('ssbos', [])]
    for b in textures:
        if not 0 <= b < MAX_TEXTURES: fail(stage['src'] + ': texture binding %d out of range' % b)
    for b in samplers:
        if not 0 <= b < MAX_TEXTURES: fail(stage['src'] + ': sampler binding %d out of range' % (b + SAMPLER_BASE))
    for b in storage:
        if not 0 <= b < 8: fail(stage['src'] + ': storage binding %d out of range' % (b + STORAGE_BASE))
    for s in r.get('ssbos', []):
        if not s.get('readonly'): fail(stage['src'] + ': storage buffer ' + s['name'] + ' must be readonly')
    outputs = [o['location'] + (o.get('array', [1])[0] - 1) for o in r.get('outputs', [])]
    return (max(textures + samplers, default=-1) + 1,
            max(storage, default=-1) + 1,
            (stage['constants'] or {}).get('size', 0),
            max(outputs, default=-1) + 1)

programs = []
for pname, vname, fname in PROGRAMS:
    v, f = stages[vname], stages[fname]
    vt, vs, vc, _  = layout(v)
    ft, fs, fc, fo = layout(f)
    # Both stages read the same constant bytes: one must be a prefix of the other.
    if v['constants'] and f['constants']:
        a, b = v['constants']['members'], f['constants']['members']
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if [(t, o) for _, t, o in shorter] != [(t, o) for _, t, o in longer[:len(shorter)]]:
            fail(pname + ': its stages declare incompatible constants blocks:\n  ' + str(a) + '\n  ' + str(b))
    constants = max(vc, fc)
    constants = (constants + 15) // 16 * 16     # whole 16-byte rows: what Metal's struct sizes round to
    if constants > MAX_CONSTANTS:
        fail(pname + ': %d bytes of constants (at most %d)' % (constants, MAX_CONSTANTS))
    longest = max([s['constants']['members'] for s in (v, f) if s['constants']], key=len, default=[])
    programs.append({ 'name': pname, 'vert': vname, 'frag': fname,
                      'textures': max(vt, ft), 'storage': max(vs, fs),
                      'constants': constants, 'outputs': fo, 'members': longest })

# ---- Bundle.h ----------------------------------------------------------------

def words(data):
    w = [int.from_bytes(data[i:i + 4], 'little') for i in range(0, len(data), 4)]
    lines = []
    for i in range(0, len(w), 8):
        lines.append('    ' + ', '.join('0x%08x' % x for x in w[i:i + 8]) + ',')
    return '\n'.join(lines)

DELIM = 'jvkmsl'
def msl_literal(text):
    if not text:
        return '""'
    if ')' + DELIM + '"' in text:
        fail('MSL contains the raw-string delimiter')
    if len(text) > 60000:
        fail('MSL too long for one MSVC string literal: %d bytes' % len(text))
    # MSVC caps a single string literal at 16380 bytes: adjacent chunks, split at lines.
    chunks, cur = [], ''
    for line in text.splitlines(keepends=True):
        if len(cur) + len(line) > 2000 and cur:
            chunks.append(cur)
            cur = ''
        cur += line
    if cur:
        chunks.append(cur)
    return '\n'.join('R"%s(%s)%s"' % (DELIM, c, DELIM) for c in chunks)

out = """/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC

 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 ------------------------------------------------------------------------------
 File: Bundle.h
 Auto-generated by shaders/compile.sh from the GLSL in shaders/ — do not edit.
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::shaders::bundle
{

// =============================================================================
// Stages
// =============================================================================

"""

for src, kind, name, translate in STAGES:
    s = stages[name]
    out += '// %s\n' % src
    out += 'alignas(4) inline constexpr uint32_t %s_spv[] = {\n%s\n};\n' % (name, words(s['spirv']))
    if translate:
        out += 'inline constexpr char %s_msl[] =\n%s;\n' % (name, msl_literal(s['msl']))
    out += '\n'

def stage_init(name):
    s = stages[name]
    if s['msl']:
        return '{ %s_spv, { %s_msl, sizeof(%s_msl) - 1 } }' % (name, name, name)
    return '{ %s_spv, {} }' % name

out += """// =============================================================================
// Programs: name, vertex, fragment, textures, storage buffers, constants
// bytes, colour outputs (reflected from the SPIR-V).
// =============================================================================

"""

for p in programs:
    out += '// %s + %s' % (p['vert'].replace('_vert', '.vert'), p['frag'].replace('_frag', '.frag'))
    if p['members']:
        out += ' — constants: ' + ', '.join('%s %s @%d' % (t, n, o) for n, t, o in p['members'])
    out += '\n'
    out += 'inline const Program %s { "%s", %s, %s, %d, %d, %d, %d };\n\n' % (
        p['name'], p['name'], stage_init(p['vert']), stage_init(p['frag']),
        p['textures'], p['storage'], p['constants'], p['outputs'])

out += """// The vertex stage of user jvk::Shaders (shader_region.vert): Vulkan SPIR-V,
// its push-constant block the users' native contract.
inline const Stage shaderRegionVert %s;

} // namespace jvk::shaders::bundle
""" % stage_init('shader_region_vert')

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(out)

for p in programs:
    print('  %-10s textures %d  storage %d  constants %3d B  outputs %d'
          % (p['name'], p['textures'], p['storage'], p['constants'], p['outputs']))
print('  wrote ' + os.path.normpath(out_path))
PYEOF
