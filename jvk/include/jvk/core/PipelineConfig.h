/*
  ==============================================================================

    PipelineConfig.h
    Configurable pipeline state for jvk Pipeline.

  ==============================================================================
*/

#pragma once

namespace jvk
{

enum class BlendMode
{
    Opaque,
    AlphaBlend,
    Additive,
    Premultiplied,
    Multiply,  // result = src * dst (color grading)
    Screen     // result = 1 - (1-src)*(1-dst) (lighten)
};

struct PipelineConfig
{
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
    VkFrontFace frontFace = VK_FRONT_FACE_CLOCKWISE;
    bool depthTestEnable = false;
    bool depthWriteEnable = false;
    VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS;
    BlendMode blendMode = BlendMode::AlphaBlend;
    std::vector<VkPushConstantRange> pushConstantRanges;

    // Stencil
    bool stencilTestEnable = false;
    VkStencilOp stencilFailOp = VK_STENCIL_OP_KEEP;
    VkStencilOp stencilPassOp = VK_STENCIL_OP_KEEP;
    VkStencilOp stencilDepthFailOp = VK_STENCIL_OP_KEEP;
    VkCompareOp stencilCompareOp = VK_COMPARE_OP_ALWAYS;
    uint32_t stencilWriteMask = 0xFF;
    uint32_t stencilCompareMask = 0xFF;
    uint32_t stencilReference = 0;

    // Color write
    bool colorWriteEnable = true;
};

} // jvk
