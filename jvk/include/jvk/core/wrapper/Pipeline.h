/*
 ----------------------------------------------------------------------------
 Copyright (c) 2026 Discreet Signals LLC
 
 ██████╗  ██╗ ███████╗  ██████╗ ██████╗  ███████╗ ███████╗ ████████╗
 ██╔══██╗ ██║ ██╔════╝ ██╔════╝ ██╔══██╗ ██╔════╝ ██╔════╝ ╚══██╔══╝
 ██║  ██║ ██║ ███████╗ ██║      ██████╔╝ █████╗   █████╗      ██║
 ██║  ██║ ██║ ╚════██║ ██║      ██╔══██╗ ██╔══╝   ██╔══╝      ██║
 ██████╔╝ ██║ ███████║ ╚██████╗ ██║  ██║ ███████╗ ███████╗    ██║
 ╚═════╝  ╚═╝ ╚══════╝  ╚═════╝ ╚═╝  ╚═╝ ╚══════╝ ╚══════╝    ╚═╝
 
 Licensed under the MIT License. See LICENSE file in the project root
 for full license text.
 
 For questions, contact gavin@discreetsignals.com
 ------------------------------------------------------------------------------
 File: Pipeline.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

class Pipeline
{
public:
    template <typename SG>
    Pipeline(VkDevice vk_device, VkRenderPass render_pass, SG&& shader_group,
             VkSampleCountFlagBits msaa = VK_SAMPLE_COUNT_4_BIT,
             VertexLayout vertex_layout = VertexLayout::positionColor(),
             PipelineConfig config = {}) :
        device(vk_device),
        renderPass(render_pass),
        shaderGroup(std::forward<SG>(shader_group)),
        msaaSamples(msaa),
        vertexLayout(std::move(vertex_layout)),
        config(std::move(config))
    {
        create();
    }
    ~Pipeline();

    VkPipeline getInternal() const { return graphicsPipeline; }
    VkPipelineLayout getLayout() const { return pipelineLayout; }
    const PipelineConfig& getConfig() const { return config; }
private:
    VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);
    void create();

    VkDevice device;
    VkRenderPass renderPass;
    shaders::ShaderGroup shaderGroup;
    VkSampleCountFlagBits msaaSamples;
    VertexLayout vertexLayout;
    PipelineConfig config;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
};

}
