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
 File: Mesh.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk::objects
{

namespace vertices
{

struct Vertex
{
    glm::vec3 position;
};

struct ColoredVertex
{
    glm::vec3 position;
    glm::vec4 color;
};

struct DirectionalVertex
{
    glm::vec3 position;
    glm::vec3 normal;
};

struct DirectionalColoredVertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 color;
};

struct PBRVertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};

} // vertices

// CPU-side mesh data (any vertex type)
template <typename V>
struct MeshData
{
    std::vector<V> vertices;
    std::vector<uint32_t> indices;
};

// GPU-resident mesh (RAII, uses core::Buffer)
class GPUMesh
{
public:
    GPUMesh() = default;

    template <typename V>
    bool upload(VkPhysicalDevice physDevice, VkDevice device, const MeshData<V>& data)
    {
        destroy();
        indexCount = static_cast<uint32_t>(data.indices.size());

        if (data.vertices.empty() || data.indices.empty())
            return false;

        bool ok = vertexBuffer.create({
            physDevice, device,
            static_cast<VkDeviceSize>(sizeof(V) * data.vertices.size()),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        });
        if (!ok) return false;
        vertexBuffer.upload(data.vertices.data(), sizeof(V) * data.vertices.size());

        ok = indexBuffer.create({
            physDevice, device,
            static_cast<VkDeviceSize>(sizeof(uint32_t) * data.indices.size()),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        });
        if (!ok) return false;
        indexBuffer.upload(data.indices.data(), sizeof(uint32_t) * data.indices.size());

        return true;
    }

    void destroy()
    {
        vertexBuffer.destroy();
        indexBuffer.destroy();
        indexCount = 0;
    }

    void bind(VkCommandBuffer cmd) const
    {
        VkBuffer buffers[] = { vertexBuffer.getBuffer() };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);
        vkCmdBindIndexBuffer(cmd, indexBuffer.getBuffer(), 0, VK_INDEX_TYPE_UINT32);
    }

    void draw(VkCommandBuffer cmd, uint32_t instanceCount = 1) const
    {
        vkCmdDrawIndexed(cmd, indexCount, instanceCount, 0, 0, 0);
    }

    uint32_t getIndexCount() const { return indexCount; }
    bool isValid() const { return vertexBuffer.isValid() && indexBuffer.isValid(); }

private:
    core::Buffer vertexBuffer;
    core::Buffer indexBuffer;
    uint32_t indexCount = 0;
};

// Primitive generators
namespace primitives
{

static inline MeshData<vertices::PBRVertex> quad(float w = 1.0f, float h = 1.0f)
{
    float hw = w * 0.5f, hh = h * 0.5f;
    return {
        {
            {{ -hw, -hh, 0.0f }, { 0, 0, 1 }, { 0, 1 }},
            {{  hw, -hh, 0.0f }, { 0, 0, 1 }, { 1, 1 }},
            {{  hw,  hh, 0.0f }, { 0, 0, 1 }, { 1, 0 }},
            {{ -hw,  hh, 0.0f }, { 0, 0, 1 }, { 0, 0 }}
        },
        { 0, 1, 2, 0, 2, 3 }
    };
}

static inline MeshData<vertices::PBRVertex> unitQuad()
{
    // Unit quad from (0,0) to (1,1) — used for 2D components
    return {
        {
            {{ 0.0f, 0.0f, 0.0f }, { 0, 0, 1 }, { 0, 1 }},
            {{ 1.0f, 0.0f, 0.0f }, { 0, 0, 1 }, { 1, 1 }},
            {{ 1.0f, 1.0f, 0.0f }, { 0, 0, 1 }, { 1, 0 }},
            {{ 0.0f, 1.0f, 0.0f }, { 0, 0, 1 }, { 0, 0 }}
        },
        { 0, 1, 2, 0, 2, 3 }
    };
}

static inline MeshData<vertices::PBRVertex> cube(float size = 1.0f)
{
    float s = size * 0.5f;
    MeshData<vertices::PBRVertex> mesh;

    // Helper to add a face (4 verts + 6 indices)
    auto addFace = [&](glm::vec3 n, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3)
    {
        uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back({ v0, n, glm::vec2(0, 1) });
        mesh.vertices.push_back({ v1, n, glm::vec2(1, 1) });
        mesh.vertices.push_back({ v2, n, glm::vec2(1, 0) });
        mesh.vertices.push_back({ v3, n, glm::vec2(0, 0) });
        mesh.indices.push_back(base);
        mesh.indices.push_back(base + 1);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 3);
    };

    // Front, Back, Right, Left, Top, Bottom
    addFace(glm::vec3( 0, 0, 1), glm::vec3(-s,-s, s), glm::vec3( s,-s, s), glm::vec3( s, s, s), glm::vec3(-s, s, s));
    addFace(glm::vec3( 0, 0,-1), glm::vec3( s,-s,-s), glm::vec3(-s,-s,-s), glm::vec3(-s, s,-s), glm::vec3( s, s,-s));
    addFace(glm::vec3( 1, 0, 0), glm::vec3( s,-s, s), glm::vec3( s,-s,-s), glm::vec3( s, s,-s), glm::vec3( s, s, s));
    addFace(glm::vec3(-1, 0, 0), glm::vec3(-s,-s,-s), glm::vec3(-s,-s, s), glm::vec3(-s, s, s), glm::vec3(-s, s,-s));
    addFace(glm::vec3( 0, 1, 0), glm::vec3(-s, s, s), glm::vec3( s, s, s), glm::vec3( s, s,-s), glm::vec3(-s, s,-s));
    addFace(glm::vec3( 0,-1, 0), glm::vec3(-s,-s,-s), glm::vec3( s,-s,-s), glm::vec3( s,-s, s), glm::vec3(-s,-s, s));
    return mesh;
}

static inline MeshData<vertices::PBRVertex> sphere(int segments = 32, int rings = 16, float radius = 1.0f)
{
    MeshData<vertices::PBRVertex> mesh;

    for (int y = 0; y <= rings; y++)
    {
        float v = static_cast<float>(y) / static_cast<float>(rings);
        float phi = v * glm::pi<float>();

        for (int x = 0; x <= segments; x++)
        {
            float u = static_cast<float>(x) / static_cast<float>(segments);
            float theta = u * 2.0f * glm::pi<float>();

            glm::vec3 pos = {
                radius * std::sin(phi) * std::cos(theta),
                radius * std::cos(phi),
                radius * std::sin(phi) * std::sin(theta)
            };
            glm::vec3 normal = glm::normalize(pos);
            mesh.vertices.push_back({ pos, normal, { u, v } });
        }
    }

    for (int y = 0; y < rings; y++)
    {
        for (int x = 0; x < segments; x++)
        {
            uint32_t a = y * (segments + 1) + x;
            uint32_t b = a + segments + 1;
            mesh.indices.push_back(a);
            mesh.indices.push_back(b);
            mesh.indices.push_back(a + 1);
            mesh.indices.push_back(b);
            mesh.indices.push_back(b + 1);
            mesh.indices.push_back(a + 1);
        }
    }
    return mesh;
}

static inline MeshData<vertices::PBRVertex> plane(int divisionsX = 1, int divisionsY = 1,
                                                    float width = 1.0f, float height = 1.0f)
{
    MeshData<vertices::PBRVertex> mesh;
    int vertsX = divisionsX + 1;
    int vertsY = divisionsY + 1;
    float hw = width * 0.5f, hh = height * 0.5f;

    for (int y = 0; y < vertsY; y++)
    {
        float v = static_cast<float>(y) / divisionsY;
        for (int x = 0; x < vertsX; x++)
        {
            float u = static_cast<float>(x) / divisionsX;
            mesh.vertices.push_back({
                { -hw + u * width, 0.0f, -hh + v * height },
                { 0, 1, 0 },
                { u, v }
            });
        }
    }

    for (int y = 0; y < divisionsY; y++)
    {
        for (int x = 0; x < divisionsX; x++)
        {
            uint32_t a = y * vertsX + x;
            uint32_t b = a + vertsX;
            mesh.indices.push_back(a);
            mesh.indices.push_back(b);
            mesh.indices.push_back(a + 1);
            mesh.indices.push_back(b);
            mesh.indices.push_back(b + 1);
            mesh.indices.push_back(a + 1);
        }
    }
    return mesh;
}

} // primitives

} // jvk::objects
