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
 File: Transform.h
 Author: Gavin Payne
 ------------------------------------------------------------------------------
*/

#pragma once

namespace jvk
{

class Transform : public Updater<Transform>
{
public:
    Transform() = default;

    // Position
    void setPosition(glm::vec3 pos) { position = pos; dirty = true; }
    glm::vec3 getPosition() const { return position; }

    // Rotation (quaternion)
    void setRotation(glm::quat rot) { rotation = rot; dirty = true; }
    void setRotation(float pitch, float yaw, float roll)
    {
        rotation = glm::quat(glm::vec3(pitch, yaw, roll));
        dirty = true;
    }
    glm::quat getRotation() const { return rotation; }

    // Scale
    void setScale(glm::vec3 s) { scale = s; dirty = true; }
    void setScale(float uniform) { scale = glm::vec3(uniform); dirty = true; }
    glm::vec3 getScale() const { return scale; }

    // Model matrix (lazy cached)
    const glm::mat4& getMatrix() const
    {
        if (dirty)
        {
            matrix = glm::translate(glm::mat4(1.0f), position)
                   * glm::mat4_cast(rotation)
                   * glm::scale(glm::mat4(1.0f), scale);
            dirty = false;
        }
        return matrix;
    }

    // World matrix with optional parent
    glm::mat4 getWorldMatrix(const Transform* parent = nullptr) const
    {
        if (parent)
            return parent->getMatrix() * getMatrix();
        return getMatrix();
    }

    // Batch update: recompute matrix if dirty
    void update(double) override { getMatrix(); }

    // Dirty flag
    void markDirty() { dirty = true; }
    bool isDirty() const { return dirty; }

    // Direct ref access for Animator/Smoother targets
    glm::vec3& positionRef() { dirty = true; return position; }
    glm::quat& rotationRef() { dirty = true; return rotation; }
    glm::vec3& scaleRef() { dirty = true; return scale; }

private:
    glm::vec3 position { 0.0f };
    glm::quat rotation { glm::identity<glm::quat>() };
    glm::vec3 scale { 1.0f };
    mutable glm::mat4 matrix { 1.0f };
    mutable bool dirty = true;
};

} // jvk
