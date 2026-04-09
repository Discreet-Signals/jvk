/*
  ==============================================================================

    Camera.h
    Camera is an Object (has Transform) with a CameraComponent.
    Use Animator/Smoother on transform for smooth movement.

  ==============================================================================
*/

#pragma once

namespace jvk
{

class CameraComponent : public ecs::ComponentBase<CameraComponent>
{
public:
    CameraComponent() = default;

    void setPerspective(float fovY, float aspect, float near = 0.1f, float far = 100.0f)
    {
        projection = glm::perspective(fovY, aspect, near, far);
        projection[1][1] *= -1.0f; // Vulkan Y-flip
        isOrtho = false;
        fov = fovY;
        aspectRatio = aspect;
        nearPlane = near;
        farPlane = far;
    }

    void setOrthographic(float left, float right, float bottom, float top,
                         float near = -1.0f, float far = 1.0f)
    {
        projection = glm::ortho(left, right, bottom, top, near, far);
        isOrtho = true;
        nearPlane = near;
        farPlane = far;
    }

    const glm::mat4& getProjection() const { return projection; }

    glm::mat4 getView(const Transform& t) const
    {
        return glm::inverse(t.getMatrix());
    }

    glm::mat4 getViewProjection(const Transform& t) const
    {
        return projection * getView(t);
    }

    glm::vec3 screenToWorldRay(const Transform& t,
                                float normalizedX, float normalizedY) const
    {
        glm::mat4 invVP = glm::inverse(projection * getView(t));
        float nearZ = isOrtho ? 0.0f : -1.0f;
        glm::vec4 nearPoint = invVP * glm::vec4(normalizedX, normalizedY, nearZ, 1.0f);
        glm::vec4 farPoint  = invVP * glm::vec4(normalizedX, normalizedY, 1.0f, 1.0f);
        nearPoint /= nearPoint.w;
        farPoint /= farPoint.w;
        return glm::normalize(glm::vec3(farPoint - nearPoint));
    }

    glm::vec3 screenToWorldRay(const Transform& t,
                                float screenX, float screenY,
                                float viewportWidth, float viewportHeight) const
    {
        float ndcX = (2.0f * screenX / viewportWidth) - 1.0f;
        float ndcY = (2.0f * screenY / viewportHeight) - 1.0f;
        return screenToWorldRay(t, ndcX, ndcY);
    }

    bool isOrthographic() const { return isOrtho; }
    float getFOV() const { return fov; }
    float getAspectRatio() const { return aspectRatio; }

private:
    glm::mat4 projection { 1.0f };
    bool isOrtho = false;
    float fov = glm::radians(45.0f);
    float aspectRatio = 1.0f;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;
};

// Camera: Object with CameraComponent
class Camera : public Object
{
public:
    Camera()
    {
        auto c = std::make_unique<CameraComponent>();
        cameraComp = c.get();
        addComponent(std::move(c));
    }

    // Convenience delegates
    void setPerspective(float fovY, float aspect, float near = 0.1f, float far = 100.0f)
    { cameraComp->setPerspective(fovY, aspect, near, far); }

    void setOrthographic(float left, float right, float bottom, float top,
                         float near = -1.0f, float far = 1.0f)
    { cameraComp->setOrthographic(left, right, bottom, top, near, far); }

    void lookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 up = { 0.0f, 1.0f, 0.0f })
    {
        transform->setPosition(eye);
        glm::mat4 viewMat = glm::lookAt(eye, center, up);
        transform->setRotation(glm::conjugate(glm::quat_cast(viewMat)));
    }

    const glm::mat4& getProjection() const { return cameraComp->getProjection(); }
    glm::mat4 getView() const { return cameraComp->getView(*transform); }
    glm::mat4 getViewProjection() const { return cameraComp->getViewProjection(*transform); }

    glm::vec3 screenToWorldRay(float normalizedX, float normalizedY) const
    { return cameraComp->screenToWorldRay(*transform, normalizedX, normalizedY); }

    glm::vec3 screenToWorldRay(float screenX, float screenY,
                                float viewportWidth, float viewportHeight) const
    { return cameraComp->screenToWorldRay(*transform, screenX, screenY, viewportWidth, viewportHeight); }

    bool isOrthographic() const { return cameraComp->isOrthographic(); }

    CameraComponent* cameraComp = nullptr;
};

} // jvk
