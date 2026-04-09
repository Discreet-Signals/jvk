/*
  ==============================================================================

    Component.h
    GPU-rendered 2D component mirroring juce::Component naming.

    A Component is backed by an Object in the scene (orthographic camera,
    quad mesh, z-order as depth). setBounds() maps to transform position+scale,
    toFront()/toBack() maps to z-depth. Hit testing uses Scene::raycast.

  ==============================================================================
*/

#pragma once

namespace jvk
{

class ComponentRenderer; // forward

class Component
{
public:
    Component() = default;
    virtual ~Component() = default;

    // --- Bounds (juce-style API) ---
    void setBounds(juce::Rectangle<float> b)
    {
        bounds = b;
        if (object)
        {
            object->transform->setPosition(glm::vec3(b.getX(), b.getY(), static_cast<float>(zOrder)));
            object->transform->setScale(glm::vec3(b.getWidth(), b.getHeight(), 1.0f));
        }
        resized();
    }

    void setBounds(float x, float y, float w, float h) { setBounds({ x, y, w, h }); }

    juce::Rectangle<float> getBounds() const { return bounds; }
    juce::Rectangle<float> getLocalBounds() const { return { 0, 0, bounds.getWidth(), bounds.getHeight() }; }

    float getWidth() const { return bounds.getWidth(); }
    float getHeight() const { return bounds.getHeight(); }
    float getX() const { return bounds.getX(); }
    float getY() const { return bounds.getY(); }

    // --- Hierarchy ---
    void addChild(Component* child)
    {
        child->parent = this;
        children.push_back(child);
    }

    void removeChild(Component* child)
    {
        child->parent = nullptr;
        children.erase(std::remove(children.begin(), children.end(), child), children.end());
    }

    Component* getParent() const { return parent; }
    const std::vector<Component*>& getChildren() const { return children; }

    // --- Z-ordering ---
    void toFront()
    {
        if (parent)
        {
            auto& siblings = parent->children;
            int maxZ = 0;
            for (auto* s : siblings)
                maxZ = std::max(maxZ, s->zOrder);
            setZOrder(maxZ + 1);
        }
    }

    void toBack()
    {
        if (parent)
        {
            auto& siblings = parent->children;
            int minZ = 0;
            for (auto* s : siblings)
                minZ = std::min(minZ, s->zOrder);
            setZOrder(minZ - 1);
        }
    }

    void setZOrder(int z)
    {
        zOrder = z;
        if (object)
            object->transform->positionRef().z = static_cast<float>(z);
    }

    int getZOrder() const { return zOrder; }

    // --- Visibility ---
    void setVisible(bool v)
    {
        visible = v;
        if (object) object->enabled = v;
    }
    bool isVisible() const { return visible; }

    void setAlpha(float a) { alpha = a; }
    float getAlpha() const { return alpha; }

    // --- Paint (override this) ---
    virtual void paint(Graphics& g) { juce::ignoreUnused(g); }
    virtual void resized() {}

    // --- Mouse input ---
    virtual void mouseDown(const juce::MouseEvent& e) { juce::ignoreUnused(e); }
    virtual void mouseUp(const juce::MouseEvent& e) { juce::ignoreUnused(e); }
    virtual void mouseDrag(const juce::MouseEvent& e) { juce::ignoreUnused(e); }
    virtual void mouseMove(const juce::MouseEvent& e) { juce::ignoreUnused(e); }
    virtual void mouseEnter(const juce::MouseEvent& e) { juce::ignoreUnused(e); }
    virtual void mouseExit(const juce::MouseEvent& e) { juce::ignoreUnused(e); }

    virtual bool hitTest(float x, float y)
    {
        return getLocalBounds().contains(x, y);
    }

    void repaint() { dirty = true; }

    // --- Underlying Object access ---
    Object* getObject() { return object; }
    Transform* getTransform() { return object ? object->transform : nullptr; }

    bool isDirty() const { return dirty; }
    void clearDirty() { dirty = false; }

private:
    friend class ComponentRenderer;

    Object* object = nullptr;
    Component* parent = nullptr;
    std::vector<Component*> children;
    juce::Rectangle<float> bounds;
    int zOrder = 0;
    bool visible = true;
    float alpha = 1.0f;
    bool dirty = true;
};

} // jvk
